"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { sessionTimeline } from "@/utils/sessionTimeline";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// Tối ưu hóa kích thước frame gửi lên backend: max width 640px để giảm độ trễ mạng
const CAPTURE_MAX_WIDTH = 640;

// Chu kỳ ghi nhận định kỳ nếu KHÔNG thay đổi cảm xúc: 10 giây
const HEARTBEAT_INTERVAL_SECONDS = 10;

// Thời gian nhận diện xác nhận nếu CÓ thay đổi cảm xúc: 3 giây
const EMOTION_CHANGE_CONFIRM_SECONDS = 3;

const EMOTION_META = {
  happy:    { emoji: "😄", label: "Vui vẻ" },
  sad:      { emoji: "😢", label: "Buồn" },
  angry:    { emoji: "😠", label: "Tức giận" },
  surprise: { emoji: "😲", label: "Ngạc nhiên" },
  fear:     { emoji: "😨", label: "Sợ hãi" },
  disgust:  { emoji: "🤢", label: "Ghê tởm" },
  neutral:  { emoji: "😐", label: "Bình thường" },
  no_face:  { emoji: "👤", label: "Mất mặt" },
};

/**
 * Component CameraStream — quản lý luồng webcam, canvas overlay,
 * vòng lặp nhận diện realtime (cách 5 frame capture 1 lần),
 * VÀ cơ chế phát kết quả:
 *   - Định kỳ 10 giây nếu KHÔNG thay đổi cảm xúc
 *   - Thời gian nhận diện 3 giây nếu CÓ thay đổi cảm xúc
 */
export default function CameraStream({ onResult, onStatusChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const offscreenCanvasRef = useRef(null);

  // Trạng thái Camera & Realtime
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isRealtime, setIsRealtime] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [snapshot, setSnapshot] = useState(null);

  // Thống kê Realtime & Session
  const [analyzedCount, setAnalyzedCount] = useState(0);
  const [reportedCount, setReportedCount] = useState(0);
  const [droppedCount, setDroppedCount] = useState(0);
  const [realtimeFps, setRealtimeFps] = useState(0);
  const [sessionTime, setSessionTime] = useState(0);

  // Trạng thái chu kỳ 10s không đổi & xác nhận đổi cảm xúc 3s
  const [secondsUntilHeartbeat, setSecondsUntilHeartbeat] = useState(HEARTBEAT_INTERVAL_SECONDS);
  const [candidateInfo, setCandidateInfo] = useState(null); // { emotion: string, remaining: number }
  const [lastReportReason, setLastReportReason] = useState(null);
  const [currentTrackedEmotion, setCurrentTrackedEmotion] = useState(null);

  // Refs điều khiển vòng lặp để tránh closure stale state
  const isCameraOnRef = useRef(false);
  const isRealtimeRef = useRef(false);
  const isProcessingRef = useRef(false);
  const videoFrameCounterRef = useRef(0);
  const frameCallbackIdRef = useRef(null);
  const lastAnalysisTimeRef = useRef(Date.now());
  const timerIntervalRef = useRef(null);

  // Refs phục vụ cơ chế ghi nhận: 10s không đổi / 3s đổi cảm xúc
  const lastReportedEmotionRef = useRef(null);
  const lastReportedTimeRef = useRef(0);
  const candidateEmotionRef = useRef(null);
  const candidateStartTimeRef = useRef(0);
  const isFirstReportRef = useRef(true);
  const windowEmotionsRef = useRef([]);
  const recentFramesRef = useRef([]);

  // Đồng bộ refs với state
  useEffect(() => {
    isCameraOnRef.current = isCameraOn;
  }, [isCameraOn]);

  useEffect(() => {
    isRealtimeRef.current = isRealtime;
  }, [isRealtime]);

  // ─── Khởi tạo / dừng đồng hồ đếm thời gian phiên & đếm ngược ───
  useEffect(() => {
    if (isRealtime) {
      sessionTimeline.start();
      timerIntervalRef.current = setInterval(() => {
        setSessionTime(sessionTimeline.getElapsedSeconds());

        const now = Date.now();

        // 1. Đếm ngược chu kỳ 10 giây khi không thay đổi cảm xúc
        if (lastReportedTimeRef.current > 0) {
          const elapsedSec = (now - lastReportedTimeRef.current) / 1000;
          const remaining = Math.max(0, Math.ceil(HEARTBEAT_INTERVAL_SECONDS - elapsedSec));
          setSecondsUntilHeartbeat(remaining);
        } else {
          setSecondsUntilHeartbeat(HEARTBEAT_INTERVAL_SECONDS);
        }

        // 2. Đếm ngược 3 giây xác nhận khi phát hiện đổi cảm xúc
        if (
          candidateStartTimeRef.current > 0 &&
          candidateEmotionRef.current &&
          candidateEmotionRef.current !== "no_face"
        ) {
          const candElapsed = (now - candidateStartTimeRef.current) / 1000;
          const candRemaining = Math.max(
            0,
            Math.ceil(EMOTION_CHANGE_CONFIRM_SECONDS - candElapsed)
          );
          setCandidateInfo({
            emotion: candidateEmotionRef.current,
            remaining: candRemaining,
          });
        } else {
          setCandidateInfo(null);
        }
      }, 200);
    } else {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    }

    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    };
  }, [isRealtime]);

  // ─── Vẽ bounding box & nhãn cảm xúc lên canvas ───
  const drawBoundingBoxes = useCallback((faces) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face) => {
      if (!face.isFace || !face.box || face.box.length < 4) return;

      const [x, y, w, h] = face.box;

      // Vẽ khung bao góc neon công nghệ cao
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 3;
      ctx.shadowColor = "rgba(6, 182, 212, 0.6)";
      ctx.shadowBlur = 12;

      const cornerLen = Math.min(w, h) * 0.22;

      ctx.beginPath();
      // Góc trên-trái
      ctx.moveTo(x, y + cornerLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLen, y);
      // Góc trên-phải
      ctx.moveTo(x + w - cornerLen, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerLen);
      // Góc dưới-phải
      ctx.moveTo(x + w, y + h - cornerLen);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w - cornerLen, y + h);
      // Góc dưới-trái
      ctx.moveTo(x + cornerLen, y + h);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x, y + h - cornerLen);
      ctx.stroke();

      // Tắt shadow để vẽ chữ sắc nét
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;

      // Nhãn cảm xúc chính
      if (face.emotions && Object.keys(face.emotions).length > 0) {
        const topEmotion = Object.entries(face.emotions).sort((a, b) => b[1] - a[1])[0];
        const label = `${topEmotion[0].toUpperCase()} ${(topEmotion[1] * 100).toFixed(0)}%`;

        ctx.font = "600 13px Inter, sans-serif";
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;

        const paddingH = 8;
        const labelHeight = 24;
        const labelY = Math.max(0, y - labelHeight - 6);

        // Nền nhãn có bo góc
        ctx.fillStyle = "rgba(6, 182, 212, 0.9)";
        ctx.beginPath();
        ctx.roundRect(x, labelY, textWidth + paddingH * 2, labelHeight, 6);
        ctx.fill();

        // Chữ
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, x + paddingH, labelY + 16);
      }
    });
  }, []);

  // ─── Hàm chụp và gửi frame phân tích ───
  const processFrame = useCallback(
    async (isFast = false) => {
      const video = videoRef.current;
      if (!video || !isCameraOnRef.current || video.videoWidth === 0) {
        return;
      }

      isProcessingRef.current = true;
      if (!isFast) setIsLoading(true);

      try {
        if (!offscreenCanvasRef.current) {
          offscreenCanvasRef.current = document.createElement("canvas");
        }
        const offCanvas = offscreenCanvasRef.current;

        const vw = video.videoWidth;
        const vh = video.videoHeight;

        // Downscale tối ưu kích thước truyền mạng
        const scale = Math.min(1, CAPTURE_MAX_WIDTH / vw);
        const captureW = Math.round(vw * scale);
        const captureH = Math.round(vh * scale);

        if (offCanvas.width !== captureW || offCanvas.height !== captureH) {
          offCanvas.width = captureW;
          offCanvas.height = captureH;
        }

        const offCtx = offCanvas.getContext("2d");
        offCtx.drawImage(video, 0, 0, captureW, captureH);

        // Chuyển sang Blob JPEG chất lượng 0.75
        const blob = await new Promise((resolve) =>
          offCanvas.toBlob(resolve, "image/jpeg", 0.75)
        );

        if (!blob) return;

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        const endpoint = `${API_BASE}/api/analyze?fast=${isFast ? "true" : "false"}`;
        const response = await fetch(endpoint, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
          // Quy đổi tọa độ bounding box ngược lại hệ tọa độ video gốc
          const scaleBackX = vw / captureW;
          const scaleBackY = vh / captureH;

          const scaledFaces = (result.faces || []).map((f) => {
            if (!f.isFace || !f.box || f.box.length < 4) return f;
            return {
              ...f,
              box: [
                Math.round(f.box[0] * scaleBackX),
                Math.round(f.box[1] * scaleBackY),
                Math.round(f.box[2] * scaleBackX),
                Math.round(f.box[3] * scaleBackY),
              ],
            };
          });

          const scaledResult = { ...result, faces: scaledFaces };

          // 1. Luôn vẽ khung bounding box để theo dõi mượt mà trên video
          drawBoundingBoxes(scaledFaces);

          // 2. Cập nhật FPS thực tế của vòng lặp phân tích
          const now = Date.now();
          const deltaSec = (now - lastAnalysisTimeRef.current) / 1000;
          lastAnalysisTimeRef.current = now;
          if (deltaSec > 0) {
            const currentFps = Math.round((1 / deltaSec) * 10) / 10;
            setRealtimeFps(currentFps);
          }
          setAnalyzedCount((prev) => prev + 1);

          // 3. XÁC ĐỊNH CẢM XÚC CỦA FRAME HIỆN TẠI & TÍCH LŨY BUFFER
          const mainFace = scaledFaces.find((f) => f.isFace);
          let currentDominant = null;
          if (mainFace && mainFace.emotions && Object.keys(mainFace.emotions).length > 0) {
            const sorted = Object.entries(mainFace.emotions).sort((a, b) => b[1] - a[1]);
            currentDominant = sorted[0][0]; // "happy", "sad", "neutral", ...
            windowEmotionsRef.current.push(mainFace.emotions);
            if (windowEmotionsRef.current.length > 50) {
              windowEmotionsRef.current.shift();
            }
          }
          setCurrentTrackedEmotion(currentDominant);

          // Cập nhật recentFramesRef để smooth nhiễu (lưu 4 frame gần nhất)
          const frameLabel = mainFace ? currentDominant : "no_face";
          recentFramesRef.current.push(frameLabel);
          if (recentFramesRef.current.length > 4) {
            recentFramesRef.current.shift();
          }

          // Lấy nhãn cảm xúc chiếm đa số trong 4 frame gần nhất để chống nhiễu
          const counts = {};
          recentFramesRef.current.forEach((em) => {
            counts[em] = (counts[em] || 0) + 1;
          });
          const activeDominant = Object.entries(counts).sort((a, b) => b[1] - a[1])[0][0];

          // 4. KIỂM TRA ĐIỀU KIỆN GHI NHẬN KẾT QUẢ:
          //    - Chu kỳ 10 giây nếu KHÔNG thay đổi cảm xúc
          //    - Thời gian nhận diện 3 giây nếu CÓ thay đổi cảm xúc
          const isManual = !isFast;
          const isFirst =
            (isFirstReportRef.current || sessionTimeline.getTimeline().length === 0) &&
            !!mainFace;

          let shouldReport = false;
          let reason = "";

          if (isManual) {
            shouldReport = true;
            reason = "manual";
          } else if (isFirst) {
            shouldReport = true;
            reason = "initial";
          } else {
            const lastEmotion = lastReportedEmotionRef.current;

            if (activeDominant === lastEmotion) {
              // A. CẢM XÚC KHÔNG ĐỔI -> Hủy candidate, kiểm tra chu kỳ 10 giây
              candidateEmotionRef.current = null;
              candidateStartTimeRef.current = 0;

              const timeSinceLastReportSec =
                lastReportedTimeRef.current > 0
                  ? (now - lastReportedTimeRef.current) / 1000
                  : 999;

              if (timeSinceLastReportSec >= HEARTBEAT_INTERVAL_SECONDS) {
                shouldReport = true;
                reason = "10s_interval";
              }
            } else {
              // B. CÓ THAY ĐỔI CẢM XÚC -> Cần duy trì nhận diện đủ 3 giây
              if (candidateEmotionRef.current !== activeDominant) {
                // Bắt đầu ứng cử viên cảm xúc mới
                candidateEmotionRef.current = activeDominant;
                candidateStartTimeRef.current = now;
              } else {
                // Cảm xúc mới đang được duy trì
                const candidateDurationSec = (now - candidateStartTimeRef.current) / 1000;
                if (candidateDurationSec >= EMOTION_CHANGE_CONFIRM_SECONDS) {
                  // ĐÃ ĐẠT ĐỦ 3 GIÂY XÁC NHẬN!
                  shouldReport = true;
                  reason = "emotion_changed";
                }
              }
            }
          }

          if (shouldReport) {
            // TÍNH TOÁN CẢM XÚC ỔN ĐỊNH TỪ BUFFER CHO MOBILENETV2
            let finalFaces = scaledFaces;
            let reportedDominant = activeDominant === "no_face" ? null : activeDominant;

            if (windowEmotionsRef.current.length > 0 && mainFace) {
              const buffer = windowEmotionsRef.current;
              // Nếu là emotion_changed, lấy các frame trong 3 giây vừa xác nhận (~18 frame gần nhất)
              const sampleFrames = reason === "emotion_changed" ? buffer.slice(-18) : buffer;
              const avgEmotions = {};
              const keys = Object.keys(sampleFrames[0]);
              keys.forEach((k) => {
                const sum = sampleFrames.reduce((acc, cur) => acc + (cur[k] || 0), 0);
                avgEmotions[k] = Math.round((sum / sampleFrames.length) * 10000) / 10000;
              });

              // Chuẩn hóa tổng xác suất = 1.0
              const totalSum = Object.values(avgEmotions).reduce((a, b) => a + b, 0);
              if (totalSum > 0) {
                Object.keys(avgEmotions).forEach((k) => {
                  avgEmotions[k] = Math.round((avgEmotions[k] / totalSum) * 10000) / 10000;
                });
              }

              const sortedAvg = Object.entries(avgEmotions).sort((a, b) => b[1] - a[1]);
              reportedDominant = sortedAvg[0][0];

              finalFaces = scaledFaces.map((f) =>
                f.isFace ? { ...f, emotions: avgEmotions } : f
              );
            }

            // Dọn sạch buffer & candidate cho chu kỳ tiếp theo
            windowEmotionsRef.current = [];
            candidateEmotionRef.current = null;
            candidateStartTimeRef.current = 0;
            setCandidateInfo(null);

            lastReportedEmotionRef.current = mainFace ? reportedDominant : "no_face";
            lastReportedTimeRef.current = now;
            isFirstReportRef.current = false;
            setSecondsUntilHeartbeat(HEARTBEAT_INTERVAL_SECONDS);

            const finalResult = { ...scaledResult, faces: finalFaces };

            // Lưu vào Session Timeline Store
            const entry = sessionTimeline.addEntry(finalResult, { reason });

            // Đưa kết quả ra ngoài component cha (ResultsPanel)
            onResult?.({
              ...finalResult,
              timelineEntry: entry,
              stats: sessionTimeline.getStats(),
              reportReason: reason,
              reportedEmotion: reportedDominant,
            });

            setReportedCount((prev) => prev + 1);
            setLastReportReason(reason);
          }
        }
      } catch (err) {
        console.error("[CameraStream] Lỗi phân tích frame:", err);
      } finally {
        isProcessingRef.current = false;
        if (!isFast) setIsLoading(false);
      }
    },
    [drawBoundingBoxes, onResult]
  );

  // ─── Vòng lặp Realtime: Cách 5 frame capture 1 lần ───
  const scheduleNextFrame = useCallback(() => {
    const video = videoRef.current;
    if (!video || !isCameraOnRef.current || !isRealtimeRef.current) {
      return;
    }

    const onFrame = () => {
      if (!isCameraOnRef.current || !isRealtimeRef.current) return;

      videoFrameCounterRef.current += 1;

      // Kiểm tra chu kỳ đúng 5 frame
      if (videoFrameCounterRef.current % 5 === 0) {
        if (isProcessingRef.current) {
          // Frame trước đang bận xử lý -> Drop frame để tránh nghẽn luồng
          setDroppedCount((prev) => prev + 1);
        } else {
          // Xử lý frame với chế độ fast realtime
          processFrame(true);
        }
      }

      scheduleNextFrame();
    };

    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      frameCallbackIdRef.current = video.requestVideoFrameCallback(onFrame);
    } else {
      frameCallbackIdRef.current = requestAnimationFrame(onFrame);
    }
  }, [processFrame]);

  // Bắt đầu hoặc dừng vòng lặp Realtime khi toggle thay đổi
  useEffect(() => {
    if (isRealtime && isCameraOn) {
      scheduleNextFrame();
    } else {
      const video = videoRef.current;
      if (frameCallbackIdRef.current) {
        if (
          video &&
          "cancelVideoFrameCallback" in HTMLVideoElement.prototype &&
          typeof frameCallbackIdRef.current === "number"
        ) {
          video.cancelVideoFrameCallback(frameCallbackIdRef.current);
        } else {
          cancelAnimationFrame(frameCallbackIdRef.current);
        }
        frameCallbackIdRef.current = null;
      }
    }
  }, [isRealtime, isCameraOn, scheduleNextFrame]);

  // ─── Bật Camera ───
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user",
        },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      streamRef.current = stream;
      setIsCameraOn(true);
      onStatusChange?.("live");
    } catch (err) {
      console.error("Không thể truy cập camera:", err);
      onStatusChange?.("error");
    }
  }, [onStatusChange]);

  // ─── Tắt Camera ───
  const stopCamera = useCallback(() => {
    setIsRealtime(false);

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    windowEmotionsRef.current = [];
    recentFramesRef.current = [];
    candidateEmotionRef.current = null;
    candidateStartTimeRef.current = 0;
    setCandidateInfo(null);
    setIsCameraOn(false);
    onStatusChange?.("off");

    // Xóa canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [onStatusChange]);

  // Cleanup khi unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // ─── Chụp ảnh thủ công (Manual Snapshot) ───
  const handleManualCapture = useCallback(async () => {
    if (!videoRef.current || !isCameraOn || isLoading) return;

    // Lưu snapshot preview
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = videoRef.current.videoWidth;
    tempCanvas.height = videoRef.current.videoHeight;
    const ctx = tempCanvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0);
    setSnapshot(tempCanvas.toDataURL("image/jpeg", 0.9));

    await processFrame(false);
  }, [isCameraOn, isLoading, processFrame]);

  // ─── Đặt lại phiên (Reset Session) ───
  const handleResetSession = useCallback(() => {
    sessionTimeline.reset();
    setAnalyzedCount(0);
    setReportedCount(0);
    setDroppedCount(0);
    setSessionTime(0);
    setSnapshot(null);
    setLastReportReason(null);
    setCurrentTrackedEmotion(null);
    setCandidateInfo(null);
    setSecondsUntilHeartbeat(HEARTBEAT_INTERVAL_SECONDS);

    windowEmotionsRef.current = [];
    recentFramesRef.current = [];
    candidateEmotionRef.current = null;
    candidateStartTimeRef.current = 0;
    lastReportedEmotionRef.current = null;
    lastReportedTimeRef.current = 0;
    isFirstReportRef.current = true;

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
    onResult?.(null);
  }, [onResult]);

  // Định dạng thời gian hiển thị (00:00)
  const formatTime = (totalSeconds) => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = Math.floor(totalSeconds % 60);
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  // Diễn giải lý do đưa kết quả
  const getReasonText = (reason) => {
    if (reason === "10s_interval") return "Định kỳ 10s (không đổi)";
    if (reason === "emotion_changed") return "Đổi cảm xúc (sau 3s)";
    if (reason === "initial") return "Khởi đầu";
    if (reason === "manual") return "Thủ công";
    return "";
  };

  return (
    <div className="glass-card camera-panel">
      {/* ─── Header Camera & Chế độ ─── */}
      <div className="camera-panel__header">
        <div className="camera-panel__header-left">
          <div className="glass-card__title" style={{ margin: 0 }}>
            Camera Feed
          </div>

          {/* Model Badge — chỉ hiển thị tên model */}
          <div className="model-selector-wrapper">
            <span className="model-selector-label">Mô hình:</span>
            <div className="model-pill-group" role="group" aria-label="Mô hình AI">
              <div className="model-pill model-pill--active" title="MobileNetV2: Nhận diện ảnh màu RGB 224×224 (7 nhãn cảm xúc)">
                <span className="model-pill__icon">⚡</span>
                <span className="model-pill__name">MobileNetV2</span>
              </div>
            </div>
          </div>
        </div>

        {/* Toggle Realtime */}
        {isCameraOn && (
          <div className="realtime-toggle-wrapper">
            <label className="switch-label" htmlFor="realtime-toggle">
              <span className="switch-title">Tự động (10s không đổi / 3s đổi)</span>
              <div className="switch-container">
                <input
                  type="checkbox"
                  id="realtime-toggle"
                  checked={isRealtime}
                  onChange={(e) => setIsRealtime(e.target.checked)}
                />
                <span className="switch-slider"></span>
              </div>
            </label>
          </div>
        )}
      </div>

      {/* ─── Video + Canvas Overlay ─── */}
      <div className="camera-wrapper">
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} />

        {/* Trạng thái Camera Badge */}
        {isCameraOn ? (
          isRealtime ? (
            <div className="camera-status camera-status--realtime">
              <span className="camera-status__dot pulse" />
              TỰ ĐỘNG (10s/3s)
            </div>
          ) : (
            <div className="camera-status camera-status--live">
              <span className="camera-status__dot" />
              LIVE
            </div>
          )
        ) : (
          <div className="camera-status camera-status--off">
            <span className="camera-status__dot" />
            OFF
          </div>
        )}

        {/* HUD Chỉ số Realtime trên video */}
        {isCameraOn && isRealtime && (
          <div className="camera-hud-stats">
            {/* Model Badge */}
            <div className="hud-badge hud-badge--model" title="Mô hình AI: MobileNetV2">
              <span className="hud-badge__label">AI:</span>
              <span className="hud-badge__value">MobileNetV2</span>
            </div>

            <div className="hud-badge" title="Thời gian phiên">
              <span className="hud-badge__label">⏱️</span>
              <span className="hud-badge__value">{formatTime(sessionTime)}</span>
            </div>

            {/* Đếm ngược chu kỳ: 10s không đổi HOẶC 3s đổi cảm xúc */}
            {candidateInfo && candidateInfo.emotion !== "no_face" ? (
              <div
                className="hud-badge hud-badge--candidate pulse"
                title="Đang xác nhận biểu cảm mới duy trì trong 3 giây để ghi nhận"
              >
                <span className="hud-badge__label">Đổi cảm xúc:</span>
                <span className="hud-badge__value">
                  {EMOTION_META[candidateInfo.emotion]?.emoji || "⚡"}{" "}
                  {EMOTION_META[candidateInfo.emotion]?.label || candidateInfo.emotion} ({candidateInfo.remaining}s/3s)
                </span>
              </div>
            ) : (
              <div
                className="hud-badge hud-badge--timer"
                title="Chu kỳ 10 giây nếu giữ nguyên cảm xúc"
              >
                <span className="hud-badge__label">Không đổi:</span>
                <span className="hud-badge__value">{secondsUntilHeartbeat}s/10s</span>
              </div>
            )}

            <div className="hud-badge" title="Số kết quả đã phát ra / Số frame đã quét">
              <span className="hud-badge__label">Kết quả:</span>
              <span className="hud-badge__value">
                #{reportedCount} <span style={{ opacity: 0.6, fontSize: "0.68rem" }}>({analyzedCount}f)</span>
              </span>
            </div>

            {lastReportReason && (
              <div className="hud-badge hud-badge--reason" title="Lý do đưa kết quả gần nhất">
                <span className="hud-badge__label">⚡</span>
                <span className="hud-badge__value">{getReasonText(lastReportReason)}</span>
              </div>
            )}

            {droppedCount > 0 && (
              <div className="hud-badge hud-badge--dropped" title="Frame bị bỏ qua để chống nghẽn">
                <span className="hud-badge__label">Drop:</span>
                <span className="hud-badge__value">{droppedCount}</span>
              </div>
            )}
          </div>
        )}

        {/* Placeholder khi camera tắt */}
        {!isCameraOn && (
          <div className="camera-placeholder">
            <div className="camera-placeholder__icon">📷</div>
            <div className="camera-placeholder__text">
              Nhấn &quot;Bật Camera&quot; để bắt đầu nhận diện
            </div>
          </div>
        )}
      </div>

      {/* ─── Thanh điều khiển ─── */}
      <div className="controls-bar">
        {!isCameraOn ? (
          <button className="btn btn--primary" onClick={startCamera} id="btn-start-camera">
            <span className="btn__icon">📹</span>
            Bật Camera
          </button>
        ) : (
          <button className="btn btn--danger" onClick={stopCamera} id="btn-stop-camera">
            <span className="btn__icon">⏹</span>
            Tắt Camera
          </button>
        )}

        {/* Nút Chụp ảnh thủ công */}
        <button
          className="btn btn--capture"
          onClick={handleManualCapture}
          disabled={!isCameraOn || isLoading || isRealtime}
          id="btn-capture"
          title={isRealtime ? "Chế độ Realtime đang tự động chạy" : "Chụp ảnh hiện tại"}
        >
          {isLoading ? (
            <span className="spinner">
              <span className="spinner__ring" />
              Đang phân tích...
            </span>
          ) : (
            <>
              <span className="btn__icon">📸</span>
              {isRealtime ? "Tự động (10s/3s)" : "Chụp & Phân tích"}
            </>
          )}
        </button>

        {/* Nút Reset Phiên */}
        {(reportedCount > 0 || sessionTime > 0) && (
          <button
            className="btn btn--outline"
            onClick={handleResetSession}
            id="btn-reset-session"
            title="Xóa dữ liệu phiên để bắt đầu phiên mới"
          >
            <span className="btn__icon">🔄</span>
            Reset Phiên
          </button>
        )}
      </div>

      {/* Snapshot preview (khi chụp thủ công) */}
      {snapshot && !isRealtime && (
        <div className="snapshot-preview">
          <div className="snapshot-preview__label">Ảnh chụp gần nhất:</div>
          <img src={snapshot} alt="Captured snapshot" />
        </div>
      )}
    </div>
  );
}
