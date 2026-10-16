"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import { sessionTimeline } from "@/utils/sessionTimeline";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// Tối ưu hóa kích thước frame gửi lên backend: max width 640px để giảm độ trễ mạng
const CAPTURE_MAX_WIDTH = 640;

// Chu kỳ cập nhật tối đa nếu cảm xúc không thay đổi (giây)
const REPORT_INTERVAL_SECONDS = 10;

/**
 * Component CameraStream — quản lý luồng webcam, canvas overlay,
 * vòng lặp nhận diện realtime (cách 5 frame capture 1 lần),
 * VÀ cơ chế phát kết quả thông minh: "Cứ 10 giây HOẶC khi phát hiện thay đổi cảm xúc mới đưa kết quả".
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

  // Trạng thái cơ chế đưa kết quả (10s / Đổi cảm xúc)
  const [secondsUntilNextReport, setSecondsUntilNextReport] = useState(REPORT_INTERVAL_SECONDS);
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

  // Refs phục vụ cơ chế "Cứ 10s hoặc khi đổi cảm xúc"
  const lastReportedEmotionRef = useRef(null);
  const lastReportedTimeRef = useRef(0);
  const isFirstReportRef = useRef(true);

  // Đồng bộ refs với state
  useEffect(() => {
    isCameraOnRef.current = isCameraOn;
  }, [isCameraOn]);

  useEffect(() => {
    isRealtimeRef.current = isRealtime;
  }, [isRealtime]);

  // ─── Khởi tạo / dừng đồng hồ đếm thời gian phiên & đếm ngược 10s ───
  useEffect(() => {
    if (isRealtime) {
      sessionTimeline.start();
      timerIntervalRef.current = setInterval(() => {
        setSessionTime(sessionTimeline.getElapsedSeconds());

        // Tính số giây còn lại cho đến mốc 10 giây định kỳ
        if (lastReportedTimeRef.current > 0) {
          const elapsedSec = (Date.now() - lastReportedTimeRef.current) / 1000;
          const remaining = Math.max(0, Math.ceil(REPORT_INTERVAL_SECONDS - elapsedSec));
          setSecondsUntilNextReport(remaining);
        } else {
          setSecondsUntilNextReport(REPORT_INTERVAL_SECONDS);
        }
      }, 500);
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

        const endpoint = `${API_BASE}/api/analyze${isFast ? "?fast=true" : ""}`;
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

          // 3. XÁC ĐỊNH CẢM XÚC CỦA FRAME HIỆN TẠI
          const mainFace = scaledFaces.find((f) => f.isFace);
          let currentDominant = null;
          if (mainFace && mainFace.emotions && Object.keys(mainFace.emotions).length > 0) {
            const sorted = Object.entries(mainFace.emotions).sort((a, b) => b[1] - a[1]);
            currentDominant = sorted[0][0]; // "happy", "sad", "neutral", ...
          }
          setCurrentTrackedEmotion(currentDominant);

          // 4. KIỂM TRA ĐIỀU KIỆN ĐƯA RA KẾT QUẢ:
          //    "Cứ cách 10 giây HOẶC khi phát hiện thay đổi cảm xúc mới đưa kết quả"
          const isManual = !isFast;
          // Luôn ghi nhận ngay mốc khởi đầu khi phát hiện mặt lần đầu hoặc timeline chưa có dữ liệu
          const isFirst = (isFirstReportRef.current || sessionTimeline.getTimeline().length === 0) && !!mainFace;

          // Thay đổi cảm xúc: cảm xúc hiện tại khác với cảm xúc lần report gần nhất
          const isEmotionChanged =
            currentDominant !== null &&
            lastReportedEmotionRef.current !== null &&
            currentDominant !== lastReportedEmotionRef.current;

          // Thay đổi trạng thái có mặt / mất mặt
          const isFaceStateChanged =
            (mainFace && lastReportedEmotionRef.current === "no_face") ||
            (!mainFace &&
              lastReportedEmotionRef.current !== null &&
              lastReportedEmotionRef.current !== "no_face");

          // Định kỳ 10 giây
          const timeSinceLastReportSec =
            lastReportedTimeRef.current > 0
              ? (now - lastReportedTimeRef.current) / 1000
              : 999;
          const is10SecondsElapsed = timeSinceLastReportSec >= REPORT_INTERVAL_SECONDS;

          const shouldReport =
            isManual || isFirst || isEmotionChanged || isFaceStateChanged || is10SecondsElapsed;

          if (shouldReport) {
            let reason = "10s_interval";
            if (isManual) reason = "manual";
            else if (isFirst) reason = "initial";
            else if (isEmotionChanged) reason = "emotion_changed";
            else if (isFaceStateChanged) reason = "face_state_changed";

            lastReportedEmotionRef.current = mainFace ? currentDominant : "no_face";
            lastReportedTimeRef.current = now;
            isFirstReportRef.current = false;
            setSecondsUntilNextReport(REPORT_INTERVAL_SECONDS);

            // Lưu vào Session Timeline Store
            const entry = sessionTimeline.addEntry(scaledResult, { reason });

            // Đưa kết quả ra ngoài component cha (ResultsPanel)
            onResult?.({
              ...scaledResult,
              timelineEntry: entry,
              stats: sessionTimeline.getStats(),
              reportReason: reason,
              reportedEmotion: currentDominant,
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
    setSecondsUntilNextReport(REPORT_INTERVAL_SECONDS);

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
    if (reason === "emotion_changed") return "Đổi cảm xúc";
    if (reason === "10s_interval") return "Định kỳ 10s";
    if (reason === "initial") return "Khởi đầu";
    if (reason === "face_state_changed") return "Đổi diện mạo";
    if (reason === "manual") return "Thủ công";
    return "";
  };

  return (
    <div className="glass-card camera-panel">
      {/* ─── Header Camera & Chế độ ─── */}
      <div className="camera-panel__header">
        <div className="glass-card__title" style={{ margin: 0 }}>
          Camera Feed
        </div>

        {/* Toggle Realtime */}
        {isCameraOn && (
          <div className="realtime-toggle-wrapper">
            <label className="switch-label" htmlFor="realtime-toggle">
              <span className="switch-title">Tự động (Đổi cảm xúc / 10s)</span>
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
              TỰ ĐỘNG (5F)
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
            <div className="hud-badge" title="Thời gian phiên">
              <span className="hud-badge__label">⏱️</span>
              <span className="hud-badge__value">{formatTime(sessionTime)}</span>
            </div>

            {/* Đếm ngược 10 giây */}
            <div
              className="hud-badge hud-badge--timer"
              title="Sẽ đưa kết quả sau số giây này nếu không đổi cảm xúc (hoặc đưa ngay khi đổi cảm xúc)"
            >
              <span className="hud-badge__label">Đổi / 10s:</span>
              <span className="hud-badge__value">{secondsUntilNextReport}s</span>
            </div>

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
              {isRealtime ? "Tự động (10s / Đổi cảm xúc)" : "Chụp & Phân tích"}
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
