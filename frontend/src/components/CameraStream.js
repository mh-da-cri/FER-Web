"use client";

import { useRef, useState, useCallback, useEffect } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

/**
 * Component CameraStream — quản lý luồng webcam, canvas overlay,
 * chụp ảnh và gửi lên backend để phân tích biểu cảm.
 */
export default function CameraStream({ onResult, onStatusChange }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [snapshot, setSnapshot] = useState(null);

  // ─── Bật Camera ───
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
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
      console.error("Khong the truy cap camera:", err);
      onStatusChange?.("error");
    }
  }, [onStatusChange]);

  // ─── Tắt Camera ───
  const stopCamera = useCallback(() => {
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

  // ─── Cleanup khi unmount ───
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  // ─── Vẽ bounding box lên canvas ───
  const drawBoundingBoxes = useCallback((faces) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    faces.forEach((face) => {
      if (!face.isFace) return;

      const [x, y, w, h] = face.box;

      // Vẽ khung bao với góc bo tròn
      ctx.strokeStyle = "#06b6d4";
      ctx.lineWidth = 3;
      ctx.shadowColor = "rgba(6, 182, 212, 0.5)";
      ctx.shadowBlur = 10;

      // Vẽ 4 góc thay vì hình chữ nhật đầy đủ — phong cách hiện đại
      const cornerLen = Math.min(w, h) * 0.2;

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

      // Reset shadow
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;

      // Hiển thị biểu cảm chính phía trên khung
      if (face.emotions && Object.keys(face.emotions).length > 0) {
        const topEmotion = Object.entries(face.emotions).sort((a, b) => b[1] - a[1])[0];
        const label = `${topEmotion[0]} ${(topEmotion[1] * 100).toFixed(0)}%`;

        ctx.font = "bold 14px Inter, sans-serif";
        const textWidth = ctx.measureText(label).width;

        // Nền cho chữ
        ctx.fillStyle = "rgba(6, 182, 212, 0.85)";
        const padding = 6;
        const labelHeight = 22;
        const labelY = y - labelHeight - 4;

        ctx.beginPath();
        ctx.roundRect(x, labelY, textWidth + padding * 2, labelHeight, 4);
        ctx.fill();

        // Chữ
        ctx.fillStyle = "#fff";
        ctx.fillText(label, x + padding, labelY + 15);
      }
    });
  }, []);

  // ─── Chụp ảnh và gửi API ───
  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !isCameraOn) return;

    setIsLoading(true);

    try {
      // Chụp frame hiện tại từ video
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = videoRef.current.videoWidth;
      tempCanvas.height = videoRef.current.videoHeight;
      const tempCtx = tempCanvas.getContext("2d");
      tempCtx.drawImage(videoRef.current, 0, 0);

      // Tạo snapshot preview
      const dataUrl = tempCanvas.toDataURL("image/jpeg", 0.9);
      setSnapshot(dataUrl);

      // Chuyển sang blob và gửi API
      const blob = await new Promise((resolve) =>
        tempCanvas.toBlob(resolve, "image/jpeg", 0.9)
      );

      const formData = new FormData();
      formData.append("file", blob, "capture.jpg");

      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (result.success) {
        onResult?.(result);
        drawBoundingBoxes(result.faces);
      } else {
        onResult?.({ success: false, message: result.detail || "Loi phan tich", faces: [] });
      }
    } catch (err) {
      console.error("Loi khi phan tich:", err);
      onResult?.({ success: false, message: "Khong the ket noi den server", faces: [] });
    } finally {
      setIsLoading(false);
    }
  }, [isCameraOn, onResult, drawBoundingBoxes]);

  return (
    <div className="glass-card">
      <div className="glass-card__title">Camera Feed</div>

      {/* Video + Canvas Overlay */}
      <div className="camera-wrapper">
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} />

        {/* Camera status indicator */}
        {isCameraOn ? (
          <div className="camera-status camera-status--live">
            <span className="camera-status__dot" />
            LIVE
          </div>
        ) : (
          <div className="camera-status camera-status--off">
            <span className="camera-status__dot" />
            OFF
          </div>
        )}

        {/* Placeholder khi camera tắt */}
        {!isCameraOn && (
          <div className="camera-placeholder">
            <div className="camera-placeholder__icon">📷</div>
            <div className="camera-placeholder__text">
              Nhấn &quot;Bật Camera&quot; để bắt đầu
            </div>
          </div>
        )}
      </div>

      {/* Nút điều khiển */}
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

        <button
          className="btn btn--capture"
          onClick={captureAndAnalyze}
          disabled={!isCameraOn || isLoading}
          id="btn-capture"
        >
          {isLoading ? (
            <span className="spinner">
              <span className="spinner__ring" />
              Đang phân tích...
            </span>
          ) : (
            <>
              <span className="btn__icon">📸</span>
              Chụp & Phân tích
            </>
          )}
        </button>
      </div>

      {/* Snapshot preview */}
      {snapshot && (
        <div className="snapshot-preview">
          <img src={snapshot} alt="Captured snapshot" />
        </div>
      )}
    </div>
  );
}
