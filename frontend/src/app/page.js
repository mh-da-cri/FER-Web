"use client";

import { useState } from "react";
import CameraStream from "@/components/CameraStream";
import ResultsPanel from "@/components/ResultsPanel";
import ChatPanel from "@/components/ChatPanel";
import { sessionTimeline } from "@/utils/sessionTimeline";

const EMOTION_LABELS = {
  happy: { label: "Vui vẻ" },
  sad: { label: "Buồn" },
  angry: { label: "Tức giận" },
  surprise: { label: "Ngạc nhiên" },
  fear: { label: "Sợ hãi" },
  disgust: { label: "Ghê tởm" },
  neutral: { label: "Bình thường" },
  fatigue: { label: "Mệt mỏi" },
};

export default function HomePage() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [cameraStatus, setCameraStatus] = useState("off");
  const [selectedEntry, setSelectedEntry] = useState(null); // Frame xem lại khi bấm mốc thời gian

  // Xử lý khi bấm vào mốc thời gian từ ChatPanel
  const handleJumpToTime = (seconds) => {
    const res = sessionTimeline.getEntryAtSecond(seconds, 3.0);
    if (res && res.entry) {
      setSelectedEntry(res.entry);
    }
  };

  return (
    <div className="app-container app-container--analysis-chat">
      {/* ─── Header ─── */}
      <header className="app-header">
        <div className="app-header__badge">
          AI-Powered • Real-time Analysis & AI Chatbot
        </div>
        <h1 className="app-header__title">AI Camera</h1>
        <p className="app-header__subtitle">
          Camera giám sát • Phân tích cảm xúc và Trợ lý AI Chatbot
        </p>
      </header>

      {/* ─── Main Content: Camera bên trái, phân tích và chatbot bên phải ─── */}
      <main className="analysis-chat-layout">
        <section className="layout-col layout-col--camera" aria-label="Khung camera giám sát">
          <CameraStream
            onResult={setAnalysisResult}
            onStatusChange={setCameraStatus}
          />
        </section>

        <section className="layout-col layout-col--analysis-chat" aria-label="Phân tích cảm xúc và AI Chatbot">
          <ResultsPanel result={analysisResult} />
          <ChatPanel onJumpToTime={handleJumpToTime} />
        </section>
      </main>

      {/* ─── Modal Xem Lại Frame khi nhấp vào mốc thời gian ─── */}
      {selectedEntry && (
        <div className="review-modal-backdrop" onClick={() => setSelectedEntry(null)}>
          <div className="review-modal" onClick={(e) => e.stopPropagation()}>
            <div className="review-modal__header">
              <div className="review-modal__title">
                Xem lại Frame tại mốc <strong>[{selectedEntry.timestamp}s]</strong>
              </div>
              <button
                className="review-modal__close-btn"
                onClick={() => setSelectedEntry(null)}
              >
                Đóng
              </button>
            </div>

            <div className="review-modal__body">
              <div className="review-info-grid">
                <div className="review-info-item">
                  <span className="review-info-label">Frame Index</span>
                  <span className="review-info-val">#{selectedEntry.frameIndex}</span>
                </div>
                <div className="review-info-item">
                  <span className="review-info-label">Thời điểm</span>
                  <span className="review-info-val">{selectedEntry.timestamp}s</span>
                </div>
                <div className="review-info-item">
                  <span className="review-info-label">Biểu cảm chính</span>
                  <span className="review-info-val review-info-val--dominant">
                    {EMOTION_LABELS[selectedEntry.dominantEmotion]?.label || selectedEntry.dominantEmotion}
                    {" "}({Math.round((selectedEntry.dominantConfidence || 0) * 100)}%)
                  </span>
                </div>
                <div className="review-info-item">
                  <span className="review-info-label">Lý do ghi nhận</span>
                  <span className="review-info-val">
                    {selectedEntry.triggerReason === "emotion_changed"
                      ? "Đổi cảm xúc"
                      : selectedEntry.triggerReason === "10s_interval"
                        ? "Định kỳ 10s"
                        : "Thủ công"}
                  </span>
                </div>
              </div>

              {/* Chi tiết phân bổ tại frame này */}
              {selectedEntry.emotions && (
                <div className="review-emotion-list">
                  <div className="review-emotion-list__title">
                    Phân phối xác suất cảm xúc tại frame này:
                  </div>
                  {Object.entries(selectedEntry.emotions)
                    .sort((a, b) => b[1] - a[1])
                    .map(([em, val]) => {
                      const cfg = EMOTION_LABELS[em] || { label: em };
                      const pct = Math.round(val * 100);
                      return (
                        <div className="review-emotion-row" key={em}>
                          <span className="review-emotion-name">
                            {cfg.label}
                          </span>
                          <span className="review-emotion-pct">{pct}%</span>
                          <div className="review-emotion-bar-track">
                            <div
                              className="review-emotion-bar-fill"
                              style={{ width: `${Math.max(pct, 2)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>

            <div className="review-modal__footer">
              <button
                className="btn btn--primary"
                onClick={() => setSelectedEntry(null)}
              >
                Đóng
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
