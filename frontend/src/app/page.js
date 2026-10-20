"use client";

import { useState } from "react";
import CameraStream from "@/components/CameraStream";
import ResultsPanel from "@/components/ResultsPanel";
import ChatPanel from "@/components/ChatPanel";
import { sessionTimeline } from "@/utils/sessionTimeline";

const EMOTION_LABELS = {
  happy: { emoji: "😄", label: "Vui vẻ" },
  sad: { emoji: "😢", label: "Buồn" },
  angry: { emoji: "😠", label: "Tức giận" },
  surprise: { emoji: "😲", label: "Ngạc nhiên" },
  fear: { emoji: "😨", label: "Sợ hãi" },
  disgust: { emoji: "🤢", label: "Ghê tởm" },
  neutral: { emoji: "😐", label: "Bình thường" },
  fatigue: { emoji: "😴", label: "Mệt mỏi" },
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
    <div className="app-container app-container--three-cols">
      {/* ─── Header ─── */}
      <header className="app-header">
        <div className="app-header__badge">
          AI-Powered • Real-time Analysis & AI Chatbot
        </div>
        <h1 className="app-header__title">AI Camera</h1>
        <p className="app-header__subtitle">
          Bảng phân tích cảm xúc (Trái) • Camera giám sát (Giữa) • Trợ lý AI Chatbot (Phải)
        </p>
      </header>

      {/* ─── Main Content: Bố cục 3 cột riêng biệt ─── */}
      <main className="three-column-layout">
        {/* CỘT TRÁI: Bảng Phân Tích Kết Quả & Thống Kê Phiên */}
        <section className="layout-col layout-col--left" aria-label="Bảng phân tích cảm xúc">
          <div className="layout-col__header">
            <h2 className="layout-col__title">📊 Phân Tích Cảm Xúc</h2>
          </div>
          <ResultsPanel result={analysisResult} />
        </section>

        {/* CỘT GIỮA: Camera Stream & Điều Khiển */}
        <section className="layout-col layout-col--center" aria-label="Khung camera giám sát">
          <CameraStream
            onResult={setAnalysisResult}
            onStatusChange={setCameraStatus}
          />
        </section>

        {/* CỘT PHẢI: AI Chatbot Phân Tích & Tra Cứu */}
        <section className="layout-col layout-col--right" aria-label="AI Trợ lý Chatbot">
          <ChatPanel onJumpToTime={handleJumpToTime} />
        </section>
      </main>

      {/* ─── Modal Xem Lại Frame khi nhấp vào mốc thời gian ─── */}
      {selectedEntry && (
        <div className="review-modal-backdrop" onClick={() => setSelectedEntry(null)}>
          <div className="review-modal" onClick={(e) => e.stopPropagation()}>
            <div className="review-modal__header">
              <div className="review-modal__title">
                ⏱️ Xem lại Frame tại mốc <strong>[{selectedEntry.timestamp}s]</strong>
              </div>
              <button
                className="review-modal__close-btn"
                onClick={() => setSelectedEntry(null)}
              >
                ✕
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
                    {EMOTION_LABELS[selectedEntry.dominantEmotion]?.emoji || ""}{" "}
                    {EMOTION_LABELS[selectedEntry.dominantEmotion]?.label || selectedEntry.dominantEmotion}
                    {" "}({Math.round((selectedEntry.dominantConfidence || 0) * 100)}%)
                  </span>
                </div>
                <div className="review-info-item">
                  <span className="review-info-label">Lý do ghi nhận</span>
                  <span className="review-info-val">
                    {selectedEntry.triggerReason === "10s_interval"
                      ? "⏱️ Định kỳ 10s (không đổi)"
                      : selectedEntry.triggerReason === "emotion_changed"
                      ? "⚡ Đổi cảm xúc (xác nhận 3s)"
                      : selectedEntry.triggerReason === "face_state_changed"
                      ? "👤 Đổi diện mạo"
                      : selectedEntry.triggerReason === "initial"
                      ? "🚀 Khởi đầu"
                      : "📸 Thủ công"}
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
                      const cfg = EMOTION_LABELS[em] || { emoji: "•", label: em };
                      const pct = Math.round(val * 100);
                      return (
                        <div className="review-emotion-row" key={em}>
                          <span className="review-emotion-name">
                            {cfg.emoji} {cfg.label}
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
