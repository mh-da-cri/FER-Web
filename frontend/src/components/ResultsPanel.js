"use client";

import { formatSessionDuration } from "@/utils/sessionTimeline";

/**
 * Component ResultsPanel — hiển thị kết quả phân tích biểu cảm của frame hiện tại
 * và tổng hợp thống kê toàn bộ phiên làm việc (Session Analytics).
 */

const EMOTION_CONFIG = {
  happy: { label: "Vui vẻ", color: "#10b981" },
  sad: { label: "Buồn", color: "#6366f1" },
  angry: { label: "Tức giận", color: "#ef4444" },
  surprise: { label: "Ngạc nhiên", color: "#f59e0b" },
  fear: { label: "Sợ hãi", color: "#8b5cf6" },
  disgust: { label: "Ghê tởm", color: "#84cc16" },
  neutral: { label: "Bình thường", color: "#06b6d4" },
};

function getTopEmotion(emotions) {
  if (!emotions || Object.keys(emotions).length === 0) return null;
  const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
  return { name: sorted[0][0], value: sorted[0][1] };
}

function formatDuration(seconds) {
  return formatSessionDuration(seconds);
}

export default function ResultsPanel({ result }) {
  const mainFace = result?.faces?.find((face) => face.isFace);
  const fatigueScore = Number(mainFace?.fatigue_score ?? 0);
  const topEmotion = mainFace ? getTopEmotion(mainFace.emotions) : null;
  const stats = result?.stats || null;
  const hasSessionStats = stats && stats.totalFrames > 0;

  return (
    <div className="sidebar">
      <div className={`results-overview${hasSessionStats ? " results-overview--with-stats" : ""}`}>
        <div className="results-overview__primary">
          {/* ─── Biểu cảm hiện tại ─── */}
          <div className="glass-card current-emotion-card">
            <div className="glass-card__title">Biểu cảm hiện tại</div>

            {topEmotion ? (
              <div className="current-emotion" key={topEmotion.name}>
                <div className="current-emotion__label">
                  {EMOTION_CONFIG[topEmotion.name]?.label || topEmotion.name}
                </div>
                <div className="current-emotion__confidence">
                  Độ tin cậy: {(topEmotion.value * 100).toFixed(1)}%
                </div>

                {/* Huy hiệu lý do cập nhật kết quả */}
                {result?.reportReason && (
                  <div className="current-emotion__reason-badge">
                    {result.reportReason === "emotion_changed" && "Đổi cảm xúc"}
                    {result.reportReason === "10s_interval" && "Định kỳ 10 giây"}
                    {result.reportReason === "face_state_changed" && "Đổi diện mạo"}
                    {result.reportReason === "manual" && "Chụp thủ công"}
                    {result.reportReason === "initial" && "Khởi đầu"}
                  </div>
                )}
              </div>
            ) : (
              <div className="current-emotion current-emotion--empty">
                <div>Chưa có dữ liệu phân tích</div>
              </div>
            )}
          </div>

          <div className="glass-card status-card">
            <div className="glass-card__title">Trạng thái</div>
            <div className="emotion-bar__header">
              <span className="emotion-bar__name">Mệt mỏi</span>
              <span className="emotion-bar__value">{(fatigueScore * 100).toFixed(1)}%</span>
            </div>
            <div className="emotion-bar__track" role="meter" aria-label="Mức độ mệt mỏi" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(fatigueScore * 100)}>
              <div
                className="emotion-bar__fill"
                style={{
                  width: `${Math.max(fatigueScore * 100, 0)}%`,
                  backgroundColor: "#ec4899",
                }}
              />
            </div>
          </div>
        </div>

        {/* ─── Thống kê phiên làm việc (Session Analytics) ─── */}
        {hasSessionStats && (
          <div className="glass-card session-stats-card">
            <div className="glass-card__title">Thống kê phiên làm việc</div>

            <div className="session-stats-grid">
              <div className="stat-item">
                <span className="stat-item__label">Thời gian</span>
                <span className="stat-item__value">{formatDuration(stats.duration)}</span>
              </div>
              <div className="stat-item">
                <span className="stat-item__label">Frames</span>
                <span className="stat-item__value">#{stats.totalFrames}</span>
              </div>
              <div className="stat-item">
                <span className="stat-item__label">Chủ đạo</span>
                <span className="stat-item__value stat-item__value--highlight">
                  {stats.dominantOverall
                    ? EMOTION_CONFIG[stats.dominantOverall.name]?.label || stats.dominantOverall.name
                    : "N/A"}
                </span>
              </div>
            </div>

            {/* Phân bổ tỷ lệ cảm xúc toàn phiên */}
            {stats.emotionPercentages && Object.keys(stats.emotionPercentages).length > 0 && (
              <div className="session-distribution">
                <div className="session-distribution__title">Phân bổ cảm xúc toàn phiên:</div>
                <div className="session-distribution__list">
                  {Object.entries(stats.emotionPercentages)
                    .sort((a, b) => b[1] - a[1])
                    .map(([emotion, pct]) => {
                      const cfg = EMOTION_CONFIG[emotion] || {
                        label: emotion,
                        color: "#94a3b8",
                      };
                      return (
                        <div className="session-dist-row" key={emotion}>
                          <div className="session-dist-row__header">
                            <span>
                              {cfg.label}
                            </span>
                            <span className="session-dist-row__pct">{pct}%</span>
                          </div>
                          <div className="session-dist-row__track">
                            <div
                              className="session-dist-row__fill"
                              style={{
                                width: `${Math.max(pct, 1)}%`,
                                backgroundColor: cfg.color,
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
