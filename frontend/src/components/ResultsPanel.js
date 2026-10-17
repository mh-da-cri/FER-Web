"use client";

/**
 * Component ResultsPanel — hiển thị kết quả phân tích biểu cảm của frame hiện tại
 * và tổng hợp thống kê toàn bộ phiên làm việc (Session Analytics).
 */

const EMOTION_CONFIG = {
  happy:    { emoji: "😄", label: "Vui vẻ", color: "#10b981" },
  sad:      { emoji: "😢", label: "Buồn", color: "#6366f1" },
  angry:    { emoji: "😠", label: "Tức giận", color: "#ef4444" },
  surprise: { emoji: "😲", label: "Ngạc nhiên", color: "#f59e0b" },
  fear:     { emoji: "😨", label: "Sợ hãi", color: "#8b5cf6" },
  disgust:  { emoji: "🤢", label: "Ghê tởm", color: "#84cc16" },
  neutral:  { emoji: "😐", label: "Bình thường", color: "#06b6d4" },
  fatigue:  { emoji: "😴", label: "Mệt mỏi", color: "#ec4899" },
};

function getTopEmotion(emotions) {
  if (!emotions || Object.keys(emotions).length === 0) return null;
  const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
  return { name: sorted[0][0], value: sorted[0][1] };
}

function formatDuration(seconds) {
  if (!seconds || seconds <= 0) return "00:00";
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

export default function ResultsPanel({ result }) {
  const faces = result?.faces || [];
  const hasFace = faces.some((f) => f.isFace);
  const mainFace = faces.find((f) => f.isFace);
  const fatigueScore = Number(mainFace?.emotions?.fatigue ?? mainFace?.fatigue_score ?? 0);
  const topEmotion = mainFace ? getTopEmotion(mainFace.emotions) : null;
  const shouldShowFatigueBadge = (topEmotion?.name === "fatigue") || (fatigueScore >= 0.35);
  const stats = result?.stats || null;

  return (
    <div className="sidebar">
      {/* ─── 1. Biểu cảm hiện tại ─── */}
      <div className="glass-card">
        <div className="glass-card__title">Biểu cảm hiện tại</div>

        {topEmotion ? (
          <div className="current-emotion" key={topEmotion.name}>
            <div className="current-emotion__icon">
              {EMOTION_CONFIG[topEmotion.name]?.emoji || "🤔"}
            </div>
            <div className="current-emotion__label">
              {EMOTION_CONFIG[topEmotion.name]?.label || topEmotion.name}
            </div>
            <div className="current-emotion__confidence">
              Độ tin cậy: {(topEmotion.value * 100).toFixed(1)}%
            </div>

            {/* Huy hiệu lý do cập nhật kết quả */}
            {result?.reportReason && (
              <div className="current-emotion__reason-badge">
                {result.reportReason === "emotion_changed" && "⚡ Đổi cảm xúc"}
                {result.reportReason === "10s_interval" && "⏱️ Định kỳ 10 giây"}
                {result.reportReason === "face_state_changed" && "👤 Đổi diện mạo"}
                {result.reportReason === "manual" && "📸 Chụp thủ công"}
                {result.reportReason === "initial" && "🚀 Khởi đầu"}
              </div>
            )}
          </div>
        ) : (
          <div className="current-emotion current-emotion--empty">
            <div className="current-emotion__icon">🔍</div>
            <div>Chưa có dữ liệu phân tích</div>
          </div>
        )}
      </div>

      {/* ─── 2. Thống kê phiên làm việc (Session Analytics) ─── */}
      {stats && stats.totalFrames > 0 && (
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
                  ? `${EMOTION_CONFIG[stats.dominantOverall.name]?.emoji || ""} ${
                      EMOTION_CONFIG[stats.dominantOverall.name]?.label || stats.dominantOverall.name
                    }`
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
                      emoji: "❓",
                      label: emotion,
                      color: "#94a3b8",
                    };
                    return (
                      <div className="session-dist-row" key={emotion}>
                        <div className="session-dist-row__header">
                          <span>
                            {cfg.emoji} {cfg.label}
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

      {/* ─── 3. Chi tiết biểu cảm frame hiện tại ─── */}
      {mainFace && mainFace.emotions && Object.keys(mainFace.emotions).length > 0 && (
        <div className="glass-card">
          <div className="glass-card__title">Chi tiết biểu cảm (Frame)</div>

          <div className="emotion-bars">
            {Object.entries(mainFace.emotions)
              .sort((a, b) => b[1] - a[1])
              .map(([emotion, value]) => {
                const config = EMOTION_CONFIG[emotion] || {
                  emoji: "❓",
                  label: emotion,
                  color: "#06b6d4",
                };
                const percent = (value * 100).toFixed(1);

                return (
                  <div className={`emotion-bar emotion-bar--${emotion}`} key={emotion}>
                    <div className="emotion-bar__header">
                      <span className="emotion-bar__name">
                        {config.emoji} {config.label}
                      </span>
                      <span className="emotion-bar__value">{percent}%</span>
                    </div>
                    <div className="emotion-bar__track">
                      <div
                        className="emotion-bar__fill"
                        style={{
                          width: `${Math.max(value * 100, 1)}%`,
                          backgroundColor: config.color,
                        }}
                      />
                    </div>
                  </div>
                );
              })}

          </div>
        </div>
      )}

      {/* ─── 4. Thông tin khuôn mặt ─── */}
      <div className="glass-card">
        <div className="glass-card__title">Phát hiện khuôn mặt</div>

        {hasFace ? (
          faces
            .filter((f) => f.isFace)
            .map((face, idx) => (
              <div className="face-info" key={idx}>
                <div className="face-info__icon face-info__icon--detected">✅</div>
                <div>
                  <div className="face-info__text">Khuôn mặt #{idx + 1}</div>
                  <div className="face-info__detail">
                    Tọa độ: [{face.box[0]}, {face.box[1]}] — Kích thước: {face.box[2]}×{face.box[3]}px
                  </div>
                </div>
              </div>
            ))
        ) : (
          <div className="face-info">
            <div className="face-info__icon face-info__icon--none">👤</div>
            <div>
              <div className="face-info__text">Chưa phát hiện khuôn mặt</div>
              <div className="face-info__detail">
                {result ? "Không có khuôn mặt trong khung hình" : "Bật camera hoặc nhấn chụp để bắt đầu"}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
