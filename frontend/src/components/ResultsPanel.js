"use client";

/**
 * Component ResultsPanel — hiển thị kết quả phân tích biểu cảm
 * với thanh tiến trình animated cho từng cảm xúc.
 */

const EMOTION_CONFIG = {
  happy:    { emoji: "😄", label: "Vui vẻ" },
  sad:      { emoji: "😢", label: "Buồn" },
  angry:    { emoji: "😠", label: "Tức giận" },
  surprise: { emoji: "😲", label: "Ngạc nhiên" },
  fear:     { emoji: "😨", label: "Sợ hãi" },
  disgust:  { emoji: "🤢", label: "Ghê tởm" },
  neutral:  { emoji: "😐", label: "Bình thường" },
};

function getTopEmotion(emotions) {
  if (!emotions || Object.keys(emotions).length === 0) return null;
  const sorted = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
  return { name: sorted[0][0], value: sorted[0][1] };
}

export default function ResultsPanel({ result }) {
  const faces = result?.faces || [];
  const hasFace = faces.some((f) => f.isFace);
  const mainFace = faces.find((f) => f.isFace);
  const topEmotion = mainFace ? getTopEmotion(mainFace.emotions) : null;

  return (
    <div className="sidebar">
      {/* ─── Biểu cảm hiện tại ─── */}
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
          </div>
        ) : (
          <div className="current-emotion current-emotion--empty">
            <div className="current-emotion__icon">🔍</div>
            <div>Chưa có dữ liệu phân tích</div>
          </div>
        )}
      </div>

      {/* ─── Thông tin khuôn mặt ─── */}
      <div className="glass-card">
        <div className="glass-card__title">Phát hiện khuôn mặt</div>

        {hasFace ? (
          faces.filter((f) => f.isFace).map((face, idx) => (
            <div className="face-info" key={idx}>
              <div className="face-info__icon face-info__icon--detected">✅</div>
              <div>
                <div className="face-info__text">Khuôn mặt #{idx + 1}</div>
                <div className="face-info__detail">
                  Vị trí: x={face.box[0]}, y={face.box[1]} — {face.box[2]}×{face.box[3]}px
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="face-info">
            <div className="face-info__icon face-info__icon--none">👤</div>
            <div>
              <div className="face-info__text">Chưa phát hiện khuôn mặt</div>
              <div className="face-info__detail">Nhấn &quot;Chụp & Phân tích&quot; để bắt đầu</div>
            </div>
          </div>
        )}
      </div>

      {/* ─── Chi tiết cảm xúc ─── */}
      {mainFace && mainFace.emotions && Object.keys(mainFace.emotions).length > 0 && (
        <div className="glass-card">
          <div className="glass-card__title">Chi tiết biểu cảm</div>

          <div className="emotion-bars">
            {Object.entries(mainFace.emotions)
              .sort((a, b) => b[1] - a[1])
              .map(([emotion, value]) => {
                const config = EMOTION_CONFIG[emotion] || { emoji: "❓", label: emotion };
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
                        style={{ width: `${Math.max(value * 100, 1)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}
