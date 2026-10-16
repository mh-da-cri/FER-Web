/**
 * EmotionChatEngine — Bộ xử lý truy vấn dữ liệu cảm xúc chính xác 100%.
 * Chuyên giải quyết các câu hỏi:
 * - "ở giây thứ X cảm xúc gì"
 * - "bao nhiêu frame là vui"
 * - "tổng kết / phân tích phiên làm việc"
 * - "cách cải thiện cảm xúc / lời khuyên tâm lý"
 */

const EMOTION_MAP = {
  vui: "happy",
  "vui vẻ": "happy",
  happy: "happy",
  buồn: "sad",
  "buồn bã": "sad",
  sad: "sad",
  giận: "angry",
  "tức giận": "angry",
  angry: "angry",
  "ngạc nhiên": "surprise",
  surprise: "surprise",
  "sợ hãi": "fear",
  sợ: "fear",
  fear: "fear",
  "ghê tởm": "disgust",
  disgust: "disgust",
  "bình thường": "neutral",
  neutral: "neutral",
  "mệt mỏi": "fatigue",
  mệt: "fatigue",
  "buồn ngủ": "fatigue",
  fatigue: "fatigue",
};

const EMOTION_INFO = {
  happy: { emoji: "😄", label: "Vui vẻ" },
  sad: { emoji: "😢", label: "Buồn" },
  angry: { emoji: "😠", label: "Tức giận" },
  surprise: { emoji: "😲", label: "Ngạc nhiên" },
  fear: { emoji: "😨", label: "Sợ hãi" },
  disgust: { emoji: "🤢", label: "Ghê tởm" },
  neutral: { emoji: "😐", label: "Bình thường" },
  fatigue: { emoji: "😴", label: "Mệt mỏi" },
};

export class EmotionChatEngine {
  constructor(timelineStore) {
    this.timelineStore = timelineStore;
  }

  /**
   * Phân tích câu hỏi người dùng và giải quyết nếu là câu hỏi số liệu hoặc lời khuyên trực tiếp.
   * @param {string} query Câu hỏi từ người dùng
   * @returns {Object|null} Kết quả trả lời trực tiếp hoặc null nếu chuyển tiếp sang Gemini API
   */
  processQuery(query) {
    if (!query || typeof query !== "string") return null;

    const text = query.trim().toLowerCase();
    const stats = this.timelineStore.getStats();
    const timeline = this.timelineStore.getTimeline();
    const hasData = timeline && timeline.length > 0;

    // ─── 1. Pattern: Lời khuyên & Cách cải thiện cảm xúc (LUÔN TRẢ LỜI ĐẦY ĐỦ) ───
    if (
      text.includes("cải thiện") ||
      text.includes("lời khuyên") ||
      text.includes("thư giãn") ||
      text.includes("stress") ||
      text.includes("căng thẳng") ||
      text.includes("bớt mệt") ||
      text.includes("làm sao để vui") ||
      text.includes("tâm trạng")
    ) {
      let sessionNote = "";
      if (hasData && stats.dominantOverall) {
        const domInfo = EMOTION_INFO[stats.dominantOverall.name];
        sessionNote =
          `\n\n📌 **Ghi nhận từ phiên làm việc**: Cảm xúc chiếm ưu thế của bạn hiện tại là **${domInfo?.emoji || ""} ${domInfo?.label || stats.dominantOverall.name}** (${stats.dominantOverall.percentage}%). `;
      }

      return {
        isDirectQuery: true,
        reply:
          `🌿 **CẨM NANG CẢI THIỆN CẢM XÚC & NĂNG LƯỢNG LÀM VIỆC**:\n\n` +
          `1. 🧘 **Kỹ thuật thở sâu 4-7-8**: Hít vào bằng mũi trong 4s, giữ hơi 7s và thở ra từ từ qua miệng trong 8s. Giúp kích hoạt hệ thần kinh phó giao cảm và giảm tức thì hormone gây căng thẳng.\n` +
          `2. 👀 **Thư giãn thị giác (Quy tắc 20-20-20)**: Cứ 20 phút nhìn màn hình, hãy nhìn ra xa 20 feet (6m) trong 20 giây để giảm mỏi cơ mắt.\n` +
          `3. 💧 **Bổ sung nước & Vận động vi mô**: Uống một ly nước mát, thả lỏng quai hàm, xoay nhẹ khớp cổ và vai để giải tỏa áp lực cơ thể.\n` +
          `4. ✨ **Kích hoạt phản hồi cơ mặt (Facial Feedback)**: Mỉm cười nhẹ trong 10-15 giây. Nghiên cứu khoa học chứng minh não bộ sẽ tự động tiết Dopamine và Endorphin giúp cải thiện tâm trạng nhanh chóng.` +
          sessionNote,
      };
    }

    // ─── 2. Pattern: Tra cứu theo giây: "ở giây thứ X", "lúc X giây", "giây X"... ───
    const secondMatch =
      text.match(/(?:ở\s*)?(?:giây\s*(?:thứ\s*)?|lúc\s*)(\d+(?:[.,]\d+)?)\s*(?:giây|s)?/i) ||
      text.match(/(\d+(?:[.,]\d+)?)\s*(?:giây|s)\b/i);

    if (secondMatch) {
      if (!hasData) {
        return {
          isDirectQuery: true,
          reply:
            "⏱️ Chưa có frame nào được ghi nhận trong phiên hiện tại. Bạn hãy bật camera và chế độ Tự động (5 frames) để bắt đầu theo dõi nhé!",
        };
      }

      const targetSec = parseFloat(secondMatch[1].replace(",", "."));
      const result = this.timelineStore.getEntryAtSecond(targetSec, 3.0);

      if (result && result.entry) {
        const e = result.entry;
        const dom = e.dominantEmotion;
        const info = EMOTION_INFO[dom] || { emoji: "🤔", label: dom || "Chưa rõ" };
        const confPercent = Math.round((e.dominantConfidence || 0) * 100);

        const otherEmotions = Object.entries(e.emotions || {})
          .filter(([k]) => k !== dom)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([k, v]) => `${EMOTION_INFO[k]?.label || k}: ${(v * 100).toFixed(0)}%`)
          .join(", ");

        return {
          isDirectQuery: true,
          reply:
            `⏱️ Tại mốc **[${e.timestamp}s]** (frame #${e.frameIndex}):\n` +
            `- Biểu cảm chính: **${info.emoji} ${info.label}** (độ tin cậy: **${confPercent}%**)\n` +
            (otherEmotions ? `- Các cảm xúc khác: ${otherEmotions}\n` : "") +
            `- Khuôn mặt: ${e.hasFace ? `Phát hiện (${e.faceCount} mặt)` : "Không phát hiện"}\n\n` +
            `👉 *Mẹo: Nhấp vào nút **[${e.timestamp}s]** để xem lại frame tương ứng.*`,
          highlightTimestamp: e.timestamp,
          entry: e,
        };
      } else {
        return {
          isDirectQuery: true,
          reply:
            `🔍 Tại giây thứ **${targetSec}s**, chưa có mốc nào được lưu gần đó ` +
            `(Phiên hiện tại dài ${stats.duration}s với ${stats.totalFrames} mốc kết quả đã phát ra).`,
        };
      }
    }

    // ─── 3. Pattern: Đếm frame theo cảm xúc: "bao nhiêu frame là vui", "mấy frame"... ───
    if (
      text.includes("bao nhiêu frame") ||
      text.includes("mấy frame") ||
      text.includes("số frame") ||
      text.includes("chiếm bao nhiêu") ||
      text.includes("tỷ lệ") ||
      text.includes("mấy lần")
    ) {
      if (!hasData) {
        return {
          isDirectQuery: true,
          reply:
            "📊 Phiên làm việc chưa ghi nhận frame nào. Bạn hãy bật camera và chế độ Tự động để hệ thống bắt đầu thống kê nhé!",
        };
      }

      for (const [key, emotion] of Object.entries(EMOTION_MAP)) {
        if (text.includes(key)) {
          const count = stats.emotionCounts[emotion] || 0;
          const pct = stats.emotionPercentages[emotion] || 0;
          const info = EMOTION_INFO[emotion] || { emoji: "❓", label: emotion };

          const matchedEntries = timeline.filter((e) => e.dominantEmotion === emotion);
          const firstEntry = matchedEntries[0];
          const lastEntry = matchedEntries[matchedEntries.length - 1];

          let extraTimes = "";
          if (firstEntry && lastEntry) {
            extraTimes =
              `\n- Lần đầu xuất hiện: mốc **[${firstEntry.timestamp}s]**\n` +
              `- Lần gần nhất: mốc **[${lastEntry.timestamp}s]**`;
          }

          return {
            isDirectQuery: true,
            reply:
              `📊 Thống kê cảm xúc **${info.emoji} ${info.label}** trong phiên:\n` +
              `- Tổng số kết quả ghi nhận: **${count} / ${stats.totalFrames} mốc**\n` +
              `- Tỷ lệ phân bổ: **${pct}%** tổng thời gian` +
              extraTimes +
              (count === 0 ? `\n\n*(Chưa phát hiện biểu cảm ${info.label} nào trong phiên này)*` : ""),
            highlightTimestamp: firstEntry ? firstEntry.timestamp : null,
          };
        }
      }
    }

    // ─── 4. Pattern: Tổng quan / Thống kê / Phân tích phiên ───
    if (
      text.includes("tổng quan") ||
      text.includes("thống kê") ||
      text.includes("tổng kết") ||
      text.includes("tổng số frame") ||
      text.includes("phân tích phiên") ||
      text.includes("phiên làm việc")
    ) {
      if (!hasData) {
        return {
          isDirectQuery: true,
          reply:
            "📊 **TRẠNG THÁI PHIÊN LÀM VIỆC**:\n\n" +
            "Hệ thống hiện đang sẵn sàng! Kết quả phân tích sẽ tự động ghi nhận khi:\n" +
            "• Bạn **thay đổi biểu cảm** trước camera (ví dụ: mỉm cười hoặc ngạc nhiên).\n" +
            "• Hoặc **định kỳ mỗi 10 giây** nếu biểu cảm giữ nguyên.\n\n" +
            "👉 Hãy bật camera và thử biểu lộ một cảm xúc để tôi bắt đầu phân tích nhé!",
        };
      }

      const dominant = stats.dominantOverall;
      const domInfo = dominant
        ? EMOTION_INFO[dominant.name] || { emoji: "✨", label: dominant.name }
        : null;

      let breakdownText = "";
      for (const [em, pct] of Object.entries(stats.emotionPercentages || {})) {
        const inf = EMOTION_INFO[em] || { emoji: "•", label: em };
        breakdownText += `  • ${inf.emoji} ${inf.label}: **${pct}%** (${stats.emotionCounts[em] || 0} lần)\n`;
      }

      return {
        isDirectQuery: true,
        reply:
          `📈 **BẢN TỔNG HỢP PHIÊN LÀM VIỆC HIỆN TẠI**:\n\n` +
          `- ⏱️ Thời gian phiên: **${Math.floor(stats.duration / 60)}m ${Math.floor(stats.duration % 60)}s**\n` +
          `- 🎞️ Tổng số mốc ghi nhận: **${stats.totalFrames} mốc**\n` +
          `- 🌟 Cảm xúc chủ đạo nhất: **${domInfo ? `${domInfo.emoji} ${domInfo.label} (${dominant.percentage}%)` : "Chưa xác định"}**\n\n` +
          `**Chi tiết phân bổ biểu cảm:**\n${breakdownText || "  (Đang thu thập dữ liệu...)"}\n` +
          `👉 *Bạn có thể hỏi thêm: "ở giây thứ X cảm xúc gì?" hoặc "vui vẻ chiếm bao nhiêu frame?".*`,
      };
    }

    // Chuyển sang LLM Gemini xử lý nếu là câu hỏi phân tích tự do
    return null;
  }
}
