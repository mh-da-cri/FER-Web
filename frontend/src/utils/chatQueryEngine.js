/**
 * EmotionChatEngine — Bộ xử lý truy vấn dữ liệu cảm xúc chính xác 100%.
 * Tự động trả lời tất cả các câu hỏi số liệu, phân tích phiên, trạng thái,
 * thời điểm, mệt mỏi, tập trung, so sánh, chào hỏi và hướng dẫn.
 *
 * NGUYÊN TẮC HOẠT ĐỘNG:
 * 1. Được thì TỰ TRẢ LỜI ngay lập tức (0ms latency, không tốn quota, dữ liệu chuẩn xác 100%).
 * 2. Tự trả lời thêm rất nhiều câu hỏi phong phú ngoài các gợi ý mặc định.
 * 3. Nếu là câu hỏi mở / triết lý / phân tích chuyên sâu ngoài quy tắc -> trả về null để chuyển gọi Google Gemini API.
 */

import { formatSessionDuration } from "@/utils/sessionTimeline";

export const EMOTION_MAP = {
  vui: "happy",
  "vui vẻ": "happy",
  cuoi: "happy",
  cười: "happy",
  happy: "happy",
  joy: "happy",
  buon: "sad",
  "buồn bã": "sad",
  buồn: "sad",
  sad: "sad",
  gian: "angry",
  giận: "angry",
  "tức giận": "angry",
  tuc: "angry",
  tức: "angry",
  angry: "angry",
  "ngạc nhiên": "surprise",
  "ngac nhien": "surprise",
  surprise: "surprise",
  "sợ hãi": "fear",
  "so hai": "fear",
  so: "fear",
  sợ: "fear",
  fear: "fear",
  "ghê tởm": "disgust",
  "ghe tom": "disgust",
  disgust: "disgust",
  "bình thường": "neutral",
  "binh thuong": "neutral",
  neutral: "neutral",
  "mệt mỏi": "fatigue",
  "met moi": "fatigue",
  met: "fatigue",
  mệt: "fatigue",
  "buồn ngủ": "fatigue",
  "buon ngu": "fatigue",
  fatigue: "fatigue",
  sleepy: "fatigue",
  tired: "fatigue",
};

export const EMOTION_INFO = {
  happy: { emoji: "😄", label: "Vui vẻ" },
  sad: { emoji: "😢", label: "Buồn" },
  angry: { emoji: "😠", label: "Tức giận" },
  surprise: { emoji: "😲", label: "Ngạc nhiên" },
  fear: { emoji: "😨", label: "Sợ hãi" },
  disgust: { emoji: "🤢", label: "Ghê tởm" },
  neutral: { emoji: "😐", label: "Bình thường" },
  fatigue: { emoji: "😴", label: "Mệt mỏi" },
};

const NO_EMOTION_DATA_REPLY =
  "Chưa có dữ liệu cảm xúc trong phiên để trả lời câu hỏi này. Bạn hãy bật camera và ghi nhận một vài kết quả trước nhé!";

/**
 * Loại bỏ dấu tiếng Việt và chuẩn hóa về chữ thường để so sánh linh hoạt
 */
export function normalizeText(str) {
  if (!str || typeof str !== "string") return "";
  return str
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/đ/g, "d")
    .replace(/Đ/g, "D")
    .toLowerCase()
    .trim();
}

export class EmotionChatEngine {
  constructor(timelineStore) {
    this.timelineStore = timelineStore;
  }

  /**
   * Phân tích câu hỏi người dùng và tự trả lời nếu câu hỏi thuộc phạm vi số liệu/quy tắc.
   * Nếu không trả lời được -> trả về null để hệ thống gọi Google Gemini API.
   * @param {string} query Câu hỏi từ người dùng
   * @returns {Object|null} Kết quả trả lời trực tiếp hoặc null
   */
  processQuery(query) {
    if (!query || typeof query !== "string") return null;

    const rawText = query.trim().toLowerCase();
    const normText = normalizeText(query);
    const stats = this.timelineStore.getStats();
    const timeline = this.timelineStore.getTimeline();
    const hasData = timeline && timeline.length > 0;
    const hasEmotionData = Boolean(stats.dominantOverall);

    // ─────────────────────────────────────────────────────────────
    // NHÓM 1: GIAO TIẾP XÃ HỘI, CHÀO HỎI, CẢM ƠN, TẠM BIỆT & HƯỚNG DẪN
    // ─────────────────────────────────────────────────────────────

    // 1.1 Chào hỏi
    if (
      normText === "xin chao" ||
      normText === "chao ban" ||
      normText === "chao bot" ||
      normText === "chao tro ly" ||
      normText === "chao ai" ||
      normText === "hello" ||
      normText === "hi" ||
      normText === "alo" ||
      normText === "hey" ||
      normText.startsWith("xin chao") ||
      normText.startsWith("chao ban") ||
      normText.startsWith("hello") ||
      normText.startsWith("hi bot")
    ) {
      const durationText = formatSessionDuration(stats.duration);
      const faceText = stats.faceFrames > 0
        ? `Camera đã ghi nhận **${stats.faceFrames} frame** có khuôn mặt`
        : "Camera chưa ghi nhận khuôn mặt nào";

      return {
        isDirectQuery: true,
        reply:
          `👋 **Xin chào bạn!** Tôi là **AI Trợ Lý Phân Tích Cảm Xúc**.\n\n` +
          `Hiện tại phiên làm việc đã chạy được **${durationText}** (${faceText}).\n\n` +
          `Tôi có thể tự động trả lời nhanh các câu hỏi về:\n` +
          `• 📊 Tóm tắt & thống kê cảm xúc toàn phiên\n` +
          `• 😴 Đánh giá dấu hiệu mệt mỏi, buồn ngủ\n` +
          `• 🎯 Mức độ tập trung làm việc\n` +
          `• ⏱️ Tra cứu cảm xúc tại giây bất kỳ (ví dụ: *"ở giây thứ 10 cảm xúc gì?"*)\n` +
          `• 🕒 Tìm thời điểm vui, buồn, mệt (ví dụ: *"khi nào tôi vui?"*)\n` +
          `• 💡 Lời khuyên cải thiện tâm trạng và thư giãn\n\n` +
          `Các câu hỏi phân tích tâm lý sâu hoặc trò chuyện mở sẽ được tôi chuyển tiếp đến **Google Gemini AI** để giải đáp chi tiết. Bạn cần tôi hỗ trợ điều gì?`,
      };
    }

    // 1.2 Cảm ơn
    if (
      normText.includes("cam on") ||
      normText.includes("thank you") ||
      normText.includes("thanks") ||
      normText === "ok cam on" ||
      normText === "da cam on"
    ) {
      return {
        isDirectQuery: true,
        reply:
          `😊 **Rất vui được đồng hành cùng bạn!**\n\n` +
          `Chúc bạn có một buổi làm việc thật hiệu quả, duy trì tinh thần tích cực và thoải mái. ` +
          `Nếu thấy mệt mỏi hay căng thẳng, đừng quên nghỉ mắt và thư giãn vài phút nhé!`,
      };
    }

    // 1.3 Tạm biệt
    if (
      normText.includes("tam biet") ||
      normText === "bye" ||
      normText === "bye bye" ||
      normText === "goodbye"
    ) {
      return {
        isDirectQuery: true,
        reply:
          `👋 **Tạm biệt bạn!**\n\n` +
          `Cảm ơn bạn đã sử dụng hệ thống AI Camera. Hãy nghỉ ngơi đầy đủ và hẹn gặp lại bạn trong phiên làm việc tiếp theo nhé!`,
      };
    }

    // 1.4 Giới thiệu bản thân & Hướng dẫn sử dụng
    if (
      normText.includes("ban la ai") ||
      normText.includes("ban lam duoc gi") ||
      normText.includes("tro ly la gi") ||
      normText.includes("huong dan su dung") ||
      normText.includes("he thong nay la gi") ||
      normText.includes("co the hoi gi") ||
      normText.includes("hoi duoc gi") ||
      normText.includes("tinh nang")
    ) {
      return {
        isDirectQuery: true,
        reply:
          `🤖 **GIỚI THIỆU & HƯỚNG DẪN TRỢ LÝ AI CẢM XÚC**:\n\n` +
          `Tôi là trợ lý AI tích hợp trong hệ thống AI Camera, kết hợp giữa **Bộ phân tích dữ liệu trực tiếp 100% chính xác** và **Google Gemini LLM**.\n\n` +
          `**Các câu hỏi tôi có thể tự trả lời ngay:**\n` +
          `1. 📊 **Tóm tắt phiên**: *"Tóm tắt phiên làm việc"*, *"Tổng quan cảm xúc"*\n` +
          `2. 😴 **Đánh giá mệt mỏi**: *"Tôi có mệt không?"*, *"Có dấu hiệu buồn ngủ không?"*\n` +
          `3. 🎯 **Mức độ tập trung**: *"Tôi có tập trung không?"*, *"Đánh giá sự tập trung"*\n` +
          `4. 🌟 **Cảm xúc chủ đạo**: *"Cảm xúc nào nhiều nhất?"*, *"Cảm xúc nào ít nhất?"*\n` +
          `5. ⏱️ **Tra cứu mốc thời gian**: *"Ở giây thứ 10 cảm xúc gì?"*, *"Giây 5 biểu cảm gì?"*\n` +
          `6. 🕒 **Tìm thời điểm**: *"Khi nào tôi vui?"*, *"Lúc nào tôi mệt mỏi?"*\n` +
          `7. 📈 **Đếm frame**: *"Vui vẻ chiếm bao nhiêu frame?"*, *"Tỷ lệ mệt mỏi là bao nhiêu?"*\n` +
          `8. ⚖️ **So sánh**: *"So sánh vui vẻ và mệt mỏi"*, *"Vui hay buồn nhiều hơn?"*\n` +
          `9. 💡 **Lời khuyên**: *"Lời khuyên cải thiện tâm trạng"*, *"Cách thư giãn"*\n\n` +
          `*Đối với các câu hỏi tâm sự, kiến thức tâm lý mở hoặc phân tích tự do khác, tôi sẽ tự động chuyển tiếp tới Google Gemini AI để giải đáp!*`,
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 2: CÁC CÂU NGOÀI GỢI Ý — ĐÁNH GIÁ MỆT MỎI & BUỒN NGỦ
    // (Bao gồm cả gợi ý số 4: "Tôi có dấu hiệu mệt mỏi không?")
    // ─────────────────────────────────────────────────────────────
    if (
      normText.includes("met moi") ||
      normText.includes("co met khong") ||
      normText.includes("toi co met") ||
      normText.includes("dau hieu met") ||
      normText.includes("buon ngu") ||
      normText.includes("co buon ngu khong") ||
      normText.includes("toi co buon ngu") ||
      normText.includes("ue oai") ||
      normText.includes("muc do met") ||
      normText.includes("co bi stress") ||
      normText.includes("stress khong") ||
      normText.includes("trong toi co met")
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const fatigueCount = stats.emotionCounts.fatigue || 0;
      const fatiguePct = stats.emotionPercentages.fatigue || 0;
      const totalFrames = stats.totalFrames || 0;

      // Tìm mốc mệt mỏi gần nhất
      const fatigueEntries = timeline.filter((e) => e.dominantEmotion === "fatigue");
      const lastFatigue = fatigueEntries[fatigueEntries.length - 1];

      if (fatiguePct >= 25 || (stats.dominantOverall && stats.dominantOverall.name === "fatigue")) {
        let lastTimeText = "";
        if (lastFatigue) {
          lastTimeText = `\n- Mốc mệt mỏi gần nhất: **[${lastFatigue.timestamp}s]** (độ tin cậy ${(lastFatigue.dominantConfidence * 100).toFixed(0)}%)`;
        }

        return {
          isDirectQuery: true,
          reply:
            `⚠️ **CẢNH BÁO DẤU HIỆU MỆT MỎI:**\n\n` +
            `- Biểu cảm mệt mỏi / buồn ngủ được ghi nhận trong **${fatiguePct}%** tổng số mốc (${fatigueCount}/${totalFrames} lần).` +
            lastTimeText +
            `\n- Điểm mệt mỏi trung bình toàn phiên: **${stats.fatiguePercentage}%**.\n\n` +
            `💡 **Lời khuyên tức thì:**\n` +
            `1. Hãy rời mắt khỏi màn hình khoảng 20 giây và nhìn ra xa 6 mét (quy tắc 20-20-20).\n` +
            `2. Uống một ngụm nước lọc mát để đánh thức các giác quan.\n` +
            `3. Đứng dậy vươn vai nhẹ nhàng, hít thở sâu 3 nhịp trước khi tiếp tục làm việc nhé!`,
          highlightTimestamp: lastFatigue ? lastFatigue.timestamp : null,
        };
      } else {
        return {
          isDirectQuery: true,
          reply:
            `✅ **BẠN ĐANG DUY TRÌ TRẠNG THÁI TỐT:**\n\n` +
            `- Tỷ lệ mệt mỏi chỉ chiếm **${fatiguePct}%** (${fatigueCount}/${totalFrames} mốc), hoàn toàn dưới ngưỡng cảnh báo (25%).\n` +
            `- Biểu cảm chủ đạo hiện tại của bạn là **${EMOTION_INFO[stats.dominantOverall.name]?.emoji || ""} ${EMOTION_INFO[stats.dominantOverall.name]?.label || stats.dominantOverall.name}** (${stats.dominantOverall.percentage}%).\n\n` +
            `💡 Tiếp tục duy trì nhịp làm việc thoải mái và tư thế ngồi chuẩn nhé!`,
        };
      }
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 3: CÁC CÂU NGOÀI GỢI Ý — ĐÁNH GIÁ MỨC ĐỘ TẬP TRUNG
    // ─────────────────────────────────────────────────────────────
    if (
      normText.includes("tap trung") ||
      normText.includes("co tap trung khong") ||
      normText.includes("muc do tap trung") ||
      normText.includes("nghiem tuc") ||
      normText.includes("co bi phan tam") ||
      normText.includes("phan tam")
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const neutralPct = stats.emotionPercentages.neutral || 0;
      const happyPct = stats.emotionPercentages.happy || 0;
      const fatiguePct = stats.emotionPercentages.fatigue || 0;
      const faceRate = stats.totalFrames > 0
        ? Math.round((stats.faceFrames / stats.totalFrames) * 100)
        : 0;

      // Tính chỉ số tập trung: biểu cảm trung tính (làm việc) kết hợp nhận diện khuôn mặt liên tục
      const focusScore = Math.min(
        100,
        Math.round((neutralPct * 0.7 + happyPct * 0.3) * (faceRate / 100) * 1.1)
      );

      let assessment = "";
      if (focusScore >= 70) {
        assessment = "🌟 **Mức độ tập trung rất cao!** Bạn đang duy trì sự chú ý liên tục vào công việc.";
      } else if (focusScore >= 45) {
        assessment = "🎯 **Mức độ tập trung khá tốt.** Có một số mốc biểu cảm biến đổi nhẹ nhưng nhìn chung ổn định.";
      } else {
        assessment = "⚠️ **Mức độ tập trung đang giảm sút.** Bạn có thể đang bị mệt mỏi hoặc phân tâm khỏi màn hình.";
      }

      return {
        isDirectQuery: true,
        reply:
          `🎯 **ĐÁNH GIÁ MỨC ĐỘ TẬP TRUNG TRONG PHIÊN:**\n\n` +
          `${assessment}\n\n` +
          `- Chỉ số tập trung ước tính: **${focusScore}/100**\n` +
          `- Tỷ lệ trạng thái làm việc điềm tĩnh (Bình thường): **${neutralPct}%**\n` +
          `- Tỷ lệ hiện diện khuôn mặt trước camera: **${faceRate}%**\n` +
          `- Tỷ lệ mệt mỏi / buồn ngủ: **${fatiguePct}%**\n\n` +
          `💡 *Mẹo: Nếu cần hoàn thành công việc quan trọng, bạn có thể áp dụng phương pháp Pomodoro (25 phút tập trung cao độ, 5 phút nghỉ ngắn).*`,
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 4: CÁC CÂU NGOÀI GỢI Ý — CẢM XÚC CHỦ ĐẠO / NHIỀU NHẤT / ÁP ĐẢO
    // ─────────────────────────────────────────────────────────────
    if (
      (normText.includes("cam xuc") || normText.includes("bieu cam") || normText.includes("tam trang")) &&
      (normText.includes("nhieu nhat") ||
        normText.includes("chu dao") ||
        normText.includes("chinh la gi") ||
        normText.includes("chiem uu the") ||
        normText.includes("pho bien nhat") ||
        normText.includes("cao nhat") ||
        normText.includes("ap dao"))
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const dominant = stats.dominantOverall;
      const info = EMOTION_INFO[dominant.name] || { emoji: "✨", label: dominant.name };

      return {
        isDirectQuery: true,
        reply:
          `🌟 **CẢM XÚC CHỦ ĐẠO XUẤT HIỆN NHIỀU NHẤT:**\n\n` +
          `- Biểu cảm: **${info.emoji} ${info.label}**\n` +
          `- Số lượng ghi nhận: **${dominant.count} / ${stats.totalFrames} mốc**\n` +
          `- Tỷ lệ phân bổ: **${dominant.percentage}%** toàn bộ thời lượng phiên (${formatSessionDuration(stats.duration)}).\n\n` +
          `👉 *Bạn có thể hỏi thêm: "Khi nào tôi ${info.label.toLowerCase()}?" để xem danh sách các mốc thời gian cụ thể.*`,
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 5: CÁC CÂU NGOÀI GỢI Ý — CẢM XÚC ÍT NHẤT / THẤP NHẤT
    // ─────────────────────────────────────────────────────────────
    if (
      (normText.includes("cam xuc") || normText.includes("bieu cam")) &&
      (normText.includes("it nhat") || normText.includes("thap nhat") || normText.includes("hiem nhat"))
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const counts = stats.emotionCounts || {};
      const allEmotions = Object.keys(EMOTION_INFO);

      // Sắp xếp tăng dần theo count
      const sorted = allEmotions
        .map((em) => ({
          name: em,
          count: counts[em] || 0,
          pct: stats.emotionPercentages[em] || 0,
        }))
        .sort((a, b) => a.count - b.count);

      const lowest = sorted[0];
      const info = EMOTION_INFO[lowest.name] || { emoji: "•", label: lowest.name };

      return {
        isDirectQuery: true,
        reply:
          `📉 **CẢM XÚC XUẤT HIỆN ÍT NHẤT:**\n\n` +
          `- Biểu cảm: **${info.emoji} ${info.label}**\n` +
          `- Số lượng ghi nhận: **${lowest.count} mốc** (${lowest.pct}%).\n` +
          (lowest.count === 0
            ? `*(Trong toàn bộ phiên làm việc, camera chưa từng ghi nhận biểu cảm này).*`
            : ""),
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 6: CÁC CÂU NGOÀI GỢI Ý — TÌM THỜI ĐIỂM (KHI NÀO TÔI VUI? LÚC NÀO TÔI BUỒN?)
    // ─────────────────────────────────────────────────────────────
    if (
      normText.startsWith("khi nao") ||
      normText.startsWith("luc nao") ||
      normText.startsWith("thoi diem nao") ||
      normText.includes("khi nao toi") ||
      normText.includes("luc nao toi") ||
      normText.includes("thoi diem nao toi") ||
      normText.includes("toi vui luc nao") ||
      normText.includes("toi cuoi luc nao") ||
      normText.includes("toi met luc nao") ||
      normText.includes("toi cuoi khi nao") ||
      normText.includes("co luc nao toi") ||
      normText.includes("co khi nao toi") ||
      normText.includes("co luc nao vui") ||
      normText.includes("co luc nao met") ||
      normText.includes("toi co cuoi khong") ||
      normText.includes("toi co vui khong")
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      // Xác định cảm xúc được hỏi
      let matchedEmotion = null;
      for (const [key, emotion] of Object.entries(EMOTION_MAP)) {
        if (normText.includes(normalizeText(key))) {
          matchedEmotion = emotion;
          break;
        }
      }

      if (!matchedEmotion) {
        // Mặc định kiểm tra vui vẻ nếu hỏi "cười"
        if (normText.includes("cuoi")) matchedEmotion = "happy";
      }

      if (matchedEmotion) {
        const info = EMOTION_INFO[matchedEmotion] || { emoji: "✨", label: matchedEmotion };
        const matchedEntries = timeline.filter((e) => e.dominantEmotion === matchedEmotion);

        if (matchedEntries.length === 0) {
          return {
            isDirectQuery: true,
            reply:
              `🔍 Trong phiên hiện tại (**${formatSessionDuration(stats.duration)}**), camera **chưa ghi nhận mốc nào** bạn biểu cảm **${info.emoji} ${info.label}**.\n\n` +
              `👉 Cảm xúc chủ đạo được ghi nhận là: **${EMOTION_INFO[stats.dominantOverall.name]?.label || stats.dominantOverall.name}** (${stats.dominantOverall.percentage}%).`,
          };
        }

        const first = matchedEntries[0];
        const last = matchedEntries[matchedEntries.length - 1];

        // Lấy danh sách tối đa 6 mốc đại diện
        const sampleTimestamps = matchedEntries.slice(0, 6).map((e) => `[${e.timestamp}s]`);
        const moreCount = matchedEntries.length > 6 ? ` và ${matchedEntries.length - 6} mốc khác` : "";

        return {
          isDirectQuery: true,
          reply:
            `⏱️ **CÁC THỜI ĐIỂM GHI NHẬN BIỂU CẢM ${info.emoji} ${info.label.toUpperCase()}:**\n\n` +
            `- Tổng số lần ghi nhận: **${matchedEntries.length} mốc** (${stats.emotionPercentages[matchedEmotion] || 0}% thời gian phiên).\n` +
            `- Các mốc cụ thể: ${sampleTimestamps.join(", ")}${moreCount}.\n` +
            `- Lần đầu tiên: mốc **[${first.timestamp}s]** (độ tin cậy ${(first.dominantConfidence * 100).toFixed(0)}%).\n` +
            `- Lần gần nhất: mốc **[${last.timestamp}s]** (độ tin cậy ${(last.dominantConfidence * 100).toFixed(0)}%).\n\n` +
            `👉 *Mẹo: Nhấp vào bất kỳ nút **[Xs]** nào ở trên để xem lại frame tương ứng trên camera.*`,
          highlightTimestamp: first.timestamp,
        };
      }
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 7: CÁC CÂU NGOÀI GỢI Ý — CẢM XÚC HIỆN TẠI / GẦN NHẤT & FRAME ĐẦU TIÊN
    // ─────────────────────────────────────────────────────────────
    if (
      normText.includes("hien tai") ||
      normText.includes("moi nhat") ||
      normText.includes("vua roi") ||
      normText.includes("frame gan nhat") ||
      normText.includes("vua xong") ||
      normText.includes("bay gio") ||
      normText.includes("luc nay")
    ) {
      if (!hasData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const latestEntry = timeline[timeline.length - 1];
      if (!latestEntry || !latestEntry.dominantEmotion) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const info = EMOTION_INFO[latestEntry.dominantEmotion] || {
        emoji: "✨",
        label: latestEntry.dominantEmotion,
      };

      return {
        isDirectQuery: true,
        reply:
          `⏱️ **KẾT QUẢ TẠI FRAME GẦN NHẤT [${latestEntry.timestamp}s]:**\n\n` +
          `- Biểu cảm ghi nhận: **${info.emoji} ${info.label}**\n` +
          `- Độ tin cậy nhận diện: **${Math.round((latestEntry.dominantConfidence || 0) * 100)}%**\n` +
          `- Số khuôn mặt phát hiện: **${latestEntry.faceCount || 1}**\n` +
          `- Điểm mệt mỏi tại frame này: **${latestEntry.fatigueScore ? (latestEntry.fatigueScore * 100).toFixed(0) + "%" : "Không đáng kể"}**\n\n` +
          `👉 *Nhấp vào nút **[${latestEntry.timestamp}s]** để xem lại frame này.*`,
        highlightTimestamp: latestEntry.timestamp,
        entry: latestEntry,
      };
    }

    if (
      normText.includes("ban dau") ||
      normText.includes("luc bat dau") ||
      normText.includes("frame dau tien") ||
      normText.includes("dau phien") ||
      normText.includes("khoi dau")
    ) {
      if (!hasData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const firstEntry = timeline[0];
      if (!firstEntry || !firstEntry.dominantEmotion) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const info = EMOTION_INFO[firstEntry.dominantEmotion] || {
        emoji: "✨",
        label: firstEntry.dominantEmotion,
      };

      return {
        isDirectQuery: true,
        reply:
          `⏱️ **KẾT QUẢ TẠI FRAME ĐẦU TIÊN [${firstEntry.timestamp}s]:**\n\n` +
          `- Biểu cảm xuất phát điểm: **${info.emoji} ${info.label}**\n` +
          `- Độ tin cậy: **${Math.round((firstEntry.dominantConfidence || 0) * 100)}%**\n\n` +
          `👉 *Nhấp vào nút **[${firstEntry.timestamp}s]** để xem lại thời điểm bắt đầu phiên.*`,
        highlightTimestamp: firstEntry.timestamp,
        entry: firstEntry,
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 8: CÁC CÂU NGOÀI GỢI Ý — SO SÁNH 2 CẢM XÚC
    // ─────────────────────────────────────────────────────────────
    if (
      normText.includes("so sanh") ||
      (normText.includes("hay") && (normText.includes("nhieu hon") || normText.includes("it hon")))
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      // Tìm 2 cảm xúc xuất hiện trong câu
      const detectedEmotions = [];
      for (const [key, em] of Object.entries(EMOTION_MAP)) {
        if (normText.includes(normalizeText(key)) && !detectedEmotions.includes(em)) {
          detectedEmotions.push(em);
          if (detectedEmotions.length === 2) break;
        }
      }

      if (detectedEmotions.length === 2) {
        const emA = detectedEmotions[0];
        const emB = detectedEmotions[1];
        const infoA = EMOTION_INFO[emA] || { emoji: "•", label: emA };
        const infoB = EMOTION_INFO[emB] || { emoji: "•", label: emB };

        const countA = stats.emotionCounts[emA] || 0;
        const countB = stats.emotionCounts[emB] || 0;
        const pctA = stats.emotionPercentages[emA] || 0;
        const pctB = stats.emotionPercentages[emB] || 0;

        let verdict = "";
        if (countA > countB) {
          verdict = `Biểu cảm **${infoA.label}** chiếm ưu thế hơn **${infoB.label}** (cao hơn ${(pctA - pctB).toFixed(1)}%).`;
        } else if (countB > countA) {
          verdict = `Biểu cảm **${infoB.label}** chiếm ưu thế hơn **${infoA.label}** (cao hơn ${(pctB - pctA).toFixed(1)}%).`;
        } else {
          verdict = `Cả hai biểu cảm **${infoA.label}** và **${infoB.label}** xuất hiện tương đương nhau trong phiên.`;
        }

        return {
          isDirectQuery: true,
          reply:
            `⚖️ **SO SÁNH BIỂU CẢM: ${infoA.emoji} ${infoA.label} VS ${infoB.emoji} ${infoB.label}**\n\n` +
            `- **${infoA.emoji} ${infoA.label}**: **${countA} mốc** (${pctA}%)\n` +
            `- **${infoB.emoji} ${infoB.label}**: **${countB} mốc** (${pctB}%)\n\n` +
            `👉 **Kết luận:** ${verdict}`,
        };
      }
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 9: CÁC CÂU NGOÀI GỢI Ý — TỶ LỆ TÍCH CỰC VS TIÊU CỰC / NĂNG LƯỢNG
    // ─────────────────────────────────────────────────────────────
    if (
      normText.includes("tich cuc") ||
      normText.includes("tieu cuc") ||
      normText.includes("nang luong")
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const positivePct = stats.emotionPercentages.happy || 0;
      const neutralPct = stats.emotionPercentages.neutral || 0;
      const negativePct = Math.round(
        ((stats.emotionPercentages.sad || 0) +
          (stats.emotionPercentages.fatigue || 0) +
          (stats.emotionPercentages.angry || 0) +
          (stats.emotionPercentages.fear || 0) +
          (stats.emotionPercentages.disgust || 0)) *
          10
      ) / 10;

      let energyStatus = "";
      if (positivePct >= 40) {
        energyStatus = "🌟 **Năng lượng phiên rất tích cực!** Bạn có tinh thần hào hứng và vui vẻ.";
      } else if (negativePct >= 35) {
        energyStatus = "⚠️ **Năng lượng đang có dấu hiệu suy giảm.** Cảm xúc tiêu cực/mệt mỏi chiếm tỷ lệ đáng kể.";
      } else {
        energyStatus = "🎯 **Năng lượng ổn định, điềm tĩnh.** Bạn đang duy trì trạng thái làm việc tập trung.";
      }

      return {
        isDirectQuery: true,
        reply:
          `⚡ **PHÂN BỔ NĂNG LƯỢNG & TÍNH TÍCH CỰC TRONG PHIÊN:**\n\n` +
          `${energyStatus}\n\n` +
          `- 🟢 **Tích cực (Vui vẻ):** **${positivePct}%**\n` +
          `- ⚪ **Trung tính (Bình thường):** **${neutralPct}%**\n` +
          `- 🔴 **Cần lưu ý (Mệt mỏi, buồn, căng thẳng):** **${negativePct}%**\n\n` +
          `💡 *Nếu tỷ lệ lưu ý vượt quá 30%, bạn nên uống thêm nước và nghỉ giải lao ngắn nhé!*`,
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 10: CÁC CÂU NGOÀI GỢI Ý — THAY ĐỔI / CHUYỂN BIẾN CẢM XÚC
    // ─────────────────────────────────────────────────────────────
    if (
      normText.includes("thay doi cam xuc") ||
      normText.includes("chuyen bien") ||
      normText.includes("bien doi") ||
      normText.includes("doi cam xuc") ||
      normText.includes("dao dong") ||
      normText.includes("tam trang co thay doi")
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      // Tìm các mốc đổi cảm xúc
      const transitions = [];
      for (let i = 1; i < timeline.length; i++) {
        const prev = timeline[i - 1];
        const curr = timeline[i];
        if (curr.dominantEmotion && prev.dominantEmotion && curr.dominantEmotion !== prev.dominantEmotion) {
          transitions.push({
            timestamp: curr.timestamp,
            from: prev.dominantEmotion,
            to: curr.dominantEmotion,
          });
        }
      }

      if (transitions.length === 0) {
        const dom = stats.dominantOverall;
        return {
          isDirectQuery: true,
          reply:
            `📈 **TÂM TRẠNG CỰC KỲ ỔN ĐỊNH:**\n\n` +
            `Trong toàn bộ phiên (**${formatSessionDuration(stats.duration)}**), bạn duy trì một biểu cảm xuyên suốt là **${EMOTION_INFO[dom.name]?.label || dom.name}** mà hầu như không có sự biến đổi đột ngột nào.`,
        };
      }

      const sampleTransitions = transitions.slice(-4).map((t) => {
        const fromLabel = EMOTION_INFO[t.from]?.label || t.from;
        const toLabel = EMOTION_INFO[t.to]?.label || t.to;
        return `  • Mốc **[${t.timestamp}s]**: Chuyển từ *${fromLabel}* sang *${toLabel}*`;
      });

      return {
        isDirectQuery: true,
        reply:
          `🔄 **CÁC MỐC CHUYỂN BIẾN CẢM XÚC TRONG PHIÊN:**\n\n` +
          `- Tổng số lần chuyển đổi cảm xúc: **${transitions.length} lần**\n` +
          `- Các mốc biến đổi gần đây:\n${sampleTransitions.join("\n")}\n\n` +
          `👉 *Nhấp vào bất kỳ mốc **[Xs]** nào để xem lại frame camera lúc thay đổi.*`,
        highlightTimestamp: transitions[transitions.length - 1].timestamp,
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 11: CÁC CÂU NGOÀI GỢI Ý — THỜI LƯỢNG & SỐ LƯỢNG KẾT QUẢ
    // ─────────────────────────────────────────────────────────────
    if (
      normText.includes("keo dai bao lau") ||
      normText.includes("chay duoc bao lau") ||
      normText.includes("bao nhieu phut") ||
      normText.includes("bao nhieu giay") ||
      normText.includes("thoi gian phien") ||
      normText.includes("tong thoi gian") ||
      normText.includes("bao nhieu frame") ||
      normText.includes("tong so frame") ||
      normText.includes("bao nhieu ket qua") ||
      normText.includes("chup bao nhieu")
    ) {
      if (!hasData) {
        return {
          isDirectQuery: true,
          reply: `⏱️ Phiên làm việc hiện tại chưa có dữ liệu. Bạn hãy bật camera để bắt đầu theo dõi nhé!`,
        };
      }

      return {
        isDirectQuery: true,
        reply:
          `⏱️ **TIẾN ĐỘ & THỜI LƯỢNG PHIÊN HIỆN TẠI:**\n\n` +
          `- Thời lượng phiên đã trôi qua: **${formatSessionDuration(stats.duration)}**\n` +
          `- Tổng số mốc phân tích đã lưu: **${stats.totalFrames} mốc kết quả**\n` +
          `- Số frame phát hiện khuôn mặt: **${stats.faceFrames} / ${stats.totalFrames}** (${stats.totalFrames > 0 ? Math.round((stats.faceFrames / stats.totalFrames) * 100) : 0}%)\n` +
          `- Tốc độ lưu kết quả: Chế độ Tự động (5 frames) hoặc khi có thay đổi cảm xúc.`,
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 12: CÁC CÂU NGOÀI GỢI Ý — NHẬN DIỆN KHUÔN MẶT & CAMERA
    // ─────────────────────────────────────────────────────────────
    if (
      normText.includes("phat hien khuon mat") ||
      normText.includes("co thay mat toi khong") ||
      normText.includes("co bat duoc mat") ||
      normText.includes("may khuon mat") ||
      normText.includes("may nguoi") ||
      normText.includes("nhan dien duoc khong") ||
      normText.includes("do tin cay") ||
      normText.includes("do chinh xac")
    ) {
      if (!hasData) {
        return {
          isDirectQuery: true,
          reply: `📷 Chưa có dữ liệu nhận diện. Bạn hãy bật camera trên bảng điều khiển ở giữa để bắt đầu nhận diện khuôn mặt nhé!`,
        };
      }

      const faceRate = stats.totalFrames > 0
        ? Math.round((stats.faceFrames / stats.totalFrames) * 100)
        : 0;

      // Tính độ tin cậy trung bình của các frame có mặt
      const faceEntries = timeline.filter((e) => e.hasFace && e.dominantConfidence);
      const avgConf = faceEntries.length > 0
        ? Math.round(
            (faceEntries.reduce((sum, e) => sum + e.dominantConfidence, 0) /
              faceEntries.length) *
              100
          )
        : 0;

      return {
        isDirectQuery: true,
        reply:
          `👁️ **TRẠNG THÁI NHẬN DIỆN KHUÔN MẶT:**\n\n` +
          `- Tỷ lệ bắt diện khuôn mặt: **${faceRate}%** (${stats.faceFrames}/${stats.totalFrames} mốc)\n` +
          `- Độ tin cậy trung bình: **${avgConf}%**\n` +
          `- Khả năng theo dõi: Hệ thống tự động theo dõi khuôn mặt chính diện, hỗ trợ trích xuất 8 trạng thái cảm xúc theo thời gian thực.\n\n` +
          `👉 *Để nhận diện tối ưu, bạn hãy giữ khuôn mặt ở góc nhìn trực diện và đủ ánh sáng nhé!*`,
      };
    }

    // ─────────────────────────────────────────────────────────────
    // NHÓM 13: CÁC GỢI Ý MẶC ĐỊNH ĐÃ CÓ SẴN (ĐẢM BẢO HOẠT ĐỘNG CHUẨN XÁC)
    // ─────────────────────────────────────────────────────────────

    // 13.1 Lời khuyên dựa trên cảm xúc thường gặp nhất
    if (
      normText.includes("cai thien") ||
      normText.includes("loi khuyen") ||
      normText.includes("thu gian") ||
      normText.includes("stress") ||
      normText.includes("cang thang") ||
      normText.includes("bot met") ||
      normText.includes("lam sao de vui") ||
      normText.includes("tam trang")
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      const dominant = stats.dominantOverall;
      const domInfo = EMOTION_INFO[dominant.name] || { label: dominant.name };
      const adviceByEmotion = {
        happy: "Hãy duy trì nhịp làm việc đang phù hợp và ghi lại điều đang giúp bạn thấy vui.",
        neutral: "Thử nghỉ mắt ngắn, uống nước và xen kẽ thời gian tập trung với vài phút vận động nhẹ.",
        fatigue: "Hãy nghỉ ngắn, nhìn ra xa để thư giãn mắt, uống nước và cân nhắc đứng dậy vận động nhẹ.",
        sad: "Hãy dành thời gian cho một hoạt động nhỏ khiến bạn dễ chịu hoặc chia sẻ với người bạn tin tưởng nếu cần.",
        angry: "Thử tạm rời tác nhân gây căng thẳng, thở chậm vài nhịp và quay lại khi đã bình tĩnh hơn.",
        fear: "Thử dừng lại, thở chậm và xác định một việc nhỏ, cụ thể mà bạn có thể chủ động làm ngay lúc này.",
        disgust: "Bạn có thể đổi không gian hoặc chuyển sang một hoạt động nhẹ nhàng hơn trong ít phút.",
        surprise: "Hãy dành một chút thời gian ổn định nhịp thở và xem điều gì vừa khiến bạn bất ngờ.",
      };

      return {
        isDirectQuery: true,
        reply:
          `💡 Cảm xúc được ghi nhận nhiều nhất trong phiên là **${domInfo.label}** (${dominant.percentage}%).\n\n` +
          `**Lời khuyên phù hợp:** ${adviceByEmotion[dominant.name] || "Hãy nghỉ ngắn, thở chậm và chọn một hoạt động nhẹ nhàng phù hợp với bạn."}`,
      };
    }

    // 13.2 Tra cứu theo giây: "ở giây thứ X", "lúc X giây", "giây X"...
    const secondMatch =
      rawText.match(/(?:ở\s*)?(?:giây\s*(?:thứ\s*)?|lúc\s*)(\d+(?:[.,]\d+)?)\s*(?:giây|s)?/i) ||
      rawText.match(/(\d+(?:[.,]\d+)?)\s*(?:giây|s)\b/i);

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
        if (!e.dominantEmotion) {
          return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
        }
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

    // 13.3 Đếm frame theo cảm xúc: "bao nhiêu frame là vui", "mấy frame"...
    if (
      normText.includes("bao nhieu frame") ||
      normText.includes("may frame") ||
      normText.includes("so frame") ||
      normText.includes("chiem bao nhieu") ||
      normText.includes("ty le") ||
      normText.includes("may lan")
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
      }

      for (const [key, emotion] of Object.entries(EMOTION_MAP)) {
        if (normText.includes(normalizeText(key))) {
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

    // 13.4 Tổng quan / Thống kê / Phân tích phiên
    if (
      normText.includes("tong quan") ||
      normText.includes("thong ke") ||
      normText.includes("tong ket") ||
      normText.includes("tom tat") ||
      normText.includes("phan tich phien") ||
      normText.includes("phien lam viec")
    ) {
      if (!hasEmotionData) {
        return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
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
          `- ⏱️ Thời gian phiên: **${formatSessionDuration(stats.duration)}**\n` +
          `- 🎞️ Tổng số mốc ghi nhận: **${stats.totalFrames} mốc**\n` +
          `- 🌟 Cảm xúc chủ đạo nhất: **${domInfo ? `${domInfo.emoji} ${domInfo.label} (${dominant.percentage}%)` : "Chưa xác định"}**\n\n` +
          `**Chi tiết phân bổ biểu cảm:**\n${breakdownText || "  (Đang thu thập dữ liệu...)"}\n` +
          `👉 *Bạn có thể hỏi thêm: "ở giây thứ X cảm xúc gì?", "khi nào tôi vui?", hoặc "tôi có mệt mỏi không?".*`,
      };
    }

    // Kiểm tra câu hỏi chung về cảm xúc khi chưa có dữ liệu
    if (
      !hasEmotionData &&
      ["cam xuc", "tam trang", "tinh than", "met moi", "cang thang", "stress", "buon", "vui ve"]
        .some((keyword) => normText.includes(keyword))
    ) {
      return { isDirectQuery: true, reply: NO_EMOTION_DATA_REPLY };
    }

    // ─────────────────────────────────────────────────────────────
    // NẾU CÂU HỎI KHÔNG THUỘC CÁC MẪU TỰ TRẢ LỜI Ở TRÊN:
    // -> TRẢ VỀ NULL ĐỂ HỆ THỐNG GỌI API GEMINI (AI LLM) XỬ LÝ!
    // ─────────────────────────────────────────────────────────────
    return null;
  }
}
