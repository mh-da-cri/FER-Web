"""
Module Dịch Vụ AI Chatbot — Phân tích tâm trạng và cảm xúc phiên làm việc.
Tích hợp xử lý tự động câu hỏi số liệu/trạng thái và gọi Google Gemini API khi gặp câu hỏi mở.
"""

import json
import logging
import os
import re
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_ADVICE_BY_EMOTION = {
    "happy": "Hãy duy trì nhịp làm việc đang phù hợp và ghi lại điều đang giúp bạn thấy vui.",
    "neutral": "Thử nghỉ mắt ngắn, uống nước và xen kẽ thời gian tập trung với vài phút vận động nhẹ.",
    "fatigue": "Hãy nghỉ ngắn, nhìn ra xa để thư giãn mắt, uống nước và cân nhắc đứng dậy vận động nhẹ.",
    "sad": "Hãy dành thời gian cho một hoạt động nhỏ khiến bạn dễ chịu hoặc chia sẻ với người bạn tin tưởng nếu cần.",
    "angry": "Thử tạm rời tác nhân gây căng thẳng, thở chậm vài nhịp và quay lại khi đã bình tĩnh hơn.",
    "fear": "Thử dừng lại, thở chậm và xác định một việc nhỏ, cụ thể mà bạn có thể chủ động làm ngay lúc này.",
    "disgust": "Bạn có thể đổi không gian hoặc chuyển sang một hoạt động nhẹ nhàng hơn trong ít phút.",
    "surprise": "Hãy dành một chút thời gian ổn định nhịp thở và xem điều gì vừa khiến bạn bất ngờ.",
}

_EMOTION_LABELS = {
    "happy": "vui vẻ",
    "neutral": "bình thường",
    "fatigue": "mệt mỏi",
    "sad": "buồn",
    "angry": "tức giận",
    "fear": "sợ hãi",
    "disgust": "khó chịu",
    "surprise": "ngạc nhiên",
}

_EMOTION_EMOJIS = {
    "happy": "😄",
    "neutral": "😐",
    "fatigue": "😴",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "disgust": "🤢",
    "surprise": "😲",
}

_EMOTION_KEYWORDS = {
    "vui": "happy",
    "vui ve": "happy",
    "cuoi": "happy",
    "happy": "happy",
    "buon": "sad",
    "buon ba": "sad",
    "sad": "sad",
    "gian": "angry",
    "tuc gian": "angry",
    "tuc": "angry",
    "angry": "angry",
    "ngac nhien": "surprise",
    "surprise": "surprise",
    "so hai": "fear",
    "so": "fear",
    "fear": "fear",
    "ghe tom": "disgust",
    "disgust": "disgust",
    "binh thuong": "neutral",
    "neutral": "neutral",
    "met moi": "fatigue",
    "met": "fatigue",
    "buon ngu": "fatigue",
    "fatigue": "fatigue",
    "tired": "fatigue",
}


def _strip_diacritics(text: str) -> str:
    """Loại bỏ dấu tiếng Việt để so khớp câu hỏi linh hoạt."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    stripped = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return stripped.replace("đ", "d").replace("Đ", "D").lower().strip()


class ChatAnalysisService:
    """Service điều phối phân tích cảm xúc: Tự trả lời các câu số liệu & Gọi Gemini cho câu hỏi mở."""

    def __init__(self):
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()

    @staticmethod
    def _summary_value(
        summary: Optional[Dict[str, Any]],
        *keys: str,
        default: Any = None,
    ) -> Any:
        if not summary:
            return default
        for key in keys:
            if key in summary:
                return summary[key]
        return default

    @classmethod
    def _summary_emotion_data(cls, summary: Optional[Dict[str, Any]]) -> tuple:
        dominant = cls._summary_value(
            summary, "dominant_emotion", "dominantOverall", default={}
        ) or {}
        percentages = cls._summary_value(
            summary, "emotion_percentages", "emotionPercentages", default={}
        ) or {}
        total_frames = cls._summary_value(summary, "total_frames", "totalFrames", default=0) or 0
        return dominant, percentages, total_frames

    @staticmethod
    def _clean_reply(reply: str) -> str:
        return reply.replace("*", "").replace("•", "-")

    @staticmethod
    def _format_duration(seconds: Any) -> str:
        total_seconds = max(0, int(seconds or 0))
        if total_seconds < 60:
            return f"{total_seconds} giây"
        return f"{total_seconds // 60} phút {total_seconds % 60} giây"

    def _call_gemini_api(
        self,
        user_message: str,
        timeline_summary: Optional[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> Optional[str]:
        """Gửi request tới Gemini REST API bằng urllib (không phụ thuộc SDK ngoài)."""
        if not self.api_key:
            logger.warning("Không có GEMINI_API_KEY, bỏ qua gọi Gemini API.")
            return None

        system_instruction = (
            "Bạn là AI Trợ Lý Phân Tích Biểu Cảm của hệ thống AI Camera. "
            "Dựa trên dữ liệu biểu cảm được ghi nhận trong phiên để: "
            "1. Mô tả xu hướng biểu cảm, không chẩn đoán hoặc khẳng định trạng thái tâm lý của người dùng. "
            "2. Đưa ra lời khuyên thực tế (thư giãn mắt, uống nước, nghỉ ngơi, duy trì cảm hứng). "
            "3. Khi nhắc đến bất kỳ mốc thời gian cụ thể nào, BẮT BUỘC định dạng rõ ràng trong ngoặc vuông "
            "như [12.5s] hoặc [45.0s] để người dùng có thể nhấp chuột xem lại frame đó trên camera. "
            "Chỉ nhận xét cảm xúc dựa trên dữ liệu phiên được cung cấp; nếu không có dữ liệu cảm xúc, "
            "hãy nói rõ chưa có dữ liệu và không suy đoán. Nếu được hỏi lời khuyên, hãy cá nhân hóa "
            "theo cảm xúc xuất hiện nhiều nhất trong phiên. Thời lượng dưới 60 giây ghi bằng giây; "
            "từ 60 giây trở lên ghi bằng phút và giây. "
            "Sử dụng tiếng Việt tự nhiên, súc tích, đồng cảm và chuyên nghiệp."
        )

        # Xây dựng ngữ cảnh dữ liệu phiên
        summary_context = ""
        if timeline_summary:
            dominant, percentages, total_frames = self._summary_emotion_data(timeline_summary)
            summary_context = (
                f"\n[DỮ LIỆU PHIÊN LÀM VIỆC HIỆN TẠI]:\n"
                f"- Thời lượng phiên: {self._format_duration(self._summary_value(timeline_summary, 'duration_seconds', 'duration', default=0))}\n"
                f"- Tổng số mốc kết quả: {total_frames}\n"
                f"- Cảm xúc chủ đạo: {json.dumps(dominant, ensure_ascii=False)}\n"
                f"- Phân bổ % cảm xúc: {json.dumps(percentages, ensure_ascii=False)}\n"
                f"- Các mốc chuyển đổi gần đây: {json.dumps(timeline_summary.get('recent_events', []), ensure_ascii=False)}\n"
            )
            if not total_frames or not dominant:
                summary_context += "- Chưa có dữ liệu cảm xúc được ghi nhận trong phiên.\n"
        else:
            summary_context = "\n[DỮ LIỆU PHIÊN LÀM VIỆC HIỆN TẠI]:\n- Chưa có dữ liệu cảm xúc được ghi nhận trong phiên.\n"

        contents = []

        # Thêm lịch sử trò chuyện
        for msg in history[-6:]:
            role = "user" if msg.get("role") in ["user", "human"] else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg.get("content", "")}]
            })

        # Thêm câu hỏi hiện tại kèm ngữ cảnh
        current_prompt = f"{summary_context}\nCâu hỏi của người dùng: {user_message}"
        contents.append({
            "role": "user",
            "parts": [{"text": current_prompt}]
        })

        request_body = {
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": 600,
            }
        }

        data_bytes = json.dumps(request_body).encode("utf-8")

        # Thử danh sách các model khả dụng
        candidate_models = [self.model, "gemini-3.8-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]
        tried_models = set()

        for model_name in candidate_models:
            if not model_name or model_name in tried_models:
                continue
            tried_models.add(model_name)

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent?key={self.api_key}"
            )
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    candidates = resp_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    logger.warning("Gemini API đạt giới hạn hạn mức (HTTP 429 Too Many Requests).")
                    break
                elif exc.code == 404:
                    logger.warning("Model %s trả về HTTP 404, thử model tiếp theo...", model_name)
                    continue
                else:
                    logger.error("Gemini API trả về HTTP %s (%s)", exc.code, exc.reason)
                    break
            except Exception as exc:
                logger.warning("Lỗi gọi Gemini API (%s): %s", model_name, exc)
                break

        return None

    def _try_direct_answer(
        self,
        message: str,
        timeline_summary: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Tự động trả lời câu hỏi nếu thuộc danh mục số liệu, trạng thái, chào hỏi hoặc quy tắc.
        Trả về chuỗi câu trả lời nếu tự trả lời được, hoặc None nếu không trả lời được.
        """
        norm = _strip_diacritics(message)
        dominant, percentages, total_frames = self._summary_emotion_data(timeline_summary)
        duration = self._summary_value(timeline_summary, "duration_seconds", "duration", default=0)
        has_emotion = bool(total_frames and dominant)

        # 1. Chào hỏi
        greeting_phrases = ("xin chao", "chao ban", "chao bot", "chao tro ly")
        if (
            norm in ("hi", "hello", "alo", "hey", "chao")
            or any(w in norm for w in greeting_phrases)
            or norm.startswith("chao ")
            or bool(re.search(r"\b(hi|hello|hey|alo)\b", norm))
        ):
            duration_str = self._format_duration(duration)
            return (
                f"👋 **Xin chào!** Tôi là **AI Trợ Lý Phân Tích Cảm Xúc**.\n\n"
                f"Phiên làm việc hiện tại đã chạy được **{duration_str}** với **{total_frames} mốc dữ liệu**.\n"
                f"Bạn có thể hỏi tôi về tóm tắt phiên, mức độ mệt mỏi, độ tập trung, hoặc tra cứu theo mốc thời gian nhé!"
            )

        # 2. Cảm ơn & Tạm biệt
        if any(w in norm for w in ("cam on", "thank you", "thanks")):
            return "😊 Rất vui được đồng hành cùng bạn! Chúc bạn làm việc hiệu quả và luôn thoải mái nhé."
        if any(w in norm for w in ("tam biet", "bye", "goodbye")):
            return "👋 Tạm biệt bạn! Hãy nghỉ ngơi đầy đủ và hẹn gặp lại trong phiên làm việc tiếp theo nhé!"

        # 3. Giới thiệu bản thân
        if any(w in norm for w in ("ban la ai", "ban lam duoc gi", "huong dan su dung", "he thong nay la gi")):
            return (
                "🤖 **AI Trợ Lý Cảm Xúc Camera**:\n\n"
                "- Tự động trả lời số liệu chính xác 100%: Tóm tắt phiên, tỷ lệ cảm xúc, đánh giá mệt mỏi, mức độ tập trung, tra cứu theo giây.\n"
                "- Tích hợp Google Gemini AI cho các phân tích tâm lý chuyên sâu và câu hỏi mở."
            )

        # 4. Đánh giá Mệt mỏi & Buồn ngủ
        if any(w in norm for w in ("co met khong", "toi co met", "dau hieu met", "buon ngu", "co buon ngu", "met moi", "ue oai")):
            if not has_emotion:
                return "Chưa có dữ liệu cảm xúc trong phiên để đánh giá mức độ mệt mỏi. Hãy bật camera và ghi nhận kết quả trước nhé!"
            fatigue_pct = percentages.get("fatigue", 0)
            if fatigue_pct >= 25 or (isinstance(dominant, dict) and dominant.get("name") == "fatigue"):
                return (
                    f"⚠️ **Cảnh báo dấu hiệu mệt mỏi:**\n\n"
                    f"- Biểu cảm mệt mỏi/buồn ngủ chiếm **{fatigue_pct}%** thời lượng phiên làm việc.\n"
                    f"💡 Lời khuyên: Hãy nhìn ra xa khoảng 20 giây (quy tắc 20-20-20), uống một ngụm nước và vươn vai nhẹ nhàng nhé!"
                )
            return (
                f"✅ **Bạn đang duy trì trạng thái khá tốt:**\n\n"
                f"- Tỷ lệ mệt mỏi chỉ chiếm **{fatigue_pct}%** (dưới ngưỡng cảnh báo 25%).\n"
                f"- Biểu cảm chủ đạo là **{_EMOTION_LABELS.get(dominant.get('name') if isinstance(dominant, dict) else dominant, 'bình thường')}**."
            )

        # 5. Đánh giá Mức độ Tập trung
        if any(w in norm for w in ("tap trung", "co tap trung khong", "muc do tap trung", "nghiem tuc", "phan tam")):
            if not has_emotion:
                return "Chưa có dữ liệu cảm xúc trong phiên để đánh giá mức độ tập trung. Hãy bật camera trước nhé!"
            neutral_pct = percentages.get("neutral", 0)
            happy_pct = percentages.get("happy", 0)
            fatigue_pct = percentages.get("fatigue", 0)
            focus_score = min(100, int(neutral_pct * 0.7 + happy_pct * 0.3))
            return (
                f"🎯 **Đánh giá mức độ tập trung:**\n\n"
                f"- Điểm tập trung ước tính: **{focus_score}/100**\n"
                f"- Trạng thái bình thường (tập trung làm việc): **{neutral_pct}%**\n"
                f"- Trạng thái mệt mỏi/uể oải: **{fatigue_pct}%**\n\n"
                f"💡 Lời khuyên: Tiếp tục duy trì nhịp độ làm việc và tư thế thoải mái."
            )

        # 6. Cảm xúc chủ đạo / nhiều nhất
        if (any(w in norm for w in ("cam xuc", "bieu cam", "tam trang")) and
                any(w in norm for w in ("nhieu nhat", "chu dao", "chinh la gi", "chiem uu the", "pho bien nhat", "ap dao"))):
            if not has_emotion:
                return "Chưa có dữ liệu cảm xúc trong phiên để xác định cảm xúc chủ đạo. Hãy bật camera trước nhé!"
            dom_name = dominant.get("name") if isinstance(dominant, dict) else str(dominant)
            dom_pct = dominant.get("percentage", percentages.get(dom_name, 0)) if isinstance(dominant, dict) else percentages.get(dom_name, 0)
            dom_label = _EMOTION_LABELS.get(dom_name, dom_name)
            dom_emoji = _EMOTION_EMOJIS.get(dom_name, "✨")
            return (
                f"🌟 **Cảm xúc chủ đạo xuất hiện nhiều nhất:**\n\n"
                f"- Biểu cảm: **{dom_emoji} {dom_label.title()}**\n"
                f"- Tỷ lệ phân bổ: **{dom_pct}%** toàn bộ thời lượng phiên ({self._format_duration(duration)})."
            )

        # 7. Cảm xúc ít nhất
        if (any(w in norm for w in ("cam xuc", "bieu cam")) and
                any(w in norm for w in ("it nhat", "thap nhat", "hiem nhat"))):
            if not has_emotion:
                return "Chưa có dữ liệu cảm xúc để xác định cảm xúc ít nhất."
            sorted_pcts = sorted(percentages.items(), key=lambda x: x[1])
            if sorted_pcts:
                least_name, least_pct = sorted_pcts[0]
                least_label = _EMOTION_LABELS.get(least_name, least_name)
                least_emoji = _EMOTION_EMOJIS.get(least_name, "•")
                return (
                    f"📉 **Cảm xúc xuất hiện ít nhất:**\n\n"
                    f"- Biểu cảm: **{least_emoji} {least_label.title()}**\n"
                    f"- Tỷ lệ phân bổ: **{least_pct}%**."
                )

        # 8. Tỷ lệ tích cực vs tiêu cực / Năng lượng
        if any(w in norm for w in ("tich cuc", "tieu cuc", "nang luong")):
            if not has_emotion:
                return "Chưa có dữ liệu cảm xúc trong phiên để đánh giá tỷ lệ tích cực."
            pos = percentages.get("happy", 0)
            neu = percentages.get("neutral", 0)
            neg = sum(percentages.get(e, 0) for e in ("sad", "fatigue", "angry", "fear", "disgust"))
            return (
                f"⚡ **Phân bổ năng lượng cảm xúc:**\n\n"
                f"- 🟢 Tích cực (Vui vẻ): **{pos}%**\n"
                f"- ⚪ Trung tính (Bình thường): **{neu}%**\n"
                f"- 🔴 Cần lưu ý (Mệt mỏi, buồn, căng thẳng): **{neg:.1f}%**"
            )

        # 9. Thời lượng & Số lượng mốc
        if any(w in norm for w in ("keo dai bao lau", "chay duoc bao lau", "thoi gian phien", "tong so frame", "bao nhieu frame", "bao nhieu ket qua")):
            return (
                f"⏱️ **Tiến độ phiên làm việc hiện tại:**\n\n"
                f"- Thời lượng phiên: **{self._format_duration(duration)}**\n"
                f"- Tổng số mốc ghi nhận: **{total_frames} mốc kết quả**."
            )

        # 10. Đếm frame theo cảm xúc
        if any(w in norm for w in ("bao nhieu frame", "may frame", "chiem bao nhieu", "ty le")):
            for key, em in _EMOTION_KEYWORDS.items():
                if key in norm:
                    if not has_emotion:
                        return "Chưa có dữ liệu cảm xúc trong phiên."
                    pct = percentages.get(em, 0)
                    em_label = _EMOTION_LABELS.get(em, em)
                    em_emoji = _EMOTION_EMOJIS.get(em, "•")
                    return (
                        f"📊 Biểu cảm **{em_emoji} {em_label}** chiếm **{pct}%** tổng thời lượng phiên làm việc."
                    )

        # 11. Tổng kết / Tóm tắt phiên
        if any(w in norm for w in ("tong quan", "thong ke", "tong ket", "tom tat", "phan tich phien")):
            if not has_emotion:
                return "Chưa có dữ liệu cảm xúc trong phiên để tóm tắt. Hãy bật camera trước nhé!"
            dom_name = dominant.get("name") if isinstance(dominant, dict) else str(dominant)
            dom_pct = dominant.get("percentage", percentages.get(dom_name, 0)) if isinstance(dominant, dict) else percentages.get(dom_name, 0)
            lines = [f"📈 **Bản tổng hợp phiên làm việc** ({self._format_duration(duration)} / {total_frames} mốc):\n"]
            lines.append(f"- Cảm xúc chủ đạo: **{_EMOTION_LABELS.get(dom_name, dom_name).title()}** ({dom_pct}%)")
            lines.append("- Phân bổ chi tiết:")
            for em, pct in percentages.items():
                if pct > 0:
                    lines.append(f"  • {_EMOTION_EMOJIS.get(em, '•')} {_EMOTION_LABELS.get(em, em).title()}: {pct}%")
            return "\n".join(lines)

        # 12. Lời khuyên cải thiện
        if any(w in norm for w in ("cai thien", "loi khuyen", "thu gian", "stress", "cang thang", "lam sao de vui", "tam trang")):
            if not has_emotion:
                return "Chưa có dữ liệu cảm xúc trong phiên để đưa ra lời khuyên phù hợp."
            emotion = dominant.get("name") if isinstance(dominant, dict) else str(dominant)
            advice = _ADVICE_BY_EMOTION.get(emotion, "Hãy nghỉ ngắn, thở chậm và chọn hoạt động nhẹ nhàng phù hợp.")
            dom_label = _EMOTION_LABELS.get(emotion, emotion)
            pct = percentages.get(emotion, 0)
            return f"Biểu cảm ghi nhận nhiều nhất là {dom_label} ({pct}%). Lời khuyên: {advice}"

        # Không thuộc diện tự trả lời được -> trả về None để gọi Gemini API
        return None

    def _generate_smart_fallback(
        self,
        user_message: str,
        timeline_summary: Optional[Dict[str, Any]],
    ) -> str:
        """Hàm dự phòng thông minh khi Gemini API không phản hồi hoặc hết hạn mức."""
        # Thử kiểm tra câu trả lời trực tiếp trước
        direct = self._try_direct_answer(user_message, timeline_summary)
        if direct:
            return direct

        dominant, percentages, total_frames = self._summary_emotion_data(timeline_summary)
        if not total_frames or not dominant:
            return (
                "Chưa có dữ liệu cảm xúc trong phiên để trả lời câu hỏi này. "
                "Hãy bật camera và ghi nhận một vài kết quả trước nhé!"
            )

        duration = self._summary_value(timeline_summary, "duration_seconds", "duration", default=0)
        dom_name = dominant.get("name", "neutral") if isinstance(dominant, dict) else str(dominant)
        dom_pct = dominant.get("percentage", percentages.get(dom_name, 0)) if isinstance(dominant, dict) else percentages.get(dom_name, 0)

        advice = _ADVICE_BY_EMOTION.get(
            dom_name,
            "Hãy nghỉ ngắn, thở chậm và chọn một hoạt động nhẹ nhàng phù hợp với bạn.",
        )

        return (
            f"🧠 **Đánh giá phiên làm việc** ({self._format_duration(duration)} / {total_frames} mốc kết quả):\n\n"
            f"- Cảm xúc chủ đạo: **{_EMOTION_LABELS.get(dom_name, dom_name).title()}** ({dom_pct}%).\n"
            f"- Lời khuyên thư giãn: {advice}\n\n"
            f"*(Lưu ý: Gemini AI tạm thời chưa phản hồi câu hỏi mở này do hạn mức, câu trả lời trên được tổng hợp từ dữ liệu phiên của bạn).* "
        )

    def answer_query(
        self,
        message: str,
        timeline_summary: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Điểm đầu vào chính:
        1. Được thì TỰ TRẢ LỜI ngay lập tức (dữ liệu phiên, số liệu, mệt mỏi, tập trung, chào hỏi).
        2. Câu không trả lời được bằng quy tắc thì GỌI API GEMINI.
        3. Nếu Gemini lỗi hoặc hạn mức -> Dùng Smart Fallback an toàn.
        """
        history = history or []

        # ─── BƯỚC 1: Được thì TỰ TRẢ LỜI ───
        direct_reply = self._try_direct_answer(message, timeline_summary)
        if direct_reply:
            return {
                "success": True,
                "reply": self._clean_reply(direct_reply),
                "source": "direct_engine",
            }

        # ─── BƯỚC 2: Câu không trả lời được -> GỌI API GEMINI ───
        gemini_reply = self._call_gemini_api(message, timeline_summary, history)
        if gemini_reply:
            return {
                "success": True,
                "reply": self._clean_reply(gemini_reply),
                "source": "gemini",
            }

        # ─── BƯỚC 3: Dự phòng thông minh nếu Gemini tạm thời gián đoạn ───
        fallback_reply = self._generate_smart_fallback(message, timeline_summary)
        return {
            "success": True,
            "reply": self._clean_reply(fallback_reply),
            "source": "smart_fallback",
        }


# Singleton instance
chat_service = ChatAnalysisService()
