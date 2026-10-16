"""
Module Dịch Vụ AI Chatbot — Phân tích tâm trạng và cảm xúc phiên làm việc.
Tích hợp Google Gemini API với cơ chế Smart Fallback đảm bảo không phụ thuộc cứng.
"""

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


class ChatAnalysisService:
    """Service điều phối phân tích cảm xúc qua Gemini API và bộ quy tắc dự phòng."""

    def __init__(self):
        # Đọc API key từ biến môi trường
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        self.gemini_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-flash:generateContent"
        )

    def _call_gemini_api(
        self,
        user_message: str,
        timeline_summary: Optional[Dict[str, Any]],
        history: List[Dict[str, str]],
    ) -> Optional[str]:
        """Gửi request tới Gemini REST API bằng urllib (không phụ thuộc SDK ngoài)."""
        if not self.api_key:
            return None

        system_instruction = (
            "Bạn là AI Trợ Lý Phân Tích Cảm Xúc & Sức Khỏe Tâm Lý của hệ thống AI Camera. "
            "Nhiệm vụ của bạn là dựa vào dữ liệu cảm xúc trong phiên làm việc của người dùng để: "
            "1. Nhận định sâu sắc về trạng thái tinh thần, mức độ tập trung, căng thẳng hoặc mệt mỏi. "
            "2. Đưa ra lời khuyên thực tế (thư giãn mắt, uống nước, nghỉ ngơi, duy trì cảm hứng). "
            "3. Khi nhắc đến bất kỳ mốc thời gian cụ thể nào, BẮT BUỘC định dạng rõ ràng trong ngoặc vuông "
            "như [12.5s] hoặc [45.0s] để người dùng có thể nhấp chuột xem lại frame đó trên camera. "
            "Sử dụng tiếng Việt tự nhiên, súc tích, đồng cảm và chuyên nghiệp."
        )

        # Xây dựng ngữ cảnh dữ liệu phiên
        summary_context = ""
        if timeline_summary:
            summary_context = (
                f"\n[DỮ LIỆU PHIÊN LÀM VIỆC HIỆN TẠI]:\n"
                f"- Thời lượng phiên: {timeline_summary.get('duration_seconds', 0)} giây\n"
                f"- Tổng số mốc kết quả: {timeline_summary.get('total_frames', 0)}\n"
                f"- Cảm xúc chủ đạo: {json.dumps(timeline_summary.get('dominant_emotion', {}), ensure_ascii=False)}\n"
                f"- Phân bổ % cảm xúc: {json.dumps(timeline_summary.get('emotion_percentages', {}), ensure_ascii=False)}\n"
                f"- Các mốc chuyển đổi gần đây: {json.dumps(timeline_summary.get('recent_events', []), ensure_ascii=False)}\n"
            )

        contents = []

        # Thêm lịch sử trò chuyện
        for msg in history[-6:]:  # Lấy tối đa 6 tin nhắn gần nhất
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
            "system_instruction": {
                "parts": [{"text": system_instruction}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 600,
            }
        }

        url = f"{self.gemini_url}?key={self.api_key}"
        data_bytes = json.dumps(request_body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                resp_json = json.loads(response.read().decode("utf-8"))
                candidates = resp_json.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
        except urllib.error.HTTPError as exc:
            print(f"[ChatService] Gemini HTTP error: {exc.code} - {exc.reason}")
        except Exception as exc:
            print(f"[ChatService] Gemini call failed: {exc}")

        return None

    def _generate_smart_fallback(
        self,
        user_message: str,
        timeline_summary: Optional[Dict[str, Any]],
    ) -> str:
        msg_lower = user_message.lower()

        # 1. Câu hỏi về lời khuyên / cách cải thiện cảm xúc & tâm trạng
        if any(w in msg_lower for w in ["cải thiện", "lời khuyên", "thư giãn", "stress", "căng thẳng", "mệt mỏi", "làm sao để vui", "tâm trạng"]):
            session_context = ""
            if timeline_summary and timeline_summary.get("total_frames", 0) > 0:
                dom = timeline_summary.get("dominant_emotion", {}).get("name", "")
                if dom:
                    session_context = f"\n\n📌 *Dữ liệu hiện tại cho thấy cảm xúc chính của bạn là: **{dom}**.*"

            return (
                "🌿 **CẨM NANG CẢI THIỆN CẢM XÚC & NĂNG LƯỢNG LÀM VIỆC**:\n\n"
                "1. 🧘 **Điều hòa hơi thở (Kỹ thuật 4-7-8)**: Hít vào bằng mũi trong 4s, giữ hơi 7s và thở ra từ từ bằng miệng trong 8s. Giúp kích hoạt hệ thần kinh phó giao cảm và giảm tức thì hormone cortisol gây căng thẳng.\n"
                "2. 👀 **Thư giãn thị giác (Quy tắc 20-20-20)**: Cứ 20 phút nhìn màn hình, hãy phóng tầm mắt ra xa 20 feet (6m) trong 20 giây để giảm mỏi cơ mắt.\n"
                "3. 💧 **Bổ sung nước & Vận động vi mô**: Uống một ly nước mát, thả lỏng quai hàm, xoay nhẹ khớp cổ và vai để giải tỏa áp lực cơ thể.\n"
                "4. ✨ **Kích hoạt phản hồi cơ mặt**: Mỉm cười nhẹ trong 10-15 giây. Nghiên cứu thần kinh học chứng minh não bộ sẽ tự động tiết Dopamine và Endorphin giúp cải thiện tâm trạng nhanh chóng."
                f"{session_context}\n\n"
                "---\n"
                "*(💡 Gợi ý: Hãy giữ camera bật ở chế độ Tự động, tôi sẽ theo dõi và nhắc bạn khi có dấu hiệu mệt mỏi!)*"
            )

        # 2. Nếu phiên chưa có dữ liệu và hỏi về phân tích phiên
        if not timeline_summary or timeline_summary.get("total_frames", 0) == 0:
            return (
                "👋 Chào bạn! Tôi là **AI Trợ lý Cảm xúc**.\n\n"
                "📊 **Trạng Thái Phiên Làm Việc**:\n"
                "Camera hiện đã sẵn sàng! Hệ thống đang hoạt động ở chế độ thông minh:\n"
                "• Tự động ghi nhận khi bạn **thay đổi biểu cảm** trước camera.\n"
                "• Hoặc **định kỳ sau mỗi 10 giây** nếu biểu cảm giữ nguyên.\n\n"
                "👉 Bạn hãy bật camera và thử mỉm cười hoặc biểu lộ một cảm xúc để tôi bắt đầu phân tích nhé!"
            )

        duration = timeline_summary.get("duration_seconds", 0)
        total_frames = timeline_summary.get("total_frames", 0)
        dominant = timeline_summary.get("dominant_emotion") or {}
        percentages = timeline_summary.get("emotion_percentages") or {}
        recent_events = timeline_summary.get("recent_events") or []

        dom_name = dominant.get("name", "neutral")
        dom_pct = dominant.get("percentage", 0)

        # Đánh giá năng lượng
        happy_pct = percentages.get("happy", 0)
        neutral_pct = percentages.get("neutral", 0)
        fatigue_pct = percentages.get("fatigue", 0)
        sad_pct = percentages.get("sad", 0)
        angry_pct = percentages.get("angry", 0)

        tone_assessment = ""
        advice = ""

        if happy_pct >= 40:
            tone_assessment = (
                f"🌟 **Trạng thái tinh thần rất tích cực!** Bạn duy trì biểu cảm vui vẻ trong "
                f"**{happy_pct}%** thời gian làm việc."
            )
            advice = "Hãy tiếp tục duy trì nguồn năng lượng tuyệt vời này nhé!"
        elif fatigue_pct >= 25:
            tone_assessment = (
                f"⚠️ **Dấu hiệu mệt mỏi đáng chú ý!** Cảm xúc mệt mỏi / buồn ngủ chiếm "
                f"**{fatigue_pct}%** thời lượng phiên."
            )
            advice = (
                "Bạn nên áp dụng quy tắc 20-20-20: nhìn ra xa 20 feet trong 20 giây, "
                "đứng dậy vận động nhẹ và uống một ly nước để mắt được thư giãn."
            )
        elif neutral_pct >= 50:
            tone_assessment = (
                f"🎯 **Trạng thái tập trung và điềm tĩnh.** Biểu cảm bình thường chiếm "
                f"**{neutral_pct}%** thời gian, cho thấy sự tập trung cao độ vào công việc."
            )
            advice = "Đừng quên chớp mắt đều đặn để tránh khô mắt khi nhìn màn hình liên tục."
        else:
            tone_assessment = (
                f"📊 **Tâm trạng có sự biến chuyển đa dạng**, với cảm xúc chủ đạo là "
                f"**{dom_name}** ({dom_pct}%)."
            )
            advice = "Hãy hít thở sâu vài nhịp nếu cảm thấy có dấu hiệu căng thẳng."

        # Trích dẫn mốc sự kiện gần đây nếu có
        events_text = ""
        if recent_events:
            event_strs = []
            for ev in recent_events[-3:]:
                ts = ev.get("timestamp", 0)
                em = ev.get("emotion", "neutral")
                event_strs.append(f"mốc **[{ts}s]** ({em})")
            events_text = f"\n\nCác mốc thay đổi cảm xúc đáng chú ý: {', '.join(event_strs)}."

        response = (
            f"🧠 **Đánh Giá Tâm Trạng Phiên Làm Việc** ({int(duration)}s / #{total_frames} mốc):\n\n"
            f"{tone_assessment}\n\n"
            f"💡 **Lời khuyên**: {advice}"
            f"{events_text}\n\n"
            f"---\n"
            f"*(💡 Gợi ý: Cấu hình `GEMINI_API_KEY` trong file `.env` để kích hoạt phản hồi AI sáng tạo không giới hạn từ Google Gemini)*"
        )
        return response

    def answer_query(
        self,
        message: str,
        timeline_summary: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Điểm đầu vào chính: Thử gọi Gemini API, nếu không thành công chuyển sang Smart Fallback.
        """
        history = history or []

        # Thử gọi Gemini
        gemini_reply = self._call_gemini_api(message, timeline_summary, history)
        if gemini_reply:
            return {
                "success": True,
                "reply": gemini_reply,
                "source": "gemini",
            }

        # Dùng Smart Fallback
        fallback_reply = self._generate_smart_fallback(message, timeline_summary)
        return {
            "success": True,
            "reply": fallback_reply,
            "source": "smart_fallback",
        }


# Singleton instance
chat_service = ChatAnalysisService()
