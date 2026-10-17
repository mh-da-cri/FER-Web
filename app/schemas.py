"""
Các schema Pydantic cho model request / response.
"""

from pydantic import BaseModel, Field


class Base64Request(BaseModel):
    """Body request khi gửi ảnh dưới dạng chuỗi Base64."""

    image_base64: str = Field(
        ...,
        description="Chuỗi ảnh mã hóa Base64. Có thể bao gồm tiền tố data-URI.",
        examples=["data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
    )


class FaceResult(BaseModel):
    """Kết quả phân tích cho một khuôn mặt đã phát hiện (hoặc không có)."""

    isFace: bool = Field(
        ...,
        description="Có phát hiện khuôn mặt hay không.",
    )
    box: list[int] = Field(
        default_factory=list,
        description="Khung bao dạng [x, y, chiều rộng, chiều cao] tính bằng pixel. Rỗng nếu không có khuôn mặt.",
    )
    emotions: dict[str, float] = Field(
        default_factory=dict,
        description="Tên cảm xúc → điểm tin cậy (0.0–1.0).",
    )
    fatigue_score: float = Field(
        default=0.0,
        description="Chỉ số mệt mỏi 0.0–1.0; luôn được hiển thị ở UI để theo dõi trạng thái mệt mỏi dù cảm xúc chủ đạo không phải fatigue.",
    )


class AnalyzeResponse(BaseModel):
    """Response cấp cao nhất trả về bởi POST /api/analyze."""

    success: bool = Field(
        ...,
        description="Request có được xử lý thành công hay không.",
    )
    message: str = Field(
        default="",
        description="Thông báo trạng thái dễ đọc.",
    )
    faces: list[FaceResult] = Field(
        default_factory=list,
        description="Danh sách kết quả phân tích theo từng khuôn mặt.",
    )


# ------------------------------------------------------------------
# Schemas cho AI Chatbot Phân Tích Cảm Xúc
# ------------------------------------------------------------------
class ChatMessage(BaseModel):
    """Tin nhắn trong đoạn hội thoại chat."""

    role: str = Field(default="user", description="'user' hoặc 'model'/'assistant'")
    content: str = Field(..., description="Nội dung tin nhắn")


class ChatAnalyzeRequest(BaseModel):
    """Yêu cầu phân tích tâm trạng phiên từ client."""

    message: str = Field(..., description="Câu hỏi hoặc yêu cầu từ người dùng")
    timeline_summary: dict | None = Field(
        default=None,
        description="Bản tóm tắt dữ liệu phiên làm việc từ SessionTimeline",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Lịch sử các tin nhắn trò chuyện trước đó",
    )


class ChatAnalyzeResponse(BaseModel):
    """Phản hồi phân tích từ AI Chatbot."""

    success: bool = Field(..., description="Trạng thái xử lý")
    reply: str = Field(..., description="Câu trả lời từ AI")
    source: str = Field(
        default="gemini",
        description="Nguồn phản hồi: 'gemini' hoặc 'smart_fallback'",
    )
