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
    model_used: str = Field(
        default="mobilenetv2",
        description="Mô hình AI đã sử dụng để phân tích ('mobilenetv2' hoặc 'ferplus').",
    )
    faces: list[FaceResult] = Field(
        default_factory=list,
        description="Danh sách kết quả phân tích theo từng khuôn mặt.",
    )


# ------------------------------------------------------------------
# Schemas cho Quản lý & Chuyển đổi Mô hình AI
# ------------------------------------------------------------------
class ModelItem(BaseModel):
    """Thông tin chi tiết của một mô hình AI."""

    id: str = Field(..., description="Mã định danh mô hình ('mobilenetv2' hoặc 'ferplus')")
    name: str = Field(..., description="Tên hiển thị thân thiện")
    description: str = Field(..., description="Mô tả tóm tắt tính năng và ưu điểm của mô hình")
    is_active: bool = Field(default=False, description="Mô hình hiện đang kích hoạt hay không")
    is_loaded: bool = Field(default=True, description="Mô hình đã được nạp sẵn vào bộ nhớ chưa")


class ModelListResponse(BaseModel):
    """Danh sách các mô hình và mô hình đang kích hoạt."""

    success: bool = Field(default=True, description="Trạng thái request")
    active_model: str = Field(..., description="ID của mô hình đang hoạt động")
    models: list[ModelItem] = Field(default_factory=list, description="Danh sách các mô hình khả dụng")


class SelectModelRequest(BaseModel):
    """Yêu cầu chuyển đổi mô hình đang hoạt động."""

    model: str = Field(..., description="ID mô hình muốn chuyển sang ('mobilenetv2' hoặc 'ferplus')")


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
