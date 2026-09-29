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
    faces: list[FaceResult] = Field(
        default_factory=list,
        description="Danh sách kết quả phân tích theo từng khuôn mặt.",
    )
