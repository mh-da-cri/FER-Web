"""
Router API — định nghĩa tất cả endpoint cho phân tích biểu cảm.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.analyzer import FaceExpressionAnalyzer
from app.schemas import AnalyzeResponse, Base64Request, FaceResult

router = APIRouter(prefix="/api", tags=["Analysis"])

# Singleton analyzer — khởi tạo một lần, tái sử dụng cho mọi request.
_analyzer = FaceExpressionAnalyzer(min_detection_confidence=0.5)


# ------------------------------------------------------------------
# POST /api/analyze  —  nhận file upload
# ------------------------------------------------------------------
@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Phân tích biểu cảm khuôn mặt từ ảnh tải lên",
    description=(
        "Tải lên file ảnh (JPEG, PNG, v.v.) và nhận kết quả "
        "phân tích cảm xúc theo từng khuôn mặt."
    ),
)
async def analyze_image_file(
    file: UploadFile = File(..., description="File ảnh cần phân tích"),
) -> AnalyzeResponse:
    """Phân tích biểu cảm khuôn mặt từ file ảnh tải lên."""

    # Kiểm tra loại nội dung file
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Loại file không hợp lệ '{file.content_type}'. Vui lòng tải lên file ảnh.",
        )

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="File tải lên rỗng.")

        image_bgr = _analyzer.decode_upload_bytes(file_bytes)
        results = _analyzer.analyze_image(image_bgr)

        return AnalyzeResponse(
            success=True,
            message=f"Phát hiện {sum(1 for r in results if r['isFace'])} khuôn mặt.",
            faces=[FaceResult(**r) for r in results],
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Phân tích thất bại: {exc}")


# ------------------------------------------------------------------
# POST /api/analyze/base64  —  nhận chuỗi Base64
# ------------------------------------------------------------------
@router.post(
    "/analyze/base64",
    response_model=AnalyzeResponse,
    summary="Phân tích biểu cảm khuôn mặt từ ảnh Base64",
    description=(
        "Gửi chuỗi ảnh mã hóa Base64 và nhận kết quả "
        "phân tích cảm xúc theo từng khuôn mặt."
    ),
)
async def analyze_image_base64(body: Base64Request) -> AnalyzeResponse:
    """Phân tích biểu cảm khuôn mặt từ ảnh mã hóa Base64."""

    if not body.image_base64 or not body.image_base64.strip():
        raise HTTPException(status_code=400, detail="Trường image_base64 rỗng.")

    try:
        image_bgr = _analyzer.decode_base64(body.image_base64)
        results = _analyzer.analyze_image(image_bgr)

        return AnalyzeResponse(
            success=True,
            message=f"Phát hiện {sum(1 for r in results if r['isFace'])} khuôn mặt.",
            faces=[FaceResult(**r) for r in results],
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Phân tích thất bại: {exc}")
