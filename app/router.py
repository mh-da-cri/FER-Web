"""
Router API — định nghĩa tất cả endpoint cho phân tích biểu cảm.
"""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.analyzer import FaceExpressionAnalyzer
from app.schemas import (
    AnalyzeResponse,
    Base64Request,
    ChatAnalyzeRequest,
    ChatAnalyzeResponse,
    FaceResult,
    ModelItem,
    ModelListResponse,
    SelectModelRequest,
)
from app.services.chat_service import chat_service

router = APIRouter(prefix="/api", tags=["Analysis"])

# Singleton analyzer — khởi tạo một lần, tái sử dụng cho mọi request.
_analyzer = FaceExpressionAnalyzer(min_detection_confidence=0.5)


# ------------------------------------------------------------------
# GET /api/models  —  Lấy danh sách các mô hình
# ------------------------------------------------------------------
@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="Lấy danh sách các mô hình AI khả dụng",
    description="Trả về danh sách mô hình cảm xúc cùng mô hình đang kích hoạt.",
)
async def get_models() -> ModelListResponse:
    """Trả về danh sách các mô hình khả dụng và mô hình đang hoạt động."""
    models_data = _analyzer.get_available_models()
    return ModelListResponse(
        success=True,
        active_model=_analyzer.get_active_model_name(),
        models=[ModelItem(**m) for m in models_data],
    )


# ------------------------------------------------------------------
# POST /api/models/select  —  Chuyển đổi mô hình hoạt động
# ------------------------------------------------------------------
@router.post(
    "/models/select",
    response_model=ModelListResponse,
    summary="Chuyển đổi mô hình AI đang sử dụng",
    description="Chuyển đổi tức thì mô hình mặc định ('mobilenetv2' hoặc 'ferplus').",
)
async def select_model(body: SelectModelRequest) -> ModelListResponse:
    """Chuyển đổi mô hình AI."""
    success = _analyzer.set_model(body.model)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Mô hình '{body.model}' không khả dụng hoặc không hợp lệ.",
        )
    models_data = _analyzer.get_available_models()
    return ModelListResponse(
        success=True,
        active_model=_analyzer.get_active_model_name(),
        models=[ModelItem(**m) for m in models_data],
    )


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
    fast: bool = False,
    model: str | None = None,
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
        results = _analyzer.analyze_image(image_bgr, fast_mode=fast, model_name=model)
        used_model = model or _analyzer.get_active_model_name()

        return AnalyzeResponse(
            success=True,
            message=f"Phát hiện {sum(1 for r in results if r['isFace'])} khuôn mặt.",
            model_used=used_model,
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
async def analyze_image_base64(
    body: Base64Request,
    fast: bool = False,
    model: str | None = None,
) -> AnalyzeResponse:
    """Phân tích biểu cảm khuôn mặt từ ảnh mã hóa Base64."""

    if not body.image_base64 or not body.image_base64.strip():
        raise HTTPException(status_code=400, detail="Trường image_base64 rỗng.")

    try:
        image_bgr = _analyzer.decode_base64(body.image_base64)
        results = _analyzer.analyze_image(image_bgr, fast_mode=fast, model_name=model)
        used_model = model or _analyzer.get_active_model_name()

        return AnalyzeResponse(
            success=True,
            message=f"Phát hiện {sum(1 for r in results if r['isFace'])} khuôn mặt.",
            model_used=used_model,
            faces=[FaceResult(**r) for r in results],
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Phân tích thất bại: {exc}")


# ------------------------------------------------------------------
# POST /api/chat/analyze  —  AI Chatbot phân tích tâm trạng phiên
# ------------------------------------------------------------------
@router.post(
    "/chat/analyze",
    response_model=ChatAnalyzeResponse,
    summary="AI Chatbot phân tích cảm xúc và tâm trạng phiên làm việc",
    description=(
        "Nhận câu hỏi từ người dùng kèm bản tóm tắt chuỗi thời gian cảm xúc, "
        "trả lời thông qua Gemini LLM hoặc bộ phân tích thông minh."
    ),
)
async def chat_analyze_session(body: ChatAnalyzeRequest) -> ChatAnalyzeResponse:
    """Endpoint AI Chatbot phân tích cảm xúc phiên làm việc."""
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="Nội dung câu hỏi không được rỗng.")

    try:
        history_list = [{"role": m.role, "content": m.content} for m in body.history]
        result = chat_service.answer_query(
            message=body.message,
            timeline_summary=body.timeline_summary,
            history=history_list,
        )
        return ChatAnalyzeResponse(
            success=result.get("success", True),
            reply=result.get("reply", ""),
            source=result.get("source", "smart_fallback"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi AI Chatbot: {exc}")

