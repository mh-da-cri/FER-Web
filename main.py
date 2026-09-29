"""
AI Camera — API Phân Tích Biểu Cảm Khuôn Mặt

Điểm đầu vào cho ứng dụng FastAPI.
Chạy bằng lệnh:  uvicorn main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.router import router as analysis_router

app = FastAPI(
    title="AI Camera — Expression Analysis API",
    description=(
        "Backend API sử dụng **MediaPipe** để nhận diện khuôn mặt và "
        "**FER** để nhận dạng biểu cảm khuôn mặt. Tải lên ảnh "
        "hoặc gửi chuỗi Base64 để nhận phân tích cảm xúc theo từng khuôn mặt."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — cho phép tất cả origin trong quá trình phát triển
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Đăng ký router
# ---------------------------------------------------------------------------
app.include_router(analysis_router)


# ---------------------------------------------------------------------------
# Kiểm tra sức khỏe hệ thống
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def health_check():
    """Endpoint kiểm tra sức khỏe hệ thống đơn giản."""
    return {
        "status": "ok",
        "service": "AI Camera — Expression Analysis API",
        "version": "1.0.0",
    }
