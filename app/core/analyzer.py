"""
Bộ Phân Tích Tổng Hợp — điều phối nhận diện khuôn mặt và phân tích cảm xúc.

Đây là điểm đầu vào chính cho pipeline Core AI.
Sử dụng MobileNetV2 (PyTorch) làm mô hình phân tích cảm xúc duy nhất.
"""

import base64
import io

import cv2
import numpy as np
from PIL import Image

from app.core.face_detector import FaceDetector
from app.core.emotion_analyzer import EmotionAnalyzer


class FaceExpressionAnalyzer:
    """
    Bộ phân tích cấp cao kết hợp nhận diện khuôn mặt MediaPipe
    với phân tích cảm xúc MobileNetV2 thành một pipeline duy nhất.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self._face_detector = FaceDetector(
            min_detection_confidence=min_detection_confidence
        )
        self._emotion_analyzer = EmotionAnalyzer()

    # ------------------------------------------------------------------
    # API Công khai
    # ------------------------------------------------------------------

    def analyze_image(
        self,
        image_bgr: np.ndarray,
        fast_mode: bool = False,
    ) -> list[dict]:
        """
        Chạy toàn bộ pipeline phân tích trên ảnh BGR.

        Args:
            image_bgr: Ảnh đầu vào BGR.
            fast_mode: Tối ưu tốc độ cho realtime (giảm TTA passes).

        Returns:
            Danh sách các dict kết quả, mỗi dict cho một khuôn mặt.
        """
        faces = self._face_detector.detect(image_bgr)

        if not faces:
            return [{"isFace": False, "box": [], "emotions": {}}]

        results = []
        for face in faces:
            box = face["box"]

            # Truyền thêm features hình học từ FaceMesh vào bộ phân tích cảm xúc
            emotions = self._emotion_analyzer.analyze_aligned(
                image_bgr,
                box,
                left_eye=face.get("left_eye"),
                right_eye=face.get("right_eye"),
                fast_mode=fast_mode,
                features=face.get("features", {}),
            )

            results.append({
                "isFace": True,
                "box": box,
                "emotions": emotions,
            })

        return results

    # ------------------------------------------------------------------
    # Các hàm hỗ trợ giải mã ảnh
    # ------------------------------------------------------------------

    @staticmethod
    def decode_base64(base64_string: str) -> np.ndarray:
        """Giải mã chuỗi ảnh Base64 thành mảng numpy BGR."""
        # Loại bỏ tiền tố data-URI nếu có (vd: "data:image/png;base64,...")
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_rgb = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return image_bgr

    @staticmethod
    def decode_upload_bytes(file_bytes: bytes) -> np.ndarray:
        """Giải mã bytes file upload thành mảng numpy BGR."""
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Không thể giải mã ảnh đã tải lên. Vui lòng kiểm tra định dạng file.")
        return image_bgr

    def close(self):
        """Giải phóng tài nguyên."""
        self._face_detector.close()
