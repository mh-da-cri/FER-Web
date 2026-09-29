"""
Bộ Phân Tích Tổng Hợp — điều phối nhận diện khuôn mặt và phân tích cảm xúc.

Đây là điểm đầu vào chính cho pipeline Core AI.
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
    với phân tích cảm xúc FER thành một pipeline duy nhất.
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self._face_detector = FaceDetector(
            min_detection_confidence=min_detection_confidence
        )
        self._emotion_analyzer = EmotionAnalyzer()

    # ------------------------------------------------------------------
    # API Công khai
    # ------------------------------------------------------------------

    def analyze_image(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Chạy toàn bộ pipeline phân tích trên ảnh BGR.

        Returns:
            Danh sách các dict kết quả, mỗi dict cho một khuôn mặt:
            [
                {
                    "isFace": true,
                    "box": [x, y, w, h],
                    "emotions": {"happy": 0.85, ...}
                },
                ...
            ]
            Nếu không tìm thấy khuôn mặt, trả về:
            [{"isFace": false, "box": [], "emotions": {}}]
        """
        faces = self._face_detector.detect(image_bgr)

        if not faces:
            return [{"isFace": False, "box": [], "emotions": {}}]

        results = []
        for face in faces:
            x, y, w, h = face["box"]

            # Cắt vùng khuôn mặt từ ảnh gốc
            face_crop = image_bgr[y : y + h, x : x + w]

            # Phân tích cảm xúc trên khuôn mặt đã cắt
            emotions = self._emotion_analyzer.analyze(face_crop)

            results.append({
                "isFace": True,
                "box": [x, y, w, h],
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
