"""
Module Phân Tích Biểu Cảm sử dụng OpenCV DNN + mô hình FER2013.

Phân tích biểu cảm khuôn mặt từ các vùng khuôn mặt đã cắt.
Không phụ thuộc tensorflow hay fer — chỉ dùng OpenCV.
"""

import os
import urllib.request

import cv2
import numpy as np


# Đường dẫn lưu mô hình
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

# Mô hình mini_XCEPTION được huấn luyện trên FER2013
# Kiến trúc đơn giản: 4 residual blocks + global avg pooling
# Phân loại 7 cảm xúc: angry, disgust, fear, happy, sad, surprise, neutral
_EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionAnalyzer:
    """Phân tích cảm xúc từ vùng khuôn mặt sử dụng OpenCV DNN."""

    def __init__(self):
        self._model = None
        self._ensure_model()

    def _ensure_model(self):
        """Tải và khởi tạo mô hình ONNX cho phân tích cảm xúc."""
        os.makedirs(_MODEL_DIR, exist_ok=True)

        model_path = os.path.join(_MODEL_DIR, "emotion_model.onnx")

        if not os.path.exists(model_path):
            print("[EmotionAnalyzer] Downloading emotion model...")
            url = (
                "https://github.com/onnx/models/raw/main/validated/vision/"
                "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
            )
            try:
                urllib.request.urlretrieve(url, model_path)
                print("[EmotionAnalyzer] Model downloaded successfully!")
            except Exception as exc:
                print(f"[EmotionAnalyzer] Failed to download model: {exc}")
                return

        try:
            self._model = cv2.dnn.readNetFromONNX(model_path)
            print("[EmotionAnalyzer] Model emotion-ferplus loaded successfully.")
        except Exception as exc:
            print(f"[EmotionAnalyzer] Error loading model: {exc}")
            self._model = None

    def analyze(self, face_image_bgr: np.ndarray) -> dict[str, float]:
        """
        Phân tích cảm xúc từ ảnh khuôn mặt đã cắt.

        Args:
            face_image_bgr: Ảnh khuôn mặt đã cắt định dạng BGR.

        Returns:
            Dictionary ánh xạ tên cảm xúc sang điểm tin cậy (0.0 - 1.0).
            Ví dụ: {"happy": 0.85, "neutral": 0.10, "sad": 0.03, ...}
            Trả về dict rỗng nếu phân tích thất bại.
        """
        if self._model is None:
            return {}

        if face_image_bgr is None or face_image_bgr.size == 0:
            return {}

        # Đảm bảo vùng cắt khuôn mặt đủ lớn để phân tích
        h, w = face_image_bgr.shape[:2]
        if h < 10 or w < 10:
            return {}

        try:
            # Mô hình emotion-ferplus yêu cầu ảnh 64x64 grayscale
            gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))

            # Chuẩn hóa và tạo blob
            blob = cv2.dnn.blobFromImage(
                resized,
                scalefactor=1.0,
                size=(64, 64),
                mean=(0,),
                swapRB=False,
                crop=False,
            )

            self._model.setInput(blob)
            output = self._model.forward()

            # Áp dụng softmax để chuyển logits thành xác suất
            scores = output[0]
            exp_scores = np.exp(scores - np.max(scores))
            probabilities = exp_scores / np.sum(exp_scores)

            # Tạo dictionary kết quả
            emotions = {}
            for label, prob in zip(_EMOTION_LABELS, probabilities):
                emotions[label] = round(float(prob), 4)

            return emotions

        except Exception:
            # Xử lý mượt mà mọi lỗi phân tích
            return {}
