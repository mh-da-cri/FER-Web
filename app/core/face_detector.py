"""
Module Nhận Diện Khuôn Mặt sử dụng MediaPipe.

Phát hiện khuôn mặt trong ảnh và trả về tọa độ khung bao (bounding box).
"""

import mediapipe as mp
import numpy as np
import cv2


class FaceDetector:
    """Phát hiện khuôn mặt trong ảnh sử dụng MediaPipe Face Detection."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=1,  # 1 = mô hình toàn phạm vi (lên đến khoảng cách 5m)
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Phát hiện khuôn mặt trong ảnh BGR.

        Args:
            image_bgr: Ảnh đầu vào định dạng BGR (mặc định của OpenCV).

        Returns:
            Danh sách các dict, mỗi dict chứa:
                - "box": [x, y, w, h] tọa độ pixel
                - "confidence": điểm tin cậy phát hiện
        """
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        results = self._detector.process(image_rgb)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                # Chuyển đổi tọa độ tương đối sang giá trị pixel tuyệt đối
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                box_w = min(int(bbox.width * w), w - x)
                box_h = min(int(bbox.height * h), h - y)

                confidence = detection.score[0] if detection.score else 0.0

                faces.append({
                    "box": [x, y, box_w, box_h],
                    "confidence": round(float(confidence), 4),
                })

        return faces

    def close(self):
        """Giải phóng tài nguyên MediaPipe."""
        self._detector.close()
