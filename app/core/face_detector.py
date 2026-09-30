"""
Module Nhận Diện Khuôn Mặt sử dụng MediaPipe.

Phát hiện khuôn mặt trong ảnh và trả về tọa độ khung bao (bounding box)
cùng với landmarks mắt và tỷ lệ nhướn lông mày.
"""

import mediapipe as mp
import numpy as np
import cv2
import math


class FaceDetector:
    """Phát hiện khuôn mặt trong ảnh sử dụng MediaPipe Face Detection & Face Mesh."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )

        # Thêm Face Mesh để phân tích chi tiết cơ mặt
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Phát hiện khuôn mặt trong ảnh BGR.

        Returns:
            Danh sách các dict chứa:
                - "box": [x, y, w, h] tọa độ pixel
                - "confidence": điểm tin cậy
                - "left_eye": (x, y)
                - "right_eye": (x, y)
                - "eyebrow_lift": Tỷ lệ khoảng cách mắt-lông mày / chiều cao mặt
                - "mouth_openness": Tỷ lệ há miệng / chiều cao mặt
                - "eye_openness": EAR (Eye Aspect Ratio) — mở to=ngạc nhiên, nheo=tức giận
                - "mouth_width_ratio": Chiều rộng / chiều cao miệng — vuông vức lộ răng=tức giận
                - "face_vertical_expansion": Tỷ lệ brow-to-chin / face_h — lớn=ngạc nhiên, nhỏ=tức giận
        """
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # 1. Chạy Face Detection để lấy bounding box
        detection_results = self._detector.process(image_rgb)

        # 2. Chạy Face Mesh để lấy landmarks chi tiết
        mesh_results = self._mesh.process(image_rgb)

        faces = []
        if detection_results.detections:
            for i, detection in enumerate(detection_results.detections):
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                box_w = min(int(bbox.width * w), w - x)
                box_h = min(int(bbox.height * h), h - y)
                confidence = detection.score[0] if detection.score else 0.0

                left_eye, right_eye = None, None
                kps = detection.location_data.relative_keypoints
                if len(kps) >= 2:
                    right_eye = (int(kps[0].x * w), int(kps[0].y * h))
                    left_eye  = (int(kps[1].x * w), int(kps[1].y * h))

                # Khởi tạo các đặc trưng
                eyebrow_lift = 0.0
                mouth_openness = 0.0
                eye_openness = 0.0
                mouth_width_ratio = 0.0
                face_vertical_expansion = 0.0

                if mesh_results.multi_face_landmarks and i < len(mesh_results.multi_face_landmarks):
                    lm = mesh_results.multi_face_landmarks[i].landmark

                    # Chiều cao mặt: từ đỉnh trán (10) đến cằm (152)
                    face_h = abs(lm[152].y - lm[10].y)

                    if face_h > 0:
                        # --- Tỷ lệ nhướn lông mày ---
                        left_brow_dist  = abs(lm[159].y - lm[105].y)
                        right_brow_dist = abs(lm[386].y - lm[334].y)
                        eyebrow_lift = ((left_brow_dist + right_brow_dist) / 2.0) / face_h

                        # --- Tỷ lệ há miệng ---
                        mouth_h = abs(lm[13].y - lm[14].y)
                        mouth_openness = mouth_h / face_h

                        # --- Đặc trưng 1: EAR - Eye Aspect Ratio ---
                        # Mắt trái: trên=159, dưới=145, trái=33, phải=133
                        # Mắt phải: trên=386, dưới=374, trái=362, phải=263
                        left_eye_h  = abs(lm[159].y - lm[145].y)
                        left_eye_w  = abs(lm[33].x  - lm[133].x) + 1e-6
                        right_eye_h = abs(lm[386].y - lm[374].y)
                        right_eye_w = abs(lm[362].x - lm[263].x) + 1e-6
                        left_ear  = left_eye_h  / left_eye_w
                        right_ear = right_eye_h / right_eye_w
                        eye_openness = (left_ear + right_ear) / 2.0

                        # --- Đặc trưng 2: Tỷ lệ rộng/cao miệng ---
                        mouth_w = abs(lm[61].x - lm[291].x)
                        if mouth_h > 1e-5:
                            mouth_width_ratio = mouth_w / mouth_h
                        else:
                            mouth_width_ratio = 10.0

                        # --- Đặc trưng 3: Khuôn mặt mở rộng theo chiều dọc ---
                        # Ngạc nhiên: lông mày nhướn cao → brow_to_chin / face_h lớn
                        # Tức giận:   lông mày kéo xuống → brow_to_chin / face_h nhỏ
                        brow_center_y = (lm[105].y + lm[334].y) / 2.0
                        chin_y        = lm[152].y
                        brow_to_chin  = abs(chin_y - brow_center_y)
                        face_vertical_expansion = brow_to_chin / face_h

                faces.append({
                    "box": [x, y, box_w, box_h],
                    "confidence": round(float(confidence), 4),
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "eyebrow_lift": float(eyebrow_lift),
                    "mouth_openness": float(mouth_openness),
                    "eye_openness": float(eye_openness),
                    "mouth_width_ratio": float(mouth_width_ratio),
                    "face_vertical_expansion": float(face_vertical_expansion),
                })

        return faces

    def close(self):
        """Giải phóng tài nguyên MediaPipe."""
        self._detector.close()
        self._mesh.close()
