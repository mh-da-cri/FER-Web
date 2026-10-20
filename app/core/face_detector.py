"""
Module Nhận Diện Khuôn Mặt sử dụng MediaPipe.

Trích xuất bounding box và các đặc trưng hình học chuẩn hóa theo tỷ lệ khuôn mặt
để bất biến với khoảng cách camera, hỗ trợ phân biệt chính xác 7 cảm xúc.
"""

import math
import cv2
import mediapipe as mp
import numpy as np


class FaceDetector:
    """Phát hiện khuôn mặt và trích xuất đặc trưng cơ mặt bằng MediaPipe Face Mesh."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector_full = self._mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=min_detection_confidence,
        )

        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=10,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _dist(a, b) -> float:
        return float(math.hypot(a.x - b.x, a.y - b.y))

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detection_results = self._detector.process(image_rgb)
        if not detection_results.detections:
            detection_results = self._detector_full.process(image_rgb)

        mesh_results = self._mesh.process(image_rgb)

        faces = []
        if not detection_results.detections:
            return faces

        mesh_faces = mesh_results.multi_face_landmarks or []

        for i, detection in enumerate(detection_results.detections):
            bbox = detection.location_data.relative_bounding_box
            x = max(0, int(bbox.xmin * w))
            y = max(0, int(bbox.ymin * h))
            box_w = max(1, min(int(bbox.width * w), w - x))
            box_h = max(1, min(int(bbox.height * h), h - y))
            confidence = detection.score[0] if detection.score else 0.0

            left_eye = right_eye = None
            kps = detection.location_data.relative_keypoints
            if len(kps) >= 2:
                right_eye = (int(kps[0].x * w), int(kps[0].y * h))
                left_eye = (int(kps[1].x * w), int(kps[1].y * h))

            f = {
                "eyebrow_lift": 0.0,
                "brow_slope": 0.0,
                "brow_inner_drop": 0.0,
                "brow_furrow": 0.27,
                "mouth_openness": 0.0,
                "eye_openness": 0.27,
                "mouth_width_ratio": 0.0,
                "mouth_corner_angle": 0.0,
                "mouth_stretch": 0.50,
                "upper_lip_raise": 0.0,
                "nose_wrinkle": 0.0,
                "face_vertical_expansion": 0.0,
            }

            if i < len(mesh_faces):
                lm = mesh_faces[i].landmark
                # Khoảng cách 2 đuôi mắt ngoài (luôn ổn định kể cả khi há miệng)
                eye_w = self._dist(lm[33], lm[263]) + 1e-6
                face_h = self._dist(lm[10], lm[152]) + 1e-6

                # 1. Độ mở mắt (EAR)
                left_ear = self._dist(lm[159], lm[145]) / (self._dist(lm[33], lm[133]) + 1e-6)
                right_ear = self._dist(lm[386], lm[374]) / (self._dist(lm[362], lm[263]) + 1e-6)
                f["eye_openness"] = float((left_ear + right_ear) / 2.0)

                # 2. Hình học miệng (chuẩn hóa theo eye_w và face_h)
                mouth_h = self._dist(lm[13], lm[14])
                mouth_w = self._dist(lm[61], lm[291])
                f["mouth_openness"] = float(mouth_h / face_h)
                f["mouth_width_ratio"] = float(mouth_w / (mouth_h + 1e-6))
                f["mouth_stretch"] = float(mouth_w / eye_w)

                # Góc khóe miệng: âm là cười (khóe miệng nhếch lên), dương là buồn (khóe miệng trễ xuống)
                mouth_center_y = (lm[13].y + lm[14].y) / 2.0
                corner_y = (lm[61].y + lm[291].y) / 2.0
                f["mouth_corner_angle"] = float((corner_y - mouth_center_y) / eye_w)

                # 3. Hình học lông mày (Dùng mốc góc mắt trong 133 & 362 cố định trên hộp sọ)
                inner_eye_y = (lm[133].y + lm[362].y) / 2.0
                # Đầu lông mày thực sự: 107, 55 (trái) và 336, 285 (phải)
                brow_head_y = (lm[107].y + lm[55].y + lm[336].y + lm[285].y) / 4.0
                # Đỉnh vòm lông mày: 105 (trái) và 334 (phải)
                brow_arch_y = (lm[105].y + lm[334].y) / 2.0

                # Độ nâng vòm lông mày (Tăng mạnh khi Surprise / Fear)
                f["eyebrow_lift"] = float((inner_eye_y - brow_arch_y) / eye_w)

                # Độ ép/hạ đầu lông mày so với vòm mày (Tăng khi Angry)
                f["brow_slope"] = float((brow_head_y - brow_arch_y) / eye_w)

                # Độ hạ thấp của đầu lông mày về phía hốc mắt trong (Bình thường ~0.16, Angry < 0.135)
                inner_brow_height = (inner_eye_y - brow_head_y) / eye_w
                f["brow_inner_drop"] = float(max(0.0, 0.165 - inner_brow_height))

                # Khoảng cách nhíu 2 đầu lông mày (Bình thường ~0.27, nhíu mày Angry/Fear giảm xuống < 0.24)
                f["brow_furrow"] = float(self._dist(lm[55], lm[285]) / eye_w)

                # 4. Độ nhăn mũi / Nâng môi trên (Chuẩn hóa theo khoảng cách 2 mắt eye_w)
                # Dùng lm[164] (chân vách ngăn mũi) và lm[0] (viền môi trên)
                nose_sub = lm[164]
                upper_lip = lm[0]
                nose_lip_ratio = self._dist(nose_sub, upper_lip) / eye_w
                f["upper_lip_raise"] = float(max(0.0, (0.105 - nose_lip_ratio) / 0.04))

                # Độ nhăn sống mũi (AU9): khoảng cách nasion (lm[168]) tới chóp mũi (lm[1]) co rút khi nhăn mũi
                nose_bridge = self._dist(lm[168], lm[1]) / eye_w
                f["nose_wrinkle"] = float(max(0.0, (0.38 - nose_bridge) / 0.08))

                f["face_vertical_expansion"] = float(abs(lm[152].y - brow_arch_y) / face_h)

            features = {k: round(float(v), 6) for k, v in f.items()}

            faces.append({
                "box": [x, y, box_w, box_h],
                "confidence": round(float(confidence), 4),
                "left_eye": left_eye,
                "right_eye": right_eye,
                **f,
                "features": features,
            })

        return faces

    def close(self):
        """Giải phóng tài nguyên MediaPipe."""
        self._detector.close()
        self._detector_full.close()
        self._mesh.close()