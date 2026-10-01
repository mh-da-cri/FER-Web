"""
Module Nhận Diện Khuôn Mặt sử dụng MediaPipe.

Ngoài bounding box, module trích xuất các đặc trưng hình học ổn định theo từng
khuôn mặt để hỗ trợ phân biệt angry / sad / disgust / fear / surprise.
"""

import mediapipe as mp
import numpy as np
import cv2
import math


class FaceDetector:
    """Phát hiện khuôn mặt và trích xuất đặc trưng cơ mặt bằng MediaPipe Face Mesh."""

    def __init__(self, min_detection_confidence: float = 0.5):
        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
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
    def _dist(a, b):
        return float(math.hypot(a.x - b.x, a.y - b.y))

    @staticmethod
    def _clamp01(x):
        return float(max(0.0, min(1.0, x)))

    def detect(self, image_bgr: np.ndarray) -> list[dict]:
        """
        Returns:
            box, confidence, eye landmarks và các đặc trưng:
              eyebrow_lift       : độ nâng lông mày so với chiều cao mặt
              brow_slope         : hướng lông mày (inner/outer)
              brow_inner_drop    : inner brow hạ xuống so với outer brow
              mouth_openness     : độ mở miệng
              mouth_width_ratio  : rộng/cao miệng
              mouth_corner_angle : hướng khóe miệng
              mouth_stretch      : kéo ngang miệng
              upper_lip_raise    : môi trên nâng lên
              nose_wrinkle       : vùng mũi-môi bị co/rút
              eye_openness       : EAR
              face_vertical_expansion
        """
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        detection_results = self._detector.process(image_rgb)
        mesh_results = self._mesh.process(image_rgb)

        faces = []
        if not detection_results.detections:
            return faces

        # Face Mesh ordering is normally aligned with detection ordering for one/multi faces.
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
                "mouth_openness": 0.0,
                "eye_openness": 0.0,
                "mouth_width_ratio": 0.0,
                "mouth_corner_angle": 0.0,
                "mouth_stretch": 0.0,
                "upper_lip_raise": 0.0,
                "nose_wrinkle": 0.0,
                "face_vertical_expansion": 0.0,
            }

            if i < len(mesh_faces):
                lm = mesh_faces[i].landmark
                # Stable face scale: inter-eye distance is less affected by camera distance.
                eye_w = self._dist(lm[33], lm[263]) + 1e-6
                face_h = self._dist(lm[10], lm[152]) + 1e-6

                # Eyes: vertical opening / horizontal width.
                left_ear = self._dist(lm[159], lm[145]) / (self._dist(lm[33], lm[133]) + 1e-6)
                right_ear = self._dist(lm[386], lm[374]) / (self._dist(lm[362], lm[263]) + 1e-6)
                f["eye_openness"] = float((left_ear + right_ear) / 2.0)

                mouth_h = self._dist(lm[13], lm[14])
                mouth_w = self._dist(lm[61], lm[291])
                f["mouth_openness"] = float(mouth_h / face_h)
                f["mouth_width_ratio"] = float(mouth_w / (mouth_h + 1e-6))
                f["mouth_stretch"] = float(mouth_w / eye_w)

                # Eyebrow geometry:
                # 105/334 = inner-ish brow, 70/300 = outer-ish brow.
                brow_inner_y = (lm[105].y + lm[334].y) / 2.0
                brow_outer_y = (lm[70].y + lm[300].y) / 2.0
                eye_center_y = (lm[159].y + lm[386].y) / 2.0
                brow_dist = eye_center_y - brow_inner_y
                f["eyebrow_lift"] = float(brow_dist / face_h)

                # Positive => inner brow is lower than outer brow (furrow/anger tendency).
                f["brow_slope"] = float((brow_inner_y - brow_outer_y) / face_h)

                # Compare inner brow against eye line. More positive means brow moved down.
                f["brow_inner_drop"] = float((brow_inner_y - eye_center_y) / face_h)

                # Mouth corners: positive => corners lower than mouth center.
                mouth_center_y = (lm[13].y + lm[14].y) / 2.0
                corner_y = (lm[61].y + lm[291].y) / 2.0
                f["mouth_corner_angle"] = float((corner_y - mouth_center_y) / face_h)

                # Upper lip raise: smaller nose-tip -> upper-lip distance = raised upper lip.
                nose_tip = lm[1]
                upper_lip = lm[13]
                nose_mouth_dist = self._dist(nose_tip, upper_lip)
                f["upper_lip_raise"] = float(max(0.0, 0.075 - nose_mouth_dist) / 0.075)

                # Nose wrinkle proxy: distance from nostril wings to upper lip.
                # This is deliberately a weak signal because lighting/skin texture varies.
                nose_left = lm[98]
                nose_right = lm[327]
                nose_width = self._dist(nose_left, nose_right) + 1e-6
                f["nose_wrinkle"] = float(max(0.0, 0.85 - nose_mouth_dist / nose_width))

                brow_center_y = (lm[105].y + lm[334].y) / 2.0
                f["face_vertical_expansion"] = float(abs(lm[152].y - brow_center_y) / face_h)

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
        self._mesh.close()
