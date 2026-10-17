"""
Bộ Phân Tích Tổng Hợp — điều phối nhận diện khuôn mặt và phân tích cảm xúc.

Đây là điểm đầu vào chính cho pipeline Core AI.
"""

import base64
import io
from collections import deque

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
        self._fatigue_window = deque(maxlen=30)

    @staticmethod
    def _compute_fatigue_score(
        eye_openness: float,
        mouth_openness: float,
        eye_low_duration: float,
        micro_sleep: bool,
        yawning: bool,
        head_drop: float,
        eyebrow_lift: float = 0.0,
        mouth_width_ratio: float = 0.0,
    ) -> float:
        """Biểu điểm mệt mỏi từ nhiều chỉ số hình học và xu hướng thời gian."""
        eye_score = 0.0
        if eye_openness < 0.18:
            eye_score += 0.5
        if eye_low_duration >= 1.2:
            eye_score += 0.5
        eye_score = float(np.clip(eye_score, 0.0, 1.0))

        mouth_score = 0.0
        if mouth_openness > 0.08:
            mouth_score += 0.45
        if mouth_width_ratio > 1.1:
            mouth_score += 0.25
        if yawning:
            mouth_score += 0.30
        mouth_score = float(np.clip(mouth_score, 0.0, 1.0))

        head_score = 0.0
        if head_drop > 0.08:
            head_score += 0.55
        if head_drop > 0.12:
            head_score += 0.45
        head_score = float(np.clip(head_score, 0.0, 1.0))

        brow_score = 0.0
        if eyebrow_lift < 0.06:
            brow_score += 0.6
        if yawning:
            brow_score += 0.4
        brow_score = float(np.clip(brow_score, 0.0, 1.0))

        fatigue = 0.38 * eye_score + 0.32 * mouth_score + 0.18 * head_score + 0.12 * brow_score
        if micro_sleep:
            fatigue = max(fatigue, 0.75)
        return float(np.clip(fatigue, 0.0, 1.0))

    @staticmethod
    def _inject_fatigue_label(emotions: dict[str, float], fatigue_score: float) -> dict[str, float]:
        """Chèn nhãn fatigue vào phân phối cảm xúc và ép đóng vai trò chủ đạo khi tín hiệu mạnh."""
        if not isinstance(emotions, dict):
            return {}

        result = dict(emotions)
        if fatigue_score < 0.45:
            return result

        fatigue_prob = float(np.clip((fatigue_score - 0.35) / 0.65, 0.0, 0.85))
        result["fatigue"] = max(result.get("fatigue", 0.0), fatigue_prob)

        if fatigue_score >= 0.75:
            # Dịch chuyển trọng số ưu tiên sang fatigue thật mạnh, để UI hiển thị rõ
            # "Mệt mỏi" thay vì bị neutral/normal lấn át.
            top_weight = float(np.clip(fatigue_score, 0.75, 1.0))
            for key in list(result):
                if key != "fatigue":
                    result[key] = float(result[key]) * (1.0 - top_weight * 0.55)
            result["fatigue"] = max(result.get("fatigue", 0.0), top_weight * 0.82)

        total = sum(result.values())
        if total > 0:
            result = {k: float(v) / total for k, v in result.items()}

        return result

    # ------------------------------------------------------------------
    # API Công khai
    # ------------------------------------------------------------------

    def analyze_image(self, image_bgr: np.ndarray, fast_mode: bool = False) -> list[dict]:
        """
        Chạy toàn bộ pipeline phân tích trên ảnh BGR.

        Args:
            image_bgr: Ảnh đầu vào BGR.
            fast_mode: Tối ưu tốc độ cho realtime (giảm TTA passes).

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
            box = face["box"]
            face_features = face.get("features", {})

            # Pipeline đầy đủ: padding → face alignment → TTA → ensemble
            # Truyền landmarks mắt để căn chỉnh khuôn mặt về góc 0°
            # Truyền các đặc trưng sinh học để phân biệt surprise / angry
            emotions = self._emotion_analyzer.analyze_aligned(
                image_bgr,
                box,
                left_eye=face.get("left_eye"),
                right_eye=face.get("right_eye"),
                eyebrow_lift=face.get("eyebrow_lift", 0.0),
                mouth_openness=face.get("mouth_openness", 0.0),
                eye_openness=face.get("eye_openness", 0.0),
                mouth_width_ratio=face.get("mouth_width_ratio", 0.0),
                face_vertical_expansion=face.get("face_vertical_expansion", 0.0),
                brow_slope=face.get("brow_slope", 0.0),
                brow_inner_drop=face.get("brow_inner_drop", 0.0),
                mouth_corner_angle=face.get("mouth_corner_angle", 0.0),
                mouth_stretch=face.get("mouth_stretch", 0.0),
                upper_lip_raise=face.get("upper_lip_raise", 0.0),
                nose_wrinkle=face.get("nose_wrinkle", 0.0),
                fast_mode=fast_mode,
            )

            # Multi-signal temporal fatigue: dựa trên cửa sổ 30 frame gần nhất
            # để phát hiện micro-sleep, ngáp và gục đầu kéo dài theo thời gian.
            snapshot = {
                "eye_openness": float(face.get("eye_openness", 0.0)),
                "mouth_openness": float(face.get("mouth_openness", 0.0)),
                "head_drop": float(face.get("head_drop", 0.0)),
                "eyebrow_lift": float(face.get("eyebrow_lift", 0.0)),
                "mouth_width_ratio": float(face.get("mouth_width_ratio", 0.0)),
            }
            self._fatigue_window.append(snapshot)

            recent = list(self._fatigue_window)
            low_ear_frames = sum(1 for v in recent if v["eye_openness"] < 0.18)
            eye_low_duration = (low_ear_frames / max(1, len(recent))) * 3.0
            micro_sleep = low_ear_frames >= 12 and eye_low_duration >= 1.2
            yawning = (
                (sum(v["mouth_openness"] for v in recent) / max(1, len(recent))) > 0.08
                and (sum(v["eyebrow_lift"] for v in recent) / max(1, len(recent))) < 0.06
                and (sum(v["mouth_width_ratio"] for v in recent) / max(1, len(recent))) > 1.1
            )
            head_drop = sum(v["head_drop"] for v in recent) / max(1, len(recent))
            eyebrow_lift = sum(v["eyebrow_lift"] for v in recent) / max(1, len(recent))
            mouth_width_ratio = sum(v["mouth_width_ratio"] for v in recent) / max(1, len(recent))
            mouth_openness = sum(v["mouth_openness"] for v in recent) / max(1, len(recent))
            eye_openness = sum(v["eye_openness"] for v in recent) / max(1, len(recent))

            fatigue_score = self._compute_fatigue_score(
                eye_openness=eye_openness,
                mouth_openness=mouth_openness,
                eye_low_duration=eye_low_duration,
                micro_sleep=micro_sleep,
                yawning=yawning,
                head_drop=head_drop,
                eyebrow_lift=eyebrow_lift,
                mouth_width_ratio=mouth_width_ratio,
            )

            emotions = self._inject_fatigue_label(emotions, fatigue_score)

            results.append({
                "isFace": True,
                "box": box,
                "emotions": emotions,
                "fatigue_score": round(float(fatigue_score), 4),
                # Các đặc trưng hình học được trả về để debug/calibrate webcam.
                "features": face.get("features", {}),
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
