"""
Bộ Phân Tích Tổng Hợp — điều phối nhận diện khuôn mặt và phân tích cảm xúc.

Đây là điểm đầu vào chính cho pipeline Core AI.
"""

import base64
import io
import time
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
        self._fatigue_window = deque()
        self._fatigue_baseline = None
        self._fatigue_level = 0.0
        self._fatigue_updated_at = None

    @staticmethod
    def _update_fatigue_level(current: float, target: float, elapsed: float, active: bool) -> float:
        """Rise quickly and monotonically while fatigue cues persist; decay gently afterward."""
        delta = float(np.clip(elapsed, 0.0, 0.5))
        if active:
            target = max(float(target), float(current))
            alpha = 1.0 - np.exp(-delta / 0.18)
            return float(current + (target - current) * alpha)
        return float(current * np.exp(-delta / 1.0))

    @staticmethod
    def _compute_fatigue_score(
        eye_low_duration: float,
        head_drop: float,
        eye_closure: float = 0.0,
        head_drop_duration: float = 0.0,
        smile_score: float = 0.0,
        eye_absolute_score: float = 0.0,
        head_absolute_score: float = 0.0,
    ) -> float:
        """Tính trạng thái mệt liên tục từ cường độ tín hiệu và thời lượng duy trì."""
        eye_severity = max(
            float(np.clip((eye_closure - 0.000005) / 0.000025, 0.0, 1.0)),
            float(np.clip(eye_absolute_score, 0.0, 1.0)),
        )
        eye_persistence = float(np.clip(eye_low_duration / 0.1, 0.0, 1.0))
        smile_discount = 1.0 - 0.85 * float(np.clip(smile_score, 0.0, 1.0))
        eye_score = eye_severity * eye_persistence * smile_discount

        head_severity = max(
            float(np.clip((head_drop - 0.000005) / 0.00002, 0.0, 1.0)),
            float(np.clip(head_absolute_score, 0.0, 1.0)),
        )
        head_persistence = float(np.clip(head_drop_duration / 0.1, 0.0, 1.0))
        head_score = head_severity * head_persistence

        combined_cue_boost = (
            0.35
            * min(eye_persistence, head_persistence)
            * float(np.sqrt(eye_severity * head_severity))
            * smile_discount
        )
        sustained_duration = min(
            float(np.clip(max(eye_low_duration - 0.1, 0.0) / 1.2, 0.0, 1.0)),
            float(np.clip(max(head_drop_duration - 0.1, 0.0) / 1.2, 0.0, 1.0)),
        )
        sustained_boost = (
            0.45
            * sustained_duration
            * float(np.sqrt(eye_severity * head_severity))
            * smile_discount
        )
        eye_state_floor = 0.0
        if eye_low_duration >= 0.01 and eye_closure > 0.00000001:
            eye_progress = float(np.clip((eye_low_duration - 0.01) / 0.49, 0.0, 1.0))
            eye_state_floor = (0.52 + 0.48 * eye_progress) * smile_discount

        head_state_floor = 0.0
        if head_drop_duration >= 0.01 and (head_drop > 0.00000001 or head_absolute_score > 0.15):
            head_progress = float(np.clip((head_drop_duration - 0.01) / 0.49, 0.0, 1.0))
            head_state_floor = 0.52 + 0.48 * head_progress

        paired_state_floor = 0.0
        if (
            eye_low_duration >= 0.01
            and head_drop_duration >= 0.01
            and (eye_closure > 0.00000001 or eye_absolute_score > 0.05)
            and (head_drop > 0.00000001 or head_absolute_score > 0.15)
        ):
            paired_duration = min(eye_low_duration, head_drop_duration)
            paired_progress = float(np.clip((paired_duration - 0.01) / 0.49, 0.0, 1.0))
            paired_state_floor = (0.82 + 0.18 * paired_progress) * smile_discount

        return float(np.clip(
            max(
                0.50 * eye_score + 0.20 * head_score + combined_cue_boost + sustained_boost,
                eye_state_floor,
                head_state_floor,
                paired_state_floor,
            ),
            0.0,
            1.0,
        ))

    @staticmethod
    def _summarize_fatigue_window(recent: list[dict], eye_baseline: float, head_baseline: float) -> dict[str, float]:
        """Summarize active cues without averaging them down with unrelated normal frames."""
        eye_closures = [max(0.0, eye_baseline - frame["eye_openness"]) for frame in recent]
        head_drops = [max(0.0, frame["head_drop"] - head_baseline) for frame in recent]
        eye_active = [
            i for i, value in enumerate(eye_closures)
            if value > 0.00001
            or (
                recent[i]["eye_openness"] < 0.68
                and (head_drops[i] > 0.00001 or recent[i]["head_drop"] > 0.52)
            )
        ]
        head_active = [
            i for i, value in enumerate(head_drops)
            if value > 0.00001 or recent[i]["head_drop"] > 0.52
        ]

        eye_low_duration = 0.0
        head_drop_duration = 0.0
        for i, (previous, current) in enumerate(zip(recent, recent[1:])):
            delta = min(max(current["timestamp"] - previous["timestamp"], 0.0), 0.5)
            if (
                (eye_closures[i] > 0.00001 or (
                    previous["eye_openness"] < 0.68
                    and (head_drops[i] > 0.00001 or previous["head_drop"] > 0.52)
                ))
                and (eye_closures[i + 1] > 0.00001 or (
                    current["eye_openness"] < 0.68
                    and (head_drops[i + 1] > 0.00001 or current["head_drop"] > 0.52)
                ))
            ):
                eye_low_duration += delta
            if (
                (head_drops[i] > 0.00001 or previous["head_drop"] > 0.52)
                and (head_drops[i + 1] > 0.00001 or current["head_drop"] > 0.52)
            ):
                head_drop_duration += delta
        active_smiles = [recent[i]["smile_score"] for i in eye_active]
        active_eye_openness = [recent[i]["eye_openness"] for i in eye_active]
        active_head_values = [recent[i]["head_drop"] for i in head_active]
        eye_absolute_score = (
            float(np.clip((0.68 - float(np.mean(active_eye_openness))) / 0.20, 0.0, 1.0))
            if active_eye_openness
            else 0.0
        )
        head_absolute_score = (
            float(np.clip((float(np.mean(active_head_values)) - 0.52) / 0.14, 0.0, 1.0))
            if active_head_values
            else 0.0
        )

        return {
            "eye_closure": float(np.mean([eye_closures[i] for i in eye_active])) if eye_active else 0.0,
            "eye_low_duration": eye_low_duration,
            "head_drop": float(np.mean([head_drops[i] for i in head_active])) if head_active else 0.0,
            "head_drop_duration": head_drop_duration,
            "smile_score": float(np.mean(active_smiles)) if active_smiles else 0.0,
            "eye_absolute_score": eye_absolute_score,
            "head_absolute_score": head_absolute_score,
        }

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
            self._fatigue_window.clear()
            self._fatigue_baseline = None
            self._fatigue_level = 0.0
            self._fatigue_updated_at = None
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

            # Fatigue is estimated from temporal eye-closure and head-drop cues.
            snapshot = {
                "timestamp": time.monotonic(),
                "eye_openness": float(face.get("eye_openness", 0.0)),
                "head_drop": float(face.get("head_drop", 0.0)),
                "smile_score": max(
                    float(np.clip((emotions.get("happy", 0.0) - 0.35) / 0.45, 0.0, 1.0)),
                    float(np.clip((-face.get("mouth_corner_angle", 0.0) - 0.005) / 0.035, 0.0, 1.0))
                    * float(np.clip((face.get("mouth_stretch", 0.0) - 0.25) / 0.25, 0.0, 1.0)),
                ),
            }
            self._fatigue_window.append(snapshot)
            while (
                self._fatigue_window
                and snapshot["timestamp"] - self._fatigue_window[0]["timestamp"] > 4.0
            ):
                self._fatigue_window.popleft()

            recent = list(self._fatigue_window)
            if self._fatigue_baseline is None and len(recent) >= 3:
                calibration = recent[:3]
                self._fatigue_baseline = {
                    "eye_openness": float(np.median([v["eye_openness"] for v in calibration])),
                    "head_drop": float(np.median([v["head_drop"] for v in calibration])),
                }

            elapsed = recent[-1]["timestamp"] - recent[0]["timestamp"] if len(recent) > 1 else 0.0
            eye_baseline = self._fatigue_baseline["eye_openness"] if self._fatigue_baseline else 0.0
            head_baseline = self._fatigue_baseline["head_drop"] if self._fatigue_baseline else 0.0
            signals = self._summarize_fatigue_window(recent, eye_baseline, head_baseline)

            fatigue_score = 0.0
            if self._fatigue_baseline and elapsed >= 0.1:
                fatigue_score = self._compute_fatigue_score(
                    **signals,
                )
            now = snapshot["timestamp"]
            update_delta = 0.1 if self._fatigue_updated_at is None else now - self._fatigue_updated_at
            self._fatigue_updated_at = now
            fatigue_cues_active = (
                signals["eye_low_duration"] >= 0.01
                and (signals["eye_closure"] > 0.00000001 or signals["eye_absolute_score"] > 0.05)
            ) or (
                signals["head_drop_duration"] >= 0.01
                and (signals["head_drop"] > 0.00000001 or signals["head_absolute_score"] > 0.15)
            )
            self._fatigue_level = self._update_fatigue_level(
                self._fatigue_level,
                fatigue_score,
                update_delta,
                fatigue_cues_active,
            )

            results.append({
                "isFace": True,
                "box": box,
                "emotions": emotions,
                "fatigue_score": round(float(self._fatigue_level), 4),
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
