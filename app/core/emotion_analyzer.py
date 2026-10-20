"""
Module Phân Tích Biểu Cảm — MobileNetV2 + CBAM (PyTorch)
Tích hợp:
  - Prior Logit Adjustment (Bù mất cân bằng mẫu của RAF-DB cho Fear & Angry)
  - Neutral Deadzone (Khóa ổn định khi mặt bình thường)
  - FaceMesh Geometric Fusion & EMA Temporal Smoothing (Chống nhảy cảm xúc)
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as tv_models

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

_MODEL_FILENAME = "mobilenetv2_cbam_rafdb_best.pth"
_NUM_CLASSES = 7
# Thứ tự index: 0:surprise, 1:fear, 2:disgust, 3:happy, 4:sad, 5:angry, 6:neutral
_LABELS = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
_TEMPERATURE = 1.05

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MAX_ALIGN_ANGLE_DEG = 30.0
_EMA_ALPHA = 0.65  # Hệ số làm mượt theo thời gian (0.65: vừa chống giật vừa phản hồi biểu cảm nhanh)


class ChannelAttention(torch.nn.Module):
    def __init__(self, in_planes: int = 1280, ratio: int = 16):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(torch.nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(torch.nn.Module):
    def __init__(self, in_planes: int = 1280, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class MobileNetV2_CBAM(torch.nn.Module):
    def __init__(self, num_classes: int = _NUM_CLASSES):
        super().__init__()
        base = tv_models.mobilenet_v2(weights=None)
        self.features = base.features
        self.cbam = CBAM(1280, ratio=16, kernel_size=7)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(1280, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.cbam(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)


class EmotionAnalyzer:
    def __init__(self):
        self._model: torch.nn.Module | None = None
        self._ema_probs: np.ndarray | None = None
        self._load_model()

    def _load_model(self):
        os.makedirs(_MODEL_DIR, exist_ok=True)
        candidate_filenames = [
            _MODEL_FILENAME,
            "mobilenetv2_cbam_rafdb_best.pth",
            "mobilenetv2_rafdb_best.pth",
        ]

        model_path = None
        for fn in candidate_filenames:
            candidate_path = os.path.join(_MODEL_DIR, fn)
            if os.path.exists(candidate_path):
                model_path = candidate_path
                break

        if model_path is None and os.path.exists(_MODEL_DIR):
            for file in os.listdir(_MODEL_DIR):
                if file.endswith(".pth"):
                    model_path = os.path.join(_MODEL_DIR, file)
                    break

        if not model_path or not os.path.exists(model_path):
            print(f"[EmotionAnalyzer] Model file not found: {os.path.join(_MODEL_DIR, _MODEL_FILENAME)}")
            return

        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            has_cbam = any("cbam" in k for k in state_dict.keys())

            if has_cbam:
                model = MobileNetV2_CBAM(num_classes=_NUM_CLASSES)
                arch_name = "MobileNetV2 + CBAM"
            else:
                model = tv_models.mobilenet_v2(weights=None)
                model.classifier[1] = torch.nn.Linear(model.last_channel, _NUM_CLASSES)
                arch_name = "MobileNetV2"

            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
            print(f"[EmotionAnalyzer] {arch_name} ({os.path.basename(model_path)}) loaded successfully.")
        except Exception as exc:
            print(f"[EmotionAnalyzer] Error loading model from {model_path}: {exc}")

    def is_ready(self) -> bool:
        return self._model is not None

    @staticmethod
    def _align_face(
        face_bgr: np.ndarray,
        left_eye: tuple[int, int] | None,
        right_eye: tuple[int, int] | None,
        face_box: list[int],
    ) -> np.ndarray:
        if left_eye is None or right_eye is None:
            return face_bgr

        ox, oy = face_box[0], face_box[1]
        p1 = (left_eye[0] - ox, left_eye[1] - oy)
        p2 = (right_eye[0] - ox, right_eye[1] - oy)

        if p1[0] <= p2[0]:
            eye_left_img, eye_right_img = p1, p2
        else:
            eye_left_img, eye_right_img = p2, p1

        dx = eye_right_img[0] - eye_left_img[0]
        dy = eye_right_img[1] - eye_left_img[1]
        if dx < 2:
            return face_bgr

        angle = float(np.degrees(np.arctan2(dy, dx)))
        if abs(angle) > _MAX_ALIGN_ANGLE_DEG:
            return face_bgr

        h, w = face_bgr.shape[:2]
        center = ((eye_left_img[0] + eye_right_img[0]) // 2, (eye_left_img[1] + eye_right_img[1]) // 2)
        center = (max(0, min(center[0], w - 1)), max(0, min(center[1], h - 1)))

        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        return cv2.warpAffine(
            face_bgr, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    @staticmethod
    def _preprocess(face_image_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        norm = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
        return blob.astype(np.float32)

    def _infer(self, face_image_bgr: np.ndarray) -> np.ndarray | None:
        if self._model is None:
            return None
        try:
            blob = self._preprocess(face_image_bgr)
            tensor_input = torch.from_numpy(blob)
            with torch.no_grad():
                output = self._model(tensor_input)
            return output[0].cpu().numpy()
        except Exception as exc:
            print(f"[EmotionAnalyzer] Infer error: {exc}")
            return None

    @staticmethod
    def _calibrate_logits(logits: np.ndarray, features: dict | None = None) -> np.ndarray:
        """
        Mô hình MobileNetV2 (huấn luyện trên RAF-DB) là mô hình Deep Learning đầu-cuối
        đã tự học toàn diện các đặc trưng cảm xúc 7 lớp trực tiếp từ pixel ảnh 224x224.
        Dựa 100% vào kết quả dự đoán của mô hình, loại bỏ toàn bộ luật cứng hình học
        để không làm méo mó hoặc nhảy nhãn biểu cảm tự nhiên của người dùng.
        """
        return logits

    @staticmethod
    def _softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        scaled = logits / max(temperature, 0.1)
        exp_s = np.exp(scaled - np.max(scaled))
        return exp_s / np.sum(exp_s)

    def _ensemble_logits(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """Chạy 2 pass (ảnh gốc + lật ngang) để vừa nhanh vừa không tạo viền xoay giả."""
        logits_orig = self._infer(face_bgr)
        flipped = cv2.flip(face_bgr, 1)
        logits_flip = self._infer(flipped)

        if logits_orig is not None and logits_flip is not None:
            return (logits_orig + logits_flip) * 0.5
        return logits_orig if logits_orig is not None else logits_flip

    def analyze(self, face_image_bgr: np.ndarray, fast_mode: bool = False) -> dict[str, float]:
        if not self.is_ready() or face_image_bgr is None or face_image_bgr.size == 0:
            return {}
        h, w = face_image_bgr.shape[:2]
        if h < 20 or w < 20:
            return {}

        logits = self._ensemble_logits(face_image_bgr)
        if logits is None:
            return {}
        calibrated = self._calibrate_logits(logits)
        return self._logits_to_emotions(calibrated, use_ema=fast_mode)

    def analyze_aligned(
        self,
        full_image_bgr: np.ndarray,
        box: list[int],
        left_eye: tuple[int, int] | None = None,
        right_eye: tuple[int, int] | None = None,
        fast_mode: bool = False,
        features: dict | None = None,
        **_kwargs,
    ) -> dict[str, float]:
        if not self.is_ready():
            return {}

        x, y, bw, bh = box
        img_h, img_w = full_image_bgr.shape[:2]

        # Mở rộng 20% để giữ trọn vẹn khuôn mặt (trán, cằm, má) theo đúng tỷ lệ chuẩn của RAF-DB
        pad_x = int(bw * 0.20)
        pad_y = int(bh * 0.20)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + bw + pad_x)
        y2 = min(img_h, y + bh + pad_y)
        face_crop = full_image_bgr[y1:y2, x1:x2]

        if face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            return {}

        aligned = self._align_face(
            face_crop,
            left_eye=left_eye,
            right_eye=right_eye,
            face_box=[x1, y1, x2 - x1, y2 - y1],
        )

        logits = self._ensemble_logits(aligned)
        if logits is None:
            return {}

        calibrated_logits = self._calibrate_logits(logits, features=features)
        # Bật EMA smoothing khi chạy luồng camera (fast_mode=True) để chống nhảy nhãn giữa các frame
        return self._logits_to_emotions(calibrated_logits, use_ema=fast_mode)

    def _logits_to_emotions(self, logits: np.ndarray, use_ema: bool = True) -> dict[str, float]:
        probabilities = self._softmax_temperature(logits, _TEMPERATURE)

        if use_ema:
            if self._ema_probs is None:
                self._ema_probs = probabilities
            else:
                self._ema_probs = _EMA_ALPHA * probabilities + (1.0 - _EMA_ALPHA) * self._ema_probs
            probs_out = self._ema_probs
        else:
            probs_out = probabilities

        emotions = {label: float(prob) for label, prob in zip(_LABELS, probs_out)}
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: round(v / total, 4) for k, v in emotions.items()}
        return emotions

    def analyze_with_padding(self, full_image_bgr: np.ndarray, box: list[int]) -> dict[str, float]:
        return self.analyze_aligned(full_image_bgr, box)