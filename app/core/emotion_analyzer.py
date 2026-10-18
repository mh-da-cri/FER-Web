"""
Module Phân Tích Biểu Cảm sử dụng OpenCV DNN hỗ trợ đa mô hình:
  1. MobileNetV2 (RGB 224x224, 7 nhãn: Surprise, Fear, Disgust, Happy, Sad, Angry, Neutral) - Tối ưu tốc độ, nhận diện màu sắc
  2. FER+ (Grayscale 64x64, 8 nhãn Microsoft FER+) - Mô hình chuẩn gốc

Không phụ thuộc tensorflow hay PyTorch runtime khi inference — chỉ dùng OpenCV DNN thuần túy.

Cải tiến:
  - Hỗ trợ chuyển đổi linh hoạt (Hot-swap) giữa các mô hình mà không cần khởi động lại
  - Face Alignment: xoay khuôn mặt về góc nghiêng 0° theo trục mắt
  - TTA (Test-Time Augmentation): ensemble inference gốc + flip + slight rotations
  - Temperature softmax: phân phối mềm hơn
  - Hình học Face Mesh: brow / eye / mouth / upper-lip features hiệu chỉnh nhẹ logits
"""

import os
import urllib.request

import cv2
import numpy as np


# Đường dẫn lưu mô hình
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Cấu hình chi tiết cho từng mô hình
_MODEL_CONFIGS = {
    "mobilenetv2": {
        "filename": "mobilenetv2_opencv.onnx",
        "name": "MobileNetV2 (RGB 224x224)",
        "description": "Nhẹ, mượt mà, tối ưu thời gian thực với ảnh màu RGB 224x224 (7 nhãn)",
        "input_size": (224, 224),
        "is_rgb": True,
        "labels": ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"],
        "temperature": 1.0,
        "prior_biases": {
            "surprise": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "angry": 0.0,
            "neutral": 0.0,
        },
    },
    "ferplus": {
        "filename": "emotion_model.onnx",
        "name": "FER+ ONNX (64x64 Grayscale)",
        "description": "Mô hình FER+ gốc của Microsoft (8 nhãn gộp về 7 nhãn)",
        "input_size": (64, 64),
        "is_rgb": False,
        "labels": ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"],
        "temperature": 1.4,
        "prior_biases": {
            "neutral": -0.55,
            "happy": -0.15,
            "surprise": 0.10,
            "sad": 0.25,
            "angry": 0.25,
            "disgust": 0.25,
            "fear": 0.25,
            "contempt": 0.05,
        },
    },
}

# ImageNet normalization constants cho MobileNetV2
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Padding ratio khi cắt khuôn mặt
_FACE_PADDING = 0.15

# Góc xoay tối đa cho face alignment — nếu vượt quá, bỏ qua alignment
_MAX_ALIGN_ANGLE_DEG = 30.0

# TTA: danh sách góc xoay nhỏ để augment (độ)
_TTA_ANGLES = [0.0, -5.0, 5.0]


class EmotionAnalyzer:
    """Phân tích cảm xúc từ vùng khuôn mặt sử dụng OpenCV DNN hỗ trợ đa mô hình."""

    def __init__(self, default_model: str = "mobilenetv2"):
        self._models: dict[str, cv2.dnn.Net] = {}
        self._active_model_id = default_model if default_model in _MODEL_CONFIGS else "mobilenetv2"
        self._ensure_models()

    # ------------------------------------------------------------------
    # Model management & loading
    # ------------------------------------------------------------------

    def _ensure_models(self):
        """Khởi tạo và nạp sẵn các mô hình ONNX để chuyển đổi tức thì."""
        os.makedirs(_MODEL_DIR, exist_ok=True)

        for model_id, config in _MODEL_CONFIGS.items():
            model_path = os.path.join(_MODEL_DIR, config["filename"])

            # Tự động tải emotion_model.onnx nếu chưa có
            if model_id == "ferplus" and not os.path.exists(model_path):
                print(f"[EmotionAnalyzer] Downloading {config['name']}...")
                url = (
                    "https://github.com/onnx/models/raw/main/validated/vision/"
                    "body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
                )
                try:
                    urllib.request.urlretrieve(url, model_path)
                    print(f"[EmotionAnalyzer] {config['name']} downloaded successfully!")
                except Exception as exc:
                    print(f"[EmotionAnalyzer] Failed to download {config['name']}: {exc}")
                    continue

            if os.path.exists(model_path):
                try:
                    net = cv2.dnn.readNetFromONNX(model_path)
                    self._models[model_id] = net
                    print(f"[EmotionAnalyzer] Model '{model_id}' ({config['name']}) loaded successfully.")
                except Exception as exc:
                    print(f"[EmotionAnalyzer] Error loading model '{model_id}' from {model_path}: {exc}")

        # Đảm bảo active_model_id hợp lệ
        if self._active_model_id not in self._models and self._models:
            self._active_model_id = next(iter(self._models.keys()))

    def get_active_model(self) -> str:
        """Trả về ID của mô hình đang hoạt động."""
        return self._active_model_id

    def set_active_model(self, model_id: str) -> bool:
        """
        Chuyển đổi mô hình đang hoạt động ('mobilenetv2' hoặc 'ferplus').
        """
        if model_id in self._models:
            self._active_model_id = model_id
            print(f"[EmotionAnalyzer] Switched active model to: {model_id}")
            return True
        elif model_id in _MODEL_CONFIGS:
            # Thử nạp lại nếu trước đó chưa nạp
            model_path = os.path.join(_MODEL_DIR, _MODEL_CONFIGS[model_id]["filename"])
            if os.path.exists(model_path):
                try:
                    net = cv2.dnn.readNetFromONNX(model_path)
                    self._models[model_id] = net
                    self._active_model_id = model_id
                    return True
                except Exception as exc:
                    print(f"[EmotionAnalyzer] Failed to load model {model_id}: {exc}")
        return False

    def get_available_models(self) -> list[dict]:
        """Lấy danh sách các mô hình khả dụng kèm thông tin mô tả."""
        result = []
        for mid, cfg in _MODEL_CONFIGS.items():
            result.append({
                "id": mid,
                "name": cfg["name"],
                "description": cfg["description"],
                "is_active": mid == self._active_model_id,
                "is_loaded": mid in self._models,
            })
        return result

    # ------------------------------------------------------------------
    # Face Alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _align_face(
        face_bgr: np.ndarray,
        left_eye: tuple[int, int] | None,
        right_eye: tuple[int, int] | None,
        face_box: list[int],
    ) -> np.ndarray:
        """
        Xoay khuôn mặt về góc nghiêng 0° dựa trên vị trí hai mắt.
        """
        if left_eye is None or right_eye is None:
            return face_bgr

        ox, oy = face_box[0], face_box[1]
        le = (left_eye[0] - ox, left_eye[1] - oy)
        re = (right_eye[0] - ox, right_eye[1] - oy)

        dx = re[0] - le[0]
        dy = re[1] - le[1]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        if abs(angle) > _MAX_ALIGN_ANGLE_DEG:
            return face_bgr

        h, w = face_bgr.shape[:2]
        center = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2)
        center = (
            max(0, min(center[0], w - 1)),
            max(0, min(center[1], h - 1)),
        )

        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        aligned = cv2.warpAffine(
            face_bgr, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return aligned

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_mobilenet(face_image_bgr: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh cho MobileNetV2:
          - BGR -> RGB
          - Resize 224x224 (INTER_LINEAR)
          - Chuẩn hoá ImageNet: (img / 255 - mean) / std
          - Shape: (1, 3, 224, 224)
        """
        rgb = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        norm = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
        return blob.astype(np.float32)

    @staticmethod
    def _preprocess_ferplus(face_image_bgr: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh cho FER+:
          - BGR -> Grayscale
          - Resize 64x64 LANCZOS4
          - Shape: (1, 1, 64, 64)
        """
        gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0,
            size=(64, 64),
            mean=(0,),
            swapRB=False,
            crop=False,
        )
        return blob

    # ------------------------------------------------------------------
    # Inference (single pass)
    # ------------------------------------------------------------------

    def _infer(self, face_image_bgr: np.ndarray, model_id: str) -> np.ndarray | None:
        """Chạy một lần inference trên mô hình chỉ định, trả về logits thô."""
        net = self._models.get(model_id)
        if net is None:
            return None

        try:
            if model_id == "mobilenetv2":
                blob = self._preprocess_mobilenet(face_image_bgr)
            else:
                blob = self._preprocess_ferplus(face_image_bgr)

            net.setInput(blob)
            output = net.forward()
            return output[0].copy()
        except Exception as exc:
            print(f"[EmotionAnalyzer] Infer error on model '{model_id}': {exc}")
            return None

    # ------------------------------------------------------------------
    # Post-processing & TTA
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Softmax với temperature scaling."""
        scaled = logits / max(temperature, 0.1)
        exp_s = np.exp(scaled - np.max(scaled))
        return exp_s / np.sum(exp_s)

    @staticmethod
    def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
        """Xoay ảnh quanh tâm một góc nhỏ (dùng cho TTA)."""
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        return cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def _ensemble_logits(
        self,
        face_bgr: np.ndarray,
        model_id: str,
        fast_mode: bool = False,
    ) -> np.ndarray | None:
        """
        Test-Time Augmentation: chạy inference trên ảnh gốc + flip ngang
        (+ slight rotations nếu fast_mode=False) rồi trung bình logits.
        """
        all_logits = []

        if fast_mode:
            # Pass 1: ảnh gốc
            logits_orig = self._infer(face_bgr, model_id)
            if logits_orig is not None:
                all_logits.append(logits_orig)
            # Pass 2: flip ngang
            flipped = cv2.flip(face_bgr, 1)
            logits_flip = self._infer(flipped, model_id)
            if logits_flip is not None:
                all_logits.append(logits_flip)
        else:
            # Pass 1: ảnh gốc + slight rotations
            for angle in _TTA_ANGLES:
                augmented = self._rotate_image(face_bgr, angle) if angle != 0.0 else face_bgr
                logits = self._infer(augmented, model_id)
                if logits is not None:
                    all_logits.append(logits)

            # Pass 2: flip ngang
            flipped = cv2.flip(face_bgr, 1)
            logits_flip = self._infer(flipped, model_id)
            if logits_flip is not None:
                all_logits.append(logits_flip)

        if not all_logits:
            return None

        return np.mean(all_logits, axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        face_image_bgr: np.ndarray,
        fast_mode: bool = False,
        model_name: str | None = None,
    ) -> dict[str, float]:
        """
        Phân tích cảm xúc từ ảnh khuôn mặt đã cắt (không có alignment).
        """
        target_model = model_name if (model_name and model_name in self._models) else self._active_model_id
        if target_model not in self._models:
            return {}
        if face_image_bgr is None or face_image_bgr.size == 0:
            return {}
        h, w = face_image_bgr.shape[:2]
        if h < 20 or w < 20:
            return {}

        logits = self._ensemble_logits(face_image_bgr, target_model, fast_mode=fast_mode)
        if logits is None:
            return {}

        return self._logits_to_emotions(logits, target_model)

    def analyze_aligned(
        self,
        full_image_bgr: np.ndarray,
        box: list[int],
        left_eye: tuple[int, int] | None = None,
        right_eye: tuple[int, int] | None = None,
        eyebrow_lift: float = 0.0,
        mouth_openness: float = 0.0,
        eye_openness: float = 0.0,
        mouth_width_ratio: float = 0.0,
        face_vertical_expansion: float = 0.0,
        brow_slope: float = 0.0,
        brow_inner_drop: float = 0.0,
        mouth_corner_angle: float = 0.0,
        mouth_stretch: float = 0.0,
        upper_lip_raise: float = 0.0,
        nose_wrinkle: float = 0.0,
        fast_mode: bool = False,
        model_name: str | None = None,
    ) -> dict[str, float]:
        """
        Pipeline đầy đủ: padding → face alignment → TTA → ensemble → softmax.
        """
        target_model = model_name if (model_name and model_name in self._models) else self._active_model_id
        if target_model not in self._models:
            return {}

        x, y, bw, bh = box
        img_h, img_w = full_image_bgr.shape[:2]

        # Bước 1: padding mở rộng bounding box
        pad_x = int(bw * _FACE_PADDING)
        pad_y = int(bh * _FACE_PADDING)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + bw + pad_x)
        y2 = min(img_h, y + bh + pad_y)
        face_crop = full_image_bgr[y1:y2, x1:x2]

        if face_crop.size == 0:
            return {}

        # Bước 2: face alignment — xoay về góc 0° theo trục mắt
        aligned = self._align_face(
            face_crop,
            left_eye=left_eye,
            right_eye=right_eye,
            face_box=[x1, y1, x2 - x1, y2 - y1],
        )

        # Bước 3: TTA ensemble
        if aligned.shape[0] < 20 or aligned.shape[1] < 20:
            return {}

        logits = self._ensemble_logits(aligned, target_model, fast_mode=fast_mode)
        if logits is None:
            return {}

        return self._logits_to_emotions(
            logits,
            target_model,
            eyebrow_lift=eyebrow_lift,
            mouth_openness=mouth_openness,
            eye_openness=eye_openness,
            mouth_width_ratio=mouth_width_ratio,
            face_vertical_expansion=face_vertical_expansion,
            brow_slope=brow_slope,
            brow_inner_drop=brow_inner_drop,
            mouth_corner_angle=mouth_corner_angle,
            mouth_stretch=mouth_stretch,
            upper_lip_raise=upper_lip_raise,
            nose_wrinkle=nose_wrinkle,
        )

    def _logits_to_emotions(
        self,
        logits: np.ndarray,
        model_id: str,
        eyebrow_lift: float = 0.0,
        mouth_openness: float = 0.0,
        eye_openness: float = 0.0,
        mouth_width_ratio: float = 0.0,
        face_vertical_expansion: float = 0.0,
        brow_slope: float = 0.0,
        brow_inner_drop: float = 0.0,
        mouth_corner_angle: float = 0.0,
        mouth_stretch: float = 0.0,
        upper_lip_raise: float = 0.0,
        nose_wrinkle: float = 0.0,
    ) -> dict[str, float]:
        """Chuyển logits thành xác suất chuẩn 7 cảm xúc, kết hợp heuristic hình học."""
        config = _MODEL_CONFIGS.get(model_id, _MODEL_CONFIGS["mobilenetv2"])
        labels = config["labels"]
        # Đối với MobileNetV2: Tắt toàn bộ tính toán hình học, dựa 100% vào kết quả train của model
        if model_id == "mobilenetv2":
            probabilities = self._softmax_temperature(logits, config.get("temperature", 1.0))
            emotions = {label: float(prob) for label, prob in zip(labels, probabilities)}
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: round(v / total, 4) for k, v in emotions.items()}
            return emotions

        prior_biases_map = config["prior_biases"]

        # Các hàm ramp tạo điểm mềm
        def ramp(v, lo, hi):
            if hi <= lo:
                return 0.0
            return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

        def inv_ramp(v, lo, hi):
            return 1.0 - ramp(v, lo, hi)

        angry_score = 0.0
        sad_score = 0.0
        disgust_score = 0.0
        fear_score = 0.0
        surprise_score = 0.0
        happy_score = 0.0

        # Smile detection: khóe miệng nhếch lên khi mouth_corner_angle âm
        corners_up = ramp(-mouth_corner_angle, 0.015, 0.045)
        horizontal_stretch = ramp(mouth_stretch, 0.35, 0.60)
        not_smiling = inv_ramp(-mouth_corner_angle, 0.0, 0.020)

        # Happy signature: khóe miệng nâng lên + miệng dãn ngang
        happy_score += 2.5 * corners_up
        happy_score += 1.5 * corners_up * horizontal_stretch

        # Brow features
        angry_score += 2.0 * ramp(brow_inner_drop, 0.005, 0.035)
        angry_score += 1.5 * ramp(brow_slope, 0.005, 0.025)
        sad_score += 1.6 * inv_ramp(brow_inner_drop, -0.025, 0.0)
        surprise_score += 2.2 * ramp(eyebrow_lift, 0.105, 0.145)

        # Eyes features
        surprise_score += 2.4 * ramp(eye_openness, 0.31, 0.39)
        angry_score += 1.9 * inv_ramp(eye_openness, 0.235, 0.285)
        sad_score += 0.7 * inv_ramp(eye_openness, 0.22, 0.27)

        # Mouth opening
        vertical_open = ramp(mouth_openness, 0.045, 0.12)
        # Surprise: há miệng mở theo chiều dọc (hình chữ O tròn)
        surprise_score += 2.0 * vertical_open * inv_ramp(mouth_width_ratio, 3.5, 6.0)

        # Corners down: khóe miệng trễ xuống
        corners_down = ramp(mouth_corner_angle, 0.008, 0.030)
        sad_score += 2.0 * corners_down
        disgust_score += 1.2 * corners_down
        angry_score += 0.8 * corners_down

        # Fear: CHỈ kích hoạt khi người đó KHÔNG cười và khóe miệng trễ xuống
        fear_score += 1.5 * ramp(eyebrow_lift, 0.090, 0.125) * not_smiling
        fear_score += 1.5 * ramp(eye_openness, 0.29, 0.36) * not_smiling
        fear_score += 1.8 * vertical_open * horizontal_stretch * not_smiling * corners_down

        # Disgust signature: CHỈ kích hoạt khi người đó KHÔNG cười
        disgust_score += 2.0 * ramp(upper_lip_raise, 0.15, 0.45) * not_smiling
        disgust_score += 1.5 * ramp(nose_wrinkle, 0.12, 0.35) * not_smiling

        # Bản đồ heuristic theo tên nhãn
        heuristic_map = {
            "angry": angry_score,
            "disgust": disgust_score,
            "fear": fear_score,
            "happy": happy_score,
            "sad": sad_score,
            "surprise": surprise_score,
            "neutral": 0.0,
            "contempt": 0.0,
        }

        # Vector hóa theo thứ tự nhãn của mô hình hiện tại
        prior_vec = np.array([prior_biases_map.get(label, 0.0) for label in labels], dtype=np.float32)

        has_mesh = any(
            abs(v) > 1e-4
            for v in [
                eyebrow_lift,
                mouth_openness,
                eye_openness,
                mouth_width_ratio,
                face_vertical_expansion,
                brow_slope,
                brow_inner_drop,
                mouth_corner_angle,
                mouth_stretch,
                upper_lip_raise,
                nose_wrinkle,
            ]
        )

        if has_mesh:
            heur_vec = np.array([heuristic_map.get(label, 0.0) for label in labels], dtype=np.float32)
            # Centering heuristic để tránh lạm phát xác suất tổng thể
            heur_vec -= np.mean(heur_vec)
            # MobileNetV2 có biên độ logits hẹp hơn FER+, dùng scale và clip nhẹ hơn để không lấn át mô hình
            scale = 0.35 if model_id == "mobilenetv2" else 0.75
            max_clip = 0.8 if model_id == "mobilenetv2" else 2.0
            heur_vec = np.clip(heur_vec * scale, -max_clip, max_clip)
        else:
            heur_vec = np.zeros_like(prior_vec)

        calibrated_logits = logits + prior_vec + heur_vec
        probabilities = self._softmax_temperature(calibrated_logits, config["temperature"])

        # Chuyển đổi thành từ điển cảm xúc
        emotions: dict[str, float] = {}
        for label, prob in zip(labels, probabilities):
            emotions[label] = float(prob)

        # Nếu có nhãn 'contempt' (mô hình FER+), gộp vào 'disgust'
        if "contempt" in emotions:
            contempt_val = emotions.pop("contempt")
            emotions["disgust"] = emotions.get("disgust", 0.0) + contempt_val

        # Chuẩn hóa lại tổng xác suất về 1.0 và làm tròn 4 chữ số
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: round(v / total, 4) for k, v in emotions.items()}

        return emotions

    # Giữ lại để backward compatibility
    def analyze_with_padding(
        self,
        full_image_bgr: np.ndarray,
        box: list[int],
    ) -> dict[str, float]:
        """Backward-compatible wrapper — không có alignment."""
        return self.analyze_aligned(full_image_bgr, box)
