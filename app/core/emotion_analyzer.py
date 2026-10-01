"""
Module Phân Tích Biểu Cảm sử dụng OpenCV DNN + mô hình FER+.

Phân tích biểu cảm khuôn mặt từ các vùng khuôn mặt đã cắt.
Không phụ thuộc tensorflow hay fer — chỉ dùng OpenCV.

Cải tiến:
  - Face Alignment: xoay khuôn mặt về góc nghiêng 0° theo trục mắt
  - TTA (Test-Time Augmentation): ensemble inference gốc + flip + slight rotations
  - Temperature softmax: phân phối mềm hơn
  - Hình học Face Mesh: brow / eye / mouth / upper-lip features để tách cảm xúc khó
  - Heuristic scoring chỉ hiệu chỉnh nhẹ logits của FER+ thay vì ghi đè model
"""

import os
import urllib.request

import cv2
import numpy as np


# Đường dẫn lưu mô hình
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Mô hình emotion-ferplus-8.onnx (FER+)
# 8 class theo đúng thứ tự output của model:
# 0=neutral, 1=happiness, 2=surprise, 3=sadness, 4=anger, 5=disgust, 6=fear, 7=contempt
_EMOTION_LABELS = ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]

# Temperature scaling: > 1 làm phẳng phân phối để các cảm xúc thiểu số có cơ hội hơn
_TEMPERATURE = 1.4

# Padding ratio khi cắt khuôn mặt — lấy thêm ngữ cảnh xung quanh
_FACE_PADDING = 0.15

# Góc xoay tối đa cho face alignment — nếu vượt quá, bỏ qua alignment
_MAX_ALIGN_ANGLE_DEG = 30.0

# TTA: danh sách góc xoay nhỏ để augment (độ)
# flip ngang được xử lý riêng vì cần đổi nhãn left/right nếu có
_TTA_ANGLES = [0.0, -5.0, 5.0]


class EmotionAnalyzer:
    """Phân tích cảm xúc từ vùng khuôn mặt sử dụng OpenCV DNN."""

    def __init__(self):
        self._model = None
        self._ensure_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

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

        Khuôn mặt bị nghiêng làm mô hình mất đi tính bất biến với góc xoay,
        dẫn đến nhận diện sai. Alignment giải quyết điều này.

        Args:
            face_bgr   : Vùng khuôn mặt đã cắt (sau padding).
            left_eye   : Tọa độ mắt trái trong ảnh GỐC (toàn cục).
            right_eye  : Tọa độ mắt phải trong ảnh GỐC (toàn cục).
            face_box   : [x, y, w, h] của vùng crop trong ảnh gốc (sau padding).

        Returns:
            Ảnh khuôn mặt đã được căn chỉnh (cùng kích thước với đầu vào).
        """
        if left_eye is None or right_eye is None:
            return face_bgr

        # Chuyển tọa độ global → local (trong crop)
        ox, oy = face_box[0], face_box[1]
        le = (left_eye[0] - ox, left_eye[1] - oy)
        re = (right_eye[0] - ox, right_eye[1] - oy)

        # Tính góc nghiêng giữa hai mắt
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        angle = float(np.degrees(np.arctan2(dy, dx)))

        # Bỏ qua nếu góc quá lớn (landmark không đáng tin cậy)
        if abs(angle) > _MAX_ALIGN_ANGLE_DEG:
            return face_bgr

        # Tâm xoay = trung điểm hai mắt
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
    def _preprocess(face_image_bgr: np.ndarray) -> np.ndarray:
        """
        Chuẩn bị ảnh khuôn mặt theo pipeline của FER+:
          1. Grayscale
          2. Resize 64×64 với LANCZOS4
        LƯU Ý: Không dùng CLAHE hay Unsharp Mask ở đây vì FER+ được huấn luyện
        trên dữ liệu ảnh thô. Tăng độ nét/tương phản sẽ làm các nếp nhăn trán
        đậm lên, khiến mô hình nhầm lẫn mạnh sang 'Angry'.
        """
        gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LANCZOS4)
        return resized.astype(np.float32)

    # ------------------------------------------------------------------
    # Inference (single pass)
    # ------------------------------------------------------------------

    def _infer(self, face_image_bgr: np.ndarray) -> np.ndarray | None:
        """
        Chạy một lần inference, trả về logits thô (shape: 8,).
        Trả None nếu có lỗi.
        """
        try:
            preprocessed = self._preprocess(face_image_bgr)
            blob = cv2.dnn.blobFromImage(
                preprocessed,
                scalefactor=1.0,
                size=(64, 64),
                mean=(0,),
                swapRB=False,
                crop=False,
            )
            self._model.setInput(blob)
            output = self._model.forward()   # (1, 8)
            return output[0].copy()          # (8,)
        except Exception as exc:
            print(f"[EmotionAnalyzer] Infer error: {exc}")
            return None

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Softmax với temperature scaling.
        T > 1 → phân phối phẳng hơn (các cảm xúc cạnh tranh nhau nhiều hơn)
        T = 1 → softmax thông thường
        T < 1 → mô hình "tự tin" hơn vào 1 cảm xúc
        """
        scaled = logits / temperature
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

    def _ensemble_logits(self, face_bgr: np.ndarray) -> np.ndarray | None:
        """
        Test-Time Augmentation: chạy inference trên nhiều augmented version
        của cùng một khuôn mặt rồi ensemble bằng cách cộng logits.

        Augmentations:
          - Ảnh gốc
          - Flip ngang (mirror)
          - Xoay ±5° nhỏ

        Ensemble logits (trước softmax) cho kết quả tốt hơn ensemble probabilities
        vì tránh bị bão hòa ở đầu ra.
        """
        all_logits = []

        # Pass 1: ảnh gốc + slight rotations
        for angle in _TTA_ANGLES:
            augmented = self._rotate_image(face_bgr, angle) if angle != 0.0 else face_bgr
            logits = self._infer(augmented)
            if logits is not None:
                all_logits.append(logits)

        # Pass 2: flip ngang
        flipped = cv2.flip(face_bgr, 1)
        logits_flip = self._infer(flipped)
        if logits_flip is not None:
            all_logits.append(logits_flip)

        if not all_logits:
            return None

        # Ensemble bằng trung bình logits (trước softmax)
        return np.mean(all_logits, axis=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, face_image_bgr: np.ndarray) -> dict[str, float]:
        """
        Phân tích cảm xúc từ ảnh khuôn mặt đã cắt (không có alignment).

        Args:
            face_image_bgr: Ảnh khuôn mặt đã cắt định dạng BGR.

        Returns:
            Dictionary ánh xạ tên cảm xúc sang xác suất (0.0 - 1.0).
        """
        if self._model is None:
            return {}
        if face_image_bgr is None or face_image_bgr.size == 0:
            return {}
        h, w = face_image_bgr.shape[:2]
        if h < 20 or w < 20:
            return {}

        logits = self._ensemble_logits(face_image_bgr)
        if logits is None:
            return {}

        return self._logits_to_emotions(logits)

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
    ) -> dict[str, float]:
        """
        Pipeline đầy đủ: padding → face alignment → TTA → ensemble → softmax.

        Args:
            full_image_bgr : Ảnh gốc đầy đủ định dạng BGR.
            box            : [x, y, w, h] bounding box khuôn mặt trong ảnh gốc.
            left_eye       : Tọa độ mắt trái trong ảnh gốc (từ face detector).
            right_eye      : Tọa độ mắt phải trong ảnh gốc (từ face detector).
            eyebrow_lift   : Tỷ lệ nhướn lông mày để phân biệt ngạc nhiên / tức giận.
            mouth_openness : Tỷ lệ há miệng để củng cố biểu cảm ngạc nhiên.
        """
        if self._model is None:
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

        logits = self._ensemble_logits(aligned)
        if logits is None:
            return {}

        return self._logits_to_emotions(
            logits, eyebrow_lift, mouth_openness,
            eye_openness, mouth_width_ratio, face_vertical_expansion,
            brow_slope, brow_inner_drop, mouth_corner_angle,
            mouth_stretch, upper_lip_raise, nose_wrinkle,
        )

    def _logits_to_emotions(
        self,
        logits: np.ndarray,
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
        """Chuyển logits thành xác suất, kết hợp model với đặc trưng hình học.

        Các heuristic chỉ là tín hiệu phụ. Model FER+ vẫn là nguồn chính.
        """

        # Bias nhẹ hơn bản cũ: tránh ép cảm xúc khó thành một nhãn chỉ vì prior.
        prior_biases = np.array([
            -0.55, -0.15, 0.10, 0.25, 0.25, 0.25, 0.25, 0.05
        ], dtype=np.float32)
        
        # ---------------------------------------------------------------
        # Hình học phân biệt cảm xúc
        #
        # Ý tưởng:
        # angry  = inner brow kéo xuống + mắt nheo + môi ép/khóe xuống
        # sad    = inner brow nâng + khóe miệng xuống + mắt hơi cụp
        # disgust= môi trên nâng + vùng mũi co + khóe miệng xuống
        # fear   = mắt mở + miệng kéo ngang + brow nâng vừa
        # surprise = mắt mở rất rộng + brow nâng rõ + miệng mở theo chiều dọc
        #
        # Các hàm ramp chỉ tạo điểm mềm, không dùng ngưỡng nhị phân cứng.
        # ---------------------------------------------------------------
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

        # Brow: slope/drop là tín hiệu quan trọng hơn khoảng cách brow-eye tuyệt đối.
        angry_score += 2.4 * ramp(brow_inner_drop, 0.005, 0.035)
        angry_score += 1.8 * ramp(brow_slope, 0.005, 0.025)
        sad_score += 1.6 * inv_ramp(brow_inner_drop, -0.025, 0.0)
        surprise_score += 2.2 * ramp(eyebrow_lift, 0.105, 0.145)
        fear_score += 1.1 * ramp(eyebrow_lift, 0.090, 0.125)

        # Eyes.
        surprise_score += 2.4 * ramp(eye_openness, 0.31, 0.39)
        fear_score += 1.8 * ramp(eye_openness, 0.29, 0.36)
        angry_score += 1.9 * inv_ramp(eye_openness, 0.235, 0.285)
        sad_score += 0.7 * inv_ramp(eye_openness, 0.22, 0.27)

        # Mouth opening / shape.
        vertical_open = ramp(mouth_openness, 0.045, 0.12)
        horizontal_stretch = ramp(mouth_stretch, 0.34, 0.55)
        wide_ratio = ramp(mouth_width_ratio, 3.0, 5.0)

        # Surprise: "O" vertical opening; fear: horizontally stretched opening.
        surprise_score += 1.7 * vertical_open * wide_ratio
        fear_score += 2.2 * vertical_open * horizontal_stretch
        angry_score += 0.9 * ramp(mouth_openness, 0.025, 0.07) * inv_ramp(mouth_width_ratio, 1.4, 2.6)

        # Corners down: sad/disgust; corners compressed/down can also accompany anger.
        corners_down = ramp(mouth_corner_angle, 0.008, 0.030)
        sad_score += 2.0 * corners_down
        disgust_score += 1.2 * corners_down
        angry_score += 0.8 * corners_down

        # Disgust signature: upper lip raised / nose-mouth region contracted.
        disgust_score += 2.8 * ramp(upper_lip_raise, 0.12, 0.45)
        disgust_score += 1.8 * ramp(nose_wrinkle, 0.08, 0.32)

        # Avoid interpreting every open mouth as fear/surprise.
        # Fear gets extra weight only when eyes are wide AND mouth is stretched.
        fear_score += 0.9 * ramp(eye_openness, 0.29, 0.36) * horizontal_stretch

        # Convert scores to bounded logit adjustments. Model remains dominant.
        heuristic = np.array([
            0.0,
            0.0,
            surprise_score,
            sad_score,
            angry_score,
            disgust_score,
            fear_score,
            0.0,
        ], dtype=np.float32)

        # Center the heuristic scores so no global probability inflation occurs.
        heuristic -= np.mean(heuristic)
        heuristic = np.clip(heuristic * 0.75, -2.0, 2.0)
        prior_biases += heuristic

        calibrated_logits = logits + prior_biases
        
        probabilities = self._softmax_temperature(calibrated_logits, _TEMPERATURE)

        # Gộp "contempt" vào "disgust"
        contempt_idx = _EMOTION_LABELS.index("contempt")
        contempt_prob = float(probabilities[contempt_idx])

        emotions: dict[str, float] = {}
        for label, prob in zip(_EMOTION_LABELS, probabilities):
            if label == "contempt":
                continue
            emotions[label] = float(prob)

        emotions["disgust"] = emotions.get("disgust", 0.0) + contempt_prob

        # Re-normalize
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


