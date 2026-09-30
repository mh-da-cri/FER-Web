"""
Module Phân Tích Biểu Cảm sử dụng OpenCV DNN + mô hình FER+.

Phân tích biểu cảm khuôn mặt từ các vùng khuôn mặt đã cắt.
Không phụ thuộc tensorflow hay fer — chỉ dùng OpenCV.

Cải tiến:
  - Face Alignment: xoay khuôn mặt về góc nghiêng 0° theo trục mắt
  - TTA (Test-Time Augmentation): ensemble inference gốc + flip + slight rotations
  - CLAHE preprocessing: cân bằng histogram cục bộ
  - Temperature softmax: phân phối mềm hơn cho các cảm xúc thiểu số
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
        )

    def _logits_to_emotions(
        self,
        logits: np.ndarray,
        eyebrow_lift: float = 0.0,
        mouth_openness: float = 0.0,
        eye_openness: float = 0.0,
        mouth_width_ratio: float = 0.0,
        face_vertical_expansion: float = 0.0,
    ) -> dict[str, float]:
        """Chuyển logits thô thành dict cảm xúc với xác suất đã normalize."""
        
        # Logit Prior Calibration (LPC)
        # Bù trừ bias của dataset FER2013/FER+ (quá nhiều neutral và happy)
        # Giảm logits của majority classes, tăng cho minority classes
        # Thứ tự: ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"]
        prior_biases = np.array([
            -1.0,  # neutral (giảm mạnh)
            -0.5,  # happy (giảm vừa)
             0.3,  # surprise
             0.5,  # sad
             0.5,  # angry (tăng base để cạnh tranh tốt hơn)
             0.5,  # disgust
             0.5,  # fear
             0.0   # contempt
        ])
        
        # ================================================================
        # Hệ thống điểm số đa đặc trưng: Ngạc Nhiên vs Tức Giận
        # Mỗi đặc trưng đóng góp độc lập -> kết hợp → điều chỉnh logits
        # ================================================================
        surprise_idx = _EMOTION_LABELS.index("surprise")
        angry_idx    = _EMOTION_LABELS.index("angry")

        angry_score    = 0.0
        surprise_score = 0.0

        # --- Đặc trưng 1: Lông mày (trọng số: 3.0) ---
        if eyebrow_lift > 0:
            if eyebrow_lift < 0.100:
                angry_score += 3.0       # Nhíu mày = tức giận
            elif eyebrow_lift > 0.118:
                surprise_score += 3.0    # Nhướng mày cao = ngạc nhiên
            elif eyebrow_lift > 0.108:
                surprise_score += 1.5    # Nhướng nhẹ
            # 0.100-0.108: mập mờ, không tích điểm

        # --- Đặc trưng 2: Mắt - EAR (trọng số: 2.5) ---
        if eye_openness > 0:
            if eye_openness > 0.35:
                surprise_score += 2.5    # Mắt mở thất to
            elif eye_openness > 0.28:
                surprise_score += 1.2    # Mắt mở vừa
            elif eye_openness < 0.24:
                angry_score += 2.5       # Mắt nheo mạnh
            elif eye_openness < 0.30:
                angry_score += 1.5       # Mắt hơi nheo (phổ biến khi tức)

        # --- Đặc trưng 3: Hình dạng miệng (trọng số: 2.0) ---
        # mouth_width_ratio nhỏ = miệng vuông vức la hét (tức giận)
        # mouth_width_ratio lớn = miệng ô tròn gọn (ngạc nhiên)
        if mouth_openness > 0.04:
            if mouth_width_ratio < 1.5:
                angry_score += 2.0
            elif mouth_width_ratio < 2.5:
                angry_score += 0.8
            elif mouth_width_ratio > 4.0:
                surprise_score += 2.0
            elif mouth_width_ratio > 3.0:
                surprise_score += 1.0

        # --- Đặc trưng 4: Tổng thể khuôn mặt ---
        # DISABLED: ngưỡng chưa được calibrate thực tế, gây nhiễu.
        # face_vertical_expansion bình thường nằm trong zone 0.70-0.75
        # khiến bất kỳ khuôn mặt nào cũng bị cộng surprise_score mà không có lý do.
        # pass

        # --- Đặc trưng 5: Há miệng (phụ trợ: 1.0) ---
        if mouth_openness > 0.07 and eyebrow_lift >= 0.108:
            surprise_score += 1.0

        # --- Áp dụng vào logits qua score_diff ---
        score_diff = angry_score - surprise_score
        if score_diff > 1.0:
            boost = min(score_diff * 0.6, 4.0)
            prior_biases[angry_idx]    += boost
            prior_biases[surprise_idx] -= boost
        elif score_diff < -1.0:
            boost = min(-score_diff * 0.6, 4.0)
            prior_biases[surprise_idx] += boost
            prior_biases[angry_idx]    -= boost
        elif score_diff > 0.3:
            prior_biases[angry_idx]    += 0.5
        elif score_diff < -0.3:
            prior_biases[surprise_idx] += 0.5

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


