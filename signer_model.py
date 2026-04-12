from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================
# ПАРАМЕТРЫ ДЛЯ BUKVA (адаптированные)
# ============================================
SEQUENCE_LENGTH = 64  # trimmed видео короче (было 155)
NUM_HANDS = 2
NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURE_SHAPE = (SEQUENCE_LENGTH, NUM_HANDS, NUM_LANDMARKS, NUM_COORDS)
MIN_HAND_ACTIVITY_RATIO = 0.01
AUGMENT_REPEATS = 3  # увеличена аугментация
DEFAULT_TOP_K = 5
MAX_TOP_K = 10
UNSURE_CONFIDENCE_THRESHOLD = 0.20
TTA_FRAME_SHIFTS = (-3, 0, 3)  # меньше сдвигов для коротких видео

# Гиперпараметры LSTM (оптимизированы для 33 классов)
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3
TRAIN_BATCH_SIZE = 32
TRAIN_MAX_EPOCHS = 60
TRAIN_PATIENCE = 8
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 1e-4
VALIDATION_SIZE = 0.15
TRAIN_LABEL_SMOOTHING = 0.05
TRAIN_GRAD_CLIP_NORM = 1.0

# Буквы русского алфавита (33 класса)
RUSSIAN_LETTERS = [
    'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
    'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф',
    'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'
]


@dataclass
class TrainResult:
    samples: int
    classes: int
    train_accuracy: float
    test_accuracy: float
    total_matched_samples: int
    skipped_non_letters: int


class LSTMSequenceClassifier(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        pooled_size = hidden_size * 4
        self.head = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Linear(pooled_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pooled_mean = out.mean(dim=1)
        pooled_max = out.max(dim=1).values
        pooled = torch.cat([pooled_mean, pooled_max], dim=1)
        return self.head(pooled)


class SignTranslatorModel:
    def __init__(
            self,
            annotations_path: str | Path = "annotations.tsv",  # ← изменено для Bukva
            keypoints_dir: str | Path = "bukva_keypoints",  # ← изменено для Bukva
            model_path: str | Path = "models/bukva_model.pth",  # ← изменено
            encoder_path: str | Path = "models/bukva_encoder.joblib",  # ← изменено
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.keypoints_dir = Path(keypoints_dir)
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)

        self.model: LSTMSequenceClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._input_size: int | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._mp_hands = mp.solutions.hands

    def load_if_exists(self) -> bool:
        if not self.model_path.exists() or not self.encoder_path.exists():
            return False

        checkpoint = torch.load(self.model_path, map_location=self._device, weights_only=False)
        input_size = int(checkpoint["input_size"])
        num_classes = int(checkpoint["num_classes"])

        model = LSTMSequenceClassifier(
            input_size=input_size,
            hidden_size=int(checkpoint["hidden_size"]),
            num_layers=int(checkpoint["num_layers"]),
            num_classes=num_classes,
            dropout=float(checkpoint["dropout"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self._device)
        model.eval()

        self.model = model
        self._feature_mean = np.array(checkpoint["feature_mean"], dtype=np.float32)
        self._feature_std = np.array(checkpoint["feature_std"], dtype=np.float32)
        self._input_size = input_size
        self.label_encoder = joblib.load(self.encoder_path)
        return True

    def train(self) -> TrainResult:
        X, y_text, total_matched_samples, skipped_non_letters = self._load_dataset()

        if len(X) < 10:
            raise ValueError("Too few samples found for training.")

        print(f"📊 Загружено {len(X)} образцов, {len(np.unique(y_text))} классов")

        X_train_raw, X_test, y_train_text, y_test_text = train_test_split(
            X,
            y_text,
            test_size=0.2,
            random_state=42,
            stratify=y_text,
        )

        X_train, y_train_aug_text = self._augment_sequence_samples(
            X_train_raw,
            y_train_text,
            repeats=AUGMENT_REPEATS,
        )

        X_fit, X_val, y_fit_text, y_val_text = train_test_split(
            X_train,
            y_train_aug_text,
            test_size=VALIDATION_SIZE,
            random_state=42,
            stratify=y_train_aug_text,
        )

        label_encoder = LabelEncoder()
        label_encoder.fit(y_text)
        y_fit = label_encoder.transform(y_fit_text)
        y_val = label_encoder.transform(y_val_text)
        y_train = label_encoder.transform(y_train_aug_text)
        y_test = label_encoder.transform(y_test_text)

        num_classes = len(label_encoder.classes_)
        print(f"🎯 Количество классов: {num_classes}")

        # Нормализация признаков
        feature_mean = X_fit.reshape(-1, X_fit.shape[-1]).mean(axis=0).astype(np.float32)
        feature_std = X_fit.reshape(-1, X_fit.shape[-1]).std(axis=0).astype(np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

        X_fit_norm = ((X_fit - feature_mean) / feature_std).astype(np.float32)
        X_val_norm = ((X_val - feature_mean) / feature_std).astype(np.float32)
        X_train_norm = ((X_train - feature_mean) / feature_std).astype(np.float32)
        X_test_norm = ((X_test - feature_mean) / feature_std).astype(np.float32)

        input_size = int(X_train_norm.shape[-1])

        model = LSTMSequenceClassifier(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_LAYERS,
            num_classes=num_classes,
            dropout=LSTM_DROPOUT,
        ).to(self._device)

        # Веса классов для балансировки
        class_counts = np.bincount(y_fit, minlength=num_classes).astype(np.float32)
        class_counts = np.maximum(class_counts, 1.0)
        class_weights = class_counts.sum() / (num_classes * class_counts)

        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32, device=self._device),
            label_smoothing=TRAIN_LABEL_SMOOTHING,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
        )

        fit_dataset = TensorDataset(
            torch.tensor(X_fit_norm, dtype=torch.float32),
            torch.tensor(y_fit, dtype=torch.long),
        )
        fit_loader = DataLoader(fit_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        X_val_tensor = torch.tensor(X_val_norm, dtype=torch.float32, device=self._device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long, device=self._device)

        print(f"\n🚀 Начало обучения LSTM...")
        print(f"   Train samples: {len(X_fit_norm)}")
        print(f"   Val samples: {len(X_val_norm)}")
        print(f"   Input size: {input_size}")
        print(f"   Device: {self._device}\n")

        best_val_accuracy = -1.0
        best_state: dict | None = None
        epochs_without_improvement = 0

        for epoch in range(TRAIN_MAX_EPOCHS):
            model.train()
            for batch_x, batch_y in fit_loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=TRAIN_GRAD_CLIP_NORM)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_tensor)
                val_pred = torch.argmax(val_logits, dim=1)
                val_accuracy = float((val_pred == y_val_tensor).float().mean().item())

            scheduler.step(val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
                print(f"   Epoch {epoch + 1:3d} | Val Acc: {val_accuracy:.4f} ✨")
            else:
                epochs_without_improvement += 1
                if (epoch + 1) % 10 == 0:
                    print(
                        f"   Epoch {epoch + 1:3d} | Val Acc: {val_accuracy:.4f} (no improvement: {epochs_without_improvement})")

                if epochs_without_improvement >= TRAIN_PATIENCE:
                    print(f"\n⏹️ Early stopping at epoch {epoch + 1}")
                    break

        if best_state is None:
            raise ValueError("LSTM training failed to produce a valid checkpoint.")

        model.load_state_dict(best_state)
        model.eval()

        train_pred = self._predict_classes(model, X_train_norm)
        test_pred = self._predict_classes(model, X_test_norm)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        self.model = model
        self.label_encoder = label_encoder
        self._feature_mean = feature_mean
        self._feature_std = feature_std
        self._input_size = input_size

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_size": input_size,
                "hidden_size": LSTM_HIDDEN_SIZE,
                "num_layers": LSTM_LAYERS,
                "dropout": LSTM_DROPOUT,
                "num_classes": num_classes,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
            },
            self.model_path,
        )
        joblib.dump(self.label_encoder, self.encoder_path)

        return TrainResult(
            samples=len(X),
            classes=num_classes,
            train_accuracy=float(train_accuracy),
            test_accuracy=float(test_accuracy),
            total_matched_samples=total_matched_samples,
            skipped_non_letters=skipped_non_letters,
        )

    def predict_top_k_from_keypoints(
            self,
            keypoints: np.ndarray,
            top_k: int = DEFAULT_TOP_K,
    ) -> List[Tuple[str, float]]:
        self._ensure_loaded()
        self._ensure_has_hand_activity(keypoints)
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        probabilities = self._predict_probabilities_with_tta(keypoints)
        top_indices = np.argsort(probabilities)[::-1][:top_k]

        labels = self.label_encoder.inverse_transform(top_indices)
        return [(str(label), float(probabilities[idx])) for label, idx in zip(labels, top_indices)]

    def predict_from_keypoints(
            self,
            keypoints: np.ndarray,
            top_k: int = DEFAULT_TOP_K,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        top_predictions = self.predict_top_k_from_keypoints(keypoints, top_k=top_k)
        label, confidence = top_predictions[0]

        if confidence < UNSURE_CONFIDENCE_THRESHOLD:
            label = "UNSURE"

        return label, confidence, top_predictions

    def _predict_probabilities_with_tta(self, keypoints: np.ndarray) -> np.ndarray:
        probs: List[np.ndarray] = []
        for shift in TTA_FRAME_SHIFTS:
            shifted = self._shift_sequence(keypoints, shift)
            features = self._extract_sequence_features(shifted)
            probs.append(self._predict_probabilities(features))

        return np.mean(np.stack(probs, axis=0), axis=0)

    def _predict_probabilities(self, sequence_features: np.ndarray) -> np.ndarray:
        if self.model is None or self._feature_mean is None or self._feature_std is None:
            raise ValueError("Model is not loaded.")

        normalized = ((sequence_features - self._feature_mean) / self._feature_std).astype(np.float32)
        x_tensor = torch.tensor(normalized[None, ...], dtype=torch.float32, device=self._device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy().astype(np.float32)

        return probs

    def _predict_classes(self, model: LSTMSequenceClassifier, X_seq: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(X_seq, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            logits = model(x_tensor)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_from_video(
            self,
            video_path: str | Path,
            top_k: int = DEFAULT_TOP_K,
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        keypoints = self.extract_keypoints_from_video(video_path)
        return self.predict_from_keypoints(keypoints, top_k=top_k)

    @staticmethod
    def _hand_activity_ratio(keypoints: np.ndarray) -> float:
        arr = np.array(keypoints, dtype=np.float32)
        if arr.shape != FEATURE_SHAPE:
            raise ValueError(f"Expected keypoints shape {FEATURE_SHAPE}, got {arr.shape}")
        return float(np.count_nonzero(arr) / arr.size)

    def _ensure_has_hand_activity(self, keypoints: np.ndarray) -> None:
        ratio = self._hand_activity_ratio(keypoints)
        if ratio < MIN_HAND_ACTIVITY_RATIO:
            raise ValueError(
                "Too few hand landmarks detected in video. "
                "Record a clearer gesture with hand fully visible in frame."
            )

    def extract_keypoints_from_video(
            self,
            video_path: str | Path,
            start_frame: int | None = None,
            end_frame: int | None = None,
    ) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        if start_frame is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))

        raw_frames: List[np.ndarray] = []

        with self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=NUM_HANDS,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
        ) as hands:
            while True:
                if end_frame is not None:
                    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if current_frame > int(end_frame):
                        break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                frame_points = np.zeros((NUM_HANDS, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)

                if results.multi_hand_landmarks:
                    for hand_i, hand_landmarks in enumerate(results.multi_hand_landmarks[:NUM_HANDS]):
                        for lm_i, lm in enumerate(hand_landmarks.landmark[:NUM_LANDMARKS]):
                            frame_points[hand_i, lm_i, 0] = lm.x
                            frame_points[hand_i, lm_i, 1] = lm.y
                            frame_points[hand_i, lm_i, 2] = lm.z

                raw_frames.append(frame_points)

        cap.release()

        if not raw_frames:
            return np.zeros(FEATURE_SHAPE, dtype=np.float32)

        sequence = self._resample_frames(raw_frames, SEQUENCE_LENGTH)
        return sequence

    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray, int, int]:
        annotations = self._read_annotations()
        annotations["attachment_id"] = annotations["attachment_id"].astype(str)

        X_list: List[np.ndarray] = []
        y_list: List[str] = []
        total_matched_samples = len(list(self.keypoints_dir.glob("*.npy")))
        skipped_non_letters = 0

        for npy_path in sorted(self.keypoints_dir.glob("*.npy")):
            attachment_id = npy_path.stem

            matches = annotations[annotations["attachment_id"] == attachment_id]
            if matches.empty:
                continue

            label = str(matches.iloc[0]["text"]).strip()
            label = label.upper()

            # Фильтруем только буквы (для Bukva это все записи, но оставим для безопасности)
            if label not in RUSSIAN_LETTERS:
                skipped_non_letters += 1
                continue

            keypoints = np.load(npy_path)
            keypoints = self._coerce_shape(keypoints)
            X_list.append(self._extract_sequence_features(keypoints))
            y_list.append(label)

        if not X_list:
            raise ValueError("No samples found in keypoints directory.")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)

        print(f"📊 Загружено {len(X)} образцов для {len(np.unique(y))} букв")

        return X, y, total_matched_samples, skipped_non_letters

    @staticmethod
    def _augment_sequence_samples(
            X: np.ndarray,
            y: np.ndarray,
            repeats: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if repeats <= 0:
            return X, y

        rng = np.random.default_rng(42)
        augmented_x = [X]
        augmented_y = [y]

        non_zero_mask = X != 0

        for _ in range(repeats):
            noise = rng.normal(loc=0.0, scale=0.01, size=X.shape).astype(np.float32)
            scale = rng.uniform(0.92, 1.08, size=(X.shape[0], 1, 1)).astype(np.float32)

            shifted = X.copy()
            shifts = rng.integers(-4, 5, size=shifted.shape[0])

            for sample_i, temporal_shift in enumerate(shifts):
                shift = int(temporal_shift)
                if shift > 0:
                    pad = np.repeat(shifted[sample_i:sample_i + 1, :1, :], shift, axis=1)
                    shifted[sample_i:sample_i + 1] = np.concatenate(
                        [pad, shifted[sample_i:sample_i + 1, :-shift, :]],
                        axis=1,
                    )
                elif shift < 0:
                    trailing = abs(shift)
                    pad = np.repeat(shifted[sample_i:sample_i + 1, -1:, :], trailing, axis=1)
                    shifted[sample_i:sample_i + 1] = np.concatenate(
                        [shifted[sample_i:sample_i + 1, trailing:, :], pad],
                        axis=1,
                    )

                if float(rng.random()) < 0.5:
                    shifted[sample_i] = SignTranslatorModel._mirror_sequence_features(shifted[sample_i])

            variant = shifted
            variant = variant * scale
            variant = variant + noise * non_zero_mask

            augmented_x.append(variant.astype(np.float32))
            augmented_y.append(y)

        return np.vstack(augmented_x), np.concatenate(augmented_y)

    @staticmethod
    def _mirror_sequence_features(sequence_features: np.ndarray) -> np.ndarray:
        arr = np.array(sequence_features, dtype=np.float32, copy=True)
        if arr.ndim != 2:
            return arr

        hand_dim = NUM_LANDMARKS * NUM_COORDS
        pos_dim = NUM_HANDS * hand_dim
        if arr.shape[1] != pos_dim * 2:
            return arr

        x_idx = np.arange(0, hand_dim, NUM_COORDS)

        pos = arr[:, :pos_dim].copy()
        vel = arr[:, pos_dim:].copy()

        pos_left = pos[:, :hand_dim].copy()
        pos_right = pos[:, hand_dim:].copy()
        vel_left = vel[:, :hand_dim].copy()
        vel_right = vel[:, hand_dim:].copy()

        pos[:, :hand_dim] = pos_right
        pos[:, hand_dim:] = pos_left
        vel[:, :hand_dim] = vel_right
        vel[:, hand_dim:] = vel_left

        pos[:, x_idx] *= -1.0
        pos[:, hand_dim + x_idx] *= -1.0
        vel[:, x_idx] *= -1.0
        vel[:, hand_dim + x_idx] *= -1.0

        return np.concatenate([pos, vel], axis=1).astype(np.float32)

    def _read_annotations(self) -> pd.DataFrame:
        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                return pd.read_csv(self.annotations_path, sep="\t", encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(self.annotations_path, sep="\t")

    def _coerce_shape(self, keypoints: np.ndarray) -> np.ndarray:
        array = np.array(keypoints, dtype=np.float32)

        if array.shape == FEATURE_SHAPE:
            return array

        if array.ndim == 4:
            frames = [array[i] for i in range(array.shape[0])]
            return self._resample_frames(frames, SEQUENCE_LENGTH)

        if array.ndim == 2 and array.shape[1] == NUM_HANDS * NUM_LANDMARKS * NUM_COORDS:
            frames = []
            for row in array:
                frames.append(row.reshape(NUM_HANDS, NUM_LANDMARKS, NUM_COORDS))
            return self._resample_frames(frames, SEQUENCE_LENGTH)

        raise ValueError(f"Unsupported keypoints shape: {array.shape}")

    @staticmethod
    def _resample_frames(frames: List[np.ndarray], target_len: int) -> np.ndarray:
        if len(frames) == 0:
            return np.zeros(FEATURE_SHAPE, dtype=np.float32)

        indices = np.linspace(0, len(frames) - 1, target_len).astype(int)
        sampled = np.stack([frames[i] for i in indices]).astype(np.float32)
        return sampled

    @staticmethod
    def _shift_sequence(keypoints: np.ndarray, shift_frames: int) -> np.ndarray:
        arr = np.array(keypoints, dtype=np.float32, copy=True)
        if arr.shape != FEATURE_SHAPE:
            raise ValueError(f"Expected keypoints shape {FEATURE_SHAPE}, got {arr.shape}")

        shift = int(shift_frames)
        if shift == 0:
            return arr

        if shift > 0:
            pad = np.repeat(arr[:1], shift, axis=0)
            return np.concatenate([pad, arr[:-shift]], axis=0)

        trailing = abs(shift)
        pad = np.repeat(arr[-1:], trailing, axis=0)
        return np.concatenate([arr[trailing:], pad], axis=0)

    @staticmethod
    def _canonicalize_hands_order(frame_points: np.ndarray) -> np.ndarray:
        frame = np.array(frame_points, dtype=np.float32, copy=True)
        if frame.shape != (NUM_HANDS, NUM_LANDMARKS, NUM_COORDS):
            return frame

        hand0 = frame[0]
        hand1 = frame[1]
        if np.count_nonzero(hand0) == 0 or np.count_nonzero(hand1) == 0:
            return frame

        if hand0[0, 0] > hand1[0, 0]:
            return np.stack([hand1, hand0], axis=0)
        return frame

    def _normalize_sequence(self, keypoints: np.ndarray) -> np.ndarray:
        arr = np.array(keypoints, dtype=np.float32, copy=True)
        if arr.shape != FEATURE_SHAPE:
            raise ValueError(f"Expected keypoints shape {FEATURE_SHAPE}, got {arr.shape}")

        for frame_i in range(SEQUENCE_LENGTH):
            arr[frame_i] = self._canonicalize_hands_order(arr[frame_i])

            for hand_i in range(NUM_HANDS):
                hand = arr[frame_i, hand_i]
                if np.count_nonzero(hand) == 0:
                    continue

                wrist = hand[0].copy()
                hand -= wrist

                scale = np.linalg.norm(hand[9] - hand[0])
                if scale < 1e-4:
                    scale = np.linalg.norm(hand[5] - hand[0])
                if scale > 1e-4:
                    hand /= scale

                arr[frame_i, hand_i] = hand

        return arr

    def _extract_sequence_features(self, keypoints: np.ndarray) -> np.ndarray:
        normalized = self._normalize_sequence(keypoints)

        velocity = np.diff(normalized, axis=0, prepend=normalized[:1])
        seq = normalized.reshape(normalized.shape[0], -1)
        vel_seq = velocity.reshape(velocity.shape[0], -1)
        return np.concatenate([seq, vel_seq], axis=1).astype(np.float32)

    def _ensure_loaded(self) -> None:
        if (
                self.model is None
                or self.label_encoder is None
                or self._feature_mean is None
                or self._feature_std is None
        ):
            if not self.load_if_exists():
                raise ValueError(
                    "Model is not trained yet. Call /api/train first or run train script."
                )


def train_and_save_default() -> TrainResult:
    translator = SignTranslatorModel()
    return translator.train()


if __name__ == "__main__":
    result = train_and_save_default()
    print(
        f"\n{'=' * 50}"
        f"\n✅ Training completed!"
        f"\n   Samples: {result.samples}"
        f"\n   Classes: {result.classes}"
        f"\n   Train accuracy: {result.train_accuracy:.3f}"
        f"\n   Test accuracy: {result.test_accuracy:.3f}"
        f"\n{'=' * 50}"
    )