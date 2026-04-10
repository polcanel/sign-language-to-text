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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEQUENCE_LENGTH = 155
NUM_HANDS = 2
NUM_LANDMARKS = 21
NUM_COORDS = 3
FEATURE_SHAPE = (SEQUENCE_LENGTH, NUM_HANDS, NUM_LANDMARKS, NUM_COORDS)
MIN_HAND_ACTIVITY_RATIO = 0.01
AUGMENT_REPEATS = 4


@dataclass
class TrainResult:
    samples: int
    classes: int
    train_accuracy: float
    test_accuracy: float
    total_matched_samples: int
    skipped_non_letters: int


class SignTranslatorModel:
    def __init__(
        self,
        annotations_path: str | Path = "annotations.csv",
        keypoints_dir: str | Path = "slovo_keypoints",
        model_path: str | Path = "models/model.joblib",
        encoder_path: str | Path = "models/label_encoder.joblib",
    ) -> None:
        self.annotations_path = Path(annotations_path)
        self.keypoints_dir = Path(keypoints_dir)
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path)

        self.model: Pipeline | None = None
        self.label_encoder: LabelEncoder | None = None

        self._mp_hands = mp.solutions.hands

    def load_if_exists(self) -> bool:
        if not self.model_path.exists() or not self.encoder_path.exists():
            return False

        self.model = joblib.load(self.model_path)
        self.label_encoder = joblib.load(self.encoder_path)
        return True

    def train(self) -> TrainResult:
        X, y_text, total_matched_samples, skipped_non_letters = self._load_dataset()

        if len(X) < 10:
            raise ValueError("Too few samples found for training.")

        X_train_raw, X_test, y_train_text, y_test_text = train_test_split(
            X,
            y_text,
            test_size=0.2,
            random_state=42,
            stratify=y_text,
        )

        X_train, y_train_text = self._augment_flat_samples(
            X_train_raw,
            y_train_text,
            repeats=AUGMENT_REPEATS,
        )

        label_encoder = LabelEncoder()
        label_encoder.fit(y_text)
        y_train = label_encoder.transform(y_train_text)
        y_test = label_encoder.transform(y_test_text)

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    Pipeline(
                        steps=[
                            (
                                "pca",
                                PCA(n_components=0.98, svd_solver="full", random_state=42),
                            ),
                            (
                                "lr",
                                LogisticRegression(
                                    max_iter=1500,
                                    class_weight="balanced",
                                    solver="lbfgs",
                                ),
                            ),
                        ]
                    ),
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        self.model = pipeline
        self.label_encoder = label_encoder

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.label_encoder, self.encoder_path)

        return TrainResult(
            samples=len(X),
            classes=len(label_encoder.classes_),
            train_accuracy=float(train_accuracy),
            test_accuracy=float(test_accuracy),
            total_matched_samples=total_matched_samples,
            skipped_non_letters=skipped_non_letters,
        )

    def predict_from_keypoints(self, keypoints: np.ndarray) -> Tuple[str, float]:
        self._ensure_loaded()
        self._ensure_has_hand_activity(keypoints)
        flat = self._flatten_keypoints(keypoints)

        probabilities = self.model.predict_proba(flat)[0]
        class_idx = int(np.argmax(probabilities))
        label = self.label_encoder.inverse_transform([class_idx])[0]
        confidence = float(probabilities[class_idx])
        return label, confidence

    def predict_from_video(self, video_path: str | Path) -> Tuple[str, float]:
        keypoints = self.extract_keypoints_from_video(video_path)
        return self.predict_from_keypoints(keypoints)

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
            
            keypoints = np.load(npy_path)

            # Align shape if needed.
            keypoints = self._coerce_shape(keypoints)
            X_list.append(self._flatten_keypoints(keypoints)[0])
            y_list.append(label)

        if not X_list:
            raise ValueError("No samples found in keypoints directory.")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        return X, y, total_matched_samples, skipped_non_letters

    @staticmethod
    def _augment_flat_samples(
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
            noise = rng.normal(loc=0.0, scale=0.008, size=X.shape).astype(np.float32)
            scale = rng.uniform(0.92, 1.08, size=(X.shape[0], 1)).astype(np.float32)

            variant = X.copy()
            variant = variant * scale
            variant = variant + noise * non_zero_mask

            augmented_x.append(variant.astype(np.float32))
            augmented_y.append(y)

        return np.vstack(augmented_x), np.concatenate(augmented_y)

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
    def _flatten_keypoints(keypoints: np.ndarray) -> np.ndarray:
        arr = np.array(keypoints, dtype=np.float32)
        if arr.shape != FEATURE_SHAPE:
            raise ValueError(f"Expected keypoints shape {FEATURE_SHAPE}, got {arr.shape}")
        return arr.reshape(1, -1)

    def _ensure_loaded(self) -> None:
        if self.model is None or self.label_encoder is None:
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
        f"Training completed: samples={result.samples}, "
        f"classes={result.classes}, "
        f"train_acc={result.train_accuracy:.3f}, "
        f"test_acc={result.test_accuracy:.3f}"
    )
