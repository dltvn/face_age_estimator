from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.core.face_detection_retinaface import detect_and_align

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_LOCAL_DIR = PROJECT_ROOT / ".cache" / "hf_models"
HF_MODEL_REPO_ID = "d-ltvn/chronolens-models"
DEFAULT_IMAGE_SIZE = (224, 224)
AGE_BINS = np.arange(117, dtype=np.float32)
RACE_LABELS = {
    0: "white",
    1: "black",
    2: "asian",
    3: "indian",
    4: "others",
}


@dataclass
class ModelPaths:
    """Container for all inference model file paths."""

    gender: Path
    age_agnostic: Path
    age_male: Path
    age_female: Path
    race: Path

    @classmethod
    def from_root(cls, root: Path) -> "ModelPaths":
        """Build model paths from a shared models root directory."""
        return cls(
            gender=root / "gender_classifier" / "best_model.keras",
            age_agnostic=(root / "baseline_age_model_improved" / "best_baseline_age_model_improved.keras"),
            age_male=root / "male_age_model" / "best_male_age_model.keras",
            age_female=root / "female_age_model" / "best_female_age_model.keras",
            race=root / "race_classifier" / "best_race_model.keras",
        )


@dataclass
class GenderPrediction:
    """Gender prediction result."""

    gender: str
    confidence: float
    prob_female: float
    prob_male: float


@dataclass
class AgePrediction:
    """Age distribution prediction result."""

    distribution: list[float]
    predicted_age: int
    confidence: float


@dataclass
class RacePrediction:
    """Race prediction result."""

    race: str
    confidence: float
    probabilities: dict[str, float]


@dataclass
class PreprocessResult:
    """Face preprocessing output for downstream inference."""

    aligned_bgr: np.ndarray
    info: dict[str, Any]


class InferenceService:
    """Inference service for preprocessing + model predictions."""

    def __init__(self, paths: ModelPaths | None = None) -> None:
        self.paths = paths or self._resolve_model_paths()
        self._validate_paths()
        self.gender_model = tf.keras.models.load_model(str(self.paths.gender), compile=False)
        self.age_agnostic_model = tf.keras.models.load_model(str(self.paths.age_agnostic), compile=False)
        self.age_male_model = tf.keras.models.load_model(str(self.paths.age_male), compile=False)
        self.age_female_model = tf.keras.models.load_model(str(self.paths.age_female), compile=False)
        self.race_model = (
            tf.keras.models.load_model(str(self.paths.race), compile=False)
            if self.paths.race.exists()
            else None
        )
        if self.race_model is None:
            logger.warning(
                "Race model file not found at %s; race endpoint will be unavailable.",
                self.paths.race,
            )
        self._warmup_models()
        logger.info("Models loaded successfully.")

    @staticmethod
    def _missing_paths(paths: ModelPaths) -> list[str]:
        """Return missing file paths from a ModelPaths container."""
        return [
            str(path)
            for path in (
                paths.gender,
                paths.age_agnostic,
                paths.age_male,
                paths.age_female,
            )
            if not path.exists()
        ]

    @staticmethod
    def _download_models_snapshot(repo_id: str) -> Path:
        """Download required model files from Hugging Face Hub."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "huggingface_hub is required for remote model download."
            ) from exc

        local_dir = Path(DEFAULT_HF_LOCAL_DIR)
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "gender_classifier/best_model.keras",
                "baseline_age_model_improved/best_baseline_age_model_improved.keras",
                "male_age_model/best_male_age_model.keras",
                "female_age_model/best_female_age_model.keras",
                "race_classifier/best_race_model.keras",
            ],
        )
        return Path(snapshot_dir)

    def _resolve_model_paths(self) -> ModelPaths:
        """Resolve model paths by downloading from Hugging Face Hub."""

        logger.info("Downloading model files from HF repo '%s'.", HF_MODEL_REPO_ID)
        snapshot_root = self._download_models_snapshot(repo_id=HF_MODEL_REPO_ID)
        hub_paths = ModelPaths.from_root(snapshot_root)
        missing_hub = self._missing_paths(hub_paths)
        if missing_hub:
            raise FileNotFoundError(
                f"Missing model file(s) after HF download from {HF_MODEL_REPO_ID}: {missing_hub}"
            )
        logger.info("Using downloaded model files from %s", snapshot_root)
        return hub_paths

    def _validate_paths(self) -> None:
        missing = self._missing_paths(self.paths)
        if missing:
            raise FileNotFoundError(f"Missing model file(s): {missing}")

    @staticmethod
    def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
        """Decode image bytes into a BGR numpy array."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("Could not decode image.")
        return image_bgr

    @staticmethod
    def encode_jpeg(image_bgr: np.ndarray, quality: int = 95) -> bytes:
        """Encode BGR image to JPEG bytes."""
        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Could not encode image to JPEG.")
        return buf.tobytes()

    @staticmethod
    def preprocess_face(image_bgr: np.ndarray) -> PreprocessResult:
        """Detect and align one face from input image."""
        aligned_bgr, info = detect_and_align(
            image_bgr=image_bgr,
            output_size=DEFAULT_IMAGE_SIZE,
            min_score=0.85,
            pick="largest",
            margin=0.50,
            return_info=True,
        )
        if aligned_bgr is None:
            raise ValueError("No face detected.")
        return PreprocessResult(aligned_bgr=aligned_bgr, info=info)

    @staticmethod
    def _to_model_tensor(aligned_bgr: np.ndarray) -> np.ndarray:
        """Convert aligned BGR image to normalized ResNet50 model tensor."""
        image_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if image_rgb.shape[:2] != DEFAULT_IMAGE_SIZE:
            # Keep model input shape stable to reduce retracing overhead.
            image_rgb = cv2.resize(image_rgb, DEFAULT_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
        image_rgb = preprocess_input(image_rgb)
        return np.expand_dims(image_rgb, axis=0)

    @staticmethod
    def _run_model(model: tf.keras.Model, tensor: np.ndarray) -> np.ndarray:
        """Run one forward pass using direct model call for lower overhead."""
        outputs = model(tf.convert_to_tensor(tensor, dtype=tf.float32), training=False)
        return np.asarray(outputs)

    def _warmup_models(self) -> None:
        """Run one dry forward pass per model to avoid first-request latency."""
        dummy = np.zeros((1, DEFAULT_IMAGE_SIZE[0], DEFAULT_IMAGE_SIZE[1], 3), dtype=np.float32)
        for model in (
            self.gender_model,
            self.age_agnostic_model,
            self.age_male_model,
            self.age_female_model,
            self.race_model,
        ):
            if model is not None:
                _ = self._run_model(model, dummy)

    def predict_gender(self, aligned_bgr: np.ndarray) -> GenderPrediction:
        """Predict gender from aligned face image."""
        tensor = self._to_model_tensor(aligned_bgr)
        pred = self._run_model(self.gender_model, tensor).squeeze()
        prob_female = float(np.clip(pred, 0.0, 1.0))
        prob_male = float(1.0 - prob_female)
        gender = "female" if prob_female >= 0.5 else "male"
        confidence = float(max(prob_female, prob_male))
        return GenderPrediction(
            gender=gender,
            confidence=confidence,
            prob_female=prob_female,
            prob_male=prob_male,
        )

    @staticmethod
    def _decode_age_distribution(dist: np.ndarray) -> AgePrediction:
        """Decode age prediction outputs from a 117-bin distribution."""
        distribution = dist.astype(np.float32).reshape(-1)
        if distribution.shape[0] != AGE_BINS.shape[0]:
            raise ValueError(
                f"Expected {AGE_BINS.shape[0]} age bins, got {distribution.shape[0]}"
            )
        distribution = distribution / np.clip(distribution.sum(), 1e-8, None)
        predicted_age = int(np.argmax(distribution))
        confidence = float(np.max(distribution))
        return AgePrediction(
            distribution=distribution.tolist(),
            predicted_age=predicted_age,
            confidence=confidence,
        )

    def predict_age_agnostic(self, aligned_bgr: np.ndarray) -> AgePrediction:
        """Predict age distribution with the gender-agnostic age model."""
        tensor = self._to_model_tensor(aligned_bgr)
        dist = self._run_model(self.age_agnostic_model, tensor)
        return self._decode_age_distribution(dist)

    def predict_age_gender_specific(
        self,
        aligned_bgr: np.ndarray,
        gender: str,
    ) -> AgePrediction:
        """Predict age distribution using male/female specific model."""
        tensor = self._to_model_tensor(aligned_bgr)
        if gender == "female":
            dist = self._run_model(self.age_female_model, tensor)
        else:
            dist = self._run_model(self.age_male_model, tensor)
        return self._decode_age_distribution(dist)

    def predict_race(self, aligned_bgr: np.ndarray) -> RacePrediction:
        """Predict race from aligned face image."""
        if self.race_model is None:
            raise FileNotFoundError(
                f"Race model file not found at {self.paths.race}. "
                "Add race_classifier/best_race_model.keras to the model repo."
            )

        tensor = self._to_model_tensor(aligned_bgr)
        pred = self._run_model(self.race_model, tensor).reshape(-1)
        if pred.shape[0] != len(RACE_LABELS):
            raise ValueError(
                f"Expected {len(RACE_LABELS)} race logits, got {pred.shape[0]}"
            )

        probs = pred.astype(np.float32)
        probs = probs / np.clip(probs.sum(), 1e-8, None)
        pred_idx = int(np.argmax(probs))
        race = RACE_LABELS[pred_idx]
        confidence = float(np.max(probs))
        probabilities = {
            RACE_LABELS[idx]: float(probs[idx])
            for idx in range(len(RACE_LABELS))
        }
        return RacePrediction(
            race=race,
            confidence=confidence,
            probabilities=probabilities,
        )


@lru_cache(maxsize=1)
def get_inference_service() -> InferenceService:
    """Return a cached singleton inference service."""
    return InferenceService()
