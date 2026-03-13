from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Landmarks5 = np.ndarray  # shape (5, 2), dtype float32

_CANONICAL_SIZE: tuple[int, int] = (112, 112)

_LANDMARK_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],  # left_eye
        [73.5318, 51.5014],  # right_eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # mouth_left
        [70.7299, 92.2041],  # mouth_right
    ],
    dtype=np.float32,
)

# MediaPipe Face Mesh indices for the 5 canonical landmarks
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
_MP_LEFT_EYE_IDX   = 468  # refined left eye center (requires refine_landmarks=True)
_MP_RIGHT_EYE_IDX  = 473  # refined right eye center
_MP_NOSE_IDX       = 1    # nose tip
_MP_MOUTH_L_IDX    = 61   # left mouth corner
_MP_MOUTH_R_IDX    = 291  # right mouth corner


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class DetectedFace:
    box: np.ndarray        # (4,) [x1, y1, x2, y2]
    prob: float
    landmarks: Landmarks5  # (5, 2)


# ---------------------------------------------------------------------------
# Alignment helpers
# ---------------------------------------------------------------------------

def _get_landmark_template(output_size: tuple[int, int] = _CANONICAL_SIZE) -> np.ndarray:
    w, h = output_size
    template = _LANDMARK_TEMPLATE.copy()
    template[:, 0] *= w / 112.0
    template[:, 1] *= h / 112.0
    return template


def estimate_affine_from_5pts(src_landmarks: Landmarks5) -> np.ndarray:
    if src_landmarks.shape != (5, 2):
        raise ValueError(f"Expected landmarks shape (5, 2), got {src_landmarks.shape}")
    dst = _get_landmark_template(_CANONICAL_SIZE)
    M, _ = cv2.estimateAffinePartial2D(
        src_landmarks.astype(np.float32), dst, method=cv2.LMEDS,
    )
    if M is None:
        raise RuntimeError("Affine estimation failed — degenerate landmark positions.")
    return M.astype(np.float32)


def warp_face(
    image: np.ndarray,
    M_2x3: np.ndarray,
    output_size: tuple[int, int] = _CANONICAL_SIZE,
) -> np.ndarray:
    w, h = output_size
    return cv2.warpAffine(
        image, M_2x3, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------------------------------------------------------------------------
# MediaPipe detector (primary — Python 3.13 compatible, precise landmarks)
# ---------------------------------------------------------------------------

# Default model path — can be overridden via env var or constructor arg
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".mediapipe", "face_landmarker.task"
)


class FaceDetectorMediaPipe:
    """MediaPipe Face Landmarker detector.

    Gives precise 478-point landmarks (including refined eye centers).
    Works on Python 3.13, no TensorFlow required.

    Setup
    -----
    1. pip install mediapipe
    2. Download the model (~30MB):
       https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
       Save to: ~/.mediapipe/face_landmarker.task
       (or pass model_path= to the constructor)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        min_detection_confidence: float = 0.3,
        min_presence_confidence: float = 0.3,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
        except ImportError:
            raise ImportError("pip install mediapipe")

        path = model_path or os.environ.get("MP_FACE_LANDMARKER_MODEL", _DEFAULT_MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MediaPipe model not found at: {path}\n"
                "Download from:\n"
                "  https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/1/face_landmarker.task\n"
                f"Save to: {path}"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp
        logger.info("FaceDetectorMediaPipe ready | model=%s", path)

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        import mediapipe as mp

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return []

        H, W = image_bgr.shape[:2]
        faces = []

        for face_lm in result.face_landmarks:
            lm = face_lm  # list of NormalizedLandmark

            # Extract 5 key landmarks (denormalize to pixel coords)
            def px(idx):
                return np.array([lm[idx].x * W, lm[idx].y * H], dtype=np.float32)

            left_eye  = px(_MP_LEFT_EYE_IDX)
            right_eye = px(_MP_RIGHT_EYE_IDX)
            nose      = px(_MP_NOSE_IDX)
            mouth_l   = px(_MP_MOUTH_L_IDX)
            mouth_r   = px(_MP_MOUTH_R_IDX)

            landmarks = np.array([left_eye, right_eye, nose, mouth_l, mouth_r],
                                 dtype=np.float32)

            # Derive bounding box from all landmarks
            xs = np.array([l.x * W for l in lm])
            ys = np.array([l.y * H for l in lm])
            x1, y1 = max(0, xs.min()), max(0, ys.min())
            x2, y2 = min(W, xs.max()), min(H, ys.max())
            box = np.array([x1, y1, x2, y2], dtype=np.float32)

            # Use box area fraction as confidence proxy
            area_ratio = ((x2 - x1) * (y2 - y1)) / (W * H)
            prob = min(0.99, area_ratio * 3)

            faces.append(DetectedFace(box=box, prob=prob, landmarks=landmarks))

        faces.sort(key=lambda f: f.prob, reverse=True)
        return faces

    def detect_and_align(
        self,
        image_bgr: np.ndarray,
        min_prob: float = 0.0,
        pick: str = "best",
        margin: float = 0.35,
        return_resized: Optional[tuple[int, int]] = None,
    ) -> dict[str, Any]:
        faces = self.detect(image_bgr)
        if not faces:
            return {"faces": [], "selected": None}

        if pick == "largest":
            def _area(f: DetectedFace) -> float:
                x1, y1, x2, y2 = f.box
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)
            selected = max(faces, key=_area)
        else:
            selected = faces[0]

        H, W = image_bgr.shape[:2]
        x1, y1, x2, y2 = selected.box.astype(int)
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad_x, pad_y = int(bw * margin), int(bh * margin)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(W, x2 + pad_x)
        cy2 = min(H, y2 + pad_y)

        crop = image_bgr[cy1:cy2, cx1:cx2].copy()
        lm = selected.landmarks.copy()
        lm[:, 0] -= cx1
        lm[:, 1] -= cy1

        M = estimate_affine_from_5pts(lm)
        aligned = warp_face(crop, M, output_size=_CANONICAL_SIZE)

        result: dict[str, Any] = {
            "faces": faces,
            "selected": selected,
            "affine": M,
            "aligned_bgr": aligned,
            "crop_box": np.array([cx1, cy1, cx2, cy2], dtype=np.int32),
        }
        if return_resized is not None:
            rw, rh = return_resized
            result["aligned_bgr_resized"] = cv2.resize(
                aligned, (rw, rh), interpolation=cv2.INTER_CUBIC
            )
        return result


# ---------------------------------------------------------------------------
# Haar Cascade detector (fallback — no model file needed)
# ---------------------------------------------------------------------------

def _estimate_landmarks_from_box(x1, y1, x2, y2, gray_crop):
    bw, bh = x2 - x1, y2 - y1
    left_eye  = np.array([x1 + bw * 0.30, y1 + bh * 0.37], dtype=np.float32)
    right_eye = np.array([x1 + bw * 0.70, y1 + bh * 0.37], dtype=np.float32)
    nose      = np.array([x1 + bw * 0.50, y1 + bh * 0.60], dtype=np.float32)
    mouth_l   = np.array([x1 + bw * 0.35, y1 + bh * 0.80], dtype=np.float32)
    mouth_r   = np.array([x1 + bw * 0.65, y1 + bh * 0.80], dtype=np.float32)

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    upper_half = gray_crop[: gray_crop.shape[0] // 2, :]
    eyes = eye_cascade.detectMultiScale(upper_half, scaleFactor=1.1, minNeighbors=3)
    if len(eyes) >= 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        ex1, ey1, ew1, eh1 = eyes[0]
        ex2, ey2, ew2, eh2 = eyes[1]
        left_eye  = np.array([x1 + ex1 + ew1/2, y1 + ey1 + eh1/2], dtype=np.float32)
        right_eye = np.array([x1 + ex2 + ew2/2, y1 + ey2 + eh2/2], dtype=np.float32)
        eye_cy = (left_eye[1] + right_eye[1]) / 2
        nose    = np.array([x1 + bw * 0.50, eye_cy + bh * 0.23], dtype=np.float32)
        mouth_l = np.array([x1 + bw * 0.35, eye_cy + bh * 0.43], dtype=np.float32)
        mouth_r = np.array([x1 + bw * 0.65, eye_cy + bh * 0.43], dtype=np.float32)

    return np.array([left_eye, right_eye, nose, mouth_l, mouth_r], dtype=np.float32)


class FaceDetectorHaar:
    """OpenCV Haar Cascade fallback. No model download needed."""

    def __init__(self, scale_factor=1.1, min_neighbors=3, min_face_ratio=0.15):
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        )
        self._scale_factor  = scale_factor
        self._min_neighbors = min_neighbors
        self._min_face_ratio = min_face_ratio

    def detect(self, image_bgr):
        gray = cv2.equalizeHist(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
        min_size = max(20, int(min(image_bgr.shape[:2]) * self._min_face_ratio))
        dets = self._face_cascade.detectMultiScale(
            gray, scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors, minSize=(min_size, min_size),
        )
        faces = []
        if len(dets) == 0:
            return faces
        for (x, y, w, h) in dets:
            x1, y1, x2, y2 = x, y, x+w, y+h
            lm = _estimate_landmarks_from_box(x1, y1, x2, y2, gray[y1:y2, x1:x2])
            area_ratio = (w * h) / (image_bgr.shape[0] * image_bgr.shape[1])
            faces.append(DetectedFace(
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                prob=min(0.99, area_ratio * 4),
                landmarks=lm,
            ))
        faces.sort(key=lambda f: f.prob, reverse=True)
        return faces

    def detect_and_align(self, image_bgr, min_prob=0.0, pick="best",
                         margin=0.35, return_resized=None):
        faces = self.detect(image_bgr)
        if not faces:
            return {"faces": [], "selected": None}
        selected = max(faces, key=lambda f: (f.box[2]-f.box[0])*(f.box[3]-f.box[1])) \
                   if pick == "largest" else faces[0]
        H, W = image_bgr.shape[:2]
        x1, y1, x2, y2 = selected.box.astype(int)
        bw, bh = max(1, x2-x1), max(1, y2-y1)
        pad_x, pad_y = int(bw*margin), int(bh*margin)
        cx1, cy1 = max(0, x1-pad_x), max(0, y1-pad_y)
        cx2, cy2 = min(W, x2+pad_x), min(H, y2+pad_y)
        crop = image_bgr[cy1:cy2, cx1:cx2].copy()
        lm = selected.landmarks.copy()
        lm[:, 0] -= cx1; lm[:, 1] -= cy1
        M = estimate_affine_from_5pts(lm)
        aligned = warp_face(crop, M, output_size=_CANONICAL_SIZE)
        result = {"faces": faces, "selected": selected, "affine": M,
                  "aligned_bgr": aligned,
                  "crop_box": np.array([cx1, cy1, cx2, cy2], dtype=np.int32)}
        if return_resized:
            result["aligned_bgr_resized"] = cv2.resize(
                aligned, return_resized, interpolation=cv2.INTER_CUBIC)
        return result


# ---------------------------------------------------------------------------
# Smart factory — uses MediaPipe if model exists, falls back to Haar
# ---------------------------------------------------------------------------

def create_detector(model_path: Optional[str] = None) -> Any:
    """Return the best available detector.

    Uses MediaPipe if face_landmarker.task is available, else Haar Cascade.
    """
    path = model_path or os.environ.get("MP_FACE_LANDMARKER_MODEL", _DEFAULT_MODEL_PATH)
    if os.path.exists(path):
        logger.info("Using MediaPipe detector")
        return FaceDetectorMediaPipe(model_path=path)
    logger.warning("MediaPipe model not found — falling back to Haar Cascade. "
                   "Download face_landmarker.task for better accuracy.")
    return FaceDetectorHaar(scale_factor=1.05, min_neighbors=3)


# Aliases
FaceDetector = create_detector
FaceDetectorMTCNN = FaceDetectorHaar  # legacy alias
FaceDetectorRetinaFace = FaceDetectorHaar  # legacy alias