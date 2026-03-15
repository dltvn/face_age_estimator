from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

try:
    from mtcnn.mtcnn import MTCNN  # TensorFlow/Keras-based implementation
except Exception as e:
    raise ImportError("Install mtcnn (TF/Keras): pip install mtcnn tensorflow") from e

logger = logging.getLogger(__name__)

# Landmarks array shape: (5, 2)
# Order: left_eye, right_eye, nose, mouth_left, mouth_right
Landmarks5 = np.ndarray

Landmarks5 = np.ndarray  # shape (5, 2), dtype float32

@dataclass
class DetectedFace:
    """
    Represents one detected face.

    box:
        Bounding box in [x1, y1, x2, y2] format.
    prob:
        Detection confidence (0 to 1).
    landmarks:
        The 5 facial keypoints returned by MTCNN.
    """

    box: np.ndarray
    prob: float
    landmarks: Landmarks5


# -------------------------------------------------------------
# This defines where we WANT the 5 facial landmarks to end up
# after alignment.
#
# These coordinates come from ArcFace.
# Think of this as the "ideal face layout".
# -------------------------------------------------------------
def _template_5pts(output_size: tuple[int, int]) -> np.ndarray:
    ref = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041],  # right mouth
        ],
        dtype=np.float32,
    )

    # Scale template to match output image size
    w, h = output_size
    template = _LANDMARK_TEMPLATE.copy()
    template[:, 0] *= w / 112.0
    template[:, 1] *= h / 112.0
    return template


# -------------------------------------------------------------
# Using the detected 5 landmarks, compute an affine transform
# that maps them to the canonical template.
#
# This transform handles:
#   - rotation
#   - scaling
#   - translation
#
# It does NOT change pose (no 3D correction).
# -------------------------------------------------------------
def estimate_affine_from_5pts(
    src_landmarks: Landmarks5, output_size: tuple[int, int]
) -> np.ndarray:
    if src_landmarks.shape != (5, 2):
        raise ValueError(f"Expected (5,2) landmarks, got {src_landmarks.shape}")

    dst = _template_5pts(output_size=output_size)

    M, _ = cv2.estimateAffinePartial2D(
        src_landmarks.astype(np.float32), dst, method=cv2.LMEDS,
    )

    if M is None:
        raise RuntimeError("Could not estimate affine transform from landmarks.")

    return M.astype(np.float32)


# -------------------------------------------------------------
# Makes sure the bounding box does not go outside the image.
# Prevents slicing errors and weird crops.
# -------------------------------------------------------------
def _clip_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


# -------------------------------------------------------------
# Instead of filling empty areas with black (ugly),
# we use the average color of the crop.
# This makes the padding look natural.
# -------------------------------------------------------------
def _mean_border_value(image_bgr: np.ndarray) -> tuple[int, int, int]:
    mean = image_bgr.reshape(-1, 3).mean(axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


# -------------------------------------------------------------
# Applies the affine transform to actually align the face.
#
# The result is always a fixed-size image (e.g., 224x224).
# -------------------------------------------------------------
def warp_face(
    image_bgr: np.ndarray, M_2x3: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
    w, h = output_size
    border_value = _mean_border_value(image_bgr)

    return cv2.warpAffine(
        image_bgr,
        M_2x3,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

# Default model path — can be overridden via env var or constructor arg
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.expanduser("~"), ".mediapipe", "face_landmarker.task"
)

# -------------------------------------------------------------
# Creates a square crop centered on the face.
#
# Why square?
# Because alignment works best when the face region
# is roughly symmetric and centered.
#
# Why margin?
# Because without margin, parts of the face can get
# cut off after rotation.
# -------------------------------------------------------------
def _square_crop_with_margin(
    image_bgr: np.ndarray, box_xyxy: np.ndarray, margin: float
):
    h, w = image_bgr.shape[:2]
    box = _clip_xyxy(box_xyxy, width=w, height=h)
    x1, y1, x2, y2 = box

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    # Make the crop square using the larger dimension
    side = max(bw, bh) * (1.0 + margin * 2.0)

    # Center of the face
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    cx1 = int(round(cx - side / 2.0))
    cy1 = int(round(cy - side / 2.0))
    cx2 = int(round(cx + side / 2.0))
    cy2 = int(round(cy + side / 2.0))

    # Keep crop inside image
    cx1 = max(0, cx1)
    cy1 = max(0, cy1)
    cx2 = min(w, cx2)
    cy2 = min(h, cy2)

    crop = image_bgr[cy1:cy2, cx1:cx2].copy()

    offset_xy = np.array([cx1, cy1], dtype=np.float32)
    crop_box = np.array([cx1, cy1, cx2, cy2], dtype=np.int32)

    return crop, offset_xy, crop_box


class FaceDetectorMTCNN:
    """
    Face detector + aligner using MTCNN.

    Pipeline:
      1. Detect face + 5 keypoints
      2. Pick best or largest face
      3. Square crop with margin
      4. Shift landmarks into crop space
      5. Estimate affine transform
      6. Warp to fixed size
    """

    def __init__(self) -> None:
        self.detector = MTCNN()

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """
        Runs MTCNN detection and returns a list of faces.
        """
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        dets = self.detector.detect_faces(rgb) or []

        faces = []

        for d in dets:
            conf = float(d.get("confidence", 0.0))
            x, y, w_box, h_box = d["box"]

            x1, y1 = float(x), float(y)
            x2, y2 = float(x + w_box), float(y + h_box)

            kp = d.get("keypoints", {})
            if not kp:
                continue

            lm = np.array(
                [
                    kp["left_eye"],
                    kp["right_eye"],
                    kp["nose"],
                    kp["mouth_left"],
                    kp["mouth_right"],
                ],
                dtype=np.float32,
            )

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
        output_size: tuple[int, int] = (224, 224),
        min_prob: float = 0.85,
        pick: str = "largest",
        margin: float = 0.50,
    ) -> dict[str, Any]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected a 3-channel BGR image, got shape {image_bgr.shape}"
            )

        faces = self.detect(image_bgr)
        if not faces:
            return {"faces": [], "selected": None}

        filtered = [f for f in faces if f.prob >= min_prob]
        if not filtered:
            logger.warning(
                "No detections met min_prob=%.2f; falling back to all %d detection(s).",
                min_prob,
                len(faces),
            )
        faces_f = filtered or faces

        # Choose which face to align
        if pick == "best":
            selected = faces_f[0]
        else:
            selected = max(
                faces_f, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1])
            )

        # Step 1: Square crop around face
        crop_bgr, offset_xy, crop_box = _square_crop_with_margin(
            image_bgr, selected.box, margin
        )

        # Step 2: Convert landmarks into crop coordinate system
        lm_crop = selected.landmarks.astype(np.float32) - offset_xy

        # Step 3: Compute transform that moves face into template layout
        M = estimate_affine_from_5pts(lm_crop, output_size)

        # Step 4: Warp image using that transform
        aligned = warp_face(crop_bgr, M, output_size)

        return {
            "faces": faces,
            "selected": selected,
            "affine": M,
            "aligned_bgr": aligned,
            "crop_box": crop_box,
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