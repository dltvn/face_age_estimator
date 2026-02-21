from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

try:
    from mtcnn.mtcnn import MTCNN  # TensorFlow/Keras-based
except Exception as e:
    raise ImportError("Install mtcnn (TF/Keras): pip install mtcnn tensorflow") from e

logger = logging.getLogger(__name__)

Landmarks5 = np.ndarray  # shape (5, 2)

# ArcFace / InsightFace canonical size (template defined in this space)
_CANONICAL_SIZE: tuple[int, int] = (112, 112)


@dataclass
class DetectedFace:
    """A single face detection result.

    Attributes:
        box: Bounding box as [x1, y1, x2, y2] pixel coordinates (float32, shape (4,)).
        prob: Detection confidence score in [0, 1].
        landmarks: Five facial keypoints as (x, y) pairs (float32, shape (5, 2)).
            Order: left_eye, right_eye, nose, mouth_left, mouth_right.
    """

    box: np.ndarray  # (4,) => [x1, y1, x2, y2]
    prob: float
    landmarks: Landmarks5  # (5, 2)


def _template_5pts(output_size: tuple[int, int] = _CANONICAL_SIZE) -> np.ndarray:
    """Return the canonical 5-point facial landmark template scaled to output_size."""
    ref = np.array(
        [
            [38.2946, 51.6963],  # left eye centre
            [73.5318, 51.5014],  # right eye centre
            [56.0252, 71.7366],  # nose tip
            [41.5493, 92.3655],  # left mouth corner
            [70.7299, 92.2041],  # right mouth corner
        ],
        dtype=np.float32,
    )

    w, h = output_size
    ref[:, 0] *= w / 112.0
    ref[:, 1] *= h / 112.0
    return ref


def estimate_affine_from_5pts(src_landmarks: Landmarks5) -> np.ndarray:
    """Estimate 2x3 affine transform mapping detected landmarks to canonical template (112x112)."""
    if src_landmarks.shape != (5, 2):
        raise ValueError(f"Expected (5,2) landmarks, got {src_landmarks.shape}")

    dst = _template_5pts(output_size=_CANONICAL_SIZE)
    M, _ = cv2.estimateAffinePartial2D(
        src_landmarks.astype(np.float32),
        dst.astype(np.float32),
        method=cv2.LMEDS,
    )
    if M is None:
        raise RuntimeError("Could not estimate affine transform from landmarks.")
    return M.astype(np.float32)


def warp_face(
    image: np.ndarray,
    M_2x3: np.ndarray,
    output_size: tuple[int, int] = _CANONICAL_SIZE,
) -> np.ndarray:
    """Apply a 2x3 affine transform to produce an aligned face crop.

    - Cubic interpolation for better detail.
    - Constant borders to avoid reflection/mirroring artifacts.
    """
    w, h = output_size
    return cv2.warpAffine(
        image,
        M_2x3,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


class FaceDetectorMTCNN:
    """MTCNN-based face detector with optional alignment."""

    def __init__(self) -> None:
        self.detector = MTCNN()

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in a BGR image."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected HxWx3 image (BGR)")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        dets = self.detector.detect_faces(rgb) or []

        faces: list[DetectedFace] = []
        for d in dets:
            conf = float(d.get("confidence", 0.0))

            x, y, w, h = d["box"]
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)

            kp = d.get("keypoints", {})
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

            faces.append(
                DetectedFace(
                    box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    prob=conf,
                    landmarks=lm,
                )
            )

        faces.sort(key=lambda f: f.prob, reverse=True)
        return faces

    def detect_and_align(
        self,
        image_bgr: np.ndarray,
        min_prob: float = 0.90,
        pick: str = "best",  # "best" | "largest"
        return_resized: Optional[tuple[int, int]] = None,  # e.g. (224,224) for display
        margin: float = 0.35,  # extra context around face box to prevent out-of-bounds warps
    ) -> dict[str, Any]:
        """Detect faces and return an affine-aligned crop of the selected face.

        Key idea:
            Crop around the selected face with margin first, then align the crop.
            This avoids the transform sampling outside the image (which caused the
            ugly reflection artifacts).
        """
        faces = self.detect(image_bgr)
        if not faces:
            return {"faces": [], "selected": None}

        faces_f = [f for f in faces if f.prob >= min_prob]
        if not faces_f:
            logger.warning(
                "No face met min_prob=%.2f (best=%.2f); using all %d detection(s).",
                min_prob,
                faces[0].prob,
                len(faces),
            )
            faces_f = faces

        if pick == "largest":

            def area(face: DetectedFace) -> float:
                x1, y1, x2, y2 = face.box
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)

            selected = max(faces_f, key=area)
        else:
            selected = faces_f[0]

        # --- Crop with margin ---
        H, W = image_bgr.shape[:2]
        x1, y1, x2, y2 = selected.box.astype(int)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = int(bw * margin)
        pad_y = int(bh * margin)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(W, x2 + pad_x)
        cy2 = min(H, y2 + pad_y)

        crop = image_bgr[cy1:cy2, cx1:cx2].copy()

        # Shift landmarks into crop coordinate system
        lm = selected.landmarks.copy()
        lm[:, 0] -= cx1
        lm[:, 1] -= cy1

        # Estimate affine in canonical 112x112 space and warp the cropped image
        M = estimate_affine_from_5pts(lm)
        aligned = warp_face(crop, M, output_size=_CANONICAL_SIZE)

        out: dict[str, Any] = {
            "faces": faces,
            "selected": selected,
            "affine": M,
            "aligned_bgr": aligned,  # 112x112
            "crop_box": np.array([cx1, cy1, cx2, cy2], dtype=np.int32),
        }

        if return_resized is not None:
            rw, rh = return_resized
            out["aligned_bgr_resized"] = cv2.resize(
                aligned,
                (rw, rh),
                interpolation=cv2.INTER_CUBIC,
            )

        return out