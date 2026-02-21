from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from mtcnn.mtcnn import MTCNN  # TensorFlow/Keras-based
except Exception as e:
    raise ImportError("Install mtcnn (TF/Keras): pip install mtcnn tensorflow") from e

logger = logging.getLogger(__name__)

# Type alias for a 5-point landmark array: [left_eye, right_eye, nose, mouth_left, mouth_right]
# Each point is an (x, y) pixel coordinate, so shape is (5, 2).
Landmarks5 = np.ndarray  # shape (5, 2)


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


def _template_5pts(output_size: tuple[int, int] = (112, 112)) -> np.ndarray:
    """Return the canonical 5-point facial landmark template scaled to output_size.

    The reference coordinates are taken from the ArcFace/insightface alignment
    standard (originally defined for a 112x112 crop). They place both eyes on a
    horizontal line roughly one-third from the top, the nose tip at centre, and
    the two mouth corners symmetric about the vertical midline.

    Reference:
        Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
        Recognition", CVPR 2019. Template from insightface:
        https://github.com/deepinsight/insightface

    Args:
        output_size: Target crop dimensions as (width, height). Coordinates are
            linearly scaled from the canonical 112x112 space.

    Returns:
        Destination landmark array of shape (5, 2) in float32.
    """
    # Canonical 112x112 reference points (x, y):
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


def estimate_affine_from_5pts(
    src_landmarks: Landmarks5,
    output_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Estimate a 2-D partial affine transform that maps detected landmarks to
    the canonical face template.

    Uses LMEDS (Least Median of Squares) for robustness against outlier
    keypoints (e.g. occluded eye or mouth corner).

    Args:
        src_landmarks: Detected 5-point landmarks as (x, y) pixel coordinates,
            shape (5, 2), float32.
        output_size: Target crop dimensions as (width, height). Must match the
            output_size used when calling warp_face.

    Returns:
        2x3 affine matrix M (float32) suitable for cv2.warpAffine.

    Raises:
        ValueError: If src_landmarks does not have shape (5, 2).
        RuntimeError: If cv2 cannot estimate a valid transform (degenerate input).
    """
    if src_landmarks.shape != (5, 2):
        raise ValueError(f"Expected (5,2) landmarks, got {src_landmarks.shape}")

    dst = _template_5pts(output_size=output_size)
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
    output_size: tuple[int, int] = (112, 112),
) -> np.ndarray:
    """Apply a 2x3 affine transform to produce an aligned face crop.

    Color-space agnostic: operates on raw pixel values regardless of whether
    the input is BGR, RGB, or grayscale. The caller is responsible for keeping
    track of the color space (the typical contract in this pipeline is BGR in,
    BGR out).

    Args:
        image: Input image array of shape (H, W, C) or (H, W).
        M_2x3: 2x3 affine matrix as returned by estimate_affine_from_5pts.
        output_size: Output crop dimensions as (width, height). Must match the
            output_size used when estimating M_2x3.

    Returns:
        Warped image of shape (height, width, C) in the same dtype as input.
        Out-of-bounds pixels are filled with 0 (black).
    """
    w, h = output_size
    return cv2.warpAffine(image, M_2x3, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


class FaceDetectorMTCNN:
    """MTCNN-based face detector with optional alignment.

    Wraps the TensorFlow/Keras `mtcnn` package to provide detection and
    affine alignment in a single call. Internally converts BGR→RGB before
    passing frames to MTCNN (which requires RGB input), then works in BGR
    for all subsequent geometry operations so that the aligned crop can be
    passed directly to OpenCV-based preprocessing code.

    Typical usage::

        detector = FaceDetectorMTCNN()
        result = detector.detect_and_align(bgr_frame)
        aligned_bgr = result["aligned_bgr"]  # ready for preprocessing
    """

    def __init__(self) -> None:
        self.detector = MTCNN()

    def detect(self, image_bgr: np.ndarray) -> list[DetectedFace]:
        """Detect all faces in a BGR image.

        Converts to RGB internally because the `mtcnn` package requires RGB
        input. Bounding boxes and landmarks are returned in the original image's
        pixel coordinate space (not the RGB copy). Results are sorted by
        confidence descending so index 0 is always the most confident detection.

        Note: No confidence threshold is applied here — all detections are
        returned. Use the min_prob parameter in detect_and_align, or filter the
        list yourself, to discard low-confidence results.

        Args:
            image_bgr: Input image in BGR channel order, shape (H, W, 3), uint8.

        Returns:
            List of DetectedFace objects sorted by prob descending. Empty list
            if no faces are found.

        Raises:
            ValueError: If the input array is not a 3-channel HxWx3 image.
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected HxWx3 image (BGR)")

        # MTCNN requires RGB; convert before detection only — all subsequent
        # geometry (bounding boxes, landmarks, warping) uses the original BGR frame.
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # detector.detect_faces returns a list of dicts:
        # {'box': [x, y, w, h], 'confidence': float, 'keypoints': {name: (x, y), ...}}
        dets = self.detector.detect_faces(rgb) or []

        faces: list[DetectedFace] = []
        for d in dets:
            conf = float(d.get("confidence", 0.0))

            # Convert MTCNN's (x, y, w, h) box to (x1, y1, x2, y2) corner format.
            x, y, w, h = d["box"]
            x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)

            kp = d.get("keypoints", {})
            # Stack keypoints in the fixed order expected by estimate_affine_from_5pts
            # and the ArcFace template: left_eye, right_eye, nose, mouth_left, mouth_right.
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

        # Sort highest confidence first so callers can rely on faces[0] being best.
        faces.sort(key=lambda f: f.prob, reverse=True)
        return faces

    def detect_and_align(
        self,
        image_bgr: np.ndarray,
        output_size: tuple[int, int] = (112, 112),
        min_prob: float = 0.90,
        pick: str = "best",  # "best" | "largest"
    ) -> dict[str, Any]:
        """Detect faces and return an affine-aligned crop of the selected face.

        Selection strategy:
        - Candidates are first filtered to those with prob >= min_prob.
        - If no detection passes the threshold, all detections are used as
          candidates and a warning is logged.
        - Among candidates, pick="best" selects highest confidence; pick="largest"
          selects the largest bounding-box area.

        The aligned crop is produced in BGR so it can be passed directly to
        OpenCV-based preprocessing. Downstream code (e.g. ResNet50 inference)
        must convert BGR→RGB before feeding the model.

        Args:
            image_bgr: Input image in BGR channel order, shape (H, W, 3), uint8.
            output_size: Aligned crop dimensions as (width, height). Defaults to
                (112, 112) to match the ArcFace alignment template.
            min_prob: Minimum confidence threshold for candidate selection.
                Detections below this value are excluded unless no detection
                meets the threshold, in which case all detections are used.
            pick: Face selection strategy among candidates.
                "best" — highest confidence score (default).
                "largest" — largest bounding-box area (useful for single-subject
                images where the subject may not be the most confident detection).

        Returns:
            Dict with keys:
                "faces" (list[DetectedFace]): All detections sorted by confidence.
                "selected" (DetectedFace | None): The chosen face, or None if no
                    faces were detected.
                "affine" (np.ndarray): 2x3 affine matrix used for warping (only
                    present when a face was selected).
                "aligned_bgr" (np.ndarray): Aligned BGR crop of shape
                    (height, width, 3) (only present when a face was selected).
        """
        faces = self.detect(image_bgr)
        if not faces:
            return {"faces": [], "selected": None}

        faces_f = [f for f in faces if f.prob >= min_prob]
        if not faces_f:
            # No detection met the threshold — fall back to all detections so the
            # caller still gets a result, but warn so the caller is aware.
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
            # "best": faces are already sorted by confidence descending.
            selected = faces_f[0]

        M = estimate_affine_from_5pts(selected.landmarks, output_size=output_size)
        aligned = warp_face(image_bgr, M, output_size=output_size)

        return {
            "faces": faces,
            "selected": selected,
            "affine": M,
            # BGR — caller must convert to RGB before model inference.
            "aligned_bgr": aligned,
        }
