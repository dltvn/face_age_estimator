from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from retinaface.pre_trained_models import get_model

logger = logging.getLogger(__name__)

_MODEL = None


def _get_model():
    """Load and cache the Torch RetinaFace model."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL = get_model("resnet50_2020-07-20", max_size=2048, device=device)
    _MODEL.eval()
    logger.info("Loaded RetinaFace model on %s", device)
    return _MODEL


def _run_detector(image_bgr: np.ndarray) -> list[dict]:
    """Run RetinaFace and return raw detections.

    Expected detection schema from backend:
    - `score`: float
    - `bbox`: [x1, y1, x2, y2]
    - `landmarks`: [[x, y], ...] shape (5, 2)
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Expected a 3-channel BGR image, got shape {image_bgr.shape}")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = _get_model().predict_jsons(rgb)
    if detections is None:
        return []
    if not isinstance(detections, list):
        raise RuntimeError("Unexpected RetinaFace output type; expected list.")
    return detections


def _clip_xyxy(box: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip [x1, y1, x2, y2] box to image bounds."""
    x1, y1, x2, y2 = box.astype(np.float32)
    x1 = float(np.clip(x1, 0, width - 1))
    y1 = float(np.clip(y1, 0, height - 1))
    x2 = float(np.clip(x2, 0, width - 1))
    y2 = float(np.clip(y2, 0, height - 1))
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _square_crop_with_margin(
    image_bgr: np.ndarray,
    box_xyxy: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create square crop around face and return crop with top-left offset."""
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_xyxy(box_xyxy, width=w, height=h)

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    side = max(bw, bh) * (1.0 + margin * 2.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    cx1 = max(0, int(round(cx - side / 2.0)))
    cy1 = max(0, int(round(cy - side / 2.0)))
    cx2 = min(w, int(round(cx + side / 2.0)))
    cy2 = min(h, int(round(cy + side / 2.0)))

    crop = image_bgr[cy1:cy2, cx1:cx2].copy()
    offset_xy = np.array([cx1, cy1], dtype=np.float32)
    return crop, offset_xy


def _template_5pts(output_size: tuple[int, int]) -> np.ndarray:
    """Return ArcFace 5-point template scaled to output size."""
    ref = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    w, h = output_size
    ref[:, 0] *= w / 112.0
    ref[:, 1] *= h / 112.0
    return ref


def _align_face(
    crop_bgr: np.ndarray,
    landmarks_in_crop: np.ndarray,
    output_size: tuple[int, int],
) -> np.ndarray:
    """Align crop to canonical template using 5 landmarks."""
    if landmarks_in_crop.shape != (5, 2):
        raise ValueError(f"Expected landmarks shape (5,2), got {landmarks_in_crop.shape}")

    dst = _template_5pts(output_size=output_size)
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks_in_crop.astype(np.float32),
        dst.astype(np.float32),
        method=cv2.LMEDS,
    )
    if matrix is None:
        raise RuntimeError("Could not estimate affine transform.")

    mean = crop_bgr.reshape(-1, 3).mean(axis=0)
    border = (int(mean[0]), int(mean[1]), int(mean[2]))
    w, h = output_size
    return cv2.warpAffine(
        crop_bgr,
        matrix.astype(np.float32),
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def detect_faces(image_bgr: np.ndarray, min_score: float = 0.0) -> list[dict]:
    """Return raw RetinaFace detections filtered by score threshold."""
    detections = _run_detector(image_bgr=image_bgr)
    return [d for d in detections if float(d["score"]) >= min_score]


def detect_and_align(
    image_bgr: np.ndarray,
    output_size: tuple[int, int] = (224, 224),
    min_score: float = 0.85,
    pick: str = "largest",
    margin: float = 0.50,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray | None, dict]:
    """Detect, select, crop, and align one face for inference.

    This function assumes RetinaFace detections include:
    `score`, `bbox`, and `landmarks` in the expected shapes.
    """
    detections = detect_faces(image_bgr=image_bgr, min_score=min_score)
    if not detections:
        info = {"detections": [], "selected": None}
        return (None, info) if return_info else None

    if pick == "best":
        selected = max(detections, key=lambda d: float(d["score"]))
    else:
        selected = max(
            detections,
            key=lambda d: (
                (float(d["bbox"][2]) - float(d["bbox"][0]))
                * (float(d["bbox"][3]) - float(d["bbox"][1]))
            ),
        )

    box = np.asarray(selected["bbox"], dtype=np.float32)
    landmarks = np.asarray(selected["landmarks"], dtype=np.float32)

    crop_bgr, offset_xy = _square_crop_with_margin(
        image_bgr=image_bgr,
        box_xyxy=box,
        margin=margin,
    )
    aligned = _align_face(
        crop_bgr=crop_bgr,
        landmarks_in_crop=landmarks - offset_xy,
        output_size=output_size,
    )

    info = {"detections": detections, "selected": selected}
    return (aligned, info) if return_info else aligned
