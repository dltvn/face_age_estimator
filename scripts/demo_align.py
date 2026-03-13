#!/usr/bin/env python3
"""
Demo: load an image, detect and align the face, show before/after with matplotlib.

Usage:
    python scripts/demo_align.py
    python scripts/demo_align.py path/to/image.jpg
"""

import os
import sys

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.core.face_detection import (
    FaceDetectorMediaPipe, FaceDetectorHaar, create_detector, _get_landmark_template
)

DEFAULT_IMAGE_PATH = (
    r"C:\Users\akash\OneDrive\Desktop\Port\utk_face\utk_face"
    r"\49_1_3_20170104235844116.jpg"
)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_landmarks(ax, landmarks: np.ndarray, color: str = "lime") -> None:
    labels = ["L-Eye", "R-Eye", "Nose", "Mouth-L", "Mouth-R"]
    for (x, y), label in zip(landmarks, labels):
        ax.plot(x, y, "o", color=color, markersize=5)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=6, color=color)


def draw_bbox(ax, box: np.ndarray, color: str = "red", label: str = "") -> None:
    x1, y1, x2, y2 = box.astype(int)
    rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=1.5, edgecolor=color, facecolor="none")
    ax.add_patch(rect)
    if label:
        ax.text(x1, y1 - 4, label, color=color, fontsize=7, fontweight="bold")


def try_detect(img_bgr: np.ndarray, detector):
    """Try detection, retry with upscaling if nothing found."""
    for scale in [1.0, 2.0, 3.0]:
        if scale != 1.0:
            h, w = img_bgr.shape[:2]
            img_try = cv2.resize(img_bgr, (int(w*scale), int(h*scale)),
                                 interpolation=cv2.INTER_LANCZOS4)
        else:
            img_try = img_bgr

        result = detector.detect_and_align(
            img_try, pick="best", margin=0.35, return_resized=(224, 224)
        )
        if result["selected"] is not None:
            if scale != 1.0:
                print(f"[demo] Detected at {scale}x upscale")
                result["selected"].box        /= scale
                result["selected"].landmarks  /= scale
                result["crop_box"] = (result["crop_box"].astype(float) / scale).astype(int)
            return result

    return {"faces": [], "selected": None}


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH
    print(f"[demo] Loading image: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print(f"[demo] Image size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

    # Auto-select best available detector
    detector = create_detector()
    print(f"[demo] Using detector: {type(detector).__name__}")

    result = try_detect(img_bgr, detector)

    # If MediaPipe failed, try Haar as last resort
    if result["selected"] is None and not isinstance(detector, FaceDetectorHaar):
        print("[demo] Primary detector failed, trying Haar fallback...")
        haar = FaceDetectorHaar(scale_factor=1.05, min_neighbors=1)
        result = try_detect(img_bgr, haar)

    if result["selected"] is None:
        print("[demo] WARNING: No face detected.")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.suptitle("No face detected", fontsize=12, color="red")
        ax.imshow(bgr_to_rgb(img_bgr))
        ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(os.path.dirname(os.path.abspath(image_path)),
                                "demo_align_output.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        return

    selected    = result["selected"]
    aligned_bgr = result["aligned_bgr_resized"] if "aligned_bgr_resized" in result \
                  else result["aligned_bgr"]
    crop_box    = result["crop_box"]

    print(f"[demo] {len(result['faces'])} face(s) detected. "
          f"Selected → score={selected.prob:.4f}  box={selected.box.astype(int)}")

    cx1, cy1, cx2, cy2 = np.clip(
        crop_box, [0, 0, 0, 0],
        [img_bgr.shape[1], img_bgr.shape[0], img_bgr.shape[1], img_bgr.shape[0]],
    ).astype(int)

    before_bgr = img_bgr[cy1:cy2, cx1:cx2].copy()
    lm_local = selected.landmarks.copy()
    lm_local[:, 0] -= cx1
    lm_local[:, 1] -= cy1

    bx1, by1, bx2, by2 = selected.box.astype(int)
    box_local = np.array([bx1-cx1, by1-cy1, bx2-cx1, by2-cy1], dtype=np.int32)

    detector_name = type(detector).__name__.replace("FaceDetector", "")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Face Alignment Demo  ·  {detector_name} + Affine Warp", fontsize=12)

    # Panel 1 — Original
    axes[0].imshow(bgr_to_rgb(img_bgr))
    axes[0].set_title(f"1. Original  ({img_bgr.shape[1]}×{img_bgr.shape[0]})", fontsize=9)
    axes[0].axis("off")

    # Panel 2 — Detection
    axes[1].imshow(bgr_to_rgb(before_bgr))
    axes[1].set_title(f"2. Detection  (score={selected.prob:.4f})", fontsize=9)
    draw_bbox(axes[1], box_local, color="red", label=f"{selected.prob:.2f}")
    draw_landmarks(axes[1], lm_local, color="lime")
    axes[1].axis("off")

    # Panel 3 — Aligned
    axes[2].imshow(bgr_to_rgb(aligned_bgr))
    axes[2].set_title("3. Aligned  (224×224)", fontsize=9)
    canonical_lm = _get_landmark_template(output_size=(224, 224))
    draw_landmarks(axes[2], canonical_lm, color="cyan")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(image_path)),
                            "demo_align_output.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[demo] Figure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()