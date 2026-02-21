#!/usr/bin/env python3
"""
Demo: load an un-aligned image, align it, show before/after with matplotlib.

Run:
  python scripts/demo_align.py
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt

# Make src importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set your image path
IMAGE_PATH = r"C:\Users\akash\OneDrive\Desktop\Port\utk_face\utk_face\49_1_3_20170104235844116.jpg"


def bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def main():
    print(f"Loading image: {IMAGE_PATH}")

    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    from src.core.face_detection import FaceDetectorMTCNN

    detector = FaceDetectorMTCNN()

    # Run alignment (alignment happens at 112x112 internally; crop+margin avoids border artifacts)
    result = detector.detect_and_align(
        img_bgr,
        min_prob=0.90,
        pick="best",
        return_resized=(224, 224),  # prettier for display
        margin=0.35,                # IMPORTANT: prevents “weird reflection” artifacts
    )

    # Get aligned image (prefer resized for display)
    aligned_bgr = result.get("aligned_bgr_resized")
    if aligned_bgr is None:
        aligned_bgr = result.get("aligned_bgr")

    if aligned_bgr is None:
        raise RuntimeError("No face found / alignment failed.")

    # For fair comparison, show the same crop that alignment used (if available)
    crop_box = result.get("crop_box")
    if crop_box is not None:
        cx1, cy1, cx2, cy2 = crop_box.astype(int)
        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(img_bgr.shape[1], cx2)
        cy2 = min(img_bgr.shape[0], cy2)
        before_bgr = img_bgr[cy1:cy2, cx1:cx2]
    else:
        # fallback: show detected face bbox
        selected = result.get("selected")
        if selected is not None:
            x1, y1, x2, y2 = selected.box.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_bgr.shape[1], x2)
            y2 = min(img_bgr.shape[0], y2)
            before_bgr = img_bgr[y1:y2, x1:x2]
        else:
            before_bgr = img_bgr

    # Convert for display
    before = bgr_to_rgb(before_bgr)
    after = bgr_to_rgb(aligned_bgr)

    # Show result
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Before (crop used for alignment)")
    plt.imshow(before)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("After (aligned)")
    plt.imshow(after)
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()