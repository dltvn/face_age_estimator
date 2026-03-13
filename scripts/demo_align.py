#!/usr/bin/env python3
"""
Simple demo:git pull origin main
Load one image, detect the face, align it,
and show the before/after result.

Just change IMAGE_PATH below and run:
    python scripts/demo_align.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Make sure we can import from src/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

#Change this to whatever image you want to test
IMAGE_PATH = r"C:\Users\akash\OneDrive\Desktop\Port\utk_face\utk_face\1_1_0_20170109194452834.jpg"


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    # OpenCV loads images in BGR format.
    # Matplotlib expects RGB.
    # If we don’t convert, colors will look weird.
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    image_path = Path(IMAGE_PATH)
    print(f"Loading image: {image_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read the image using OpenCV
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Import here so script still runs even if src path changes
    from src.core.face_detection import FaceDetectorMTCNN

    detector = FaceDetectorMTCNN()

    # This is where everything happens:
    # - MTCNN finds the face
    # - It picks the largest one
    # - Crops it (with margin)
    # - Uses the 5 keypoints to rotate + scale + center the face
    # - Outputs a clean 224x224 aligned image
    result = detector.detect_and_align(
        img_bgr,
        output_size=(224, 224),  # final size
        min_prob=0.90,           # ignore very weak detections
        pick="largest",          # if multiple faces, pick biggest
        margin=0.45,             # add some extra space around face
    )

    if result.get("selected") is None:
        raise RuntimeError("No face detected.")

    aligned_bgr = result["aligned_bgr"]

    # Show the original vs aligned result
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Raw Input")
    plt.imshow(bgr_to_rgb(img_bgr))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Aligned (5-point)")
    plt.imshow(bgr_to_rgb(aligned_bgr))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()