"""Quick visual test for RetinaFace preprocessing.

Usage:
    python scripts/test_retinaface_preprocess.py --image path/to/image.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.face_detection_retinaface import detect_and_align


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot original image and RetinaFace-processed aligned face."
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to an input image file.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Alias for --image.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Output face size (square). Default: 224",
    )
    parser.add_argument(
        "--min_prob",
        type=float,
        default=0.85,
        help="Minimum face confidence for selection. Default: 0.85",
    )
    parser.add_argument(
        "--pick",
        type=str,
        default="largest",
        choices=["largest", "best"],
        help="Face selection strategy when multiple faces are detected.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.50,
        help="Extra crop margin around selected face. Default: 0.50",
    )
    return parser.parse_args()


def main() -> None:
    """Run detection/alignment and display original vs processed image."""
    args = parse_args()

    image_arg = args.image or args.path
    if image_arg is None:
        raise ValueError("Please provide an image path with --image or --path.")

    image_path = Path(image_arg)
    if not image_path.is_absolute():
        image_path = (PROJECT_ROOT / image_path).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    aligned_bgr, info = detect_and_align(
        image_bgr=image_bgr,
        output_size=(args.size, args.size),
        min_score=args.min_prob,
        pick=args.pick,
        margin=args.margin,
        return_info=True,
    )

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if aligned_bgr is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(original_rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].text(0.5, 0.5, "No face detected", ha="center", va="center")
        axes[1].set_title("Processed")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        return

    aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(aligned_rgb)
    selected_score = float(info["selected"].get("score", info["selected"].get("confidence", 0.0)))
    axes[1].set_title(f"Processed (Aligned) - score: {selected_score:.3f}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
