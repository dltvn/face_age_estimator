"""Script to explore and analyze the UTKFace dataset.

This script performs comprehensive dataset exploration including:
- Checking valid and invalid images
- Image resolution analysis
- Age distribution histogram
- Race/ethnicity distribution histogram
- Display 25 random images (5x5 grid) with labels
- Generate a summary report

Usage:
    python scripts/explore_dataset.py

Configuration:
    DATASET_PATH: Path to the UTKFace images directory
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import random

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Configuration - use paths relative to script location
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "utk_face"

# Race/ethnicity mapping for UTKFace dataset
RACE_MAPPING = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}

# Gender mapping
GENDER_MAPPING = {
    0: "Male",
    1: "Female",
}


def parse_utk_filename(filename: str) -> Optional[Dict[str, any]]:
    """Parse UTKFace filename to extract metadata.

    UTKFace images are named as: [age]_[gender]_[race]_[date&time].jpg
    or [age]_[gender]_[race]_[date&time].jpg.chip.jpg for preprocessed chips.

    Args:
        filename: Name of the UTKFace image file.

    Returns:
        Dictionary containing age, gender, race, and timestamp, or None
        if the filename format is invalid.
    """
    try:
        # Handle .jpg.chip.jpg format by removing .chip.jpg suffix first
        name_without_ext = filename
        if name_without_ext.endswith(".chip.jpg"):
            name_without_ext = name_without_ext[:-9]  # Remove .chip.jpg

        # Remove remaining file extension (.jpg, .jpeg, .png)
        name_without_ext = Path(name_without_ext).stem

        # Split by underscore
        parts = name_without_ext.split("_")

        # Validate format (should have 4 parts)
        if len(parts) != 4:
            return None

        age_str, gender_str, race_str, timestamp = parts

        # Parse age
        age = int(age_str)
        if age < 0 or age > 116:
            return None

        # Parse gender
        gender_id = int(gender_str)
        if gender_id not in GENDER_MAPPING:
            return None
        gender = GENDER_MAPPING[gender_id]

        # Parse race
        race_id = int(race_str)
        if race_id not in RACE_MAPPING:
            return None
        race = RACE_MAPPING[race_id]

        return {
            "filename": filename,
            "age": age,
            "gender": gender,
            "gender_id": gender_id,
            "race": race,
            "race_id": race_id,
            "timestamp": timestamp,
        }

    except (ValueError, IndexError):
        return None


def get_image_resolution(image_path: Path) -> Optional[Tuple[int, int]]:
    """Get the resolution of an image file.

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (width, height), or None if image cannot be loaded.
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        logger.warning(f"Failed to load image {image_path.name}: {e}")
        return None


def analyze_dataset(dataset_path: Path | str) -> Dict:
    """Analyze the UTKFace dataset comprehensively.

    Args:
        dataset_path: Path to the UTKFace dataset directory.

    Returns:
        Dictionary containing analysis results.
    """
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if not dataset_dir.is_dir():
        raise ValueError(f"Path is not a directory: {dataset_path}")

    logger.info(f"Scanning dataset directory: {dataset_path}")

    # Collect all image files
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [
        f
        for f in dataset_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    logger.info(f"Found {len(image_files)} image files")

    # Separate valid and invalid files
    valid_files = []
    invalid_files = []
    valid_metadata = []
    resolutions = []
    ages = []
    races = []

    for idx, file_path in enumerate(image_files):
        if (idx + 1) % 5000 == 0:
            logger.info(f"Processing... {idx + 1}/{len(image_files)}")

        metadata = parse_utk_filename(file_path.name)
        if metadata:
            valid_files.append(file_path)
            valid_metadata.append(metadata)

            # Get image resolution (skip for speed)
            # resolution = get_image_resolution(file_path)
            # if resolution:
            #     resolutions.append(resolution)

            ages.append(metadata["age"])
            races.append(metadata["race"])
        else:
            invalid_files.append(file_path)

    # Calculate statistics
    results = {
        "total_files": len(image_files),
        "valid_count": len(valid_files),
        "invalid_count": len(invalid_files),
        "valid_percentage": (len(valid_files) / len(image_files) * 100)
        if image_files
        else 0,
        "resolutions": resolutions,
        "ages": ages,
        "races": races,
        "valid_metadata": valid_metadata,
        "valid_files": valid_files,
        "invalid_files": invalid_files,
    }

    return results


def print_summary(results: Dict) -> str:
    """Generate a summary of dataset analysis.

    Args:
        results: Dictionary containing analysis results.

    Returns:
        String containing the summary.
    """
    summary = []
    summary.append("=" * 70)
    summary.append("DATASET EXPLORATION SUMMARY")
    summary.append("=" * 70)
    summary.append("")

    # File validity
    summary.append("FILE VALIDITY")
    summary.append(f"Total files: {results['total_files']}")
    summary.append(f"Valid files: {results['valid_count']}")
    summary.append(f"Invalid files: {results['invalid_count']}")
    summary.append(f"Valid percentage: {results['valid_percentage']:.2f}%")
    summary.append("")

    # Image resolution
    summary.append("IMAGE RESOLUTION")
    summary.append(
        "Resolution analysis skipped for faster processing (use separate script if needed)"
    )
    summary.append("")

    # Age distribution
    summary.append("AGE DISTRIBUTION")
    if results["ages"]:
        ages_arr = np.array(results["ages"])
        summary.append(f"Total images: {len(results['ages'])}")
        summary.append(f"Age range: {ages_arr.min()} to {ages_arr.max()} years")
        summary.append(f"Mean age: {ages_arr.mean():.2f} years")
        summary.append(f"Median age: {np.median(ages_arr):.2f} years")
        summary.append(f"Standard deviation: {ages_arr.std():.2f} years")
        summary.append("")
    else:
        summary.append("No valid age data available")
        summary.append("")

    # Race distribution
    summary.append("RACE/ETHNICITY DISTRIBUTION")
    if results["races"]:
        from collections import Counter

        race_counter = Counter(results["races"])
        summary.append(f"Total images: {len(results['races'])}")
        summary.append("Distribution:")
        for race, count in sorted(
            race_counter.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = count / len(results["races"]) * 100
            summary.append(f"  {race}: {count} ({percentage:.2f}%)")
        summary.append("")
    else:
        summary.append("No valid race data available")
        summary.append("")

    summary.append("=" * 70)

    return "\n".join(summary)


def plot_age_distribution(ages: list, title: str = "Age Distribution") -> None:
    """Plot age distribution histogram.

    Args:
        ages: List of ages.
        title: Title for the plot.
    """
    if not ages:
        logger.warning("No age data to plot")
        return

    plt.figure(figsize=(12, 5))

    plt.hist(ages, bins=40, edgecolor="black", alpha=0.7, color="steelblue")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_race_distribution(
    races: list, title: str = "Race/Ethnicity Distribution"
) -> None:
    """Plot race/ethnicity distribution histogram.

    Args:
        races: List of race labels.
        title: Title for the plot.
    """
    if not races:
        logger.warning("No race data to plot")
        return

    from collections import Counter

    race_counter = Counter(races)
    races_sorted = sorted(race_counter.items(), key=lambda x: x[1], reverse=True)
    race_labels = [r[0] for r in races_sorted]
    race_counts = [r[1] for r in races_sorted]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        race_labels, race_counts, edgecolor="black", alpha=0.7, color="coral"
    )

    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.xlabel("Race/Ethnicity")
    plt.ylabel("Count")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def display_random_images(
    valid_files: list, valid_metadata: list, num_images: int = 10
) -> None:
    """Display random images from the dataset with their labels.

    Args:
        valid_files: List of valid image file paths.
        valid_metadata: List of metadata dictionaries corresponding to valid files.
        num_images: Number of random images to display.
    """
    if len(valid_files) < num_images:
        num_images = len(valid_files)
        logger.warning(f"Only {num_images} valid images available")

    # Get random indices
    indices = random.sample(range(len(valid_files)), num_images)
    selected_files = [valid_files[i] for i in indices]
    selected_metadata = [valid_metadata[i] for i in indices]

    # Create a grid of subplots
    cols = 5
    rows = (num_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3.2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, (file_path, metadata) in enumerate(zip(selected_files, selected_metadata)):
        try:
            img = Image.open(file_path)
            axes[idx].imshow(img)

            # Create label
            label = (
                f"Age: {metadata['age']}\n"
                f"Gender: {metadata['gender']}\n"
                f"Race: {metadata['race']}"
            )
            axes[idx].set_title(label, fontsize=10)
            axes[idx].axis("off")

        except Exception as e:
            logger.warning(f"Failed to display image {file_path.name}: {e}")
            axes[idx].text(0.5, 0.5, f"Error loading image", ha="center", va="center")
            axes[idx].axis("off")

    # Hide remaining subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """Main function for dataset exploration."""
    logger.info("Starting dataset exploration")
    logger.info(f"Dataset path: {DATASET_PATH}")

    try:
        # Analyze dataset
        results = analyze_dataset(DATASET_PATH)

        # Print summary
        summary = print_summary(results)
        print(summary)

        # Plot age distribution
        if results["ages"]:
            logger.info("Plotting age distribution histogram...")
            plot_age_distribution(results["ages"])

        # Plot race distribution
        if results["races"]:
            logger.info("Plotting race/ethnicity distribution histogram...")
            plot_race_distribution(results["races"])

        # Display random images
        if results["valid_files"]:
            logger.info("Displaying 25 random images...")
            display_random_images(
                results["valid_files"], results["valid_metadata"], num_images=25
            )

        logger.info("Dataset exploration complete")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Dataset directory not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
