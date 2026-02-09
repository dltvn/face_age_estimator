"""Script to rename .jpg.chip.jpg files to .jpg in UTKFace dataset.

This script processes all files with .chip.jpg extension and renames them
to have a single .jpg extension for consistency.

Usage:
    python scripts/rename_chip_files.py

Configuration:
    DATASET_PATH: Path to the UTKFace images directory
    DRY_RUN: If True, shows what would be renamed without actually doing it
"""

import logging
from pathlib import Path

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
DRY_RUN = False  # Set to False to actually perform renames


def rename_chip_files(dataset_path: Path | str, dry_run: bool = False) -> dict:
    """Rename all .jpg.chip.jpg files to .jpg.

    Args:
        dataset_path: Path to the directory containing UTKFace images.
        dry_run: If True, only simulate the renaming without making changes.

    Returns:
        Dictionary with statistics: renamed_count, skipped_count, error_count.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
    """
    dataset_dir = Path(dataset_path)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if not dataset_dir.is_dir():
        raise ValueError(f"Path is not a directory: {dataset_path}")

    logger.info(f"Scanning dataset directory: {dataset_path}")
    if dry_run:
        logger.info("DRY RUN MODE - No files will actually be renamed")

    # Find all .chip.jpg files
    chip_files = list(dataset_dir.glob("*.chip.jpg"))
    logger.info(f"Found {len(chip_files)} .chip.jpg files")

    renamed_count = 0
    skipped_count = 0
    error_count = 0

    for file_path in chip_files:
        try:
            # Generate new name by removing .chip.jpg and adding .jpg
            # e.g., "1_0_0_20161219140623097.jpg.chip.jpg" -> "1_0_0_20161219140623097.jpg"
            old_name = file_path.name

            # Remove .chip.jpg suffix (9 characters)
            if old_name.endswith(".chip.jpg"):
                new_name = old_name[:-9]  # Remove .chip.jpg
                if not new_name.endswith(".jpg"):
                    new_name += ".jpg"
            else:
                logger.warning(f"Unexpected file format: {old_name}")
                skipped_count += 1
                continue

            new_path = file_path.parent / new_name

            # Check if target already exists
            if new_path.exists():
                logger.warning(f"Target already exists, skipping: {new_name}")
                skipped_count += 1
                continue

            if dry_run:
                logger.info(f"Would rename: {old_name} -> {new_name}")
            else:
                file_path.rename(new_path)
                logger.debug(f"Renamed: {old_name} -> {new_name}")

            renamed_count += 1

        except Exception as e:
            logger.error(f"Error renaming {file_path.name}: {e}")
            error_count += 1

    stats = {
        "renamed_count": renamed_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
    }

    return stats


def main():
    """Main function to rename chip files."""
    logger.info("Starting UTKFace file renaming")
    logger.info(f"Dataset path: {DATASET_PATH}")

    try:
        stats = rename_chip_files(DATASET_PATH, dry_run=DRY_RUN)

        logger.info("=" * 50)
        logger.info("Renaming completed!")
        logger.info(f"Files renamed: {stats['renamed_count']}")
        logger.info(f"Files skipped: {stats['skipped_count']}")
        logger.info(f"Errors: {stats['error_count']}")

        if DRY_RUN:
            logger.info(
                "\nThis was a DRY RUN. Set DRY_RUN=False to actually rename files."
            )

        return 0

    except FileNotFoundError as e:
        logger.error(f"Dataset directory not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error during renaming: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
