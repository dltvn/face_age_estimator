"""Script to identify and delete unparseable files in UTKFace dataset.

This script scans all image files, attempts to parse their filenames,
and optionally deletes files that cannot be parsed according to the
UTKFace naming convention.

Usage:
    python scripts/delete_invalid_files.py

Configuration:
    DATASET_PATH: Path to the UTKFace images directory
    DRY_RUN: If True, shows what would be deleted without actually doing it
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict

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
DRY_RUN = True  # Set to False to actually delete files
BACKUP_DIR = (
    PROJECT_ROOT / "data" / "utk_face_invalid_backup"
)  # Optional: move instead of delete

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

    Example:
        >>> parse_utk_filename("39_1_0_20170116174525125.jpg.chip.jpg")
        {
            'filename': '39_1_0_20170116174525125.jpg.chip.jpg',
            'age': 39,
            'gender': 'Female',
            'gender_id': 1,
            'race': 'White',
            'race_id': 0,
            'timestamp': '20170116174525125'
        }
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
            logger.warning(f"Invalid filename format: {filename} (expected 4 parts)")
            return None

        age_str, gender_str, race_str, timestamp = parts

        # Parse age
        age = int(age_str)
        if age < 0 or age > 116:
            logger.warning(f"Age {age} out of valid range [0, 116] in {filename}")
            return None

        # Parse gender
        gender_id = int(gender_str)
        if gender_id not in GENDER_MAPPING:
            logger.warning(f"Invalid gender ID {gender_id} in {filename}")
            return None
        gender = GENDER_MAPPING[gender_id]

        # Parse race
        race_id = int(race_str)
        if race_id not in RACE_MAPPING:
            logger.warning(f"Invalid race ID {race_id} in {filename}")
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

    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse filename {filename}: {e}")
        return None


def identify_invalid_files(dataset_path: Path | str) -> tuple:
    """Identify files that cannot be parsed.

    Args:
        dataset_path: Path to the directory containing UTKFace images.

    Returns:
        Tuple of (valid_files, invalid_files) lists.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
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

    valid_files = []
    invalid_files = []

    for file_path in image_files:
        metadata = parse_utk_filename(file_path.name)
        if metadata:
            valid_files.append(file_path)
        else:
            invalid_files.append(file_path)
            logger.warning(f"Invalid file: {file_path.name}")

    return valid_files, invalid_files


def delete_invalid_files(
    dataset_path: Path | str,
    dry_run: bool = True,
    backup: bool = False,
    backup_dir: Path | str = None,
) -> dict:
    """Delete or move files that cannot be parsed.

    Args:
        dataset_path: Path to the directory containing UTKFace images.
        dry_run: If True, only simulate deletion without making changes.
        backup: If True, move files to backup directory instead of deleting.
        backup_dir: Path to backup directory (only used if backup=True).

    Returns:
        Dictionary with statistics: valid_count, deleted_count, error_count.
    """
    valid_files, invalid_files = identify_invalid_files(dataset_path)

    logger.info(f"Valid files: {len(valid_files)}")
    logger.info(f"Invalid files: {len(invalid_files)}")

    # Show invalid files found
    if invalid_files:
        logger.info("\nInvalid files found:")
        for f in invalid_files[:10]:  # Show first 10
            logger.info(f"  - {f.name}")
        if len(invalid_files) > 10:
            logger.info(f"  ... and {len(invalid_files) - 10} more")
    else:
        logger.info("\nNo invalid files found. Dataset is clean!")
        return {
            "valid_count": len(valid_files),
            "deleted_count": 0,
            "error_count": 0,
        }

    if dry_run:
        logger.info("\nDRY RUN MODE - No files will actually be deleted/moved")
    else:
        logger.warning(f"\nAbout to process {len(invalid_files)} files!")

    # Setup backup directory if needed
    if backup and not dry_run:
        backup_path = Path(backup_dir or BACKUP_DIR)
        backup_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup directory: {backup_path}")

    deleted_count = 0
    error_count = 0

    for file_path in invalid_files:
        try:
            if dry_run:
                if backup:
                    logger.info(f"Would move to backup: {file_path.name}")
                else:
                    logger.info(f"Would delete: {file_path.name}")
            else:
                if backup:
                    backup_path = Path(backup_dir or BACKUP_DIR)
                    target_path = backup_path / file_path.name
                    file_path.rename(target_path)
                    logger.debug(f"Moved to backup: {file_path.name}")
                else:
                    file_path.unlink()
                    logger.debug(f"Deleted: {file_path.name}")

            deleted_count += 1

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            error_count += 1

    stats = {
        "valid_count": len(valid_files),
        "deleted_count": deleted_count,
        "error_count": error_count,
    }

    return stats


def main():
    """Main function to delete invalid files."""
    logger.info("Starting invalid file identification and deletion")
    logger.info(f"Dataset path: {DATASET_PATH}")

    try:
        stats = delete_invalid_files(
            DATASET_PATH,
            dry_run=DRY_RUN,
            backup=True,  # Always backup by default
            backup_dir=BACKUP_DIR,
        )

        logger.info("\n" + "=" * 50)
        if DRY_RUN:
            logger.info("DRY RUN COMPLETE")
            logger.info("Set DRY_RUN=False to actually process files.")
            logger.info(
                "Files will be moved to backup directory (not permanently deleted)."
            )
        else:
            logger.info("DELETION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Valid files: {stats['valid_count']}")
        logger.info(f"Invalid files processed: {stats['deleted_count']}")
        logger.info(f"Errors: {stats['error_count']}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"Dataset directory not found: {e}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
