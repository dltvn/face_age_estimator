"""Script to prepare UTKFace dataset for TensorFlow training.

This script creates stratified train/val/test splits with proper structure:
1. Test set (balanced male/female, ~15% of data)
2. Train/Val sets for gender prediction (~70/30 split of remaining)
3. Gender-specific train/val sets for age prediction (with LDAE encoding)

The age labels use Label Distribution Age Encoding (LDAE) where ages are
encoded as Gaussian distributions with age-dependent standard deviation.

Usage:
    python scripts/prepare_dataset.py

Configuration:
    DATASET_PATH: Path to the UTKFace images directory
    SPLITS_DIR: Directory where split CSV files will be saved
    TEST_SIZE: Proportion of data for test set (default: 0.15)
    VAL_SIZE: Proportion of remaining data for validation (default: 0.3)
    RANDOM_SEED: Random seed for reproducibility (default: 42)
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Split configuration
TEST_SIZE = 0.15  # 15% for test set
VAL_SIZE = 0.3  # 30% of remaining for validation
RANDOM_SEED = 42

# Age encoding configuration (LDAE)
MIN_AGE = 0
MAX_AGE = 116
NUM_AGE_CLASSES = MAX_AGE - MIN_AGE + 1  # 117 classes (0-116)

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

    Args:
        filename: Name of the UTKFace image file.

    Returns:
        Dictionary containing age, gender, race, and timestamp, or None
        if the filename format is invalid.

    Example:
        >>> parse_utk_filename("39_1_0_20170116174525125.jpg")
        {
            'filename': '39_1_0_20170116174525125.jpg',
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
        if age < MIN_AGE or age > MAX_AGE:
            logger.warning(
                f"Age {age} out of valid range [{MIN_AGE}, {MAX_AGE}] in {filename}"
            )
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


def get_age_std_dev(age: int) -> float:
    """Calculate standard deviation for LDAE based on age.

    The standard deviation increases linearly with age to account for
    increasing uncertainty in age estimation for older individuals.

    Args:
        age: The age value (0-116).

    Returns:
        Standard deviation for the Gaussian distribution.
    """
    # Linear increase: σ = 1.0 + (age / 116) * 7.0
    # Young (0-20): σ ≈ 1.0-2.2
    # Middle (20-60): σ ≈ 2.2-4.6
    # Old (60+): σ ≈ 4.6-8.0
    min_std = 1.0
    max_std = 8.0
    return min_std + (age / MAX_AGE) * (max_std - min_std)


def encode_age_ldae(age: int) -> np.ndarray:
    """Encode age using Label Distribution Age Encoding (LDAE).

    Creates a Gaussian distribution centered at the true age with
    age-dependent standard deviation.

    Args:
        age: The true age value (0-116).

    Returns:
        Numpy array of shape (117,) containing the probability distribution.
    """
    std_dev = get_age_std_dev(age)
    age_range = np.arange(NUM_AGE_CLASSES)

    # Create Gaussian distribution
    distribution = np.exp(-0.5 * ((age_range - age) / std_dev) ** 2)

    # Normalize to sum to 1
    distribution = distribution / distribution.sum()

    return distribution


def load_dataset_metadata(dataset_path: Path) -> pd.DataFrame:
    """Load and parse all image files from dataset directory.

    Args:
        dataset_path: Path to the UTKFace images directory.

    Returns:
        DataFrame with parsed metadata.

    Raises:
        FileNotFoundError: If dataset_path does not exist.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if not dataset_path.is_dir():
        raise ValueError(f"Path is not a directory: {dataset_path}")

    logger.info(f"Scanning dataset directory: {dataset_path}")

    # Collect all image files
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = []

    for f in dataset_path.iterdir():
        if f.is_file():
            # Handle both regular and .chip.jpg files
            if f.suffix.lower() in image_extensions or f.name.endswith(".chip.jpg"):
                image_files.append(f.name)

    logger.info(f"Found {len(image_files)} image files")

    # Parse metadata from filenames
    metadata_list: List[Dict] = []
    failed_count = 0

    for filename in image_files:
        metadata = parse_utk_filename(filename)
        if metadata:
            metadata_list.append(metadata)
        else:
            failed_count += 1

    logger.info(f"Successfully parsed {len(metadata_list)} files")
    if failed_count > 0:
        logger.warning(f"Failed to parse {failed_count} files")

    # Create DataFrame
    df = pd.DataFrame(metadata_list)

    return df


def create_age_bins(df: pd.DataFrame, n_bins: int = 10) -> pd.Series:
    """Create age bins for stratified splitting.

    Args:
        df: DataFrame with 'age' column.
        n_bins: Number of age bins to create.

    Returns:
        Series with age bin labels.
    """
    return pd.cut(df["age"], bins=n_bins, labels=False, duplicates="drop")


def create_splits(
    df: pd.DataFrame, test_size: float, val_size: float, random_seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/val/test splits.

    Stratification is done on both gender and age bins to ensure balanced
    distribution across all splits.

    Args:
        df: Full dataset DataFrame.
        test_size: Proportion for test set.
        val_size: Proportion of remaining data for validation.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    logger.info("Creating stratified train/val/test splits...")

    # Create age bins for stratification
    df["age_bin"] = create_age_bins(df)

    # Create stratification column (gender + age_bin)
    df["stratify_col"] = df["gender_id"].astype(str) + "_" + df["age_bin"].astype(str)

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        stratify=df["stratify_col"],
    )

    logger.info(f"Test set size: {len(test_df)} ({test_size * 100:.1f}%)")

    # Second split: separate train and validation from remaining data
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_seed,
        stratify=train_val_df["stratify_col"],
    )

    logger.info(
        f"Train set size: {len(train_df)} ({(1 - test_size) * (1 - val_size) * 100:.1f}%)"
    )
    logger.info(
        f"Val set size: {len(val_df)} ({(1 - test_size) * val_size * 100:.1f}%)"
    )

    # Drop temporary columns
    for split_df in [train_df, val_df, test_df]:
        split_df.drop(columns=["age_bin", "stratify_col"], inplace=True)

    return train_df, val_df, test_df


def add_age_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add LDAE age encoding to DataFrame.

    Args:
        df: DataFrame with 'age' column.

    Returns:
        DataFrame with added 'age_distribution' column.
    """
    logger.info("Encoding ages with LDAE...")

    # Encode each age as a distribution
    age_distributions = []
    for age in df["age"]:
        distribution = encode_age_ldae(age)
        # Convert to comma-separated string for CSV storage
        dist_str = ",".join(f"{x:.6f}" for x in distribution)
        age_distributions.append(dist_str)

    df["age_distribution"] = age_distributions

    return df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
):
    """Save all dataset splits to CSV files.

    Creates:
    - test.csv: Balanced test set
    - train_gender.csv, val_gender.csv: For gender classification
    - train_age_male.csv, train_age_female.csv: For age estimation (train)
    - val_age_male.csv, val_age_female.csv: For age estimation (val)

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
        output_dir: Directory to save CSV files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving splits to {output_dir}...")

    # Save test set (no age encoding needed for test, can be used for both tasks)
    test_with_age = add_age_encoding(test_df.copy())
    test_with_age.to_csv(output_dir / "test.csv", index=False)
    logger.info(f"  Saved test.csv ({len(test_df)} samples)")

    # Save gender classification datasets
    train_df.to_csv(output_dir / "train_gender.csv", index=False)
    val_df.to_csv(output_dir / "val_gender.csv", index=False)
    logger.info(
        f"  Saved train_gender.csv ({len(train_df)} samples) and val_gender.csv ({len(val_df)} samples)"
    )

    # Add age encoding for age prediction datasets
    train_with_age = add_age_encoding(train_df.copy())
    val_with_age = add_age_encoding(val_df.copy())

    # Split by gender for age prediction
    train_male = train_with_age[train_with_age["gender_id"] == 0]
    train_female = train_with_age[train_with_age["gender_id"] == 1]
    val_male = val_with_age[val_with_age["gender_id"] == 0]
    val_female = val_with_age[val_with_age["gender_id"] == 1]

    # Save gender-specific age datasets
    train_male.to_csv(output_dir / "train_age_male.csv", index=False)
    train_female.to_csv(output_dir / "train_age_female.csv", index=False)
    val_male.to_csv(output_dir / "val_age_male.csv", index=False)
    val_female.to_csv(output_dir / "val_age_female.csv", index=False)

    logger.info(f"  Saved train_age_male.csv ({len(train_male)} samples)")
    logger.info(f"  Saved train_age_female.csv ({len(train_female)} samples)")
    logger.info(f"  Saved val_age_male.csv ({len(val_male)} samples)")
    logger.info(f"  Saved val_age_female.csv ({len(val_female)} samples)")


def print_split_statistics(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    """Print statistics for each split.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame.
    """
    logger.info("\n" + "=" * 60)
    logger.info("SPLIT STATISTICS")
    logger.info("=" * 60)

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        logger.info(f"\n{name} Set:")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  Age range: {df['age'].min()}-{df['age'].max()}")
        logger.info(f"  Mean age: {df['age'].mean():.1f}")
        logger.info("  Gender distribution:")
        for gender, count in df["gender"].value_counts().items():
            pct = count / len(df) * 100
            logger.info(f"    {gender}: {count} ({pct:.1f}%)")
        logger.info("  Race distribution:")
        for race, count in df["race"].value_counts().items():
            pct = count / len(df) * 100
            logger.info(f"    {race}: {count} ({pct:.1f}%)")


def main():
    """Main function to prepare dataset splits."""
    logger.info("=" * 60)
    logger.info("UTKFace Dataset Preparation for TensorFlow")
    logger.info("=" * 60)
    logger.info(f"Dataset path: {DATASET_PATH}")
    logger.info(f"Splits output directory: {SPLITS_DIR}")
    logger.info(f"Test size: {TEST_SIZE * 100:.0f}%")
    logger.info(f"Validation size: {VAL_SIZE * 100:.0f}%")
    logger.info(f"Random seed: {RANDOM_SEED}")

    try:
        # Load dataset
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Loading dataset metadata")
        logger.info("=" * 60)
        df = load_dataset_metadata(DATASET_PATH)

        if df.empty:
            logger.error("No valid images found in dataset!")
            return 1

        logger.info(f"Loaded {len(df)} valid samples")

        # Create splits
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Creating stratified splits")
        logger.info("=" * 60)
        train_df, val_df, test_df = create_splits(df, TEST_SIZE, VAL_SIZE, RANDOM_SEED)

        # Print statistics
        print_split_statistics(train_df, val_df, test_df)

        # Save splits
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Saving splits to CSV")
        logger.info("=" * 60)
        save_splits(train_df, val_df, test_df, SPLITS_DIR)

        logger.info("\n" + "=" * 60)
        logger.info("DATASET PREPARATION COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"\nSplit files saved to: {SPLITS_DIR}")
        logger.info("\nGenerated files:")
        logger.info("  - test.csv (with age encoding)")
        logger.info("  - train_gender.csv, val_gender.csv")
        logger.info("  - train_age_male.csv, train_age_female.csv")
        logger.info("  - val_age_male.csv, val_age_female.csv")
        logger.info(
            "\nYou can now load these with tf.data.Dataset for efficient training!"
        )

        return 0

    except FileNotFoundError as e:
        logger.error(f"Dataset directory not found: {e}")
        logger.error(f"Please ensure the UTKFace dataset is in: {DATASET_PATH}")
        return 1

    except Exception as e:
        logger.error(f"Unexpected error during preparation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
