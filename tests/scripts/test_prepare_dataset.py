# Test script for dataset preparation
"""Tests for prepare_dataset.py main functionality.

Covers:
- parse_utk_filename  (canonical copy — same logic exists in all three scripts)
- encode_age_ldae
- create_splits
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the script as a module. Because it lives in scripts/ (not a package),
# we add the project root to sys.path so the import resolves cleanly.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.prepare_dataset import (
    create_splits,
    encode_age_ldae,
    parse_utk_filename,
)


# ---------------------------------------------------------------------------
# parse_utk_filename
# ---------------------------------------------------------------------------


class TestParseUtkFilename:
    # --- valid inputs ---

    def test_valid_jpg_filename(self):
        # A correctly formatted filename should be parsed into a dict with all
        # expected fields populated and matching the encoded values.
        result = parse_utk_filename("39_1_0_20170116174525125.jpg")
        assert result is not None
        assert result["age"] == 39
        assert result["gender"] == "Female"
        assert result["gender_id"] == 1
        assert result["race"] == "White"
        assert result["race_id"] == 0
        assert result["timestamp"] == "20170116174525125"

    def test_valid_chip_jpg_filename(self):
        # The .chip.jpg double-extension format used by some UTKFace subsets
        # should be handled identically to a plain .jpg filename.
        result = parse_utk_filename("25_0_2_20170117152001294.jpg.chip.jpg")
        assert result is not None
        assert result["age"] == 25
        assert result["gender"] == "Male"
        assert result["race"] == "Asian"

    def test_age_boundary_zero(self):
        # Age 0 is the minimum valid value in the UTKFace dataset and must be accepted.
        assert parse_utk_filename("0_0_0_20170101000000000.jpg") is not None

    def test_age_boundary_116(self):
        # Age 116 is the maximum valid value in the UTKFace dataset and must be accepted.
        assert parse_utk_filename("116_1_4_20170101000000000.jpg") is not None

    def test_all_valid_race_ids(self):
        # Race IDs 0–4 are all defined in the UTKFace schema; each should parse successfully.
        for race_id in range(5):
            result = parse_utk_filename(f"30_0_{race_id}_20170101000000000.jpg")
            assert result is not None, f"race_id {race_id} should be valid"

    def test_both_valid_gender_ids(self):
        # Gender IDs 0 (Male) and 1 (Female) are the only valid values and must both parse.
        for gender_id in range(2):
            result = parse_utk_filename(f"30_{gender_id}_0_20170101000000000.jpg")
            assert result is not None, f"gender_id {gender_id} should be valid"

    # --- invalid inputs ---

    def test_returns_none_for_too_few_parts(self):
        # A filename with fewer than 4 underscore-separated parts cannot be parsed.
        assert parse_utk_filename("39_1_20170116174525125.jpg") is None

    def test_returns_none_for_too_many_parts(self):
        # A filename with more than 4 underscore-separated parts is not a valid UTKFace name.
        assert parse_utk_filename("39_1_0_extra_20170116174525125.jpg") is None

    def test_returns_none_for_age_above_max(self):
        # Age 117 exceeds the dataset maximum of 116 and should be rejected.
        assert parse_utk_filename("117_0_0_20170101000000000.jpg") is None

    def test_returns_none_for_negative_age(self):
        # Negative ages are not physiologically valid and should be rejected.
        assert parse_utk_filename("-1_0_0_20170101000000000.jpg") is None

    def test_returns_none_for_invalid_gender_id(self):
        # Gender ID 2 is not defined in the UTKFace schema and should be rejected.
        assert parse_utk_filename("30_2_0_20170101000000000.jpg") is None

    def test_returns_none_for_invalid_race_id(self):
        # Race ID 5 is outside the valid range (0–4) and should be rejected.
        assert parse_utk_filename("30_0_5_20170101000000000.jpg") is None

    def test_returns_none_for_non_numeric_age(self):
        # A non-integer age field cannot be parsed and should return None gracefully.
        assert parse_utk_filename("abc_0_0_20170101000000000.jpg") is None

    def test_returns_none_for_completely_wrong_name(self):
        # An arbitrary filename with no UTKFace structure should return None without raising.
        assert parse_utk_filename("random_image.jpg") is None


# ---------------------------------------------------------------------------
# encode_age_ldae
# ---------------------------------------------------------------------------


class TestEncodeAgeLdae:
    def test_output_shape_is_117(self):
        # The distribution must cover all 117 age classes (0–116) to match the
        # model's output layer size.
        dist = encode_age_ldae(30)
        assert dist.shape == (117,)

    def test_distribution_sums_to_one(self):
        # As a probability distribution, values must sum to 1 for the loss
        # function (KL divergence) to be well-defined.
        for age in [0, 30, 60, 116]:
            dist = encode_age_ldae(age)
            assert dist.sum() == pytest.approx(1.0, abs=1e-5), f"age={age}"

    def test_peak_is_at_true_age(self):
        # The highest probability mass must sit at the true age so the label
        # distribution is centred on the correct target.
        for age in [0, 25, 50, 90, 116]:
            dist = encode_age_ldae(age)
            assert np.argmax(dist) == age, f"peak should be at age={age}"

    def test_distribution_is_wider_for_older_ages(self):
        # LDAE intentionally increases uncertainty with age. An older age should
        # produce a higher-variance distribution than a younger one.
        young_dist = encode_age_ldae(5)
        old_dist = encode_age_ldae(100)
        age_range = np.arange(117)
        var_young = np.sum(young_dist * (age_range - 5) ** 2)
        var_old = np.sum(old_dist * (age_range - 100) ** 2)
        assert var_old > var_young

    def test_all_values_are_non_negative(self):
        # Probability values must be non-negative; a Gaussian should never
        # produce negative values but this guards against implementation errors.
        dist = encode_age_ldae(50)
        assert np.all(dist >= 0)


# ---------------------------------------------------------------------------
# create_splits
# ---------------------------------------------------------------------------


def _make_dummy_df(n: int = 200) -> pd.DataFrame:
    """Create a minimal DataFrame that mirrors the real dataset structure."""
    rng = np.random.default_rng(0)
    ages = rng.integers(0, 116, size=n)
    genders = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "filename": [
                f"{a}_{g}_0_ts{i}.jpg" for i, (a, g) in enumerate(zip(ages, genders))
            ],
            "age": ages,
            "gender_id": genders,
            "gender": ["Male" if g == 0 else "Female" for g in genders],
            "race_id": rng.integers(0, 5, size=n),
            "race": ["White"] * n,
            "timestamp": ["20170101"] * n,
        }
    )


class TestCreateSplits:
    def test_splits_are_non_overlapping(self):
        # No image should appear in more than one split; data leakage between
        # train, val, and test would invalidate evaluation metrics.
        df = _make_dummy_df(200)
        train, val, test = create_splits(
            df, test_size=0.15, val_size=0.30, random_seed=42
        )
        train_files = set(train["filename"])
        val_files = set(val["filename"])
        test_files = set(test["filename"])
        assert train_files.isdisjoint(val_files)
        assert train_files.isdisjoint(test_files)
        assert val_files.isdisjoint(test_files)

    def test_splits_cover_all_samples(self):
        # Every sample must end up in exactly one split so no data is silently dropped.
        df = _make_dummy_df(200)
        train, val, test = create_splits(
            df, test_size=0.15, val_size=0.30, random_seed=42
        )
        assert len(train) + len(val) + len(test) == len(df)

    def test_test_size_is_approximately_correct(self):
        # The test set proportion should be close to the requested 15% to ensure
        # a representative held-out evaluation set.
        df = _make_dummy_df(200)
        _, _, test = create_splits(df, test_size=0.15, val_size=0.30, random_seed=42)
        ratio = len(test) / len(df)
        assert ratio == pytest.approx(0.15, abs=0.03)

    def test_splits_contain_both_genders(self):
        # Each split must include both genders so that gender-specific age models
        # can be evaluated fairly across all three sets.
        df = _make_dummy_df(200)
        train, val, test = create_splits(
            df, test_size=0.15, val_size=0.30, random_seed=42
        )
        for name, split in [("train", train), ("val", val), ("test", test)]:
            assert split["gender_id"].nunique() == 2, f"{name} split missing a gender"

    def test_temporary_columns_are_dropped(self):
        # Internal stratification columns must not leak into the output DataFrames
        # as they are not part of the dataset schema.
        df = _make_dummy_df(200)
        train, val, test = create_splits(
            df, test_size=0.15, val_size=0.30, random_seed=42
        )
        for split in (train, val, test):
            assert "age_bin" not in split.columns
            assert "stratify_col" not in split.columns
