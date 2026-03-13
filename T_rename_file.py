
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.rename_chip_files import rename_chip_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CHIP = "25_0_1_20170117152001294.jpg.chip.jpg"
_CHIP2 = "39_1_0_20170116174525125.jpg.chip.jpg"
_PLAIN = "10_0_2_20170101000000000.jpg"  # not a chip file, should be untouched


# ---------------------------------------------------------------------------
# rename_chip_files
# ---------------------------------------------------------------------------


class TestRenameChipFiles:
    def test_dry_run_does_not_rename(self, tmp_path):
        # In dry-run mode the original .chip.jpg file must remain on disk unchanged.
        (tmp_path / _CHIP).touch()
        rename_chip_files(tmp_path, dry_run=True)
        assert (tmp_path / _CHIP).exists()

    def test_dry_run_reports_correct_count(self, tmp_path):
        # Even without making changes, the returned stats must reflect how many
        # files would have been renamed.
        (tmp_path / _CHIP).touch()
        (tmp_path / _CHIP2).touch()
        stats = rename_chip_files(tmp_path, dry_run=True)
        assert stats["renamed_count"] == 2
        assert stats["error_count"] == 0

    def test_live_run_renames_chip_file(self, tmp_path):
        # A live run must strip the .chip.jpg suffix, producing a plain .jpg file
        # and removing the original double-extension file.
        (tmp_path / _CHIP).touch()
        rename_chip_files(tmp_path, dry_run=False)
        expected = "25_0_1_20170117152001294.jpg"
        assert (tmp_path / expected).exists()
        assert not (tmp_path / _CHIP).exists()

    def test_live_run_does_not_touch_plain_jpg(self, tmp_path):
        # Files that are already plain .jpg should not be modified or deleted.
        (tmp_path / _PLAIN).touch()
        rename_chip_files(tmp_path, dry_run=False)
        assert (tmp_path / _PLAIN).exists()

    def test_skips_when_target_already_exists(self, tmp_path):
        # If the target .jpg filename already exists, the rename must be skipped
        # to avoid overwriting data, and the original chip file must be preserved.
        chip_path = tmp_path / _CHIP
        chip_path.touch()
        target = "25_0_1_20170117152001294.jpg"
        (tmp_path / target).touch()  # already present
        stats = rename_chip_files(tmp_path, dry_run=False)
        assert stats["skipped_count"] == 1
        assert chip_path.exists()

    def test_empty_directory_returns_zero_counts(self, tmp_path):
        # Running against an empty directory should succeed quietly with all
        # counters at zero.
        stats = rename_chip_files(tmp_path, dry_run=False)
        assert stats["renamed_count"] == 0
        assert stats["skipped_count"] == 0
        assert stats["error_count"] == 0

    def test_raises_on_nonexistent_directory(self, tmp_path):
        # Passing a path that does not exist must raise FileNotFoundError immediately.
        with pytest.raises(FileNotFoundError):
            rename_chip_files(tmp_path / "does_not_exist")
