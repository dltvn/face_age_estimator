"""Tests for delete_invalid_files.py main functionality.

Covers:
- identify_invalid_files
- delete_invalid_files (dry_run and live, with and without backup)

Uses pytest's tmp_path fixture so no real dataset is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.delete_invalid_files import delete_invalid_files, identify_invalid_files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _plant(directory: Path, *names: str) -> None:
    """Create empty files with the given names inside directory."""
    for name in names:
        (directory / name).touch()


_VALID = "25_0_1_20170117152001294.jpg"
_VALID2 = "39_1_0_20170116174525125.jpg"
_INVALID = "not_a_valid_name.jpg"
_INVALID2 = "badname.jpg"


# ---------------------------------------------------------------------------
# identify_invalid_files
# ---------------------------------------------------------------------------


class TestIdentifyInvalidFiles:
    def test_all_valid_files_classified_correctly(self, tmp_path):
        # A directory containing only correctly named UTKFace files should
        # produce an empty invalid list and a full valid list.
        _plant(tmp_path, _VALID, _VALID2)
        valid, invalid = identify_invalid_files(tmp_path)
        assert len(valid) == 2
        assert len(invalid) == 0

    def test_all_invalid_files_classified_correctly(self, tmp_path):
        # A directory with no parseable filenames should return an empty valid
        # list and place every file in the invalid list.
        _plant(tmp_path, _INVALID, _INVALID2)
        valid, invalid = identify_invalid_files(tmp_path)
        assert len(valid) == 0
        assert len(invalid) == 2

    def test_mixed_files_split_correctly(self, tmp_path):
        # Valid and invalid files must be classified independently; one of each
        # should produce lists of length 1 each.
        _plant(tmp_path, _VALID, _INVALID)
        valid, invalid = identify_invalid_files(tmp_path)
        assert len(valid) == 1
        assert len(invalid) == 1

    def test_empty_directory_returns_empty_lists(self, tmp_path):
        # An empty dataset directory should not raise and should return two empty lists.
        valid, invalid = identify_invalid_files(tmp_path)
        assert valid == []
        assert invalid == []

    def test_raises_on_nonexistent_directory(self, tmp_path):
        # Passing a path that does not exist must raise FileNotFoundError immediately.
        with pytest.raises(FileNotFoundError):
            identify_invalid_files(tmp_path / "does_not_exist")

    def test_non_image_files_are_ignored(self, tmp_path):
        # Non-image files (.txt, .csv) are not part of the dataset and must not
        # appear in either the valid or invalid list.
        _plant(tmp_path, "readme.txt", "data.csv", _VALID)
        valid, invalid = identify_invalid_files(tmp_path)
        assert len(valid) == 1
        assert len(invalid) == 0


# ---------------------------------------------------------------------------
# delete_invalid_files
# ---------------------------------------------------------------------------


class TestDeleteInvalidFiles:
    def test_dry_run_does_not_delete_files(self, tmp_path):
        # In dry-run mode no files should be removed; the invalid file must
        # still exist on disk after the call.
        _plant(tmp_path, _VALID, _INVALID)
        delete_invalid_files(tmp_path, dry_run=True)
        assert (tmp_path / _INVALID).exists()

    def test_dry_run_reports_correct_counts(self, tmp_path):
        # Even in dry-run mode the returned stats must accurately reflect how
        # many files would have been processed.
        _plant(tmp_path, _VALID, _INVALID, _INVALID2)
        stats = delete_invalid_files(tmp_path, dry_run=True)
        assert stats["valid_count"] == 1
        assert stats["deleted_count"] == 2
        assert stats["error_count"] == 0

    def test_live_run_deletes_invalid_files(self, tmp_path):
        # A live run must remove files that fail filename validation from disk.
        _plant(tmp_path, _VALID, _INVALID)
        delete_invalid_files(tmp_path, dry_run=False, backup=False)
        assert not (tmp_path / _INVALID).exists()

    def test_live_run_preserves_valid_files(self, tmp_path):
        # A live run must never delete files with valid UTKFace filenames.
        _plant(tmp_path, _VALID, _INVALID)
        delete_invalid_files(tmp_path, dry_run=False, backup=False)
        assert (tmp_path / _VALID).exists()

    def test_backup_moves_files_instead_of_deleting(self, tmp_path):
        # With backup=True, invalid files must be moved to the backup directory
        # rather than permanently deleted, allowing recovery if needed.
        backup_dir = tmp_path / "backup"
        _plant(tmp_path, _VALID, _INVALID)
        delete_invalid_files(
            tmp_path, dry_run=False, backup=True, backup_dir=backup_dir
        )
        assert not (tmp_path / _INVALID).exists()
        assert (backup_dir / _INVALID).exists()

    def test_no_invalid_files_returns_zero_deleted(self, tmp_path):
        # When all files are valid, the deleted and error counts should both be
        # zero to confirm nothing was touched unnecessarily.
        _plant(tmp_path, _VALID, _VALID2)
        stats = delete_invalid_files(tmp_path, dry_run=False, backup=False)
        assert stats["deleted_count"] == 0
        assert stats["error_count"] == 0
