from __future__ import annotations

import numpy as np
import pytest

from src.core import face_detection_retinaface as retinaface


def test_run_detector_rejects_non_rgb_image_shape() -> None:
    """Test detector rejects non-3-channel images before model inference."""
    gray = np.zeros((32, 32), dtype=np.uint8)

    with pytest.raises(ValueError, match="3-channel BGR image"):
        retinaface._run_detector(gray)


def test_detect_faces_filters_below_min_score(monkeypatch) -> None:
    """Test score threshold removes low-confidence detections."""
    monkeypatch.setattr(
        retinaface,
        "_run_detector",
        lambda image_bgr: [
            {"score": 0.40, "bbox": [0, 0, 10, 10], "landmarks": [[1, 1]] * 5},
            {"score": 0.95, "bbox": [0, 0, 20, 20], "landmarks": [[2, 2]] * 5},
        ],
    )

    detections = retinaface.detect_faces(
        np.zeros((10, 10, 3), dtype=np.uint8), min_score=0.8
    )

    assert len(detections) == 1
    assert detections[0]["score"] == 0.95


def test_detect_and_align_returns_none_and_info_when_no_faces(monkeypatch) -> None:
    """Test empty detections return None with stable metadata."""
    monkeypatch.setattr(retinaface, "detect_faces", lambda image_bgr, min_score: [])

    aligned, info = retinaface.detect_and_align(
        np.zeros((20, 20, 3), dtype=np.uint8),
        return_info=True,
    )

    assert aligned is None
    assert info == {"detections": [], "selected": None}


def test_detect_and_align_pick_best_uses_highest_score(monkeypatch) -> None:
    """Test pick='best' selects the highest-confidence detection."""
    low = {"score": 0.60, "bbox": [0, 0, 50, 50], "landmarks": [[20, 20]] * 5}
    high = {"score": 0.95, "bbox": [10, 10, 20, 20], "landmarks": [[14, 14]] * 5}
    monkeypatch.setattr(
        retinaface,
        "detect_faces",
        lambda image_bgr, min_score: [low, high],
    )
    monkeypatch.setattr(
        retinaface,
        "_square_crop_with_margin",
        lambda image_bgr, box_xyxy, margin: (
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.array([10.0, 10.0], dtype=np.float32),
        ),
    )
    captured = {}

    def fake_align_face(crop_bgr, landmarks_in_crop, output_size):
        captured["landmarks_in_crop"] = landmarks_in_crop
        return np.ones((output_size[1], output_size[0], 3), dtype=np.uint8)

    monkeypatch.setattr(retinaface, "_align_face", fake_align_face)

    aligned, info = retinaface.detect_and_align(
        np.zeros((60, 60, 3), dtype=np.uint8),
        pick="best",
        return_info=True,
    )

    assert aligned.shape == (224, 224, 3)
    assert info["selected"] is high
    np.testing.assert_array_equal(
        captured["landmarks_in_crop"], np.array([[4, 4]] * 5, dtype=np.float32)
    )


def test_detect_and_align_pick_largest_prefers_bbox_area(monkeypatch) -> None:
    """Test default selection uses largest bounding box area."""
    small_high_score = {
        "score": 0.99,
        "bbox": [0, 0, 10, 10],
        "landmarks": [[2, 2]] * 5,
    }
    large_lower_score = {
        "score": 0.80,
        "bbox": [0, 0, 40, 40],
        "landmarks": [[5, 5]] * 5,
    }
    monkeypatch.setattr(
        retinaface,
        "detect_faces",
        lambda image_bgr, min_score: [small_high_score, large_lower_score],
    )
    monkeypatch.setattr(
        retinaface,
        "_square_crop_with_margin",
        lambda image_bgr, box_xyxy, margin: (
            np.zeros((6, 6, 3), dtype=np.uint8),
            np.array([0.0, 0.0], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        retinaface,
        "_align_face",
        lambda crop_bgr, landmarks_in_crop, output_size: np.zeros(
            (output_size[1], output_size[0], 3),
            dtype=np.uint8,
        ),
    )

    _, info = retinaface.detect_and_align(
        np.zeros((60, 60, 3), dtype=np.uint8),
        return_info=True,
    )

    assert info["selected"] is large_lower_score
