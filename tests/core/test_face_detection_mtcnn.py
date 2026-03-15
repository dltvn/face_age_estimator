"""Tests for FaceDetectorMTCNN.detect_and_align — the public API.

MTCNN is mocked throughout so that no TensorFlow model is loaded during
testing. The mock is applied at the point of use inside
face_detection_mtcnn.py so that the real class is never instantiated.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BLANK_BGR = np.zeros((200, 200, 3), dtype=np.uint8)


def _make_mtcnn_detection(
    x: int = 10,
    y: int = 20,
    w: int = 50,
    h: int = 60,
    confidence: float = 0.95,
) -> dict:
    """Return a dict shaped like an mtcnn.detect_faces() result."""
    cx = x + w // 2
    cy = y + h // 2
    return {
        "box": [x, y, w, h],
        "confidence": confidence,
        "keypoints": {
            "left_eye": (cx - 10, cy - 10),
            "right_eye": (cx + 10, cy - 10),
            "nose": (cx, cy),
            "mouth_left": (cx - 8, cy + 10),
            "mouth_right": (cx + 8, cy + 10),
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_mtcnn():
    """Patch MTCNN inside face_detection_mtcnn so no TF model is loaded."""
    with patch("src.core.face_detection_mtcnn.MTCNN") as MockClass:
        instance = MagicMock()
        MockClass.return_value = instance
        yield instance  # callers set instance.detect_faces.return_value


@pytest.fixture
def detector(mock_mtcnn):
    """FaceDetectorMTCNN with MTCNN replaced by a mock."""
    from src.core.face_detection_mtcnn import FaceDetectorMTCNN

    return FaceDetectorMTCNN()


# ---------------------------------------------------------------------------
# FaceDetectorMTCNN.detect_and_align
# ---------------------------------------------------------------------------


class TestDetectAndAlign:
    def test_raises_on_non_3channel_input(self, detector):
        # A grayscale (2-D) array is not a valid BGR image; a ValueError should be raised.
        gray = np.zeros((200, 200), dtype=np.uint8)
        with pytest.raises(ValueError):
            detector.detect_and_align(gray)

    def test_rgb_is_passed_to_mtcnn_not_bgr(self, detector, mock_mtcnn):
        # MTCNN requires RGB input. Verify the BGR→RGB conversion happens before
        # the array is forwarded to the underlying detector.
        import cv2

        bgr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        mock_mtcnn.detect_faces.return_value = []
        detector.detect_and_align(bgr)

        received = mock_mtcnn.detect_faces.call_args[0][0]
        expected_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(received, expected_rgb)

    def test_returns_no_selected_when_no_faces(self, detector, mock_mtcnn):
        # When MTCNN finds nothing, the result should contain an empty face list
        # and selected=None so callers can safely check for a result.
        mock_mtcnn.detect_faces.return_value = []
        result = detector.detect_and_align(_BLANK_BGR)
        assert result["selected"] is None
        assert result["faces"] == []

    def test_result_has_required_keys_when_face_found(self, detector, mock_mtcnn):
        # The returned dict must always contain these four keys so that downstream
        # code (preprocessing, inference) can rely on a consistent interface.
        mock_mtcnn.detect_faces.return_value = [_make_mtcnn_detection()]
        result = detector.detect_and_align(_BLANK_BGR)
        assert "faces" in result
        assert "selected" in result
        assert "affine" in result
        assert "aligned_bgr" in result

    def test_aligned_bgr_has_correct_output_size(self, detector, mock_mtcnn):
        # The aligned crop dimensions must match the requested output_size so the
        # image is ready for the model's expected input resolution.
        mock_mtcnn.detect_faces.return_value = [_make_mtcnn_detection()]
        result = detector.detect_and_align(_BLANK_BGR, output_size=(96, 96))
        # warpAffine with dsize=(w, h) produces array of shape (h, w, c)
        h, w = result["aligned_bgr"].shape[:2]
        assert (w, h) == (96, 96)

    def test_affine_matrix_shape(self, detector, mock_mtcnn):
        # The affine matrix must be 2×3 to be usable with cv2.warpAffine.
        mock_mtcnn.detect_faces.return_value = [_make_mtcnn_detection()]
        result = detector.detect_and_align(_BLANK_BGR)
        assert result["affine"].shape == (2, 3)

    def test_pick_best_selects_highest_confidence(self, detector, mock_mtcnn):
        # With pick="best", the face with the highest confidence score should be
        # selected regardless of its position or size in the image.
        low = _make_mtcnn_detection(x=0, y=0, w=30, h=30, confidence=0.70)
        high = _make_mtcnn_detection(x=50, y=50, w=30, h=30, confidence=0.95)
        mock_mtcnn.detect_faces.return_value = [low, high]
        result = detector.detect_and_align(_BLANK_BGR, min_prob=0.0, pick="best")
        assert result["selected"].prob == pytest.approx(0.95)

    def test_pick_largest_selects_by_area(self, detector, mock_mtcnn):
        # With pick="largest", the face with the biggest bounding-box area should
        # be selected even when a smaller face has higher confidence.
        small = _make_mtcnn_detection(x=0, y=0, w=20, h=20, confidence=0.99)
        large = _make_mtcnn_detection(x=50, y=50, w=80, h=80, confidence=0.80)
        mock_mtcnn.detect_faces.return_value = [small, large]
        # min_prob=0.0 so both faces are candidates; only then does pick="largest" matter
        result = detector.detect_and_align(_BLANK_BGR, pick="largest", min_prob=0.0)
        x1, y1, x2, y2 = result["selected"].box
        assert (x2 - x1) * (y2 - y1) == pytest.approx(80.0 * 80.0)

    def test_min_prob_filters_low_confidence_faces(self, detector, mock_mtcnn):
        # Detections below min_prob should be excluded from selection, ensuring
        # the chosen face meets a minimum quality bar.
        low = _make_mtcnn_detection(x=0, y=0, w=30, h=30, confidence=0.50)
        high = _make_mtcnn_detection(x=50, y=50, w=30, h=30, confidence=0.95)
        mock_mtcnn.detect_faces.return_value = [low, high]
        result = detector.detect_and_align(_BLANK_BGR, min_prob=0.90, pick="best")
        assert result["selected"].prob == pytest.approx(0.95)

    def test_min_prob_fallback_logs_warning_and_still_returns_face(
        self, detector, mock_mtcnn, caplog
    ):
        # When no detection meets min_prob, the method should fall back to all
        # detections (rather than returning None) and log a warning so the caller
        # is aware the threshold was not met.
        mock_mtcnn.detect_faces.return_value = [_make_mtcnn_detection(confidence=0.60)]
        with caplog.at_level(logging.WARNING, logger="src.core.face_detection_mtcnn"):
            result = detector.detect_and_align(_BLANK_BGR, min_prob=0.90)

        assert result["selected"] is not None
        assert any("min_prob" in record.message for record in caplog.records)
