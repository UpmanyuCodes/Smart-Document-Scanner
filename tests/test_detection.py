"""Tests for scanner.detection."""

import cv2
import numpy as np

from scanner import detection


def _document_on_background():
    """Synthetic photo: a light 'page' rectangle on a dark background."""
    img = np.full((600, 800, 3), 30, dtype=np.uint8)      # dark desk
    cv2.rectangle(img, (150, 100), (650, 500), (235, 235, 235), -1)  # white page
    # a little text-like detail so it is not perfectly flat
    cv2.putText(img, "DOC", (300, 320), cv2.FONT_HERSHEY_SIMPLEX, 3, (20, 20, 20), 8)
    return img


def test_detects_clear_rectangle():
    result = detection.detect_document_corners(_document_on_background())
    assert result.success
    assert result.corners is not None
    assert result.corners.shape == (4, 2)
    # Corners should sit near the drawn rectangle bounds.
    xs, ys = result.corners[:, 0], result.corners[:, 1]
    assert xs.min() < 250 and xs.max() > 550
    assert ys.min() < 200 and ys.max() > 400


def test_blank_image_not_detected():
    blank = np.full((400, 400, 3), 127, dtype=np.uint8)
    result = detection.detect_document_corners(blank)
    assert not result.success
    assert result.corners is None
    assert result.message


def test_none_input_handled():
    result = detection.detect_document_corners(None)
    assert not result.success


def test_draw_corners_does_not_modify_original():
    img = _document_on_background()
    before = img.copy()
    result = detection.detect_document_corners(img)
    _ = detection.draw_corners(img, result.corners)
    assert np.array_equal(img, before)
