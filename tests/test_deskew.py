"""Tests for scanner.deskew."""

import cv2
import numpy as np

from scanner import deskew


def _lined_page(angle=0.0):
    """White page with long horizontal rules, optionally rotated."""
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    for y in range(80, 560, 40):
        cv2.line(img, (60, y), (740, y), (0, 0, 0), 3)
    if angle:
        m = cv2.getRotationMatrix2D((400, 300), angle, 1.0)
        img = cv2.warpAffine(img, m, (800, 600), borderValue=(255, 255, 255))
    return img


def test_estimates_known_tilt():
    angle = deskew.estimate_skew_angle(_lined_page(5.0))
    assert angle is not None
    assert abs(abs(angle) - 5.0) < 1.0


def test_aligned_page_not_rotated():
    img = _lined_page(0.0)
    out, applied = deskew.correct_skew(img)
    assert applied == 0.0
    assert out.shape == img.shape


def test_tilted_page_gets_corrected():
    out, applied = deskew.correct_skew(_lined_page(6.0))
    assert abs(applied) > 4.0
    residual = deskew.estimate_skew_angle(out)
    assert residual is None or abs(residual) < 1.0


def test_blank_page_returns_no_estimate():
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)
    assert deskew.estimate_skew_angle(blank) is None
    out, applied = deskew.correct_skew(blank)
    assert applied == 0.0


def test_rotation_expands_canvas():
    img = _lined_page(0.0)
    rotated = deskew.rotate_image(img, 10)
    assert rotated.shape[0] > img.shape[0] and rotated.shape[1] > img.shape[1]
