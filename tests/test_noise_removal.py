"""Tests for scanner.noise_removal and the related pipeline settings."""

import cv2
import numpy as np

from scanner import noise_removal, pipeline
from scanner.pipeline import EditSettings


def _shadowed_page():
    """White page with dark text and a strong left-to-right lighting gradient."""
    h, w = 300, 400
    gradient = np.linspace(90, 250, w, dtype=np.float32)
    page = np.tile(gradient, (h, 1)).astype(np.uint8)
    cv2.putText(page, "TEXT", (60, 170), cv2.FONT_HERSHEY_SIMPLEX, 3, 20, 8)
    return cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)


def test_median_and_gaussian_preserve_shape():
    img = _shadowed_page()
    assert noise_removal.median_denoise(img, 4).shape == img.shape  # even k coerced
    assert noise_removal.gaussian_denoise(img, 5).shape == img.shape


def test_denoise_level_zero_is_identity():
    img = _shadowed_page()
    assert noise_removal.denoise(img, 0) is img


def test_median_removes_salt_noise():
    img = np.full((50, 50), 128, dtype=np.uint8)
    img[10, 10] = 255
    img[30, 40] = 0
    out = noise_removal.median_denoise(img, 3)
    assert out[10, 10] == 128 and out[30, 40] == 128


def test_reduce_shadows_evens_background():
    img = _shadowed_page()
    gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(noise_removal.reduce_shadows(img), cv2.COLOR_BGR2GRAY)
    # Compare paper brightness at the dark (left) and bright (right) edges.
    spread_before = abs(int(gray_before[20, 380]) - int(gray_before[20, 20]))
    spread_after = abs(int(gray_after[20, 380]) - int(gray_after[20, 20]))
    assert spread_after < spread_before / 3
    # Text must survive: the text area should still contain dark pixels.
    assert gray_after[120:180, 60:340].min() < 100


def test_reduce_shadows_grayscale_input():
    gray = cv2.cvtColor(_shadowed_page(), cv2.COLOR_BGR2GRAY)
    assert noise_removal.reduce_shadows(gray).shape == gray.shape


def test_manual_margins_crop():
    img = np.zeros((200, 100, 3), dtype=np.uint8)
    s = EditSettings(margin_left=10, margin_right=10, margin_top=25, margin_bottom=25)
    out = pipeline.render_page(img, s)
    assert out.shape[:2] == (100, 80)


def test_max_margins_still_leave_content():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    s = EditSettings(margin_left=40, margin_right=40, margin_top=40, margin_bottom=40)
    assert pipeline.render_page(img, s).shape[:2] == (20, 20)


def test_out_of_range_margins_are_clamped():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    out = pipeline.apply_margins(img, EditSettings(margin_left=60, margin_right=60))
    assert out.shape[:2] == (100, 20)  # each side clamped to 40%


def test_settings_key_changes_with_values():
    a, b = EditSettings(), EditSettings(denoise=2)
    assert a.key() != b.key()
    assert a.key() == EditSettings().key()
