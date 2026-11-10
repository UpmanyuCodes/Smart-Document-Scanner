"""Tests for scanner.filters and scanner.enhancement."""

import numpy as np

from scanner import enhancement, filters


def _color():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)


def test_grayscale_is_single_channel():
    gray = filters.convert_to_grayscale(_color())
    assert gray.ndim == 2
    assert gray.shape == (80, 100)


def test_global_threshold_is_binary():
    binary = filters.global_threshold(_color(), 127)
    assert binary.ndim == 2
    assert set(np.unique(binary)).issubset({0, 255})


def test_adaptive_threshold_is_binary_and_same_size():
    img = _color()
    binary = filters.adaptive_threshold(img, block_size=20, c_value=5)  # even -> coerced
    assert binary.shape == img.shape[:2]
    assert set(np.unique(binary)).issubset({0, 255})


def test_high_contrast_single_channel():
    hc = filters.create_high_contrast(_color())
    assert hc.ndim == 2


def test_brightness_contrast_preserves_shape():
    img = _color()
    out = enhancement.adjust_brightness_contrast(img, brightness=30, contrast=1.2)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_enhance_contrast_color_stays_color():
    img = _color()
    out = enhancement.enhance_contrast(img)
    assert out.shape == img.shape


def test_sharpen_preserves_shape():
    img = _color()
    out = enhancement.sharpen_image(img, strength=1.0)
    assert out.shape == img.shape


def test_sharpen_zero_strength_noop_shape():
    img = _color()
    out = enhancement.sharpen_image(img, strength=0)
    assert out.shape == img.shape
