"""Tests for scanner.pipeline (per-page edit rendering)."""

import numpy as np

from scanner import pipeline
from scanner.pipeline import EditSettings


def _page():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (200, 150, 3), dtype=np.uint8)


def test_default_render_returns_color_image():
    img = _page()
    out = pipeline.render_page(img, EditSettings())
    assert out.shape == img.shape


def test_bw_filter_returns_binary():
    out = pipeline.render_page(_page(), EditSettings(filter_mode=pipeline.FILTER_BW))
    assert out.ndim == 2
    assert set(np.unique(out)).issubset({0, 255})


def test_grayscale_filter_single_channel():
    out = pipeline.render_page(
        _page(), EditSettings(filter_mode=pipeline.FILTER_GRAYSCALE)
    )
    assert out.ndim == 2


def test_rotate_90_swaps_dimensions():
    img = _page()  # 200x150
    out = pipeline.render_page(img, EditSettings(rotate=90))
    assert out.shape[0] == img.shape[1] and out.shape[1] == img.shape[0]


def test_crop_uses_corners():
    img = _page()
    corners = np.array([[10, 10], [140, 15], [138, 185], [12, 180]], dtype="float32")
    out = pipeline.render_page(img, EditSettings(crop=True), corners=corners)
    assert out.shape[0] > 10 and out.shape[1] > 10


def test_preview_downscales():
    img = np.full((1000, 800, 3), 128, dtype=np.uint8)
    out = pipeline.render_page(img, EditSettings(), preview_max=200)
    assert max(out.shape[:2]) <= 200


def test_original_not_modified():
    img = _page()
    before = img.copy()
    pipeline.render_page(img, EditSettings(brightness=40, contrast=1.5, sharpen=1.0))
    assert np.array_equal(img, before)


def test_settings_copy_is_independent():
    s = EditSettings(brightness=10)
    c = s.copy()
    c.brightness = 99
    assert s.brightness == 10
