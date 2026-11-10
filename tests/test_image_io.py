"""Tests for utils.image_io."""

import cv2
import numpy as np

from utils import image_io


def _sample_png_bytes():
    img = np.full((40, 60, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_decode_valid_png():
    img = image_io.decode_image(_sample_png_bytes())
    assert img is not None
    assert img.shape == (40, 60, 3)


def test_decode_garbage_returns_none():
    assert image_io.decode_image(b"not an image") is None


def test_decode_empty_returns_none():
    assert image_io.decode_image(b"") is None


def test_encode_roundtrip():
    img = np.full((20, 20, 3), 200, dtype=np.uint8)
    data = image_io.encode_image(img, ".png")
    assert data is not None and len(data) > 0
    decoded = image_io.decode_image(data)
    assert decoded is not None and decoded.shape == (20, 20, 3)


def test_bgr_to_rgb_swaps_channels():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # blue channel in BGR
    rgb = image_io.bgr_to_rgb(img)
    assert (rgb[:, :, 2] == 255).all()  # blue moves to the last channel in RGB


def test_grayscale_passthrough():
    gray = np.zeros((5, 5), dtype=np.uint8)
    assert image_io.bgr_to_rgb(gray).ndim == 2


def test_limit_resolution_downscales_large_images():
    big = np.zeros((6000, 8000, 3), dtype=np.uint8)
    out = image_io.limit_resolution(big, 3508)
    assert max(out.shape[:2]) == 3508
    assert abs(out.shape[1] / out.shape[0] - 8000 / 6000) < 0.01  # aspect kept


def test_limit_resolution_keeps_small_images_untouched():
    small = np.zeros((300, 200, 3), dtype=np.uint8)
    assert image_io.limit_resolution(small, 3508) is small


def test_image_dimensions():
    color = np.zeros((30, 50, 3), dtype=np.uint8)
    assert image_io.image_dimensions(color) == (50, 30, 3)
    gray = np.zeros((10, 20), dtype=np.uint8)
    assert image_io.image_dimensions(gray) == (20, 10, 1)
