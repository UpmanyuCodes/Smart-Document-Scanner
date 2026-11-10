"""Tests for utils.validation."""

import numpy as np

from utils import validation


def test_valid_extensions_accepted():
    for name in ("photo.jpg", "scan.JPEG", "doc.png"):
        assert validation.validate_extension(name).ok


def test_unsupported_extension_rejected():
    result = validation.validate_extension("document.pdf")
    assert not result.ok
    assert "pdf" in result.message.lower()


def test_missing_extension_rejected():
    assert not validation.validate_extension("noextension").ok


def test_file_size_limits():
    assert validation.validate_file_size(1024).ok
    assert not validation.validate_file_size(0).ok
    assert not validation.validate_file_size(validation.MAX_FILE_BYTES + 1).ok


def test_valid_image_array():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    assert validation.validate_image_array(img).ok


def test_none_image_rejected():
    assert not validation.validate_image_array(None).ok


def test_too_small_image_rejected():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    assert not validation.validate_image_array(img).ok


def test_very_large_image_warns_but_ok():
    img = np.zeros((50, validation.MAX_DIMENSION + 100, 3), dtype=np.uint8)
    result = validation.validate_image_array(img)
    assert result.ok
    assert result.warning


def test_clamp():
    assert validation.clamp(5, 0, 10) == 5
    assert validation.clamp(-3, 0, 10) == 0
    assert validation.clamp(15, 0, 10) == 10
