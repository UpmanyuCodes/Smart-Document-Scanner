"""Tests for scanner.ocr.

These do not require the Tesseract engine to be installed: they verify the
graceful-degradation paths and that results are well-formed either way.
"""

import numpy as np

from scanner import ocr


def _text_image():
    import cv2
    img = np.full((120, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Hello 123", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 0), 4)
    return img


def test_is_available_returns_tuple():
    available, info = ocr.is_available()
    assert isinstance(available, bool)
    assert isinstance(info, str) and info


def test_available_languages_is_list():
    assert isinstance(ocr.available_languages(), list)


def test_extract_text_none_image():
    result = ocr.extract_text(None)
    assert not result.success
    assert result.message


def test_extract_text_result_shape():
    # Whether or not Tesseract is installed, we get a structured OcrResult.
    result = ocr.extract_text(_text_image())
    assert isinstance(result.success, bool)
    assert isinstance(result.text, str)
    assert isinstance(result.message, str)
    if not result.success:
        # Missing engine should produce an actionable setup message.
        assert "Tesseract" in result.message or "language" in result.message.lower()


def test_confidence_path_matches_availability():
    result = ocr.extract_text_with_confidence(_text_image())
    # mean_confidence is only set when text was actually extracted.
    if result.success and result.text:
        assert result.mean_confidence is None or 0 <= result.mean_confidence <= 100
