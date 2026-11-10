"""Tests for utils.document (multi-page assembly)."""

import numpy as np

from utils import document


def _color_page(value=180, size=(60, 40)):
    h, w = size
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_pdf_from_single_page():
    data = document.images_to_pdf_bytes([_color_page()])
    assert data is not None
    assert data[:4] == b"%PDF"


def test_pdf_from_multiple_pages():
    pages = [_color_page(100), _color_page(200), _color_page(50)]
    data = document.images_to_pdf_bytes(pages)
    assert data is not None
    assert data[:4] == b"%PDF"


def test_pdf_handles_grayscale_page():
    gray = np.full((40, 30), 120, dtype=np.uint8)
    data = document.images_to_pdf_bytes([gray])
    assert data is not None and data[:4] == b"%PDF"


def test_pdf_empty_returns_none():
    assert document.images_to_pdf_bytes([]) is None


def test_zip_from_pages():
    data = document.images_to_zip_bytes([_color_page(), _color_page(90)])
    assert data is not None
    assert data[:2] == b"PK"  # ZIP local file header signature


def test_zip_empty_returns_none():
    assert document.images_to_zip_bytes([]) is None
