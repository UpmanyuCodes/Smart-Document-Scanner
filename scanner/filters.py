"""Scanner-style document filters.

Produce the familiar "scan" looks: grayscale, black & white (global and
adaptive thresholding), and high-contrast. Inputs may be color or grayscale;
outputs are new arrays.
"""

from __future__ import annotations

import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Single-channel grayscale (idempotent for already-gray images)."""
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def global_threshold(image: np.ndarray, threshold_value: int = 127) -> np.ndarray:
    """Global binary threshold. ``threshold_value`` in 0..255."""
    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(
        gray, int(threshold_value), 255, cv2.THRESH_BINARY
    )
    return binary


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Automatic global threshold via Otsu's method."""
    gray = convert_to_grayscale(image)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary


def adaptive_threshold(
    image: np.ndarray, block_size: int = 21, c_value: int = 10
) -> np.ndarray:
    """Adaptive Gaussian threshold — the classic clean 'B&W scan' look.

    Handles uneven lighting far better than a global threshold. ``block_size``
    must be odd and >= 3; it is coerced if needed.
    """
    gray = convert_to_grayscale(image)
    block = max(3, int(block_size))
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block, int(c_value),
    )


def create_high_contrast(image: np.ndarray) -> np.ndarray:
    """Grayscale with histogram equalization for a punchy, readable look."""
    gray = convert_to_grayscale(image)
    return cv2.equalizeHist(gray)
