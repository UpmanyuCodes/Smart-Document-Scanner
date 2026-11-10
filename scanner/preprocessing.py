"""Image preprocessing: grayscale, blur, and resizing for detection.

These operations prepare a working copy of the image for document detection
and other expensive steps. They never modify the input array (OpenCV returns
new arrays), so the caller's original stays intact.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale.

    If the image is already single-channel it is returned unchanged.
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _odd(kernel_size: int) -> int:
    """Gaussian/median kernels must be positive and odd."""
    k = max(1, int(kernel_size))
    return k if k % 2 == 1 else k + 1


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply a Gaussian blur to suppress high-frequency noise before edge
    detection. ``kernel_size`` is coerced to a valid odd value."""
    k = _odd(kernel_size)
    return cv2.GaussianBlur(image, (k, k), 0)


def reduce_initial_noise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Light denoising pass used before detection (Gaussian based)."""
    return gaussian_blur(image, kernel_size)


def resize_for_detection(
    image: np.ndarray, max_dimension: int = 1200
) -> tuple[np.ndarray, float]:
    """Downscale large images for faster, more stable detection.

    Returns ``(resized_image, scale)`` where ``scale`` maps the resized image
    back to the original: ``original_coord = resized_coord / scale``. Images
    already within ``max_dimension`` are returned unchanged with ``scale=1.0``.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return image, 1.0
    scale = max_dimension / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized, scale


def canny_edges(
    image: np.ndarray, low: int = 75, high: int = 200
) -> np.ndarray:
    """Canny edge map of a (grayscale or color) image, for detection and for
    the 'Processing Steps' demonstration view."""
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray, 5)
    return cv2.Canny(blurred, low, high)
