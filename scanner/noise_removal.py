"""Noise reduction and shadow (uneven illumination) removal.

Noise reduction and shadow correction are separate operations so either can
be switched off independently. Kernel sizes for shadow removal scale with the
image so the preview and the full-resolution export look the same.
"""

from __future__ import annotations

import cv2
import numpy as np


def _odd(value: int, minimum: int = 3) -> int:
    k = max(minimum, int(value))
    return k if k % 2 == 1 else k + 1


def median_denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Median filter: removes salt-and-pepper speckle while keeping edges."""
    return cv2.medianBlur(image, _odd(kernel_size))


def gaussian_denoise(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Gaussian filter: smooths fine sensor grain."""
    k = _odd(kernel_size)
    return cv2.GaussianBlur(image, (k, k), 0)


def denoise(image: np.ndarray, level: int = 1) -> np.ndarray:
    """Preset noise reduction. ``level`` 0 = off, 1 = light, 2 = strong.

    Light uses a 3x3 median; strong adds a gentle Gaussian pass. Both are
    kept small so thin text strokes survive.
    """
    if level <= 0:
        return image
    out = median_denoise(image, 3)
    if level >= 2:
        out = gaussian_denoise(out, 3)
    return out


def _flatten_channel(channel: np.ndarray, dilate_k: int, blur_k: int) -> np.ndarray:
    # Dilation erases dark text so only the paper/background remains, then a
    # large median blur gives a smooth illumination estimate.
    background = cv2.dilate(channel, np.ones((dilate_k, dilate_k), np.uint8))
    background = cv2.medianBlur(background, blur_k)
    background = np.maximum(background, 1)  # avoid division by zero
    return cv2.divide(channel, background, scale=255)


def reduce_shadows(image: np.ndarray) -> np.ndarray:
    """Even out lighting by dividing the image by its estimated background.

    Works per channel for colour images so paper turns uniformly white while
    ink colour is preserved. Kernel sizes scale with resolution.
    """
    h, w = image.shape[:2]
    short_side = min(h, w)
    dilate_k = _odd(short_side / 120, 7)
    blur_k = min(_odd(short_side / 25, 21), 255)

    if image.ndim == 2:
        return _flatten_channel(image, dilate_k, blur_k)
    channels = [_flatten_channel(c, dilate_k, blur_k) for c in cv2.split(image)]
    return cv2.merge(channels)
