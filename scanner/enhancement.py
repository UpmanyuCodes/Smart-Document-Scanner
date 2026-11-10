"""Image enhancement: brightness/contrast, CLAHE, and sharpening.

Each operation is a standalone function returning a new image. The input is
never modified. Functions accept both color (BGR) and grayscale images.
"""

from __future__ import annotations

import cv2
import numpy as np


def adjust_brightness_contrast(
    image: np.ndarray, brightness: int = 0, contrast: float = 1.0
) -> np.ndarray:
    """Linear brightness/contrast: ``out = contrast * image + brightness``.

    ``brightness`` is an additive offset in the -100..100 range (approx);
    ``contrast`` is a multiplicative gain (1.0 = unchanged). Values are
    clamped to a valid 8-bit range.
    """
    if brightness == 0 and contrast == 1.0:
        return image.copy()
    return cv2.convertScaleAbs(image, alpha=float(contrast), beta=int(brightness))


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Adaptive contrast enhancement.

    For color images, CLAHE is applied to the L (lightness) channel in LAB
    space so colours are not distorted. For grayscale, CLAHE is applied
    directly. This is the 'magic / auto' look.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    if image.ndim == 2:
        return clahe.apply(image)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Unsharp-mask sharpening. ``strength`` 0 = no change.

    Blends the image with its blurred version: a moderate, artifact-light
    sharpen rather than a harsh convolution kernel.
    """
    if strength <= 0:
        return image.copy()
    strength = float(min(strength, 3.0))  # cap to avoid harsh halos
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
