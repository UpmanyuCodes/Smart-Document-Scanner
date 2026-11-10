"""Image decoding, encoding, and format conversion helpers.

All processing in the app works on OpenCV BGR ``numpy`` arrays. Streamlit and
Pillow expect RGB, so the conversion helpers live here in one place. Decoding
never raises on bad input: it returns ``None`` so callers can show a friendly
error instead of crashing.
"""

from __future__ import annotations

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode raw image bytes into a BGR image array.

    Returns ``None`` if the bytes cannot be decoded (corrupt/unsupported),
    so the caller can report a clear error rather than crash.
    """
    if not data:
        return None
    try:
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception:
        return None
    return image


def encode_image(image: np.ndarray, ext: str = ".png") -> bytes | None:
    """Encode a BGR (or grayscale) image to bytes for download.

    ``ext`` should include the dot, e.g. ``.png`` or ``.jpg``.
    Returns ``None`` on failure.
    """
    if image is None:
        return None
    try:
        success, buffer = cv2.imencode(ext, image)
    except Exception:
        return None
    if not success:
        return None
    return buffer.tobytes()


def limit_resolution(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Return a copy no larger than ``max_dimension`` on its longest side.

    Images already within the limit are returned unchanged. Uses area
    interpolation, which is the best choice for downscaling.
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dimension:
        return image
    scale = max_dimension / float(longest)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB for display. Grayscale is passed through."""
    if image is None:
        return image
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_display(image: np.ndarray) -> np.ndarray:
    """Return an image ready for ``st.image`` (RGB, or 2-D grayscale)."""
    return bgr_to_rgb(image)


def image_dimensions(image: np.ndarray) -> tuple[int, int, int]:
    """Return ``(width, height, channels)`` for an image array."""
    if image is None:
        return (0, 0, 0)
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return (w, h, channels)
