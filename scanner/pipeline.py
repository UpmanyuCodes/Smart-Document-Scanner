"""Per-page edit pipeline.

Turns an original page image plus a set of edit settings into the final
rendered page. Geometry (crop/deskew/rotate) is applied first, then the
photometric look (brightness/contrast, filter, sharpen). The same function
drives both the live preview (downscaled) and the full-resolution export.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

from scanner import deskew, enhancement, filters, perspective

# Filter mode names shown in the UI (kept here so UI and logic agree).
FILTER_ORIGINAL = "Original"
FILTER_MAGIC = "Magic Color"
FILTER_GRAYSCALE = "Grayscale"
FILTER_BW = "Black & White"
FILTER_HIGH_CONTRAST = "High Contrast"

FILTER_MODES = [
    FILTER_ORIGINAL,
    FILTER_MAGIC,
    FILTER_GRAYSCALE,
    FILTER_BW,
    FILTER_HIGH_CONTRAST,
]


@dataclass
class EditSettings:
    """All adjustable parameters for a single page."""

    crop: bool = False            # apply perspective correction (needs corners)
    deskew: bool = False          # auto-straighten after crop
    rotate: int = 0               # manual rotation, multiple of 90 degrees
    brightness: int = 0           # -100..100
    contrast: float = 1.0         # 0.5..2.0
    sharpen: float = 0.0          # 0..3
    filter_mode: str = FILTER_ORIGINAL
    bw_block_size: int = 21       # adaptive-threshold block size for B&W
    bw_c: int = 10                # adaptive-threshold constant for B&W

    def copy(self) -> "EditSettings":
        return EditSettings(**asdict(self))


def _rotate_90s(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate by a multiple of 90 degrees (lossless)."""
    turns = (int(degrees) // 90) % 4
    if turns == 0:
        return image
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def apply_geometry(image, settings: EditSettings, corners=None) -> np.ndarray:
    """Crop (perspective), deskew, and rotate — full resolution."""
    out = image
    if settings.crop and corners is not None:
        warped = perspective.correct_perspective(image, corners)
        if warped is not None:
            out = warped
    if settings.deskew:
        out, _ = deskew.correct_skew(out)
    if settings.rotate:
        out = _rotate_90s(out, settings.rotate)
    return out


def apply_look(image, settings: EditSettings) -> np.ndarray:
    """Apply brightness/contrast, the chosen filter, then sharpening."""
    out = enhancement.adjust_brightness_contrast(
        image, settings.brightness, settings.contrast
    )

    mode = settings.filter_mode
    if mode == FILTER_MAGIC:
        out = enhancement.enhance_contrast(out)
    elif mode == FILTER_GRAYSCALE:
        out = filters.convert_to_grayscale(out)
    elif mode == FILTER_BW:
        out = filters.adaptive_threshold(out, settings.bw_block_size, settings.bw_c)
    elif mode == FILTER_HIGH_CONTRAST:
        out = filters.create_high_contrast(out)
    # FILTER_ORIGINAL: leave as colour

    # Sharpening is meaningful for continuous-tone outputs, not binary B&W.
    if settings.sharpen > 0 and mode != FILTER_BW:
        out = enhancement.sharpen_image(out, settings.sharpen)
    return out


def _downscale(image: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def render_page(
    image, settings: EditSettings, corners=None, preview_max: int | None = None
) -> np.ndarray:
    """Render a page with its edit settings.

    When ``preview_max`` is set, the geometry result is downscaled before the
    photometric look is applied — fast enough for live slider feedback. For
    export, pass ``preview_max=None`` to keep full resolution.
    """
    geo = apply_geometry(image, settings, corners)
    if preview_max is not None:
        geo = _downscale(geo, preview_max)
    return apply_look(geo, settings)
