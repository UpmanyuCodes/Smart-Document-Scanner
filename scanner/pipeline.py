"""Per-page edit pipeline.

Turns an original page image plus a set of edit settings into the final
rendered page. Geometry (crop/deskew/rotate) is applied first, then the
photometric look (brightness/contrast, filter, sharpen). The same function
drives both the live preview (downscaled) and the full-resolution export.
"""

from __future__ import annotations

from dataclasses import asdict, astuple, dataclass

import cv2
import numpy as np

from scanner import deskew, enhancement, filters, noise_removal, perspective

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
    denoise: int = 0              # 0 off, 1 light, 2 strong
    clean_background: bool = False  # shadow / uneven-lighting removal
    margin_left: int = 0          # manual crop, percent of width/height (0..40)
    margin_top: int = 0
    margin_right: int = 0
    margin_bottom: int = 0

    def copy(self) -> "EditSettings":
        return EditSettings(**asdict(self))

    def key(self) -> tuple:
        """Hashable snapshot, used for caching rendered results."""
        return astuple(self)

    def geometry_key(self) -> tuple:
        """Hashable snapshot of only the fields that change page geometry."""
        return (self.crop, self.deskew, self.rotate, self.margin_left,
                self.margin_top, self.margin_right, self.margin_bottom)


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


def apply_margins(image: np.ndarray, settings: EditSettings) -> np.ndarray:
    """Manual crop: trim a percentage from each side (fallback when automatic
    edge detection fails, or to tighten an auto-crop)."""
    h, w = image.shape[:2]
    left = int(w * min(max(settings.margin_left, 0), 40) / 100)
    right = int(w * min(max(settings.margin_right, 0), 40) / 100)
    top = int(h * min(max(settings.margin_top, 0), 40) / 100)
    bottom = int(h * min(max(settings.margin_bottom, 0), 40) / 100)
    if left + right >= w - 10 or top + bottom >= h - 10:
        return image
    return image[top:h - bottom, left:w - right]


def apply_geometry(image, settings: EditSettings, corners=None) -> np.ndarray:
    """Crop (perspective), manual trim, deskew, and rotate — full resolution."""
    out = image
    if settings.crop and corners is not None:
        warped = perspective.correct_perspective(image, corners)
        if warped is not None:
            out = warped
    out = apply_margins(out, settings)
    if settings.deskew:
        out, _ = deskew.correct_skew(out)
    if settings.rotate:
        out = _rotate_90s(out, settings.rotate)
    return out


def apply_look(image, settings: EditSettings) -> np.ndarray:
    """Clean background, denoise, brightness/contrast, filter, then sharpen."""
    out = image
    if settings.clean_background:
        out = noise_removal.reduce_shadows(out)
    if settings.denoise:
        out = noise_removal.denoise(out, settings.denoise)
    out = enhancement.adjust_brightness_contrast(
        out, settings.brightness, settings.contrast
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


def measurement_settings(settings: EditSettings) -> EditSettings:
    """Settings to use when measuring scan quality.

    Thresholding to pure black/white trivially maximizes contrast and
    sharpness statistics, so B&W pages are measured on their grayscale
    version (all other edits kept) to keep the score meaningful.
    """
    if settings.filter_mode != FILTER_BW:
        return settings
    measured = settings.copy()
    measured.filter_mode = FILTER_GRAYSCALE
    return measured


def downscale(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Shrink so the longest side is at most ``max_dim`` (never enlarges)."""
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
        geo = downscale(geo, preview_max)
    return apply_look(geo, settings)
