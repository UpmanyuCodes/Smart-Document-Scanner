"""Input and parameter validation for the Smart Document Scanner.

These helpers are deliberately UI-agnostic: they return structured results
(or raise ``ValueError``) so they can be reused by the Streamlit app, the
pipeline, and the tests without any Streamlit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Supported upload formats (lower-case, without the leading dot).
SUPPORTED_EXTENSIONS = ("jpg", "jpeg", "png")

# Practical limits. These guard against degenerate inputs rather than being
# hard physical limits; they are easy to tune in one place.
MIN_DIMENSION = 50          # px, smaller than this is not a usable document photo
MAX_DIMENSION = 10000       # px on the longest side, above this we warn/limit
MAX_FILE_BYTES = 40 * 1024 * 1024  # 40 MB upload ceiling


@dataclass
class ValidationResult:
    """Outcome of validating a candidate input.

    ``ok`` is True only when the input is safe to process. ``message`` always
    carries a human-readable explanation (empty string when everything is fine
    and there is nothing to warn about).
    """

    ok: bool
    message: str = ""
    warning: str = ""


def validate_extension(filename: str) -> ValidationResult:
    """Check that ``filename`` has a supported image extension."""
    if not filename or "." not in filename:
        return ValidationResult(False, "File has no recognizable extension.")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(SUPPORTED_EXTENSIONS)
        return ValidationResult(False, f"Unsupported file type '.{ext}'. Allowed: {allowed}.")
    return ValidationResult(True)


def validate_file_size(num_bytes: int) -> ValidationResult:
    """Check that the uploaded byte count is within limits."""
    if num_bytes <= 0:
        return ValidationResult(False, "The uploaded file is empty.")
    if num_bytes > MAX_FILE_BYTES:
        mb = MAX_FILE_BYTES / (1024 * 1024)
        return ValidationResult(False, f"File is too large (limit {mb:.0f} MB).")
    return ValidationResult(True)


def validate_image_array(image: np.ndarray) -> ValidationResult:
    """Validate a decoded image array (expected BGR or grayscale)."""
    if image is None:
        return ValidationResult(False, "The image could not be decoded (it may be corrupted).")
    if not isinstance(image, np.ndarray):
        return ValidationResult(False, "Decoded object is not a valid image.")
    if image.ndim not in (2, 3):
        return ValidationResult(False, "Image has an unexpected number of dimensions.")
    if image.size == 0:
        return ValidationResult(False, "The decoded image is empty.")

    h, w = image.shape[:2]
    if min(h, w) < MIN_DIMENSION:
        return ValidationResult(
            False,
            f"Image is too small ({w}x{h}). Minimum side length is {MIN_DIMENSION}px.",
        )

    warning = ""
    if max(h, w) > MAX_DIMENSION:
        warning = (
            f"Image is very large ({w}x{h}); it will be downscaled for faster "
            "processing where appropriate."
        )
    return ValidationResult(True, warning=warning)


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a numeric parameter into ``[low, high]`` (used for slider safety)."""
    if value < low:
        return low
    if value > high:
        return high
    return value
