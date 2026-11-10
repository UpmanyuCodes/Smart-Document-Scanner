"""OCR text extraction via Tesseract (through pytesseract).

The Tesseract *engine* is a separate native install; ``pytesseract`` is only a
wrapper. Every function here degrades gracefully when the engine or a language
pack is missing, returning a structured result with an actionable message
rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import pytesseract

from scanner import filters

SETUP_HELP = (
    "Tesseract OCR engine was not found. Install it (Windows: the UB Mannheim "
    "build at https://github.com/UB-Mannheim/tesseract/wiki), then add its "
    "folder (e.g. C:\\Program Files\\Tesseract-OCR) to your PATH and restart "
    "the app. pytesseract is only a Python wrapper and does not include the "
    "engine."
)


@dataclass
class OcrResult:
    success: bool
    text: str = ""
    message: str = ""
    mean_confidence: float | None = None
    word_count: int = 0
    languages: list[str] = field(default_factory=list)


def is_available() -> tuple[bool, str]:
    """Return ``(available, version_or_help)`` for the Tesseract engine."""
    try:
        version = pytesseract.get_tesseract_version()
        return True, str(version)
    except Exception:
        return False, SETUP_HELP


def available_languages() -> list[str]:
    """List installed Tesseract language codes (empty if engine missing)."""
    try:
        langs = pytesseract.get_languages(config="")
        # 'osd' is orientation data, not a text language.
        return [l for l in langs if l and l != "osd"]
    except Exception:
        return []


def _prepare_for_ocr(image: np.ndarray) -> np.ndarray:
    """Give Tesseract a clean binary image. Already-binary input is passed
    through; otherwise Otsu thresholding is applied to grayscale."""
    gray = filters.convert_to_grayscale(image)
    unique = np.unique(gray)
    if set(unique.tolist()).issubset({0, 255}):
        return gray
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def extract_text(image: np.ndarray, language: str = "eng") -> OcrResult:
    """Run OCR and return the recognized text (no confidence)."""
    if image is None or image.size == 0:
        return OcrResult(False, message="No image to read.")

    available, info = is_available()
    if not available:
        return OcrResult(False, message=info)

    prepared = _prepare_for_ocr(image)
    try:
        text = pytesseract.image_to_string(prepared, lang=language)
    except pytesseract.TesseractError as exc:
        msg = str(exc)
        if "Failed loading language" in msg or "could not initialize" in msg.lower():
            return OcrResult(
                False,
                message=(f"Language data for '{language}' is not installed. "
                         "Install the matching Tesseract language pack."),
            )
        return OcrResult(False, message=f"OCR failed: {msg}")
    except Exception as exc:  # noqa: BLE001 - surface any wrapper error cleanly
        return OcrResult(False, message=f"OCR failed: {exc}")

    cleaned = text.strip()
    if not cleaned:
        return OcrResult(
            True, text="", word_count=0,
            message="OCR ran but found no readable text. Try a sharper image, "
                    "the Black & White filter, or better lighting.",
        )
    return OcrResult(True, text=cleaned, word_count=len(cleaned.split()))


def extract_text_with_confidence(image: np.ndarray, language: str = "eng") -> OcrResult:
    """Run OCR and also report the engine's mean word confidence.

    The confidence is Tesseract's own estimate, not a guarantee of correctness.
    """
    base = extract_text(image, language)
    if not base.success or not base.text:
        return base

    prepared = _prepare_for_ocr(image)
    try:
        data = pytesseract.image_to_data(
            prepared, lang=language, output_type=pytesseract.Output.DICT
        )
        confidences = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit()
                       and int(c) >= 0]
        base.mean_confidence = (
            float(np.mean(confidences)) if confidences else None
        )
    except Exception:
        base.mean_confidence = None
    return base
