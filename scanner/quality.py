"""Scan Quality Analyzer.

Computes measurable image properties (brightness, contrast, sharpness, noise),
normalizes each to a 0-100 sub-score using documented thresholds, and combines
them into an estimated overall quality score.

This is a HEURISTIC estimate from image statistics — not an objective or
universal measure of document quality. Thresholds are gathered here so they
are easy to tune, and the weights sum to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from scanner import filters

# --- Scoring configuration (single place to tune) -------------------------
# Weights for the overall score; must sum to 1.0.
WEIGHTS = {
    "sharpness": 0.35,
    "contrast": 0.25,
    "brightness": 0.20,
    "noise": 0.20,
}

# Brightness: mean intensity; best inside an ideal band, penalized outside.
# Documents are mostly white paper, so a bright mean is desirable.
BRIGHTNESS_IDEAL_LOW = 140
BRIGHTNESS_IDEAL_HIGH = 235
# Contrast (std-dev): mapped 0 -> 0, this value or more -> 100.
CONTRAST_FULL = 65.0
# Sharpness (variance of Laplacian): below LOW ~ blurry, at/above HIGH ~ crisp.
SHARPNESS_LOW = 50.0
SHARPNESS_HIGH = 600.0
# Noise (estimated sigma, grey levels): 0 -> 100, this value or more -> 0.
NOISE_MAX = 10.0
# Pixels whose gradient magnitude exceeds this are treated as edges (text,
# borders) and excluded from the noise estimate.
NOISE_EDGE_THRESHOLD = 40.0


@dataclass
class QualityReport:
    brightness: float
    contrast: float
    sharpness: float
    noise: float
    scores: dict = field(default_factory=dict)   # normalized 0-100 per metric
    overall: float = 0.0
    label: str = ""
    detection_success: bool = False


def _gray(image: np.ndarray) -> np.ndarray:
    return filters.convert_to_grayscale(image)


def calculate_brightness(image: np.ndarray) -> float:
    """Mean grayscale intensity (0-255)."""
    return float(np.mean(_gray(image)))


def calculate_contrast(image: np.ndarray) -> float:
    """Grayscale standard deviation as a contrast measure."""
    return float(np.std(_gray(image)))


def calculate_sharpness(image: np.ndarray) -> float:
    """Variance of the Laplacian — a practical blur/sharpness indicator.

    Higher usually means more high-frequency detail, but text, noise and
    content all affect this value, so it is only an indicator.
    """
    return float(cv2.Laplacian(_gray(image), cv2.CV_64F).var())


def estimate_noise(image: np.ndarray) -> float:
    """Estimate the noise standard deviation (in grey levels).

    Uses Immerkaer's fast noise estimator: convolve with a Laplacian-difference
    mask that cancels smooth image structure, then
    ``sigma = sqrt(pi/2) * mean(|response|) / 6``. Strong edges (text strokes,
    page borders) would otherwise dominate, so pixels with a high Sobel
    gradient are excluded and only flat regions such as paper are measured.

    Limitation: on an image with almost no flat area, or with fine textured
    backgrounds, the estimate can still mix texture with noise.
    """
    gray = _gray(image).astype(np.float64)
    if min(gray.shape) < 3:
        return 0.0
    mask = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    response = np.abs(cv2.filter2D(gray, -1, mask))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.hypot(gx, gy) > NOISE_EDGE_THRESHOLD
    edges = cv2.dilate(edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    flat = ~edges
    flat[[0, -1], :] = False  # filter2D border responses are unreliable
    flat[:, [0, -1]] = False

    if flat.sum() < 0.05 * flat.size:  # too little flat area to measure
        flat = np.ones_like(flat)
        flat[[0, -1], :] = False
        flat[:, [0, -1]] = False
    return float(np.sqrt(np.pi / 2) * response[flat].mean() / 6.0)


# --- Normalizers (raw metric -> 0..100 sub-score) -------------------------
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_brightness(value: float) -> float:
    if BRIGHTNESS_IDEAL_LOW <= value <= BRIGHTNESS_IDEAL_HIGH:
        return 100.0
    if value < BRIGHTNESS_IDEAL_LOW:
        return _clamp01(value / BRIGHTNESS_IDEAL_LOW) * 100.0
    # too bright: fall from 100 at ideal-high to 0 at pure white
    return _clamp01((255 - value) / (255 - BRIGHTNESS_IDEAL_HIGH)) * 100.0


def _score_contrast(value: float) -> float:
    return _clamp01(value / CONTRAST_FULL) * 100.0


def _score_sharpness(value: float) -> float:
    if value <= SHARPNESS_LOW:
        return _clamp01(value / SHARPNESS_LOW) * 40.0  # blurry range caps low
    return 40.0 + _clamp01(
        (value - SHARPNESS_LOW) / (SHARPNESS_HIGH - SHARPNESS_LOW)
    ) * 60.0


def _score_noise(value: float) -> float:
    return _clamp01(1.0 - value / NOISE_MAX) * 100.0


def label_for(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 45:
        return "Fair"
    return "Poor"


def calculate_quality_score(
    image: np.ndarray, detection_success: bool = False
) -> QualityReport | None:
    """Compute all metrics and the weighted overall score for ``image``.

    Returns ``None`` for invalid input so the caller can report it safely.
    """
    if image is None or getattr(image, "size", 0) == 0:
        return None

    brightness = calculate_brightness(image)
    contrast = calculate_contrast(image)
    sharpness = calculate_sharpness(image)
    noise = estimate_noise(image)

    scores = {
        "brightness": _score_brightness(brightness),
        "contrast": _score_contrast(contrast),
        "sharpness": _score_sharpness(sharpness),
        "noise": _score_noise(noise),
    }
    overall = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
    overall = float(max(0.0, min(100.0, overall)))

    return QualityReport(
        brightness=brightness, contrast=contrast, sharpness=sharpness,
        noise=noise, scores=scores, overall=overall,
        label=label_for(overall), detection_success=detection_success,
    )
