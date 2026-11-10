"""Automatic document boundary detection.

Finds the largest plausible four-corner (quadrilateral) region in a photo,
which is treated as the document. The approach is the standard contour
pipeline: edges -> contours -> polygon approximation -> validate a quad.

This is heuristic. It will not work when the paper and background share
colours, when edges are obscured, or when the document is folded/cropped.
Callers must handle ``success == False`` (e.g. fall back to the original
image or manual corners).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from scanner import preprocessing
from utils import geometry

# Tunable thresholds (kept here so they are easy to find and adjust).
MIN_AREA_RATIO = 0.15   # quad must cover at least this fraction of the frame
MAX_AREA_RATIO = 0.99   # ...but not essentially the whole frame (likely the border)
APPROX_EPSILON = 0.02   # polygon approximation tolerance (× contour perimeter)


@dataclass
class DetectionResult:
    """Outcome of document detection.

    ``corners`` are in the *original* image coordinate system, ordered
    top-left, top-right, bottom-right, bottom-left. When ``success`` is
    False, ``corners`` is None and ``message`` explains why.
    """

    success: bool
    corners: np.ndarray | None = None
    message: str = ""
    area_ratio: float = 0.0
    edges: np.ndarray | None = field(default=None, repr=False)


def detect_document_corners(image: np.ndarray) -> DetectionResult:
    """Detect the four corners of the document in ``image``.

    Returns a :class:`DetectionResult`. Never raises on ordinary images.
    """
    if image is None or image.size == 0:
        return DetectionResult(False, message="No image to analyze.")

    # 1. Work on a downscaled copy for speed and edge stability.
    working, scale = preprocessing.resize_for_detection(image, max_dimension=1200)
    frame_area = working.shape[0] * working.shape[1]

    # 2. Grayscale -> blur -> Canny edges.
    gray = preprocessing.to_grayscale(working)
    blurred = preprocessing.gaussian_blur(gray, 5)
    edges = cv2.Canny(blurred, 75, 200)

    # 3. Close small gaps so the page outline forms a continuous contour.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 4. Find external contours, largest first.
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return DetectionResult(
            False, message="No contours found — no clear document edges.",
            edges=edges,
        )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    best = None
    best_ratio = 0.0
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        approx = cv2.approxPolyDP(contour, APPROX_EPSILON * perimeter, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        ratio = area / float(frame_area)
        if ratio < MIN_AREA_RATIO or ratio > MAX_AREA_RATIO:
            continue
        if ratio > best_ratio:
            best = approx
            best_ratio = ratio

    if best is None:
        return DetectionResult(
            False,
            message=(
                "No reliable four-corner document boundary found. The paper may "
                "blend into the background, or edges may be cut off."
            ),
            edges=edges,
        )

    # 5. Order corners and map them back to the original image coordinates.
    ordered = geometry.order_points(best.reshape(4, 2))
    corners = ordered / scale  # scale maps working -> original

    return DetectionResult(
        True,
        corners=corners.astype("float32"),
        message="Document boundary detected.",
        area_ratio=best_ratio,
        edges=edges,
    )


def draw_corners(
    image: np.ndarray,
    corners: np.ndarray,
    color=(0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Return a copy of ``image`` with the detected quad and corner dots drawn.

    The original image is not modified.
    """
    canvas = image.copy()
    if corners is None:
        return canvas
    pts = np.asarray(corners, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
    radius = max(4, thickness * 3)
    for (x, y) in corners.astype(int):
        cv2.circle(canvas, (int(x), int(y)), radius, (0, 0, 255), -1)
    return canvas
