"""Deskewing: estimate and correct small document rotations.

Uses the probabilistic Hough line transform to find dominant near-horizontal
edges, takes a robust (median) angle, and only rotates when the angle is
meaningful but within a safe correction range. The canvas is expanded so
content is not clipped.
"""

from __future__ import annotations

import cv2
import numpy as np

from scanner import preprocessing


def estimate_skew_angle(image: np.ndarray, max_angle: float = 15.0) -> float | None:
    """Estimate the skew angle in degrees (positive = counter-clockwise tilt).

    Returns ``None`` if no reliable dominant orientation is found. Only angles
    whose magnitude is within ``max_angle`` are considered, so a strong
    vertical line or an arbitrary stroke does not dominate.
    """
    gray = preprocessing.to_grayscale(image)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )
    if lines is None:
        return None

    angles = []
    # OpenCV versions differ in output shape ((N,1,4) vs (N,4)); normalize.
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Normalize to [-45, 45] and keep near-horizontal lines only.
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) <= max_angle:
            angles.append(angle)

    if not angles:
        return None
    return float(np.median(angles))


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate ``image`` by ``angle`` degrees, expanding the canvas to fit."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    border = (255, 255, 255) if image.ndim == 3 else 255
    return cv2.warpAffine(
        image, matrix, (new_w, new_h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def correct_skew(
    image: np.ndarray, max_correction_angle: float = 15.0, min_angle: float = 0.5
) -> tuple[np.ndarray, float]:
    """Deskew ``image`` if a meaningful angle is detected.

    Returns ``(result_image, applied_angle)``. ``applied_angle`` is 0.0 when
    no correction was made (already aligned, or no reliable estimate).
    """
    angle = estimate_skew_angle(image, max_correction_angle)
    if angle is None or abs(angle) < min_angle:
        return image, 0.0
    rotated = rotate_image(image, angle)
    return rotated, angle
