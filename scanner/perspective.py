"""Perspective correction (document flattening).

Given four document corners, warp the region into a rectangular, top-down
view using a homography. The output size is derived from the detected edge
lengths so resolution is preserved as much as possible.
"""

from __future__ import annotations

import cv2
import numpy as np

from utils import geometry


def correct_perspective(image: np.ndarray, corners) -> np.ndarray | None:
    """Return a top-down view of the quad defined by ``corners``.

    ``corners`` is any four (x, y) points (ordering is normalized here).
    Returns ``None`` if the corners are invalid or the computed output size
    is degenerate, so callers can fall back to the original image.
    """
    if image is None or corners is None:
        return None
    try:
        ordered = geometry.order_points(corners)
    except (ValueError, TypeError):
        return None

    (tl, tr, br, bl) = ordered

    # Output width = larger of the two horizontal edges; height similarly.
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_w = int(round(max(width_top, width_bottom)))
    out_h = int(round(max(height_left, height_right)))

    # Guard against zero / unreasonable dimensions.
    if out_w < 10 or out_h < 10:
        return None
    if out_w > 10000 or out_h > 10000:
        return None

    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )
    try:
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    except cv2.error:
        return None
    return warped
