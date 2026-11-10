"""Geometric helpers: point ordering and simple quad geometry.

Kept UI-free and dependency-light so detection, perspective correction, and
the tests can all share the same ordering logic.
"""

from __future__ import annotations

import numpy as np


def order_points(points) -> np.ndarray:
    """Order four points consistently as top-left, top-right, bottom-right,
    bottom-left.

    Uses the classic coordinate-sum / coordinate-difference trick:
      - top-left has the smallest x+y, bottom-right the largest
      - top-right has the smallest (y-x), bottom-left the largest

    Returns a ``(4, 2)`` float32 array. Raises ``ValueError`` if the input is
    not exactly four 2-D points.
    """
    pts = np.asarray(points, dtype="float32").reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("order_points expects exactly four (x, y) points.")

    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1).ravel()  # y - x
    ordered[1] = pts[np.argmin(diff)]    # top-right
    ordered[3] = pts[np.argmax(diff)]    # bottom-left
    return ordered


def quad_is_convex(points) -> bool:
    """Return True if the four ordered points form a convex quadrilateral.

    Checks that the cross product of consecutive edge vectors keeps a
    consistent sign around the polygon.
    """
    pts = np.asarray(points, dtype="float64").reshape(-1, 2)
    if pts.shape[0] != 4:
        return False
    signs = []
    for i in range(4):
        a = pts[i]
        b = pts[(i + 1) % 4]
        c = pts[(i + 2) % 4]
        u = b - a
        v = c - b
        cross = u[0] * v[1] - u[1] * v[0]  # scalar 2-D cross product
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def quad_area(points) -> float:
    """Shoelace-formula area of a quadrilateral."""
    pts = np.asarray(points, dtype="float64").reshape(-1, 2)
    if pts.shape[0] != 4:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
