"""Tests for utils.geometry."""

import numpy as np
import pytest

from utils import geometry


def test_order_points_from_shuffled():
    # Provide corners out of order; expect tl, tr, br, bl.
    pts = np.array([[200, 10], [10, 10], [10, 200], [200, 200]], dtype="float32")
    ordered = geometry.order_points(pts)
    assert np.allclose(ordered[0], [10, 10])     # top-left
    assert np.allclose(ordered[1], [200, 10])    # top-right
    assert np.allclose(ordered[2], [200, 200])   # bottom-right
    assert np.allclose(ordered[3], [10, 200])    # bottom-left


def test_order_points_rejects_wrong_count():
    with pytest.raises(ValueError):
        geometry.order_points([[0, 0], [1, 1], [2, 2]])


def test_quad_is_convex_true_for_rectangle():
    rect = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert geometry.quad_is_convex(rect)


def test_quad_is_convex_false_for_self_intersecting():
    bowtie = [[0, 0], [10, 10], [10, 0], [0, 10]]
    assert not geometry.quad_is_convex(bowtie)


def test_quad_area_rectangle():
    rect = [[0, 0], [10, 0], [10, 5], [0, 5]]
    assert geometry.quad_area(rect) == pytest.approx(50.0)
