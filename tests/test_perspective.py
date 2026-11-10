"""Tests for scanner.perspective."""

import numpy as np

from scanner import perspective


def test_correct_perspective_valid_output():
    img = np.full((400, 400, 3), 200, dtype=np.uint8)
    corners = np.array([[50, 60], [350, 40], [360, 330], [40, 340]], dtype="float32")
    warped = perspective.correct_perspective(img, corners)
    assert warped is not None
    h, w = warped.shape[:2]
    assert h > 10 and w > 10


def test_degenerate_corners_return_none():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")
    assert perspective.correct_perspective(img, corners) is None


def test_none_corners_return_none():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    assert perspective.correct_perspective(img, None) is None


def test_wrong_corner_count_return_none():
    img = np.full((200, 200, 3), 128, dtype=np.uint8)
    corners = np.array([[0, 0], [10, 0], [10, 10]], dtype="float32")
    assert perspective.correct_perspective(img, corners) is None
