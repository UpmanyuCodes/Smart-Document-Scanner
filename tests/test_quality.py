"""Tests for scanner.quality."""

import numpy as np

from scanner import quality


def _sharp_image():
    """High-frequency checkerboard -> high sharpness/contrast."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[::2, ::2] = 255
    img[1::2, 1::2] = 255
    return img


def _flat_image(value=130):
    return np.full((100, 100, 3), value, dtype=np.uint8)


def test_weights_sum_to_one():
    assert abs(sum(quality.WEIGHTS.values()) - 1.0) < 1e-9


def test_metrics_return_numbers():
    img = _sharp_image()
    assert isinstance(quality.calculate_brightness(img), float)
    assert isinstance(quality.calculate_contrast(img), float)
    assert isinstance(quality.calculate_sharpness(img), float)
    assert isinstance(quality.estimate_noise(img), float)


def test_overall_score_bounded():
    for img in (_sharp_image(), _flat_image(0), _flat_image(255), _flat_image(130)):
        report = quality.calculate_quality_score(img)
        assert report is not None
        assert 0.0 <= report.overall <= 100.0
        assert report.label in {"Excellent", "Good", "Fair", "Poor"}


def test_score_depends_on_image_not_constant():
    sharp = quality.calculate_quality_score(_sharp_image()).overall
    flat = quality.calculate_quality_score(_flat_image(130)).overall
    # A flat gray image lacks contrast/sharpness, so it must score differently.
    assert sharp != flat


def test_flat_image_low_sharpness_contrast():
    report = quality.calculate_quality_score(_flat_image(130))
    assert report.scores["contrast"] < 10
    assert report.scores["sharpness"] < 50


def test_invalid_image_returns_none():
    assert quality.calculate_quality_score(None) is None
    assert quality.calculate_quality_score(np.array([])) is None


def test_detection_flag_passed_through():
    report = quality.calculate_quality_score(_sharp_image(), detection_success=True)
    assert report.detection_success is True


def test_noise_estimate_recovers_known_sigma():
    rng = np.random.default_rng(0)
    flat = np.full((300, 300), 200.0)
    for sigma in (3, 8):
        noisy = np.clip(flat + rng.normal(0, sigma, flat.shape), 0, 255).astype(np.uint8)
        assert abs(quality.estimate_noise(noisy) - sigma) < 0.15 * sigma


def test_noise_estimate_ignores_text_edges():
    import cv2
    page = np.full((300, 300), 235, dtype=np.uint8)
    for y in range(40, 280, 30):
        cv2.putText(page, "lorem ipsum", (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 20, 2)
    assert quality.estimate_noise(page) < 1.0


def test_bw_pages_measured_on_grayscale():
    from scanner import pipeline
    from scanner.pipeline import EditSettings
    bw = EditSettings(filter_mode=pipeline.FILTER_BW, brightness=20)
    measured = pipeline.measurement_settings(bw)
    assert measured.filter_mode == pipeline.FILTER_GRAYSCALE
    assert measured.brightness == 20 and bw.filter_mode == pipeline.FILTER_BW
    magic = EditSettings(filter_mode=pipeline.FILTER_MAGIC)
    assert pipeline.measurement_settings(magic) is magic


def test_label_thresholds():
    assert quality.label_for(90) == "Excellent"
    assert quality.label_for(70) == "Good"
    assert quality.label_for(50) == "Fair"
    assert quality.label_for(10) == "Poor"
