import numpy as np

from common.calibration import (
    ProbabilityCalibrator,
    brier_score,
    calibrate_scores,
    expected_calibration_error,
    fit_temperature,
)


def test_none_calibrator_returns_clipped_scores():
    scores = np.array([0.0, 0.2, 0.8, 1.0])
    out = ProbabilityCalibrator(method="none").fit([0, 0, 1, 1], scores).transform(scores)

    assert out[0] > 0.0
    assert out[-1] < 1.0
    assert np.all(np.diff(out) > 0)


def test_platt_and_isotonic_calibrators_return_probabilities():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])

    for method in ["platt", "isotonic"]:
        out = calibrate_scores(y, scores, scores, method=method)
        assert len(out) == len(scores)
        assert np.all(out >= 0)
        assert np.all(out <= 1)


def test_temperature_scaling_fits_positive_temperature():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    temperature = fit_temperature(y, scores)
    out = calibrate_scores(y, scores, scores, method="temperature")

    assert temperature > 0
    assert np.all(out >= 0)
    assert np.all(out <= 1)


def test_brier_and_ece_are_bounded_for_probabilities():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    assert 0 <= brier_score(y, scores) <= 1
    assert 0 <= expected_calibration_error(y, scores, bins=2) <= 1

