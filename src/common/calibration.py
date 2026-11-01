from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from common.selective_metrics import as_binary_array, as_score_array

CalibrationMethod = Literal["none", "platt", "isotonic", "temperature"]


@dataclass
class ProbabilityCalibrator:
    method: CalibrationMethod
    model: object | None = None
    temperature: float | None = None

    def fit(self, y_true: Iterable[int], scores: Iterable[float]) -> "ProbabilityCalibrator":
        y = as_binary_array(y_true, "y_true")
        s = _clip_probabilities(as_score_array(scores, expected_len=len(y)))

        if self.method == "none":
            return self

        if self.method == "platt":
            model = LogisticRegression(solver="lbfgs")
            model.fit(_logit(s).reshape(-1, 1), y)
            self.model = model
            return self

        if self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(s, y)
            self.model = model
            return self

        if self.method == "temperature":
            self.temperature = fit_temperature(y, s)
            return self

        raise ValueError(f"unsupported calibration method: {self.method}")

    def transform(self, scores: Iterable[float]) -> np.ndarray:
        s = _clip_probabilities(as_score_array(scores))

        if self.method == "none":
            return s

        if self.method == "platt":
            if self.model is None:
                raise ValueError("Platt calibrator is not fitted")
            return self.model.predict_proba(_logit(s).reshape(-1, 1))[:, 1]

        if self.method == "isotonic":
            if self.model is None:
                raise ValueError("isotonic calibrator is not fitted")
            return np.asarray(self.model.predict(s), dtype=float)

        if self.method == "temperature":
            if self.temperature is None:
                raise ValueError("temperature calibrator is not fitted")
            return _sigmoid(_logit(s) / self.temperature)

        raise ValueError(f"unsupported calibration method: {self.method}")

    def fit_transform(self, y_true: Iterable[int], scores: Iterable[float]) -> np.ndarray:
        return self.fit(y_true, scores).transform(scores)


def calibrate_scores(
    y_calibration: Iterable[int],
    calibration_scores: Iterable[float],
    target_scores: Iterable[float],
    *,
    method: CalibrationMethod,
) -> np.ndarray:
    calibrator = ProbabilityCalibrator(method=method).fit(y_calibration, calibration_scores)
    return calibrator.transform(target_scores)


def fit_temperature(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = as_binary_array(y_true, "y_true")
    s = _clip_probabilities(as_score_array(scores, expected_len=len(y)))
    logits = _logit(s)

    candidates = np.unique(np.concatenate([np.logspace(-2, 2, 81), np.array([1.0])]))
    losses = np.asarray([_binary_nll(y, _sigmoid(logits / temperature)) for temperature in candidates])
    best = float(candidates[int(np.argmin(losses))])
    return best


def brier_score(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = as_binary_array(y_true, "y_true")
    s = _clip_probabilities(as_score_array(scores, expected_len=len(y)))
    return float(np.mean((s - y) ** 2))


def expected_calibration_error(y_true: Iterable[int], scores: Iterable[float], *, bins: int = 20) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    y = as_binary_array(y_true, "y_true")
    s = _clip_probabilities(as_score_array(scores, expected_len=len(y)))
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        left = edges[i]
        right = edges[i + 1]
        if i == bins - 1:
            mask = (s >= left) & (s <= right)
        else:
            mask = (s >= left) & (s < right)
        if not mask.any():
            continue
        confidence = float(s[mask].mean())
        accuracy = float(y[mask].mean())
        total += float(mask.mean()) * abs(accuracy - confidence)
    return float(total)


def _clip_probabilities(scores: np.ndarray) -> np.ndarray:
    return np.clip(scores.astype(float), 1e-12, 1.0 - 1e-12)


def _logit(scores: np.ndarray) -> np.ndarray:
    return np.log(scores / (1.0 - scores))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _binary_nll(y_true: np.ndarray, scores: np.ndarray) -> float:
    s = _clip_probabilities(scores)
    return float(-np.mean(y_true * np.log(s) + (1 - y_true) * np.log(1.0 - s)))

