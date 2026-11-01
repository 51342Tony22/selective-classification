from __future__ import annotations

from typing import Iterable

import numpy as np

from common.selective_metrics import as_binary_array, selective_metrics


def paired_bootstrap_metric_difference(
    y_true: Iterable[int],
    decisions_a: Iterable[int],
    decisions_b: Iterable[int],
    *,
    metric: str,
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 7,
    stratified: bool = True,
) -> dict[str, float]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")

    y = as_binary_array(y_true, "y_true")
    a = np.asarray(list(decisions_a), dtype=int)
    b = np.asarray(list(decisions_b), dtype=int)
    if len(y) != len(a) or len(y) != len(b):
        raise ValueError("y_true, decisions_a, and decisions_b must have the same length")

    observed_a = selective_metrics(y, a)[metric]
    observed_b = selective_metrics(y, b)[metric]
    observed_delta = float(observed_b - observed_a)

    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = _bootstrap_indices(y, rng, stratified=stratified)
        metric_a = selective_metrics(y[idx], a[idx])[metric]
        metric_b = selective_metrics(y[idx], b[idx])[metric]
        deltas[i] = metric_b - metric_a

    alpha = 1.0 - confidence_level
    low, high = np.nanquantile(deltas, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "metric": metric,
        "observed_a": float(observed_a),
        "observed_b": float(observed_b),
        "observed_delta_b_minus_a": observed_delta,
        "bootstrap_mean_delta": float(np.nanmean(deltas)),
        "ci_low": float(low),
        "ci_high": float(high),
        "iterations": float(iterations),
        "confidence_level": float(confidence_level),
        "stratified": float(bool(stratified)),
    }


def paired_bootstrap_table(
    y_true: Iterable[int],
    decisions_a: Iterable[int],
    decisions_b: Iterable[int],
    *,
    metrics: Iterable[str] = ("macro_f1", "balanced_accuracy", "mcc", "minority_recall"),
    iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 7,
    stratified: bool = True,
) -> list[dict[str, float]]:
    rows = []
    for offset, metric in enumerate(metrics):
        rows.append(
            paired_bootstrap_metric_difference(
                y_true,
                decisions_a,
                decisions_b,
                metric=metric,
                iterations=iterations,
                confidence_level=confidence_level,
                seed=seed + offset,
                stratified=stratified,
            )
        )
    return rows


def _bootstrap_indices(y_true: np.ndarray, rng: np.random.Generator, *, stratified: bool) -> np.ndarray:
    if not stratified:
        return rng.integers(0, len(y_true), size=len(y_true))

    pieces = []
    for label in [0, 1]:
        label_idx = np.flatnonzero(y_true == label)
        if len(label_idx) == 0:
            continue
        pieces.append(rng.choice(label_idx, size=len(label_idx), replace=True))
    if not pieces:
        raise ValueError("cannot bootstrap empty labels")
    return np.concatenate(pieces)

