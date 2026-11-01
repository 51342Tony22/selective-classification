from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from common.selective_metrics import (
    as_binary_array,
    as_score_array,
    evaluate_dual_threshold,
    evaluate_single_threshold,
)


@dataclass(frozen=True)
class SearchResult:
    params: dict[str, float | str]
    metrics: dict[str, float]


def threshold_candidates(
    scores: Iterable[float],
    *,
    max_candidates: int | None = 301,
    include_full_range: bool = True,
) -> np.ndarray:
    s = as_score_array(scores)
    if len(s) == 0:
        raise ValueError("scores must not be empty")

    values = np.unique(s)
    if max_candidates is not None and len(values) > max_candidates:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        values = np.unique(np.quantile(s, quantiles))

    if include_full_range:
        values = np.unique(np.concatenate(([np.nextafter(s.min(), -np.inf)], values, [np.nextafter(s.max(), np.inf)])))
    return np.sort(values.astype(float))


def optimize_single_threshold(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    metric: str = "macro_f1",
    max_candidates: int | None = 301,
) -> SearchResult:
    y = as_binary_array(y_true, "y_true")
    s = as_score_array(scores, expected_len=len(y))
    best: SearchResult | None = None
    for threshold in threshold_candidates(s, max_candidates=max_candidates):
        metrics = evaluate_single_threshold(y, s, float(threshold))
        candidate = SearchResult(params={"policy": "single", "threshold": float(threshold)}, metrics=metrics)
        if _is_better(candidate, best, metric):
            best = candidate
    if best is None:
        raise RuntimeError("single-threshold search produced no candidates")
    return best


def default_single_threshold(y_true: Iterable[int], scores: Iterable[float], threshold: float = 0.5) -> SearchResult:
    y = as_binary_array(y_true, "y_true")
    s = as_score_array(scores, expected_len=len(y))
    metrics = evaluate_single_threshold(y, s, threshold)
    return SearchResult(params={"policy": "single", "threshold": float(threshold)}, metrics=metrics)


def optimize_dual_threshold(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    abstention_budget: float,
    metric: str = "macro_f1",
    max_candidates: int | None = 301,
    tolerance: float = 1e-12,
) -> SearchResult:
    if abstention_budget < 0 or abstention_budget > 1:
        raise ValueError("abstention_budget must be between 0 and 1")

    y = as_binary_array(y_true, "y_true")
    s = as_score_array(scores, expected_len=len(y))
    candidates = threshold_candidates(s, max_candidates=max_candidates)

    best: SearchResult | None = None
    for low in candidates:
        for high in candidates:
            if low > high:
                continue
            metrics = evaluate_dual_threshold(y, s, float(low), float(high))
            if metrics["abstention_rate"] > abstention_budget + tolerance:
                continue
            candidate = SearchResult(
                params={
                    "policy": "dual",
                    "low": float(low),
                    "high": float(high),
                    "abstention_budget": float(abstention_budget),
                },
                metrics=metrics,
            )
            if _is_better(candidate, best, metric):
                best = candidate

    if best is None:
        raise RuntimeError("dual-threshold search produced no feasible candidates")
    return best


def compare_threshold_policies(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    single_metrics: Iterable[str] = ("macro_f1", "balanced_accuracy", "mcc"),
    abstention_budgets: Iterable[float] = (0.01, 0.02, 0.05, 0.10),
    max_candidates: int | None = 301,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []

    default = default_single_threshold(y_true, scores)
    rows.append(_flatten_result(default, label="single_default_0_5"))

    for metric in single_metrics:
        result = optimize_single_threshold(y_true, scores, metric=metric, max_candidates=max_candidates)
        rows.append(_flatten_result(result, label=f"single_opt_{metric}", optimization_metric=metric))

    for budget in abstention_budgets:
        result = optimize_dual_threshold(
            y_true,
            scores,
            abstention_budget=budget,
            metric="macro_f1",
            max_candidates=max_candidates,
        )
        rows.append(_flatten_result(result, label=f"dual_budget_{budget:g}", optimization_metric="macro_f1"))

    return rows


def _metric_value(result: SearchResult, metric: str) -> float:
    value = result.metrics.get(metric)
    if value is None:
        raise KeyError(f"metric not found in result: {metric}")
    return float(value)


def _is_better(candidate: SearchResult, incumbent: SearchResult | None, metric: str) -> bool:
    if incumbent is None:
        return True

    cand_value = _metric_value(candidate, metric)
    inc_value = _metric_value(incumbent, metric)
    if np.isnan(cand_value):
        return False
    if np.isnan(inc_value):
        return True
    if cand_value > inc_value + 1e-12:
        return True
    if abs(cand_value - inc_value) <= 1e-12:
        cand_coverage = float(candidate.metrics.get("coverage", 0.0))
        inc_coverage = float(incumbent.metrics.get("coverage", 0.0))
        if cand_coverage > inc_coverage + 1e-12:
            return True
        if abs(cand_coverage - inc_coverage) <= 1e-12:
            cand_width = float(candidate.metrics.get("high", 0.0)) - float(candidate.metrics.get("low", 0.0))
            inc_width = float(incumbent.metrics.get("high", 0.0)) - float(incumbent.metrics.get("low", 0.0))
            return cand_width < inc_width
    return False


def _flatten_result(
    result: SearchResult,
    *,
    label: str,
    optimization_metric: str | None = None,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"policy_label": label}
    if optimization_metric is not None:
        row["optimization_metric"] = optimization_metric
    row.update(result.params)
    row.update(result.metrics)
    return row

