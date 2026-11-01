from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np

ABSTAIN_LABEL = -1


@dataclass(frozen=True)
class ConfusionCounts:
    tn: int
    fp: int
    fn: int
    tp: int


def as_binary_array(values: Iterable[int], name: str) -> np.ndarray:
    arr = np.asarray(list(values), dtype=int)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    unique = set(arr.tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1 labels; found {sorted(unique)}")
    return arr


def as_score_array(values: Iterable[float], expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError(f"scores length {len(arr)} does not match labels length {expected_len}")
    if np.isnan(arr).any():
        raise ValueError("scores must not contain NaN")
    return arr


def single_threshold_predictions(scores: Iterable[float], threshold: float) -> np.ndarray:
    s = as_score_array(scores)
    return (s >= threshold).astype(int)


def dual_threshold_predictions(scores: Iterable[float], low: float, high: float) -> np.ndarray:
    if low > high:
        raise ValueError("low threshold must be <= high threshold")
    s = as_score_array(scores)
    decisions = np.full(len(s), ABSTAIN_LABEL, dtype=int)
    decisions[s < low] = 0
    decisions[s >= high] = 1
    return decisions


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> ConfusionCounts:
    y = as_binary_array(y_true, "y_true")
    pred = as_binary_array(y_pred, "y_pred")
    if len(y) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return ConfusionCounts(tn=tn, fp=fp, fn=fn, tp=tp)


def _safe_divide(num: float, denom: float, zero_value: float = 0.0) -> float:
    return float(num / denom) if denom else float(zero_value)


def binary_classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, float]:
    counts = confusion_counts(y_true, y_pred)
    tn, fp, fn, tp = counts.tn, counts.fp, counts.fn, counts.tp
    n = tn + fp + fn + tp

    precision_pos = _safe_divide(tp, tp + fp)
    recall_pos = _safe_divide(tp, tp + fn)
    precision_neg = _safe_divide(tn, tn + fn)
    recall_neg = _safe_divide(tn, tn + fp)

    f1_pos = _safe_divide(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    f1_neg = _safe_divide(2 * precision_neg * recall_neg, precision_neg + recall_neg)

    mcc_denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_divide(tp * tn - fp * fn, mcc_denom)

    return {
        "n": float(n),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "accuracy": _safe_divide(tp + tn, n),
        "risk": _safe_divide(fp + fn, n),
        "minority_precision": precision_pos,
        "minority_recall": recall_pos,
        "negative_precision": precision_neg,
        "negative_recall": recall_neg,
        "macro_f1": (f1_pos + f1_neg) / 2.0,
        "balanced_accuracy": (recall_pos + recall_neg) / 2.0,
        "mcc": mcc,
    }


def selective_metrics(
    y_true: Iterable[int],
    decisions: Iterable[int],
    *,
    minority_label: int = 1,
) -> dict[str, float]:
    if minority_label not in {0, 1}:
        raise ValueError("minority_label must be 0 or 1")

    y = as_binary_array(y_true, "y_true")
    pred = np.asarray(list(decisions), dtype=int)
    if pred.ndim != 1:
        raise ValueError("decisions must be one-dimensional")
    if len(y) != len(pred):
        raise ValueError("y_true and decisions must have the same length")
    unique = set(pred.tolist())
    if not unique.issubset({ABSTAIN_LABEL, 0, 1}):
        raise ValueError(f"decisions must contain only -1/0/1; found {sorted(unique)}")

    n = len(y)
    accepted = pred != ABSTAIN_LABEL
    abstained = ~accepted
    accepted_n = int(accepted.sum())
    abstained_n = int(abstained.sum())

    out = {
        "n": float(n),
        "accepted_n": float(accepted_n),
        "abstained_n": float(abstained_n),
        "coverage": _safe_divide(accepted_n, n),
        "abstention_rate": _safe_divide(abstained_n, n),
        "positive_abstention_rate": _class_abstention_rate(y, abstained, 1),
        "negative_abstention_rate": _class_abstention_rate(y, abstained, 0),
        "minority_proportion_abstained": _minority_proportion_abstained(y, abstained, minority_label),
    }

    if accepted_n == 0:
        out.update(
            {
                "tp": 0.0,
                "fp": 0.0,
                "tn": 0.0,
                "fn": 0.0,
                "accuracy": float("nan"),
                "risk": float("nan"),
                "minority_precision": float("nan"),
                "minority_recall": float("nan"),
                "negative_precision": float("nan"),
                "negative_recall": float("nan"),
                "macro_f1": float("nan"),
                "balanced_accuracy": float("nan"),
                "mcc": float("nan"),
            }
        )
        return out

    accepted_metrics = binary_classification_metrics(y[accepted], pred[accepted])
    out.update(accepted_metrics)
    out["accepted_positive_n"] = float((y[accepted] == 1).sum())
    out["accepted_negative_n"] = float((y[accepted] == 0).sum())
    return out


def evaluate_single_threshold(
    y_true: Iterable[int],
    scores: Iterable[float],
    threshold: float,
    *,
    minority_label: int = 1,
) -> dict[str, float]:
    y = as_binary_array(y_true, "y_true")
    s = as_score_array(scores, expected_len=len(y))
    decisions = single_threshold_predictions(s, threshold)
    out = selective_metrics(y, decisions, minority_label=minority_label)
    out.update({"policy": "single", "threshold": float(threshold), "low": float(threshold), "high": float(threshold)})
    return out


def evaluate_dual_threshold(
    y_true: Iterable[int],
    scores: Iterable[float],
    low: float,
    high: float,
    *,
    minority_label: int = 1,
) -> dict[str, float]:
    y = as_binary_array(y_true, "y_true")
    s = as_score_array(scores, expected_len=len(y))
    decisions = dual_threshold_predictions(s, low, high)
    out = selective_metrics(y, decisions, minority_label=minority_label)
    out.update({"policy": "dual", "threshold": float("nan"), "low": float(low), "high": float(high)})
    return out


def _class_abstention_rate(y_true: np.ndarray, abstained: np.ndarray, label: int) -> float:
    class_mask = y_true == label
    return _safe_divide(int((abstained & class_mask).sum()), int(class_mask.sum()), zero_value=float("nan"))


def _minority_proportion_abstained(y_true: np.ndarray, abstained: np.ndarray, minority_label: int) -> float:
    abstained_n = int(abstained.sum())
    return _safe_divide(int((abstained & (y_true == minority_label)).sum()), abstained_n, zero_value=float("nan"))


def risk_coverage_curve(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    positive_threshold: float = 0.5,
) -> list[dict[str, float]]:
    y = as_binary_array(y_true, "y_true")
    s = as_score_array(scores, expected_len=len(y))
    confidence = np.maximum(s, 1.0 - s)
    predicted = (s >= positive_threshold).astype(int)
    rows: list[dict[str, float]] = []
    for cutoff in np.unique(confidence):
        accepted = confidence >= cutoff
        if not accepted.any():
            continue
        metrics = binary_classification_metrics(y[accepted], predicted[accepted])
        rows.append(
            {
                "confidence_cutoff": float(cutoff),
                "coverage": float(accepted.mean()),
                "risk": metrics["risk"],
            }
        )
    rows.sort(key=lambda row: row["coverage"])
    return rows


def aurc(curve: list[dict[str, float]]) -> float:
    if not curve:
        return float("nan")
    coverage = np.asarray([row["coverage"] for row in curve], dtype=float)
    risk = np.asarray([row["risk"] for row in curve], dtype=float)
    if len(curve) == 1:
        return float(coverage[0] * risk[0])
    return float(np.trapz(risk, coverage))

