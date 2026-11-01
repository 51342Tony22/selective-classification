import math

import numpy as np

from common.selective_metrics import (
    ABSTAIN_LABEL,
    dual_threshold_predictions,
    evaluate_dual_threshold,
    evaluate_single_threshold,
    selective_metrics,
    single_threshold_predictions,
)


def test_single_threshold_predicts_all_samples():
    scores = np.array([0.1, 0.49, 0.5, 0.9])
    decisions = single_threshold_predictions(scores, 0.5)
    assert decisions.tolist() == [0, 0, 1, 1]


def test_dual_threshold_uses_half_open_abstention_region():
    scores = np.array([0.19, 0.20, 0.50, 0.80, 0.81])
    decisions = dual_threshold_predictions(scores, low=0.20, high=0.80)
    assert decisions.tolist() == [0, ABSTAIN_LABEL, ABSTAIN_LABEL, 1, 1]


def test_selective_metrics_report_coverage_and_class_abstention():
    y = np.array([1, 1, 1, 0, 0, 0])
    decisions = np.array([0, ABSTAIN_LABEL, 1, 0, ABSTAIN_LABEL, 1])
    metrics = selective_metrics(y, decisions)

    assert metrics["coverage"] == 4 / 6
    assert metrics["abstention_rate"] == 2 / 6
    assert metrics["positive_abstention_rate"] == 1 / 3
    assert metrics["negative_abstention_rate"] == 1 / 3
    assert metrics["minority_proportion_abstained"] == 1 / 2
    assert metrics["minority_recall"] == 1 / 2
    assert metrics["minority_precision"] == 1 / 2


def test_evaluate_single_threshold_has_full_coverage():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    metrics = evaluate_single_threshold(y, scores, threshold=0.5)

    assert metrics["coverage"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["mcc"] == 1.0


def test_evaluate_dual_threshold_can_abstain_without_human_review_terms():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.45, 0.55, 0.9])
    metrics = evaluate_dual_threshold(y, scores, low=0.4, high=0.6)

    assert metrics["coverage"] == 0.5
    assert metrics["abstention_rate"] == 0.5
    assert metrics["macro_f1"] == 1.0
    assert all("review" not in key for key in metrics)


def test_all_abstained_returns_nan_accepted_metrics():
    y = np.array([0, 1])
    decisions = np.array([ABSTAIN_LABEL, ABSTAIN_LABEL])
    metrics = selective_metrics(y, decisions)

    assert metrics["coverage"] == 0.0
    assert math.isnan(metrics["risk"])

