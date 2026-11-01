import numpy as np

from common.threshold_search import (
    compare_threshold_policies,
    default_single_threshold,
    optimize_dual_threshold,
    optimize_single_threshold,
    threshold_candidates,
)


def test_threshold_candidates_include_values_just_outside_score_range():
    scores = np.array([0.2, 0.4, 0.4, 0.8])
    candidates = threshold_candidates(scores, max_candidates=None)

    assert candidates[0] < 0.2
    assert candidates[-1] > 0.8
    assert 0.4 in candidates


def test_default_single_threshold_uses_point_five():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.9])
    result = default_single_threshold(y, scores)

    assert result.params["threshold"] == 0.5
    assert result.metrics["coverage"] == 1.0
    assert result.metrics["macro_f1"] == 1.0


def test_optimize_single_threshold_finds_perfect_separator():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = optimize_single_threshold(y, scores, metric="macro_f1", max_candidates=None)

    assert result.metrics["macro_f1"] == 1.0
    assert result.metrics["coverage"] == 1.0


def test_optimize_dual_threshold_respects_abstention_budget():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.linspace(0.0, 1.0, 100)
    result = optimize_dual_threshold(y, scores, abstention_budget=0.05, max_candidates=None)

    assert result.metrics["abstention_rate"] <= 0.05 + 1e-12
    assert result.metrics["coverage"] >= 0.95 - 1e-12


def test_compare_threshold_policies_returns_smoke_test_rows():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.45, 0.55, 0.8, 0.95])
    rows = compare_threshold_policies(
        y,
        scores,
        single_metrics=("macro_f1",),
        abstention_budgets=(0.05,),
        max_candidates=None,
    )

    labels = {row["policy_label"] for row in rows}
    assert labels == {"single_default_0_5", "single_opt_macro_f1", "dual_budget_0.05"}
    assert all("coverage" in row for row in rows)
    assert all("minority_recall" in row for row in rows)

