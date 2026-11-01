from __future__ import annotations

import numpy as np

from imbalance_selective.phase2_policies import (
    evaluate_phase2_policy,
    optimize_asymmetric_dual_matched_coverage,
    run_phase2_from_scores,
    sample_training_indices_for_ratio,
    select_phase2_policies,
    symmetric_dual_thresholds,
    Phase2Policy,
)


def test_symmetric_dual_thresholds_split_budget_around_boundary() -> None:
    scores = np.linspace(0.01, 1.0, 100)
    low, high = symmetric_dual_thresholds(scores, boundary=0.5, abstention_budget=0.10)
    decisions = (scores >= high).astype(int)
    decisions[(scores >= low) & (scores < high)] = -1

    assert low < 0.5 < high
    assert (decisions == -1).sum() == 10
    assert ((decisions == -1) & (scores < 0.5)).sum() == 5
    assert ((decisions == -1) & (scores >= 0.5)).sum() == 5


def test_symmetric_dual_thresholds_do_not_split_ties_or_exceed_budget() -> None:
    scores = np.array([0.1] * 3 + [0.4] * 4 + [0.5] * 5 + [0.8] * 3)
    low, high = symmetric_dual_thresholds(scores, boundary=0.5, abstention_budget=0.20)
    rejected = (scores >= low) & (scores < high)

    assert rejected.mean() <= 0.20
    assert not rejected.any()


def test_sample_training_indices_can_make_more_extreme_ratio_by_dropping_minority() -> None:
    y = np.array([1] * 10 + [0] * 50)
    selected = sample_training_indices_for_ratio(y, majority_per_minority="1:10", seed=7)

    assert int((y[selected] == 1).sum()) == 5
    assert int((y[selected] == 0).sum()) == 50


def test_evaluate_phase2_policy_reports_effective_recall() -> None:
    policy = Phase2Policy(
        label="sym",
        family="symmetric_dual",
        lower_threshold=0.4,
        upper_threshold=0.7,
        optimized_single_threshold=0.5,
        abstention_budget=0.25,
    )
    metrics = evaluate_phase2_policy([1, 1, 0, 0], [0.9, 0.5, 0.2, 0.6], policy)

    assert metrics["coverage"] == 0.5
    assert metrics["effective_minority_recall"] == 0.5
    assert metrics["positive_coverage"] == 0.5


def test_matched_asymmetric_search_respects_target_rejection_rate() -> None:
    scores = np.linspace(0.01, 1.0, 100)
    labels = (scores >= 0.55).astype(int)
    symmetric_low, symmetric_high = symmetric_dual_thresholds(
        scores,
        boundary=0.5,
        abstention_budget=0.10,
    )
    low, high = optimize_asymmetric_dual_matched_coverage(
        labels,
        scores,
        target_abstention_rate=0.10,
        coverage_tolerance=0.002,
        max_candidates=41,
        required_thresholds=(symmetric_low, symmetric_high),
    )

    rejection_rate = float(((scores >= low) & (scores < high)).mean())
    assert abs(rejection_rate - 0.10) <= 0.002


def test_select_phase2_policies_can_add_matched_baselines() -> None:
    scores = np.linspace(0.01, 1.0, 100)
    labels = (scores >= 0.6).astype(int)

    policies = select_phase2_policies(
        labels,
        scores,
        budgets=(0.10,),
        max_candidates=41,
        include_matched_coverage=True,
        include_matched_baselines=True,
        policy_seed=7,
    )
    labels_by_policy = {policy.label: policy for policy in policies}

    assert "symmetric_dual_0.1" in labels_by_policy
    assert "matched_asymmetric_dual_0.1" in labels_by_policy
    assert "matched_random_reject_0.1" in labels_by_policy
    assert "matched_maxprob_reject_0.1" in labels_by_policy
    assert "matched_margin_reject_0.1" in labels_by_policy
    assert "matched_aucross_reject_0.1" in labels_by_policy
    assert "matched_error_model_reject_0.1" in labels_by_policy

    symmetric = labels_by_policy["symmetric_dual_0.1"]
    target_rate = float(((scores >= symmetric.lower_threshold) & (scores < symmetric.upper_threshold)).mean())
    for label, policy in labels_by_policy.items():
        if not label.startswith("matched_"):
            continue
        metrics = evaluate_phase2_policy(labels, scores, policy)
        assert abs(metrics["actual_abstention_rate"] - target_rate) <= 0.011


def test_run_phase2_from_scores_adds_when_diagnostics() -> None:
    validation_scores = np.linspace(0.01, 0.99, 80)
    validation_y = (validation_scores >= 0.58).astype(int)
    test_scores = np.linspace(0.02, 0.98, 60)
    test_y = (test_scores >= 0.62).astype(int)

    rows = run_phase2_from_scores(
        validation_y,
        validation_scores,
        test_y,
        test_scores,
        dataset="toy",
        model="toy_model",
        seed=7,
        training_distribution="natural",
        budgets=(0.10,),
        max_candidates=41,
        include_matched_coverage=True,
        include_matched_baselines=True,
    )

    assert "validation_boundary_error_enrichment" in rows.columns
    assert "validation_reject_error_enrichment" in rows.columns
    assert "test_reject_error_capture" in rows.columns
    matched_random = rows[rows["policy_label"] == "matched_random_reject_0.1"].iloc[0]
    assert matched_random["validation_reject_error_capture"] >= 0.0
