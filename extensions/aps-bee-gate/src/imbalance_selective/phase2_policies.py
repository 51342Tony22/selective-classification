from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common.resampling import class_counts, parse_minority_majority_ratio
from common.selective_metrics import ABSTAIN_LABEL, dual_threshold_predictions, evaluate_single_threshold, selective_metrics
from common.threshold_search import optimize_single_threshold, threshold_candidates
from imbalance_selective.datasets import DatasetBundle
from imbalance_selective.preprocessing import build_model_pipeline


@dataclass(frozen=True)
class Phase2Policy:
    label: str
    family: str
    lower_threshold: float
    upper_threshold: float
    optimized_single_threshold: float
    abstention_budget: float
    optimization_metric: str = "balanced_selective_risk"
    metadata: dict[str, Any] = field(default_factory=dict)


INTERVAL_DECISION_FAMILIES = {
    "symmetric_dual",
    "asymmetric_dual",
    "matched_asymmetric_dual",
}
INTERVAL_REJECT_FAMILIES = {
    "matched_maxprob_reject",
    "matched_margin_reject",
    "matched_aucross_reject",
}
CUSTOM_REJECT_FAMILIES = {
    "matched_random_reject",
    "matched_error_model_reject",
}


def sample_training_indices_for_ratio(
    y: Iterable[int],
    *,
    majority_per_minority: str | float | int | None,
    seed: int,
    fixed_total: int | None = None,
) -> np.ndarray:
    labels = np.asarray(list(y), dtype=int)
    indices = np.arange(len(labels))
    if majority_per_minority is None:
        return indices

    ratio = parse_minority_majority_ratio(majority_per_minority)
    positive_idx = np.flatnonzero(labels == 1)
    negative_idx = np.flatnonzero(labels == 0)
    rng = np.random.default_rng(seed)

    if fixed_total is not None:
        target_positive = int(round(fixed_total / (ratio + 1.0)))
        target_negative = fixed_total - target_positive
        if target_positive <= 0 or target_negative <= 0:
            raise ValueError(f"fixed_total={fixed_total} is too small for ratio {majority_per_minority}")
        if target_positive > len(positive_idx) or target_negative > len(negative_idx):
            raise ValueError(
                f"fixed_total={fixed_total} cannot support ratio {majority_per_minority}: "
                f"need {target_positive} positives and {target_negative} negatives, "
                f"have {len(positive_idx)} positives and {len(negative_idx)} negatives"
            )
        selected_positive = rng.choice(positive_idx, size=target_positive, replace=False)
        selected_negative = rng.choice(negative_idx, size=target_negative, replace=False)
        return np.sort(np.concatenate([selected_positive, selected_negative]))

    target_negative = int(round(len(positive_idx) * ratio))
    if target_negative <= len(negative_idx):
        selected_negative = rng.choice(negative_idx, size=target_negative, replace=False)
        return np.sort(np.concatenate([positive_idx, selected_negative]))

    target_positive = int(np.floor(len(negative_idx) / ratio))
    if target_positive <= 0:
        raise ValueError(f"ratio {majority_per_minority} leaves no positive training examples")
    selected_positive = rng.choice(positive_idx, size=min(target_positive, len(positive_idx)), replace=False)
    return np.sort(np.concatenate([selected_positive, negative_idx]))


def select_phase2_policies(
    validation_y: np.ndarray,
    validation_scores: np.ndarray,
    *,
    budgets: tuple[float, ...] = (0.01, 0.05, 0.10),
    max_candidates: int = 201,
    include_matched_coverage: bool = False,
    include_matched_baselines: bool = False,
    matched_coverage_tolerance: float = 0.002,
    policy_seed: int = 0,
) -> list[Phase2Policy]:
    optimized_single = optimize_single_threshold(
        validation_y,
        validation_scores,
        metric="balanced_accuracy",
        max_candidates=max_candidates,
    )
    t_star = float(optimized_single.params["threshold"])
    policies = [
        Phase2Policy(
            label="default_single_0_5",
            family="default_single",
            lower_threshold=0.5,
            upper_threshold=0.5,
            optimized_single_threshold=t_star,
            abstention_budget=0.0,
            optimization_metric="fixed",
        ),
        Phase2Policy(
            label="optimized_single",
            family="optimized_single",
            lower_threshold=t_star,
            upper_threshold=t_star,
            optimized_single_threshold=t_star,
            abstention_budget=0.0,
        ),
    ]
    for budget in budgets:
        low, high = symmetric_dual_thresholds(validation_scores, boundary=t_star, abstention_budget=budget)
        symmetric_rejection_rate = float(((validation_scores >= low) & (validation_scores < high)).mean())
        policies.append(
            Phase2Policy(
                label=f"symmetric_dual_{budget:g}",
                family="symmetric_dual",
                lower_threshold=low,
                upper_threshold=high,
                optimized_single_threshold=t_star,
                abstention_budget=budget,
            )
        )
        if include_matched_coverage:
            matched_low, matched_high = optimize_asymmetric_dual_matched_coverage(
                validation_y,
                validation_scores,
                target_abstention_rate=symmetric_rejection_rate,
                coverage_tolerance=matched_coverage_tolerance,
                max_candidates=max_candidates,
                required_thresholds=(low, high),
            )
            policies.append(
                Phase2Policy(
                    label=f"matched_asymmetric_dual_{budget:g}",
                    family="matched_asymmetric_dual",
                    lower_threshold=matched_low,
                    upper_threshold=matched_high,
                    optimized_single_threshold=t_star,
                    abstention_budget=budget,
                    optimization_metric="balanced_selective_risk_matched_coverage",
                )
            )
        if include_matched_baselines:
            policies.extend(
                select_matched_coverage_baseline_policies(
                    validation_y,
                    validation_scores,
                    target_abstention_rate=symmetric_rejection_rate,
                    optimized_single_threshold=t_star,
                    budget=budget,
                    max_candidates=max_candidates,
                    coverage_tolerance=matched_coverage_tolerance,
                    policy_seed=policy_seed,
                )
            )
        low_asym, high_asym = optimize_asymmetric_dual_balanced_accuracy(
            validation_y,
            validation_scores,
            abstention_budget=budget,
            max_candidates=max_candidates,
        )
        policies.append(
            Phase2Policy(
                label=f"asymmetric_dual_{budget:g}",
                family="asymmetric_dual",
                lower_threshold=low_asym,
                upper_threshold=high_asym,
                optimized_single_threshold=t_star,
                abstention_budget=budget,
            )
        )
    return policies


def select_matched_coverage_baseline_policies(
    validation_y: Iterable[int],
    validation_scores: Iterable[float],
    *,
    target_abstention_rate: float,
    optimized_single_threshold: float,
    budget: float,
    max_candidates: int = 201,
    coverage_tolerance: float = 0.002,
    policy_seed: int = 0,
) -> list[Phase2Policy]:
    y = np.asarray(list(validation_y), dtype=int)
    s = np.asarray(list(validation_scores), dtype=float)
    policies: list[Phase2Policy] = []

    random_seed = int(policy_seed * 1000 + round(budget * 10000) + 17)
    random_priority = _random_priority_scores(len(s), seed=random_seed)
    random_cutoff = _high_score_cutoff(random_priority, target_abstention_rate)
    policies.append(
        Phase2Policy(
            label=f"matched_random_reject_{budget:g}",
            family="matched_random_reject",
            lower_threshold=float("nan"),
            upper_threshold=float("nan"),
            optimized_single_threshold=optimized_single_threshold,
            abstention_budget=budget,
            optimization_metric="matched_validation_coverage",
            metadata={"random_seed": random_seed, "priority_cutoff": random_cutoff},
        )
    )

    maxprob_low, maxprob_high = symmetric_dual_thresholds(s, boundary=0.5, abstention_budget=target_abstention_rate)
    policies.append(
        Phase2Policy(
            label=f"matched_maxprob_reject_{budget:g}",
            family="matched_maxprob_reject",
            lower_threshold=maxprob_low,
            upper_threshold=maxprob_high,
            optimized_single_threshold=optimized_single_threshold,
            abstention_budget=budget,
            optimization_metric="matched_validation_coverage",
        )
    )

    margin_low, margin_high = symmetric_dual_thresholds(s, boundary=0.5, abstention_budget=target_abstention_rate)
    policies.append(
        Phase2Policy(
            label=f"matched_margin_reject_{budget:g}",
            family="matched_margin_reject",
            lower_threshold=margin_low,
            upper_threshold=margin_high,
            optimized_single_threshold=optimized_single_threshold,
            abstention_budget=budget,
            optimization_metric="matched_validation_coverage",
        )
    )

    aucross_low, aucross_high = optimize_aucross_reject(
        y,
        s,
        target_abstention_rate=target_abstention_rate,
        coverage_tolerance=coverage_tolerance,
        max_candidates=max_candidates,
        required_thresholds=(maxprob_low, maxprob_high),
    )
    policies.append(
        Phase2Policy(
            label=f"matched_aucross_reject_{budget:g}",
            family="matched_aucross_reject",
            lower_threshold=aucross_low,
            upper_threshold=aucross_high,
            optimized_single_threshold=optimized_single_threshold,
            abstention_budget=budget,
            optimization_metric="accepted_auc_matched_coverage",
        )
    )

    error_model_params = fit_error_model_reject(
        y,
        s,
        decision_threshold=optimized_single_threshold,
        target_abstention_rate=target_abstention_rate,
    )
    policies.append(
        Phase2Policy(
            label=f"matched_error_model_reject_{budget:g}",
            family="matched_error_model_reject",
            lower_threshold=float("nan"),
            upper_threshold=float("nan"),
            optimized_single_threshold=optimized_single_threshold,
            abstention_budget=budget,
            optimization_metric="learned_error_probability_matched_coverage",
            metadata=error_model_params,
        )
    )
    return policies


def symmetric_dual_thresholds(scores: Iterable[float], *, boundary: float, abstention_budget: float) -> tuple[float, float]:
    s = np.asarray(list(scores), dtype=float)
    target_total = int(np.floor(len(s) * abstention_budget + 1e-12))
    target_lower = target_total // 2
    target_upper = target_total - target_lower

    # Include only complete score-tie blocks so realized rejection never exceeds
    # the nominal maximum budget.
    low = _lower_tie_block_threshold(s[s < boundary], boundary, target_lower)
    high = _upper_tie_block_threshold(s[s >= boundary], boundary, target_upper)

    if low > high:
        low, high = high, low
    return float(low), float(high)


def _lower_tie_block_threshold(scores: np.ndarray, boundary: float, target: int) -> float:
    if target <= 0 or len(scores) == 0:
        return float(boundary)
    values, counts = np.unique(scores, return_counts=True)
    used = 0
    low = float(boundary)
    for value, count in zip(values[::-1], counts[::-1], strict=True):
        if used + int(count) > target:
            break
        used += int(count)
        low = float(value)
    return low


def _upper_tie_block_threshold(scores: np.ndarray, boundary: float, target: int) -> float:
    if target <= 0 or len(scores) == 0:
        return float(boundary)
    values, counts = np.unique(scores, return_counts=True)
    used = 0
    high = float(boundary)
    for value, count in zip(values, counts, strict=True):
        if used + int(count) > target:
            break
        used += int(count)
        high = float(np.nextafter(value, np.inf))
    return high


def optimize_asymmetric_dual_balanced_accuracy(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    abstention_budget: float,
    max_candidates: int = 201,
) -> tuple[float, float]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    candidates = threshold_candidates(s, max_candidates=max_candidates)
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    prefix_pos = np.concatenate([[0], np.cumsum(sorted_y == 1)])
    total_pos = int(prefix_pos[-1])
    total_neg = len(y) - total_pos

    best_low = float(candidates[0])
    best_high = float(candidates[0])
    best_score = -np.inf
    best_coverage = -np.inf
    best_width = np.inf

    positions = np.searchsorted(sorted_scores, candidates, side="left")
    n = len(y)
    for low_idx, low in enumerate(candidates):
        low_pos = int(positions[low_idx])
        fn = int(prefix_pos[low_pos])
        tn = low_pos - fn
        for high_idx in range(low_idx, len(candidates)):
            high = float(candidates[high_idx])
            high_pos = int(positions[high_idx])
            abstained = high_pos - low_pos
            abstention_rate = abstained / n
            if abstention_rate > abstention_budget + 1e-12:
                continue
            tp = total_pos - int(prefix_pos[high_pos])
            fp = total_neg - (high_pos - int(prefix_pos[high_pos]))
            pos_denom = tp + fn
            neg_denom = tn + fp
            recall_pos = tp / pos_denom if pos_denom else 0.0
            recall_neg = tn / neg_denom if neg_denom else 0.0
            balanced_accuracy = (recall_pos + recall_neg) / 2.0
            coverage = 1.0 - abstention_rate
            width = high - float(low)
            if (
                balanced_accuracy > best_score + 1e-12
                or (
                    abs(balanced_accuracy - best_score) <= 1e-12
                    and (coverage > best_coverage + 1e-12 or (abs(coverage - best_coverage) <= 1e-12 and width < best_width))
                )
            ):
                best_score = balanced_accuracy
                best_coverage = coverage
                best_width = width
                best_low = float(low)
                best_high = high
    return best_low, best_high


def optimize_aucross_reject(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    target_abstention_rate: float,
    coverage_tolerance: float = 0.002,
    max_candidates: int = 201,
    required_thresholds: tuple[float, ...] = (),
) -> tuple[float, float]:
    if not 0.0 <= target_abstention_rate <= 1.0:
        raise ValueError("target_abstention_rate must be between 0 and 1")
    if coverage_tolerance < 0.0:
        raise ValueError("coverage_tolerance must be non-negative")

    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                threshold_candidates(s, max_candidates=max_candidates),
                np.asarray(required_thresholds, dtype=float),
            ]
        )
    )
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    prefix_pos = np.concatenate([[0], np.cumsum(sorted_y == 1)])
    prefix_neg = np.concatenate([[0], np.cumsum(sorted_y == 0)])
    prefix_correct = np.zeros(len(sorted_y) + 1, dtype=float)
    for idx, label in enumerate(sorted_y, start=1):
        prefix_correct[idx] = prefix_correct[idx - 1]
        if label == 1:
            prefix_correct[idx] += prefix_neg[idx - 1]

    total_pos = int(prefix_pos[-1])
    total_neg = int(prefix_neg[-1])
    total_correct = float(prefix_correct[-1])
    positions = np.searchsorted(sorted_scores, candidates, side="left")

    best_low: float | None = None
    best_high: float | None = None
    best_auc = -np.inf
    best_rate_gap = np.inf
    best_width = np.inf
    n = len(y)
    for low_idx, low in enumerate(candidates):
        low_pos = int(positions[low_idx])
        accepted_prefix_pos = int(prefix_pos[low_pos])
        accepted_prefix_neg = int(prefix_neg[low_pos])
        correct_prefix = float(prefix_correct[low_pos])
        for high_idx in range(low_idx, len(candidates)):
            high = float(candidates[high_idx])
            high_pos = int(positions[high_idx])
            abstention_rate = (high_pos - low_pos) / n
            rate_gap = abs(abstention_rate - target_abstention_rate)
            if rate_gap > coverage_tolerance + 1e-12:
                continue

            suffix_pos = total_pos - int(prefix_pos[high_pos])
            suffix_neg = total_neg - int(prefix_neg[high_pos])
            accepted_pos = accepted_prefix_pos + suffix_pos
            accepted_neg = accepted_prefix_neg + suffix_neg
            if accepted_pos == 0 or accepted_neg == 0:
                continue

            correct_suffix = total_correct - float(prefix_correct[high_pos]) - int(prefix_neg[high_pos]) * suffix_pos
            correct_cross = accepted_prefix_neg * suffix_pos
            auc = (correct_prefix + correct_suffix + correct_cross) / (accepted_pos * accepted_neg)
            width = high - float(low)
            if (
                auc > best_auc + 1e-12
                or (
                    abs(auc - best_auc) <= 1e-12
                    and (
                        rate_gap < best_rate_gap - 1e-12
                        or (abs(rate_gap - best_rate_gap) <= 1e-12 and width < best_width)
                    )
                )
            ):
                best_auc = auc
                best_rate_gap = rate_gap
                best_width = width
                best_low = float(low)
                best_high = high

    if best_low is None or best_high is None:
        raise ValueError("no threshold pair satisfies the AUCROSS matched-coverage tolerance")
    return best_low, best_high


def optimize_asymmetric_dual_matched_coverage(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    target_abstention_rate: float,
    coverage_tolerance: float = 0.002,
    max_candidates: int = 201,
    required_thresholds: tuple[float, ...] = (),
) -> tuple[float, float]:
    if not 0.0 <= target_abstention_rate <= 1.0:
        raise ValueError("target_abstention_rate must be between 0 and 1")
    if coverage_tolerance < 0.0:
        raise ValueError("coverage_tolerance must be non-negative")

    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    candidates = np.unique(
        np.concatenate(
            [
                threshold_candidates(s, max_candidates=max_candidates),
                np.asarray(required_thresholds, dtype=float),
            ]
        )
    )
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    sorted_y = y[order]
    prefix_pos = np.concatenate([[0], np.cumsum(sorted_y == 1)])
    total_pos = int(prefix_pos[-1])
    total_neg = len(y) - total_pos
    positions = np.searchsorted(sorted_scores, candidates, side="left")

    best_low: float | None = None
    best_high: float | None = None
    best_score = -np.inf
    best_rate_gap = np.inf
    best_width = np.inf
    n = len(y)
    for low_idx, low in enumerate(candidates):
        low_pos = int(positions[low_idx])
        fn = int(prefix_pos[low_pos])
        tn = low_pos - fn
        for high_idx in range(low_idx, len(candidates)):
            high = float(candidates[high_idx])
            high_pos = int(positions[high_idx])
            abstention_rate = (high_pos - low_pos) / n
            rate_gap = abs(abstention_rate - target_abstention_rate)
            if rate_gap > coverage_tolerance + 1e-12:
                continue
            tp = total_pos - int(prefix_pos[high_pos])
            fp = total_neg - (high_pos - int(prefix_pos[high_pos]))
            recall_pos = tp / (tp + fn) if tp + fn else 0.0
            recall_neg = tn / (tn + fp) if tn + fp else 0.0
            balanced_accuracy = (recall_pos + recall_neg) / 2.0
            width = high - float(low)
            if (
                balanced_accuracy > best_score + 1e-12
                or (
                    abs(balanced_accuracy - best_score) <= 1e-12
                    and (
                        rate_gap < best_rate_gap - 1e-12
                        or (abs(rate_gap - best_rate_gap) <= 1e-12 and width < best_width)
                    )
                )
            ):
                best_score = balanced_accuracy
                best_rate_gap = rate_gap
                best_width = width
                best_low = float(low)
                best_high = high

    if best_low is None or best_high is None:
        raise ValueError("no threshold pair satisfies the matched-coverage tolerance")
    return best_low, best_high


def fit_error_model_reject(
    y_true: Iterable[int],
    scores: Iterable[float],
    *,
    decision_threshold: float,
    target_abstention_rate: float,
) -> dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    errors = ((s >= decision_threshold).astype(int) != y).astype(int)
    if len(np.unique(errors)) < 2:
        priority = -np.abs(s - decision_threshold)
        cutoff = _high_score_cutoff(priority, target_abstention_rate)
        return {
            "model_type": "distance_fallback",
            "priority_cutoff": cutoff,
            "decision_threshold": float(decision_threshold),
        }

    design = np.column_stack([s, np.square(s)])
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=0,
    )
    model.fit(design, errors)
    linear = float(model.intercept_[0]) + design @ model.coef_[0]
    priority = 1.0 / (1.0 + np.exp(-linear))
    cutoff = _high_score_cutoff(priority, target_abstention_rate)
    return {
        "model_type": "quadratic_logistic",
        "priority_cutoff": cutoff,
        "decision_threshold": float(decision_threshold),
        "intercept": float(model.intercept_[0]),
        "coef_linear": float(model.coef_[0][0]),
        "coef_quadratic": float(model.coef_[0][1]),
    }


def evaluate_phase2_policy(
    y_true: Iterable[int],
    scores: Iterable[float],
    policy: Phase2Policy,
) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    decisions = _policy_decisions(s, policy)
    if policy.family in {"default_single", "optimized_single"}:
        base = evaluate_single_threshold(y, s, policy.lower_threshold)
    else:
        base = selective_metrics(y, decisions)

    accepted = decisions != ABSTAIN_LABEL
    abstained = ~accepted
    positives = y == 1
    negatives = y == 0
    true_positive_accepted = int(((decisions == 1) & positives).sum())

    lower_abstention = float(((abstained) & (s < policy.optimized_single_threshold)).mean())
    upper_abstention = float(((abstained) & (s >= policy.optimized_single_threshold)).mean())
    total_abstention = float(abstained.mean())
    allocation_asymmetry = (
        abs(lower_abstention - upper_abstention) / total_abstention if total_abstention > 0 else 0.0
    )

    out = {
        "balanced_selective_risk": 1.0 - float(base["balanced_accuracy"]),
        "macro_f1": float(base["macro_f1"]),
        "mcc": float(base["mcc"]),
        "minority_precision": float(base["minority_precision"]),
        "minority_recall": float(base["minority_recall"]),
        "specificity": float(base["negative_recall"]),
        "coverage": float(base["coverage"]),
        "positive_coverage": _safe_mean(accepted[positives]),
        "negative_coverage": _safe_mean(accepted[negatives]),
        "positive_abstention_rate": float(base["positive_abstention_rate"]),
        "negative_abstention_rate": float(base["negative_abstention_rate"]),
        "minority_share_in_abstention": float(base["minority_proportion_abstained"]),
        "effective_minority_recall": true_positive_accepted / int(positives.sum()) if positives.any() else float("nan"),
        "lower_side_abstention": lower_abstention,
        "upper_side_abstention": upper_abstention,
        "actual_abstention_rate": total_abstention,
        "allocation_asymmetry": allocation_asymmetry,
    }
    return out


def run_phase2_smoke_test(
    bundle: DatasetBundle,
    split: Any,
    *,
    model_name: str = "lightgbm",
    seed: int = 7,
    train_ratios: tuple[str | None, ...] = (None, "1:3", "1:5", "1:10"),
    budgets: tuple[float, ...] = (0.01, 0.05, 0.10),
    max_candidates: int = 201,
    fixed_controlled_train_total: int | None = None,
    include_matched_coverage: bool = False,
    include_matched_baselines: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ratio in train_ratios:
        selected_rel = sample_training_indices_for_ratio(
            bundle.y.iloc[split.train].to_numpy(),
            majority_per_minority=ratio,
            seed=seed,
            fixed_total=fixed_controlled_train_total if ratio is not None else None,
        )
        train_idx = np.sort(split.train[selected_rel])
        train_counts = class_counts(bundle.y.iloc[train_idx].to_numpy())
        model = build_model_pipeline(model_name, bundle.features.iloc[train_idx], seed=seed)
        model.fit(bundle.features.iloc[train_idx], bundle.y.iloc[train_idx])

        validation_y = bundle.y.iloc[split.threshold_selection].to_numpy()
        validation_scores = model.predict_proba(bundle.features.iloc[split.threshold_selection])[:, 1]
        test_y = bundle.y.iloc[split.test].to_numpy()
        test_scores = model.predict_proba(bundle.features.iloc[split.test])[:, 1]
        policies = select_phase2_policies(
            validation_y,
            validation_scores,
            budgets=budgets,
            max_candidates=max_candidates,
            include_matched_coverage=include_matched_coverage,
            include_matched_baselines=include_matched_baselines,
            policy_seed=seed,
        )
        rows.extend(
            phase2_rows_from_scores(
                validation_y,
                validation_scores,
                test_y,
                test_scores,
                policies,
                dataset=bundle.name,
                model=model_name,
                seed=seed,
                training_distribution="natural" if ratio is None else ratio,
                train_rows=len(train_idx),
                train_positive_n=train_counts["positives"],
                train_negative_n=train_counts["negatives"],
                realized_train_majority_per_minority=train_counts["negatives"] / train_counts["positives"],
                calibration="none",
                source="trained_in_project",
            )
        )
    return pd.DataFrame(rows)


def phase2_rows_from_scores(
    validation_y: Iterable[int],
    validation_scores: Iterable[float],
    test_y: Iterable[int],
    test_scores: Iterable[float],
    policies: list[Phase2Policy],
    *,
    dataset: str,
    model: str,
    seed: int,
    training_distribution: str,
    train_rows: int | None,
    train_positive_n: int | None,
    train_negative_n: int | None,
    realized_train_majority_per_minority: float | None,
    calibration: str = "none",
    source: str = "",
) -> list[dict[str, Any]]:
    validation_y_arr = np.asarray(list(validation_y), dtype=int)
    validation_scores_arr = np.asarray(list(validation_scores), dtype=float)
    test_y_arr = np.asarray(list(test_y), dtype=int)
    test_scores_arr = np.asarray(list(test_scores), dtype=float)
    rows: list[dict[str, Any]] = []
    for policy in policies:
        validation_metrics = evaluate_phase2_policy(validation_y_arr, validation_scores_arr, policy)
        test_metrics = evaluate_phase2_policy(test_y_arr, test_scores_arr, policy)
        validation_diagnostics = _phase2_diagnostics(
            validation_y_arr,
            validation_scores_arr,
            policy,
            actual_abstention_rate=float(validation_metrics["actual_abstention_rate"]),
        )
        test_diagnostics = _phase2_diagnostics(
            test_y_arr,
            test_scores_arr,
            policy,
            actual_abstention_rate=float(test_metrics["actual_abstention_rate"]),
        )
        row = {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "training_distribution": training_distribution,
            "train_rows": train_rows,
            "train_positive_n": train_positive_n,
            "train_negative_n": train_negative_n,
            "realized_train_majority_per_minority": realized_train_majority_per_minority,
            "calibration": calibration,
            "source": source,
            "policy_label": policy.label,
            "policy_family": policy.family,
            "optimization_metric": policy.optimization_metric,
            "abstention_budget": policy.abstention_budget,
            "optimized_single_threshold": policy.optimized_single_threshold,
            "lower_threshold": policy.lower_threshold,
            "upper_threshold": policy.upper_threshold,
            "abstention_width": policy.upper_threshold - policy.lower_threshold,
            "validation_coverage": validation_metrics["coverage"],
            "test_coverage": test_metrics["coverage"],
            "coverage_drift": test_metrics["coverage"] - validation_metrics["coverage"],
        }
        row.update(test_metrics)
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_diagnostics.items()})
        row.update({f"test_{key}": value for key, value in test_diagnostics.items()})
        rows.append(row)
    return rows


def run_phase2_from_scores(
    validation_y: Iterable[int],
    validation_scores: Iterable[float],
    test_y: Iterable[int],
    test_scores: Iterable[float],
    *,
    dataset: str,
    model: str,
    seed: int,
    training_distribution: str,
    train_rows: int | None = None,
    train_positive_n: int | None = None,
    train_negative_n: int | None = None,
    realized_train_majority_per_minority: float | None = None,
    calibration: str = "none",
    source: str = "",
    budgets: tuple[float, ...] = (0.01, 0.05, 0.10),
    max_candidates: int = 201,
    include_matched_coverage: bool = False,
    include_matched_baselines: bool = False,
) -> pd.DataFrame:
    validation_y_arr = np.asarray(list(validation_y), dtype=int)
    validation_scores_arr = np.asarray(list(validation_scores), dtype=float)
    policies = select_phase2_policies(
        validation_y_arr,
        validation_scores_arr,
        budgets=budgets,
        max_candidates=max_candidates,
        include_matched_coverage=include_matched_coverage,
        include_matched_baselines=include_matched_baselines,
        policy_seed=seed,
    )
    return pd.DataFrame(
        phase2_rows_from_scores(
            validation_y_arr,
            validation_scores_arr,
            test_y,
            test_scores,
            policies,
            dataset=dataset,
            model=model,
            seed=seed,
            training_distribution=training_distribution,
            train_rows=train_rows,
            train_positive_n=train_positive_n,
            train_negative_n=train_negative_n,
            realized_train_majority_per_minority=realized_train_majority_per_minority,
            calibration=calibration,
            source=source,
        )
    )


def _safe_mean(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def _policy_decisions(scores: np.ndarray, policy: Phase2Policy) -> np.ndarray:
    if policy.family in {"default_single", "optimized_single"}:
        return (scores >= policy.lower_threshold).astype(int)
    if policy.family in INTERVAL_DECISION_FAMILIES:
        return dual_threshold_predictions(scores, policy.lower_threshold, policy.upper_threshold)

    base_predictions = (scores >= policy.optimized_single_threshold).astype(int)
    decisions = base_predictions.copy()
    reject_mask = _policy_reject_mask(scores, policy)
    decisions[reject_mask] = ABSTAIN_LABEL
    return decisions


def _policy_reject_mask(scores: np.ndarray, policy: Phase2Policy) -> np.ndarray:
    if policy.family in INTERVAL_REJECT_FAMILIES:
        return (scores >= policy.lower_threshold) & (scores < policy.upper_threshold)
    if policy.family == "matched_random_reject":
        priority = _random_priority_scores(len(scores), seed=int(policy.metadata["random_seed"]))
        return priority >= float(policy.metadata["priority_cutoff"])
    if policy.family == "matched_error_model_reject":
        priority = _error_model_priority_scores(scores, policy.metadata)
        return priority >= float(policy.metadata["priority_cutoff"])
    raise ValueError(f"unsupported policy family: {policy.family}")


def _phase2_diagnostics(
    y_true: np.ndarray,
    scores: np.ndarray,
    policy: Phase2Policy,
    *,
    actual_abstention_rate: float,
) -> dict[str, float]:
    reference_errors = _reference_error_mask(y_true, scores, policy.optimized_single_threshold)
    overall_error_rate = float(reference_errors.mean()) if len(reference_errors) else float("nan")
    boundary_mask = _boundary_reject_mask(scores, policy.optimized_single_threshold, actual_abstention_rate)
    policy_reject_mask = _policy_decisions(scores, policy) == ABSTAIN_LABEL
    diagnostics = {"reference_error_rate": overall_error_rate}
    diagnostics.update(_reject_error_stats(reference_errors, boundary_mask, prefix="boundary"))
    diagnostics.update(_reject_error_stats(reference_errors, policy_reject_mask, prefix="reject"))
    return diagnostics


def _reference_error_mask(y_true: np.ndarray, scores: np.ndarray, decision_threshold: float) -> np.ndarray:
    return (scores >= decision_threshold).astype(int) != y_true


def _boundary_reject_mask(scores: np.ndarray, boundary: float, target_abstention_rate: float) -> np.ndarray:
    if target_abstention_rate <= 0.0:
        return np.zeros(len(scores), dtype=bool)
    low, high = symmetric_dual_thresholds(scores, boundary=boundary, abstention_budget=target_abstention_rate)
    return (scores >= low) & (scores < high)


def _reject_error_stats(reference_errors: np.ndarray, reject_mask: np.ndarray, *, prefix: str) -> dict[str, float]:
    total_errors = int(reference_errors.sum())
    reject_n = int(reject_mask.sum())
    overall_error_rate = float(reference_errors.mean()) if len(reference_errors) else float("nan")
    if reject_n == 0:
        return {
            f"{prefix}_error_rate": float("nan"),
            f"{prefix}_error_capture": 0.0,
            f"{prefix}_error_enrichment": float("nan"),
        }

    rejected_errors = int((reference_errors & reject_mask).sum())
    reject_error_rate = rejected_errors / reject_n
    error_capture = rejected_errors / total_errors if total_errors else 0.0
    error_enrichment = reject_error_rate / overall_error_rate if overall_error_rate > 0 else float("nan")
    return {
        f"{prefix}_error_rate": float(reject_error_rate),
        f"{prefix}_error_capture": float(error_capture),
        f"{prefix}_error_enrichment": float(error_enrichment),
    }


def _high_score_cutoff(values: Iterable[float], target_abstention_rate: float) -> float:
    scores = np.asarray(list(values), dtype=float)
    if len(scores) == 0:
        raise ValueError("values must not be empty")
    target_total = int(np.floor(len(scores) * target_abstention_rate + 1e-12))
    if target_total <= 0:
        return float(np.nextafter(scores.max(), np.inf))

    unique_scores, counts = np.unique(scores, return_counts=True)
    used = 0
    cutoff = float(np.nextafter(scores.max(), np.inf))
    for value, count in zip(unique_scores[::-1], counts[::-1], strict=True):
        if used + int(count) > target_total:
            break
        used += int(count)
        cutoff = float(value)
    return cutoff


def _random_priority_scores(n: int, *, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def _error_model_priority_scores(scores: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    model_type = str(metadata.get("model_type", "distance_fallback"))
    decision_threshold = float(metadata.get("decision_threshold", 0.5))
    if model_type == "quadratic_logistic":
        linear = (
            float(metadata["intercept"])
            + float(metadata["coef_linear"]) * scores
            + float(metadata["coef_quadratic"]) * np.square(scores)
        )
        return 1.0 / (1.0 + np.exp(-linear))
    return -np.abs(scores - decision_threshold)
