from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.data_audit import audit_dataframe
from common.data_split import SplitIndices
from common.resampling import class_counts, downsample_majority_indices
from common.selective_metrics import evaluate_dual_threshold, evaluate_single_threshold
from common.threshold_search import default_single_threshold, optimize_dual_threshold, optimize_single_threshold
from imbalance_selective.datasets import DatasetBundle, load_config, load_dataset_from_config
from imbalance_selective.preprocessing import build_model_pipeline
from imbalance_selective.smoke_test import build_split


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    train_ratio: str | None = None


DEFAULT_SEEDS = (7, 17, 177)
DEFAULT_EXPERIMENTS = (
    ExperimentSpec(label="natural", train_ratio=None),
    ExperimentSpec(label="controlled_1_to_3", train_ratio="1:3"),
)
DEFAULT_BUDGETS = (0.01, 0.05, 0.10)
DEFAULT_MODELS = ("logistic_regression", "lightgbm", "mlp")


def run_dataset_experiments(
    config_path: str | Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    experiments: tuple[ExperimentSpec, ...] = DEFAULT_EXPERIMENTS,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    model_names: tuple[str, ...] = DEFAULT_MODELS,
    max_candidates: int = 201,
) -> pd.DataFrame:
    config = load_config(config_path)
    bundle = load_dataset_from_config(config)
    audit = audit_dataframe(
        bundle.frame,
        dataset_name=bundle.name,
        target_candidates=[bundle.target],
        positive_label=1,
        month_column=bundle.metadata.get("month_column"),
        source_path=bundle.metadata.get("path"),
    )

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        split = build_split(bundle, config, seed=seed)
        for model_name in model_names:
            for experiment in experiments:
                rows.extend(
                    run_one_experiment(
                        bundle,
                        split,
                        seed=seed,
                        model_name=model_name,
                        experiment=experiment,
                        budgets=budgets,
                        max_candidates=max_candidates,
                        source_rows=int(audit["rows"]),
                        source_positive_rate=float(audit["positive_rate"]),
                    )
                )
    return pd.DataFrame(rows)


def run_one_experiment(
    bundle: DatasetBundle,
    split: SplitIndices,
    *,
    seed: int,
    model_name: str,
    experiment: ExperimentSpec,
    budgets: tuple[float, ...],
    max_candidates: int,
    source_rows: int,
    source_positive_rate: float,
) -> list[dict[str, Any]]:
    train_idx = split.train
    if experiment.train_ratio is not None:
        selected_rel = downsample_majority_indices(
            bundle.y.iloc[split.train].to_numpy(),
            majority_per_minority=experiment.train_ratio,
            seed=seed,
        )
        train_idx = np.sort(split.train[selected_rel])

    train_counts = class_counts(bundle.y.iloc[train_idx].to_numpy())
    positive_n = train_counts["positives"]
    negative_n = train_counts["negatives"]
    realized_ratio = negative_n / positive_n if positive_n else float("nan")

    model = build_model_pipeline(model_name, bundle.features.iloc[train_idx], seed=seed)
    model.fit(bundle.features.iloc[train_idx], bundle.y.iloc[train_idx])

    threshold_y = bundle.y.iloc[split.threshold_selection].to_numpy()
    threshold_scores = model.predict_proba(bundle.features.iloc[split.threshold_selection])[:, 1]
    test_y = bundle.y.iloc[split.test].to_numpy()
    test_scores = model.predict_proba(bundle.features.iloc[split.test])[:, 1]

    policies = [
        (
            "single_default_0_5",
            default_single_threshold(threshold_y, threshold_scores, threshold=0.5).params,
            "fixed",
        ),
        (
            "single_opt_macro_f1",
            optimize_single_threshold(
                threshold_y,
                threshold_scores,
                metric="macro_f1",
                max_candidates=max_candidates,
            ).params,
            "macro_f1",
        ),
    ]
    for budget in budgets:
        policies.append(
            (
                f"dual_budget_{budget:g}",
                optimize_dual_threshold(
                    threshold_y,
                    threshold_scores,
                    abstention_budget=budget,
                    metric="macro_f1",
                    max_candidates=max_candidates,
                ).params,
                "macro_f1",
            )
        )

    return [
        _evaluate_policy_row(
            test_y,
            test_scores,
            params,
            dataset=bundle.name,
            model_name=model_name,
            seed=seed,
            policy_label=policy_label,
            optimization_metric=optimization_metric,
            experiment=experiment,
            train_rows=len(train_idx),
            train_positive_n=positive_n,
            train_negative_n=negative_n,
            realized_train_majority_per_minority=realized_ratio,
            source_rows=source_rows,
            source_positive_rate=source_positive_rate,
        )
        for policy_label, params, optimization_metric in policies
    ]


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["macro_f1", "balanced_accuracy", "mcc", "minority_recall", "minority_precision", "coverage"]
    group_cols = ["dataset", "model", "experiment", "policy_label"]
    grouped = results.groupby(group_cols, dropna=False)
    pieces = []
    for metric in metrics:
        stat = grouped[metric].agg(["mean", "std"]).rename(
            columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}
        )
        pieces.append(stat)
    out = pd.concat(pieces, axis=1).reset_index()
    return out.sort_values(["dataset", "model", "experiment", "policy_label"]).reset_index(drop=True)


def write_dataset_tables(results: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for dataset, subset in results.groupby("dataset", sort=True):
        aggregate = aggregate_results(subset)
        path = out_dir / f"{dataset}_summary.csv"
        aggregate.to_csv(path, index=False)
        written.append(path)
    return written


def run_all(
    config_paths: list[str | Path],
    *,
    output_dir: str | Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    model_names: tuple[str, ...] = DEFAULT_MODELS,
    max_candidates: int = 201,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = pd.concat(
        [
            run_dataset_experiments(
                path,
                seeds=seeds,
                model_names=model_names,
                max_candidates=max_candidates,
            )
            for path in config_paths
        ],
        ignore_index=True,
    )
    summary = aggregate_results(all_results)
    all_results.to_csv(out_dir / "all_results.csv", index=False)
    summary.to_csv(out_dir / "summary_by_dataset_experiment_policy.csv", index=False)
    write_dataset_tables(all_results, out_dir / "dataset_tables")
    return all_results, summary


def _evaluate_policy_row(
    y_true: np.ndarray,
    scores: np.ndarray,
    params: dict[str, float | str],
    *,
    dataset: str,
    model_name: str,
    seed: int,
    policy_label: str,
    optimization_metric: str,
    experiment: ExperimentSpec,
    train_rows: int,
    train_positive_n: int,
    train_negative_n: int,
    realized_train_majority_per_minority: float,
    source_rows: int,
    source_positive_rate: float,
) -> dict[str, Any]:
    if params["policy"] == "single":
        metrics = evaluate_single_threshold(y_true, scores, float(params["threshold"]))
    elif params["policy"] == "dual":
        metrics = evaluate_dual_threshold(y_true, scores, float(params["low"]), float(params["high"]))
    else:
        raise ValueError(f"unsupported policy params: {params}")

    row: dict[str, Any] = {
        "dataset": dataset,
        "source_rows": source_rows,
        "source_positive_rate": source_positive_rate,
        "model": model_name,
        "seed": seed,
        "calibration": "none",
        "experiment": experiment.label,
        "requested_train_ratio": experiment.train_ratio or "",
        "train_rows": train_rows,
        "train_positive_n": train_positive_n,
        "train_negative_n": train_negative_n,
        "realized_train_majority_per_minority": realized_train_majority_per_minority,
        "policy_label": policy_label,
        "optimization_metric": optimization_metric,
    }
    row.update(metrics)
    return row
