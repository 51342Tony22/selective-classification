from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common.data_audit import audit_dataframe
from common.data_split import (
    SplitIndices,
    repeated_stratified_cv_indices,
    stratified_predefined_test_split,
    stratified_train_calibration_threshold_test_split,
    temporal_split_indices,
)
from common.resampling import class_counts, downsample_majority_indices
from common.threshold_search import default_single_threshold, optimize_dual_threshold, optimize_single_threshold
from common.selective_metrics import evaluate_dual_threshold, evaluate_single_threshold
from imbalance_selective.datasets import DatasetBundle, load_config, load_dataset_from_config
from imbalance_selective.preprocessing import build_logistic_regression_pipeline


def build_split(bundle: DatasetBundle, config: dict[str, Any], *, seed: int = 7) -> SplitIndices:
    split_config = config.get("dataset", {}).get("split", {})
    split_type = split_config.get("type")

    if split_type == "temporal":
        month_column = bundle.metadata.get("month_column")
        if not month_column:
            raise ValueError("temporal split requires month_column metadata")
        return temporal_split_indices(bundle.frame[month_column])

    if split_type == "stratified_train_calibration_validation_test":
        return stratified_train_calibration_threshold_test_split(
            bundle.y,
            train_size=float(split_config.get("train", 0.60)),
            calibration_size=float(split_config.get("calibration", 0.15)),
            threshold_selection_size=float(split_config.get("threshold_selection", 0.15)),
            test_size=float(split_config.get("test", 0.10)),
            seed=seed,
        )

    if split_type == "predefined_train_test":
        partition_column = bundle.metadata.get("partition_column")
        if not partition_column or partition_column not in bundle.frame.columns:
            raise ValueError("predefined_train_test split requires partition metadata")
        return stratified_predefined_test_split(
            bundle.y,
            bundle.frame[partition_column],
            train_label=split_config.get("train_label", "train"),
            test_label=split_config.get("test_label", "test"),
            train_size=float(split_config.get("train", 0.70)),
            calibration_size=float(split_config.get("calibration", 0.15)),
            threshold_selection_size=float(split_config.get("threshold_selection", 0.15)),
            seed=seed,
        )

    if split_type == "repeated_stratified_cv":
        folds = repeated_stratified_cv_indices(
            bundle.y,
            n_splits=int(split_config.get("folds", 5)),
            n_repeats=int(split_config.get("repeats", 3)),
            seed=seed,
        )
        first = folds[0]
        train_pool = np.asarray(first["train"], dtype=int)
        test = np.asarray(first["test"], dtype=int)
        inner = stratified_train_calibration_threshold_test_split(bundle.y.iloc[train_pool].to_numpy(), seed=seed)
        return SplitIndices(
            train=np.sort(train_pool[inner.train]),
            calibration=np.sort(train_pool[inner.calibration]),
            threshold_selection=np.sort(train_pool[inner.threshold_selection]),
            test=np.sort(test),
        )

    raise ValueError(f"unsupported split type: {split_type}")


def run_smoke_test_on_bundle(
    bundle: DatasetBundle,
    split: SplitIndices,
    *,
    seed: int = 7,
    abstention_budget: float = 0.05,
    max_candidates: int | None = 101,
    train_ratio: str | float | int | None = None,
) -> pd.DataFrame:
    train_idx = split.train
    experiment = "natural_imbalance"
    if train_ratio is not None:
        selected_rel = downsample_majority_indices(
            bundle.y.iloc[split.train].to_numpy(),
            majority_per_minority=train_ratio,
            seed=seed,
        )
        train_idx = np.sort(split.train[selected_rel])
        experiment = f"controlled_train_{train_ratio}"
    train_counts = class_counts(bundle.y.iloc[train_idx].to_numpy())
    train_positive_n = train_counts["positives"]
    train_negative_n = train_counts["negatives"]
    realized_ratio = train_negative_n / train_positive_n if train_positive_n else float("nan")

    model = build_logistic_regression_pipeline(bundle.features.iloc[train_idx], seed=seed)
    model.fit(bundle.features.iloc[train_idx], bundle.y.iloc[train_idx])

    threshold_scores = model.predict_proba(bundle.features.iloc[split.threshold_selection])[:, 1]
    threshold_y = bundle.y.iloc[split.threshold_selection].to_numpy()
    test_scores = model.predict_proba(bundle.features.iloc[split.test])[:, 1]
    test_y = bundle.y.iloc[split.test].to_numpy()

    default_result = default_single_threshold(threshold_y, threshold_scores, threshold=0.5)
    single_result = optimize_single_threshold(
        threshold_y,
        threshold_scores,
        metric="macro_f1",
        max_candidates=max_candidates,
    )
    dual_result = optimize_dual_threshold(
        threshold_y,
        threshold_scores,
        abstention_budget=abstention_budget,
        metric="macro_f1",
        max_candidates=max_candidates,
    )

    rows = [
        _row_from_metrics(
            evaluate_single_threshold(test_y, test_scores, float(default_result.params["threshold"])),
            dataset=bundle.name,
            policy_label="single_default_0_5",
            seed=seed,
            selected_on="threshold_selection",
            experiment=experiment,
            train_rows=len(train_idx),
            train_positive_n=train_positive_n,
            train_negative_n=train_negative_n,
            requested_train_ratio=str(train_ratio) if train_ratio is not None else "",
            realized_train_majority_per_minority=realized_ratio,
        ),
        _row_from_metrics(
            evaluate_single_threshold(test_y, test_scores, float(single_result.params["threshold"])),
            dataset=bundle.name,
            policy_label="single_opt_macro_f1",
            seed=seed,
            selected_on="threshold_selection",
            experiment=experiment,
            train_rows=len(train_idx),
            train_positive_n=train_positive_n,
            train_negative_n=train_negative_n,
            requested_train_ratio=str(train_ratio) if train_ratio is not None else "",
            realized_train_majority_per_minority=realized_ratio,
        ),
        _row_from_metrics(
            evaluate_dual_threshold(test_y, test_scores, float(dual_result.params["low"]), float(dual_result.params["high"])),
            dataset=bundle.name,
            policy_label=f"dual_budget_{abstention_budget:g}",
            seed=seed,
            selected_on="threshold_selection",
            experiment=experiment,
            train_rows=len(train_idx),
            train_positive_n=train_positive_n,
            train_negative_n=train_negative_n,
            requested_train_ratio=str(train_ratio) if train_ratio is not None else "",
            realized_train_majority_per_minority=realized_ratio,
        ),
    ]
    return pd.DataFrame(rows)


def run_smoke_test_from_config(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    seed: int = 7,
    train_ratio: str | float | int | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    bundle = load_dataset_from_config(config)
    split = build_split(bundle, config, seed=seed)
    results = run_smoke_test_on_bundle(bundle, split, seed=seed, train_ratio=train_ratio)

    audit = audit_dataframe(
        bundle.frame,
        dataset_name=bundle.name,
        target_candidates=[bundle.target],
        positive_label=1,
        month_column=bundle.metadata.get("month_column"),
        source_path=bundle.metadata.get("path"),
    )
    results.insert(0, "source_rows", audit["rows"])
    results.insert(1, "source_positive_rate", audit["positive_rate"])

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out, index=False)
    return results


def _row_from_metrics(
    metrics: dict[str, float],
    *,
    dataset: str,
    policy_label: str,
    seed: int,
    selected_on: str,
    experiment: str,
    train_rows: int,
    train_positive_n: int,
    train_negative_n: int,
    requested_train_ratio: str,
    realized_train_majority_per_minority: float,
) -> dict[str, float | str | int]:
    keep = [
        "coverage",
        "abstention_rate",
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "minority_recall",
        "minority_precision",
        "positive_abstention_rate",
        "negative_abstention_rate",
        "minority_proportion_abstained",
        "low",
        "high",
        "threshold",
    ]
    row: dict[str, float | str | int] = {
        "dataset": dataset,
        "model": "logistic_regression",
        "seed": seed,
        "calibration": "none",
        "policy_label": policy_label,
        "selected_on": selected_on,
        "experiment": experiment,
        "train_rows": train_rows,
        "train_positive_n": train_positive_n,
        "train_negative_n": train_negative_n,
        "requested_train_ratio": requested_train_ratio,
        "realized_train_majority_per_minority": realized_train_majority_per_minority,
    }
    row.update({key: metrics.get(key, float("nan")) for key in keep})
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the minimal imbalance-selective smoke test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratio", default=None, help="Optional controlled training imbalance, e.g. 1:10.")
    args = parser.parse_args(argv)

    results = run_smoke_test_from_config(args.config, output_path=args.output, seed=args.seed, train_ratio=args.train_ratio)
    print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
