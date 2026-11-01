from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CALIBRATION_COLUMNS = {
    "none": "prob_none",
    "platt": "prob_platt",
    "isotonic": "prob_isotonic",
    "temperature": "prob_temperature",
}
MODELS = ("logistic_regression", "lightgbm", "mlp")
SEEDS = (7, 17, 177)
AUDITED_POLICIES = ("symmetric_dual_0.1", "asymmetric_dual_0.1", "matched_asymmetric_dual_0.1")


def controlled_imbalance_table(differences: pd.DataFrame) -> pd.DataFrame:
    controlled = differences[
        differences["training_distribution"].astype(str).str.match(r"^1:\d+$")
        & differences["policy_label"].isin(["symmetric_dual_0.1", "asymmetric_dual_0.1"])
        & differences["comparison"].isin(["dual_minus_optimized_single", "asymmetric_minus_symmetric"])
    ].copy()
    grouped = (
        controlled.groupby(
            ["dataset", "training_distribution", "comparison", "policy_label"], dropna=False
        )["delta_balanced_selective_risk"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped["majority_per_minority"] = grouped["training_distribution"].str.split(":").str[1].astype(int)
    return grouped.sort_values(
        ["dataset", "majority_per_minority", "comparison", "policy_label"]
    ).reset_index(drop=True)


def single_threshold_table(results: pd.DataFrame) -> pd.DataFrame:
    metrics = ["balanced_selective_risk", "macro_f1", "minority_recall"]
    keys = ["dataset", "model", "seed", "training_distribution", "calibration"]
    selected = results[results["policy_label"].isin(["default_single_0_5", "optimized_single"])]
    wide = selected.pivot(index=keys, columns="policy_label", values=metrics)
    rows = []
    for dataset, group in wide.groupby(level="dataset"):
        row: dict[str, float | int | str] = {"dataset": dataset, "n_units": len(group)}
        for metric in metrics:
            default = group[(metric, "default_single_0_5")]
            optimized = group[(metric, "optimized_single")]
            row[f"default_{metric}_mean"] = float(default.mean())
            row[f"optimized_{metric}_mean"] = float(optimized.mean())
            row[f"delta_{metric}_mean"] = float((optimized - default).mean())
        optimized_rows = selected[
            (selected["dataset"] == dataset) & (selected["policy_label"] == "optimized_single")
        ]
        row["optimized_threshold_mean"] = float(optimized_rows["optimized_single_threshold"].mean())
        row["optimized_threshold_std"] = float(optimized_rows["optimized_single_threshold"].std())
        rows.append(row)
    return pd.DataFrame(rows)


def class_conditional_coverage_table(results: pd.DataFrame) -> pd.DataFrame:
    selected = results[
        results["policy_label"].isin(
            ["optimized_single", "symmetric_dual_0.1", "asymmetric_dual_0.1"]
        )
    ]
    return (
        selected.groupby(["dataset", "policy_label"], dropna=False)
        .agg(
            n_units=("coverage", "size"),
            coverage_mean=("coverage", "mean"),
            positive_coverage_mean=("positive_coverage", "mean"),
            negative_coverage_mean=("negative_coverage", "mean"),
            effective_minority_recall_mean=("effective_minority_recall", "mean"),
        )
        .reset_index()
    )


def tie_budget_audit(results: pd.DataFrame) -> pd.DataFrame:
    selected = results[results["policy_family"].str.contains("dual", na=False)].copy()
    selected["validation_rejection"] = 1.0 - selected["validation_coverage"]
    selected["budget_overshoot"] = selected["validation_rejection"] - selected["abstention_budget"]
    return (
        selected.groupby(["dataset", "calibration", "policy_family"], dropna=False)
        .agg(
            n=("budget_overshoot", "size"),
            max_validation_rejection=("validation_rejection", "max"),
            max_budget_overshoot=("budget_overshoot", "max"),
            min_budget_overshoot=("budget_overshoot", "min"),
        )
        .reset_index()
    )


def calibration_audit(
    results: pd.DataFrame,
    prediction_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(prediction_root)
    score_rows: list[dict[str, float | int | str]] = []
    set_rows: list[dict[str, float | int | str]] = []
    baf_results = results[results["dataset"] == "baf_full"]

    for model in MODELS:
        for seed in SEEDS:
            model_results = baf_results[(baf_results["model"] == model) & (baf_results["seed"] == seed)]
            for split in ("validation", "test"):
                predictions = pd.read_parquet(root / model / f"seed_{seed}" / f"{split}.parquet")
                base_scores = predictions[CALIBRATION_COLUMNS["none"]].to_numpy(dtype=float)
                row_ids = predictions["row_id"].to_numpy()
                base_sets = _policy_sets(model_results, predictions, split, "none", row_ids)

                for calibration, column in CALIBRATION_COLUMNS.items():
                    scores = predictions[column].to_numpy(dtype=float)
                    unique, counts = np.unique(scores, return_counts=True)
                    score_rows.append(
                        {
                            "model": model,
                            "seed": seed,
                            "split": split,
                            "calibration": calibration,
                            "spearman_vs_none": _rank_correlation(base_scores, scores),
                            "n_rows": len(scores),
                            "n_unique_scores": len(unique),
                            "n_tie_groups": int((counts > 1).sum()),
                            "largest_tie_group": int(counts.max()),
                        }
                    )
                    calibrated_sets = _policy_sets(
                        model_results, predictions, split, calibration, row_ids
                    )
                    for policy_label in AUDITED_POLICIES:
                        base_accepted, base_rejected = base_sets[policy_label]
                        accepted, rejected = calibrated_sets[policy_label]
                        set_rows.append(
                            {
                                "model": model,
                                "seed": seed,
                                "split": split,
                                "calibration": calibration,
                                "policy_label": policy_label,
                                "accepted_jaccard_vs_none": _jaccard(base_accepted, accepted),
                                "rejected_jaccard_vs_none": _jaccard(base_rejected, rejected),
                                "accepted_n": len(accepted),
                                "rejected_n": len(rejected),
                            }
                        )

    return pd.DataFrame(score_rows), pd.DataFrame(set_rows)


def _policy_sets(
    model_results: pd.DataFrame,
    predictions: pd.DataFrame,
    split: str,
    calibration: str,
    row_ids: np.ndarray,
) -> dict[str, tuple[set[int], set[int]]]:
    scores = predictions[CALIBRATION_COLUMNS[calibration]].to_numpy(dtype=float)
    sets: dict[str, tuple[set[int], set[int]]] = {}
    for policy_label in AUDITED_POLICIES:
        row = model_results[
            (model_results["calibration"] == calibration)
            & (model_results["policy_label"] == policy_label)
        ].iloc[0]
        rejected_mask = (scores >= float(row["lower_threshold"])) & (
            scores < float(row["upper_threshold"])
        )
        rejected = set(map(int, row_ids[rejected_mask]))
        accepted = set(map(int, row_ids[~rejected_mask]))
        sets[policy_label] = accepted, rejected
    return sets


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    return float(left_rank.corr(right_rank))


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create reporting and calibration-audit artifacts.")
    parser.add_argument(
        "--results", default="results_imbalance_selective/final_phase2/all_phase2_results.csv"
    )
    parser.add_argument(
        "--diffs", default="results_imbalance_selective/final_phase2/paired_differences.csv"
    )
    parser.add_argument(
        "--prediction-root",
        default="/Users/starsfalling/Desktop/Project/baf-rcast-study/outputs/predictions",
    )
    parser.add_argument(
        "--output-dir",
        default="results_imbalance_selective/final_phase2/paper_artifacts/revision_audit",
    )
    args = parser.parse_args(argv)

    results = pd.read_csv(args.results)
    differences = pd.read_csv(args.diffs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    controlled_imbalance_table(differences).to_csv(
        output / "controlled_imbalance_effects.csv", index=False
    )
    single_threshold_table(results).to_csv(
        output / "default_vs_optimized_single.csv", index=False
    )
    class_conditional_coverage_table(results).to_csv(
        output / "class_conditional_coverage.csv", index=False
    )
    tie_budget_audit(results).to_csv(output / "tie_budget_audit.csv", index=False)
    score_audit, set_audit = calibration_audit(results, args.prediction_root)
    score_audit.to_csv(output / "calibration_score_audit.csv", index=False)
    set_audit.to_csv(output / "calibration_policy_set_audit.csv", index=False)

    print(f"wrote revision audit artifacts to {output}")
    print(single_threshold_table(results).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
