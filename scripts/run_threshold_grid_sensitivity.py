from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imbalance_selective.datasets import load_config, load_dataset_from_config
from imbalance_selective.phase2_policies import run_phase2_from_scores
from imbalance_selective.preprocessing import build_model_pipeline
from imbalance_selective.smoke_test import build_split


SEEDS = (7, 17, 177)
GRIDS = (41, 101, 201)


def run_baf_full(prediction_root: str | Path) -> list[pd.DataFrame]:
    root = Path(prediction_root)
    chunks = []
    for seed in SEEDS:
        validation = pd.read_parquet(root / "lightgbm" / f"seed_{seed}" / "validation.parquet")
        test = pd.read_parquet(root / "lightgbm" / f"seed_{seed}" / "test.parquet")
        for grid in GRIDS:
            chunk = run_phase2_from_scores(
                validation["y_true"].to_numpy(),
                validation["prob_none"].to_numpy(),
                test["y_true"].to_numpy(),
                test["prob_none"].to_numpy(),
                dataset="baf_full",
                model="lightgbm",
                seed=seed,
                training_distribution="natural_full",
                calibration="none",
                source="threshold_grid_sensitivity",
                budgets=(0.01, 0.05),
                max_candidates=grid,
            )
            chunk["candidate_grid"] = grid
            chunks.append(chunk)
    return chunks


def run_online_shoppers() -> list[pd.DataFrame]:
    config = load_config("configs/imbalance_selective/online_shoppers.yaml")
    bundle = load_dataset_from_config(config)
    chunks = []
    for seed in SEEDS:
        split = build_split(bundle, config, seed=seed)
        model = build_model_pipeline("lightgbm", bundle.features.iloc[split.train], seed=seed)
        model.fit(bundle.features.iloc[split.train], bundle.y.iloc[split.train])
        validation_y = bundle.y.iloc[split.threshold_selection].to_numpy()
        validation_scores = model.predict_proba(bundle.features.iloc[split.threshold_selection])[:, 1]
        test_y = bundle.y.iloc[split.test].to_numpy()
        test_scores = model.predict_proba(bundle.features.iloc[split.test])[:, 1]
        for grid in GRIDS:
            chunk = run_phase2_from_scores(
                validation_y,
                validation_scores,
                test_y,
                test_scores,
                dataset=bundle.name,
                model="lightgbm",
                seed=seed,
                training_distribution="natural",
                calibration="none",
                source="threshold_grid_sensitivity",
                budgets=(0.01, 0.05),
                max_candidates=grid,
            )
            chunk["candidate_grid"] = grid
            chunks.append(chunk)
    return chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 threshold-grid sensitivity.")
    parser.add_argument(
        "--prediction-root",
        default="/Users/starsfalling/Desktop/Project/baf-rcast-study/outputs/predictions",
    )
    parser.add_argument(
        "--output",
        default="results_imbalance_selective/final_phase2/paper_artifacts/threshold_grid_sensitivity.csv",
    )
    args = parser.parse_args(argv)

    results = pd.concat(
        run_baf_full(args.prediction_root) + run_online_shoppers(),
        ignore_index=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary = (
        results[results["policy_family"].isin(["optimized_single", "symmetric_dual", "asymmetric_dual"])]
        .groupby(["dataset", "candidate_grid", "policy_family", "abstention_budget"])
        .agg(
            balanced_selective_risk_mean=("balanced_selective_risk", "mean"),
            coverage_mean=("coverage", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output.with_name("threshold_grid_sensitivity_summary.csv"), index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
