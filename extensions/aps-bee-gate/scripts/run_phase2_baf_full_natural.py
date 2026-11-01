from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imbalance_selective.phase2_policies import run_phase2_from_scores


MODELS = ("logistic_regression", "lightgbm", "mlp")
SEEDS = (7, 17, 177)
CALIBRATIONS = {
    "none": "prob_none",
    "platt": "prob_platt",
    "isotonic": "prob_isotonic",
    "temperature": "prob_temperature",
}


def run_baf_full_natural(
    *,
    prediction_root: str | Path,
    baf_path: str | Path,
    max_candidates: int = 201,
    include_matched_coverage: bool = False,
    include_matched_baselines: bool = False,
) -> pd.DataFrame:
    root = Path(prediction_root)
    baf = pd.read_parquet(baf_path, columns=["fraud_bool", "month"])
    train = baf[baf["month"].isin([0, 1, 2, 3, 4])]
    train_positive_n = int(train["fraud_bool"].sum())
    train_rows = len(train)
    train_negative_n = train_rows - train_positive_n
    ratio = train_negative_n / train_positive_n

    rows = []
    for model in MODELS:
        for seed in SEEDS:
            validation = pd.read_parquet(root / model / f"seed_{seed}" / "validation.parquet")
            test = pd.read_parquet(root / model / f"seed_{seed}" / "test.parquet")
            for calibration, score_column in CALIBRATIONS.items():
                result = run_phase2_from_scores(
                    validation["y_true"].to_numpy(),
                    validation[score_column].to_numpy(),
                    test["y_true"].to_numpy(),
                    test[score_column].to_numpy(),
                    dataset="baf_full",
                    model=model,
                    seed=seed,
                    training_distribution="natural_full",
                    train_rows=train_rows,
                    train_positive_n=train_positive_n,
                    train_negative_n=train_negative_n,
                    realized_train_majority_per_minority=ratio,
                    calibration=calibration,
                    source="old_project_predictions",
                    max_candidates=max_candidates,
                    include_matched_coverage=include_matched_coverage,
                    include_matched_baselines=include_matched_baselines,
                )
                rows.append(result)
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 2 natural full-BAF evaluation from reused predictions.")
    parser.add_argument(
        "--prediction-root",
        default="/Users/starsfalling/Desktop/Project/baf-rcast-study/outputs/predictions",
    )
    parser.add_argument("--baf-path", default="data/raw/baf/Base.parquet")
    parser.add_argument("--max-candidates", type=int, default=201)
    parser.add_argument("--include-matched-coverage", action="store_true")
    parser.add_argument("--include-matched-baselines", action="store_true")
    parser.add_argument("--output", default="results_imbalance_selective/final_phase2/baf_full_natural_results.csv")
    args = parser.parse_args(argv)

    results = run_baf_full_natural(
        prediction_root=args.prediction_root,
        baf_path=args.baf_path,
        max_candidates=args.max_candidates,
        include_matched_coverage=args.include_matched_coverage,
        include_matched_baselines=args.include_matched_baselines,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"wrote {len(results)} rows to {out}")
    print(results.groupby(["model", "calibration"])["balanced_selective_risk"].min().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
