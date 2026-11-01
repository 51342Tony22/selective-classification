from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imbalance_selective.datasets import load_config, load_dataset_from_config
from imbalance_selective.phase2_policies import run_phase2_smoke_test
from imbalance_selective.smoke_test import build_split


MODELS = ("logistic_regression", "lightgbm", "mlp")
SEEDS = (7, 17, 177)


def run_aps_extension(config_path: str | Path, *, max_candidates: int = 201) -> pd.DataFrame:
    config = load_config(config_path)
    bundle = load_dataset_from_config(config)
    chunks = []
    for seed in SEEDS:
        split = build_split(bundle, config, seed=seed)
        for model in MODELS:
            chunks.append(
                run_phase2_smoke_test(
                    bundle,
                    split,
                    model_name=model,
                    seed=seed,
                    train_ratios=(None,),
                    budgets=(0.01, 0.05, 0.10),
                    max_candidates=max_candidates,
                    include_matched_coverage=True,
                    include_matched_baselines=True,
                )
            )
    return pd.concat(chunks, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the APS independent-dataset extension.")
    parser.add_argument("--config", default="configs/imbalance_selective/aps_failure.yaml")
    parser.add_argument("--max-candidates", type=int, default=201)
    parser.add_argument(
        "--output",
        default="results_imbalance_selective/aps_bee_gate_extension/aps_results.csv",
    )
    args = parser.parse_args(argv)

    results = run_aps_extension(args.config, max_candidates=args.max_candidates)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"wrote {len(results)} rows to {output}")
    print(
        results.groupby(["model", "policy_label"])["balanced_selective_risk"]
        .mean()
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
