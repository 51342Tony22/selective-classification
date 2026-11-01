from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imbalance_selective.datasets import load_config, load_dataset_from_config
from imbalance_selective.phase2_policies import run_phase2_smoke_test
from imbalance_selective.smoke_test import build_split


DATASET_SPECS = {
    "baf_50k": {
        "config": "configs/imbalance_selective/baf_50k.yaml",
        "ratios": (None, "1:3", "1:5", "1:10", "1:20", "1:50"),
        "fixed_total": 1352,
    },
    "online_shoppers": {
        "config": "configs/imbalance_selective/online_shoppers.yaml",
        "ratios": (None, "1:3", "1:5", "1:10"),
        "fixed_total": 4580,
    },
    "secom": {
        "config": "configs/imbalance_selective/secom.yaml",
        "ratios": (None, "1:3", "1:5", "1:10"),
        "fixed_total": 200,
    },
}
MODELS = ("logistic_regression", "lightgbm", "mlp")
SEEDS = (7, 17, 177)


def run_local_phase2(
    *,
    max_candidates: int = 201,
    include_matched_coverage: bool = False,
    include_matched_baselines: bool = False,
) -> pd.DataFrame:
    chunks = []
    for spec in DATASET_SPECS.values():
        config = load_config(spec["config"])
        bundle = load_dataset_from_config(config)
        for seed in SEEDS:
            split = build_split(bundle, config, seed=seed)
            for model in MODELS:
                chunk = run_phase2_smoke_test(
                    bundle,
                    split,
                    model_name=model,
                    seed=seed,
                    train_ratios=spec["ratios"],
                    fixed_controlled_train_total=spec["fixed_total"],
                    max_candidates=max_candidates,
                    include_matched_coverage=include_matched_coverage,
                    include_matched_baselines=include_matched_baselines,
                )
                chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local Phase 2 final experiments on BAF 50k, Online, and SECOM.")
    parser.add_argument("--max-candidates", type=int, default=201)
    parser.add_argument("--include-matched-coverage", action="store_true")
    parser.add_argument("--include-matched-baselines", action="store_true")
    parser.add_argument("--output", default="results_imbalance_selective/final_phase2/local_50k_controlled_results.csv")
    args = parser.parse_args(argv)

    results = run_local_phase2(
        max_candidates=args.max_candidates,
        include_matched_coverage=args.include_matched_coverage,
        include_matched_baselines=args.include_matched_baselines,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"wrote {len(results)} rows to {out}")
    print(results.groupby(["dataset", "model"])["balanced_selective_risk"].min().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
