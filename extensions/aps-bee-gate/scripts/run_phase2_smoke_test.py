from __future__ import annotations

import argparse
from pathlib import Path

from imbalance_selective.datasets import load_config, load_dataset_from_config
from imbalance_selective.phase2_policies import run_phase2_smoke_test
from imbalance_selective.smoke_test import build_split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase 2 method-lock smoke test.")
    parser.add_argument("--config", default="configs/imbalance_selective/online_shoppers.yaml")
    parser.add_argument("--model", default="lightgbm", choices=["logistic_regression", "lightgbm", "mlp"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--train-ratios", nargs="+", default=["natural", "1:3", "1:5", "1:10"])
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--max-candidates", type=int, default=101)
    parser.add_argument("--output", default="results_imbalance_selective/phase2_method_smoke/online_lightgbm_seed7.csv")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    bundle = load_dataset_from_config(config)
    split = build_split(bundle, config, seed=args.seed)
    ratios = tuple(None if ratio == "natural" else ratio for ratio in args.train_ratios)
    results = run_phase2_smoke_test(
        bundle,
        split,
        model_name=args.model,
        seed=args.seed,
        train_ratios=ratios,
        budgets=tuple(args.budgets),
        max_candidates=args.max_candidates,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(results.to_string(index=False))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
