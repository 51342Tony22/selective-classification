from __future__ import annotations

import argparse
from pathlib import Path

from imbalance_selective.main_experiment import run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the first main 50k logistic-regression experiment set.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/imbalance_selective/baf_50k.yaml",
            "configs/imbalance_selective/online_shoppers.yaml",
            "configs/imbalance_selective/secom.yaml",
        ],
    )
    parser.add_argument("--output-dir", default="results_imbalance_selective/main_50k_three_models")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 177])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "lightgbm", "mlp"],
        choices=["logistic_regression", "lightgbm", "mlp"],
    )
    parser.add_argument("--max-candidates", type=int, default=201)
    args = parser.parse_args(argv)

    results, summary = run_all(
        args.configs,
        output_dir=Path(args.output_dir),
        seeds=tuple(args.seeds),
        model_names=tuple(args.models),
        max_candidates=args.max_candidates,
    )
    print(f"wrote {len(results)} policy rows")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
