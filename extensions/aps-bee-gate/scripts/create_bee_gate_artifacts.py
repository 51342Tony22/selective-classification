from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from imbalance_selective.bee_gate import build_bee_gate_units, summarize_bee_gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create validation-only BEE gate artifacts.")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--gate-threshold", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default="results_imbalance_selective/aps_bee_gate_extension/paper_artifacts",
    )
    args = parser.parse_args(argv)

    results = pd.concat([pd.read_csv(path) for path in args.results], ignore_index=True)
    natural = results[results["training_distribution"].astype(str) == "natural"].copy()
    units = build_bee_gate_units(natural, gate_threshold=args.gate_threshold)
    summary = summarize_bee_gate(units)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    units.to_csv(output / "bee_gate_units.csv", index=False)
    summary.to_csv(output / "bee_gate_summary.csv", index=False)
    summary[summary["abstention_budget"] == 0.10].to_csv(
        output / "bee_gate_summary_10pct.csv", index=False
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
