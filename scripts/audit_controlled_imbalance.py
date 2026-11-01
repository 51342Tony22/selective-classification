from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common.resampling import class_counts
from imbalance_selective.datasets import load_config, load_dataset_from_config
from imbalance_selective.smoke_test import build_split


def audit_configs(config_paths: list[str], *, seed: int = 7, ratios: tuple[int, ...] = (3, 5, 10, 20, 50)) -> pd.DataFrame:
    rows = []
    for config_path in config_paths:
        config = load_config(config_path)
        bundle = load_dataset_from_config(config)
        split = build_split(bundle, config, seed=seed)
        counts = class_counts(bundle.y.iloc[split.train].to_numpy())
        positives = counts["positives"]
        negatives = counts["negatives"]
        natural_ratio = negatives / positives
        common_fixed_total = _max_common_fixed_total(positives, negatives, ratios)

        for ratio in ratios:
            fixed_pos = int(common_fixed_total / (ratio + 1))
            fixed_neg = common_fixed_total - fixed_pos
            fixed_ratio_feasible = fixed_pos <= positives and fixed_neg <= negatives and fixed_pos > 0
            if ratio <= natural_ratio:
                mode = "majority_downsample_only"
                kept_pos = positives
                kept_neg = min(negatives, positives * ratio)
            else:
                mode = "minority_downsample_required"
                kept_neg = negatives
                kept_pos = int(negatives / ratio)
            rows.append(
                {
                    "dataset": bundle.name,
                    "seed": seed,
                    "natural_train_rows": len(split.train),
                    "train_positive_n": positives,
                    "train_negative_n": negatives,
                    "natural_majority_per_minority": natural_ratio,
                    "target_ratio": f"1:{ratio}",
                    "feasible_without_duplication": kept_pos > 0 and kept_neg > 0,
                    "sampling_mode": mode,
                    "minimal_drop_positive_n": positives - kept_pos,
                    "minimal_drop_negative_n": negatives - kept_neg,
                    "minimal_drop_train_rows": kept_pos + kept_neg,
                    "fixed_common_train_rows": common_fixed_total,
                    "fixed_common_positive_n": fixed_pos,
                    "fixed_common_negative_n": fixed_neg,
                    "fixed_common_feasible": fixed_ratio_feasible,
                    "positive_count_warning": fixed_pos < 20,
                }
            )
    return pd.DataFrame(rows)


def write_markdown(audit: pd.DataFrame, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Controlled Imbalance Feasibility",
        "",
        "This audit checks whether target training ratios can be formed without duplicating minority examples.",
        "",
    ]
    for dataset, subset in audit.groupby("dataset", sort=True):
        first = subset.iloc[0]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"- Natural train rows: {int(first['natural_train_rows'])}",
                f"- Positive train rows: {int(first['train_positive_n'])}",
                f"- Negative train rows: {int(first['train_negative_n'])}",
                f"- Natural majority-per-minority ratio: {first['natural_majority_per_minority']:.2f}",
                f"- Maximum common fixed train size across audited ratios: {int(first['fixed_common_train_rows'])}",
                "",
                _markdown_table(
                    subset[
                        [
                            "target_ratio",
                            "sampling_mode",
                            "minimal_drop_train_rows",
                            "minimal_drop_positive_n",
                            "minimal_drop_negative_n",
                            "fixed_common_positive_n",
                            "fixed_common_negative_n",
                            "positive_count_warning",
                        ]
                    ]
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Recommendation",
            "",
            "- Main common controlled ratios: `1:3`, `1:5`, `1:10`.",
            "- BAF can support `1:20` and `1:50` as dataset-specific extreme-ratio checks.",
            "- Online Shoppers can support extreme ratios only by discarding positive examples when the target ratio is more imbalanced than the natural training distribution.",
            "- SECOM should not be the main evidence for extreme controlled ratios because fixed-ratio cells leave very few positive examples.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def _max_common_fixed_total(positives: int, negatives: int, ratios: tuple[int, ...]) -> int:
    limits = []
    for ratio in ratios:
        limits.append((ratio + 1) * positives)
        limits.append(int(((ratio + 1) / ratio) * negatives))
    return int(min(limits))


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit controlled imbalance feasibility.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/imbalance_selective/baf_50k.yaml",
            "configs/imbalance_selective/online_shoppers.yaml",
            "configs/imbalance_selective/secom.yaml",
        ],
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-csv", default="results_imbalance_selective/reports/controlled_imbalance_feasibility.csv")
    parser.add_argument("--output-md", default="CONTROLLED_IMBALANCE_FEASIBILITY.md")
    args = parser.parse_args(argv)

    audit = audit_configs(args.configs, seed=args.seed)
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(csv_path, index=False)
    write_markdown(audit, args.output_md)
    print(audit.to_string(index=False))
    print(f"\nwrote {csv_path}")
    print(f"wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
