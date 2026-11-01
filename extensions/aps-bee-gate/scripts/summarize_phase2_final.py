from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


METRICS = [
    "balanced_selective_risk",
    "macro_f1",
    "mcc",
    "minority_precision",
    "minority_recall",
    "effective_minority_recall",
    "coverage",
    "validation_coverage",
    "positive_coverage",
    "negative_coverage",
    "allocation_asymmetry",
    "validation_boundary_error_enrichment",
    "validation_reject_error_enrichment",
    "validation_reject_error_capture",
]


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "model", "training_distribution", "calibration", "policy_label", "policy_family"]
    pieces = []
    grouped = results.groupby(group_cols, dropna=False)
    for metric in METRICS:
        pieces.append(grouped[metric].agg(["mean", "std"]).rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    return pd.concat(pieces, axis=1).reset_index()


def paired_differences(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "model", "seed", "training_distribution", "calibration"]
    rows = []
    for key, group in results.groupby(keys, dropna=False):
        by_policy = group.set_index("policy_label")
        if "optimized_single" in by_policy.index:
            single = by_policy.loc["optimized_single"]
            for label in by_policy.index:
                if label in {"default_single_0_5", "optimized_single"}:
                    continue
                row = by_policy.loc[label]
                comparison = "dual_minus_optimized_single"
                if label.startswith("matched_"):
                    comparison = "matched_policy_minus_optimized_single"
                rows.append(_diff_row(key, label, row, single, comparison=comparison))
        for budget in ["0.01", "0.05", "0.1"]:
            sym = f"symmetric_dual_{budget}"
            asym = f"asymmetric_dual_{budget}"
            matched = f"matched_asymmetric_dual_{budget}"
            if sym in by_policy.index and asym in by_policy.index:
                rows.append(_diff_row(key, asym, by_policy.loc[asym], by_policy.loc[sym], comparison="asymmetric_minus_symmetric"))
            if sym in by_policy.index and matched in by_policy.index:
                rows.append(
                    _diff_row(
                        key,
                        matched,
                        by_policy.loc[matched],
                        by_policy.loc[sym],
                        comparison="matched_asymmetric_minus_symmetric",
                    )
                )
            if sym in by_policy.index:
                for label in by_policy.index:
                    match = re.fullmatch(r"matched_(?!asymmetric_dual)(.+)_(0\.01|0\.05|0\.1)", label)
                    if match and match.group(2) == budget:
                        rows.append(
                            _diff_row(
                                key,
                                label,
                                by_policy.loc[label],
                                by_policy.loc[sym],
                                comparison="matched_baseline_minus_symmetric",
                            )
                        )
    return pd.DataFrame(rows)


def write_figures(summary: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    best = summary.loc[summary.groupby("dataset")["balanced_selective_risk_mean"].idxmin()]
    path = out / "best_balanced_selective_risk_by_dataset.svg"
    path.write_text(_bar_svg(best, "balanced_selective_risk_mean", "Best Balanced Selective Risk by Dataset"), encoding="utf-8")
    written.append(path)
    mechanism = summary[summary["policy_family"].isin(["symmetric_dual", "asymmetric_dual"])]
    path = out / "symmetric_vs_asymmetric_risk.svg"
    path.write_text(_mechanism_svg(mechanism), encoding="utf-8")
    written.append(path)
    return written


def write_key_tables(summary: pd.DataFrame, diffs: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    primary = summary[
        summary["policy_family"].isin(
            ["default_single", "optimized_single", "symmetric_dual", "asymmetric_dual"]
        )
    ]
    best = primary.loc[primary.groupby("dataset")["balanced_selective_risk_mean"].idxmin()]
    best.to_csv(out / "table_best_by_dataset.csv", index=False)

    baf_none = primary[(primary["dataset"] == "baf_full") & (primary["calibration"] == "none")]
    baf_best = baf_none.loc[baf_none.groupby("model")["balanced_selective_risk_mean"].idxmin()]
    baf_best.to_csv(out / "table_baf_full_none_by_model.csv", index=False)

    delta_cols = [column for column in diffs.columns if column.startswith("delta_")]
    paired_means = (
        diffs.groupby(["dataset", "comparison", "policy_label"], dropna=False)[delta_cols]
        .mean()
        .reset_index()
    )
    paired_means["n"] = diffs.groupby(
        ["dataset", "comparison", "policy_label"], dropna=False
    ).size().to_numpy()
    paired_means.to_csv(out / "table_paired_comparison_means.csv", index=False)


def _diff_row(key: tuple, label: str, left: pd.Series, right: pd.Series, *, comparison: str) -> dict[str, float | str | int]:
    dataset, model, seed, training_distribution, calibration = key
    row = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "training_distribution": training_distribution,
        "calibration": calibration,
        "comparison": comparison,
        "policy_label": label,
    }
    for metric in METRICS:
        row[f"delta_{metric}"] = float(left[metric] - right[metric])
    return row


def _bar_svg(data: pd.DataFrame, metric: str, title: str) -> str:
    width = 980
    height = 420
    left = 90
    top = 54
    bottom = 100
    plot_h = height - top - bottom
    plot_w = width - left - 30
    rows = data.sort_values(metric)
    max_value = max(1.0, float(rows[metric].max()))
    bar_w = plot_w / max(len(rows), 1) * 0.58
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
    ]
    for i, (_, row) in enumerate(rows.iterrows()):
        x = left + (i + 0.25) * (plot_w / len(rows))
        h = float(row[metric]) / max_value * plot_h
        y = top + plot_h - h
        label = f"{row['dataset']}\\n{row['model']}\\n{row['policy_label']}"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#3f7f83"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{float(row[metric]):.3f}</text>')
        for j, text in enumerate(label.split("\\n")):
            parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 18 + 14*j}" text-anchor="middle" font-family="Arial" font-size="10">{text}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _mechanism_svg(data: pd.DataFrame) -> str:
    collapsed = data.groupby(["dataset", "policy_family"], as_index=False)["balanced_selective_risk_mean"].min()
    width = 980
    height = 420
    left = 90
    top = 54
    bottom = 80
    plot_h = height - top - bottom
    plot_w = width - left - 30
    datasets = sorted(collapsed["dataset"].unique().tolist())
    colors = {"symmetric_dual": "#2f6f73", "asymmetric_dual": "#b65f3b"}
    max_value = max(1.0, float(collapsed["balanced_selective_risk_mean"].max()))
    group_w = plot_w / len(datasets)
    bar_w = 42
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="20">Symmetric vs Asymmetric Dual Risk</text>',
    ]
    for i, dataset in enumerate(datasets):
        center = left + i * group_w + group_w / 2
        for j, family in enumerate(["symmetric_dual", "asymmetric_dual"]):
            row = collapsed[(collapsed["dataset"] == dataset) & (collapsed["policy_family"] == family)]
            if row.empty:
                continue
            value = float(row.iloc[0]["balanced_selective_risk_mean"])
            h = value / max_value * plot_h
            x = center - bar_w + j * bar_w
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 5}" height="{h:.1f}" fill="{colors[family]}"/>')
            parts.append(f'<text x="{x + (bar_w-5)/2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12">{dataset}</text>')
    parts.append('<rect x="90" y="375" width="14" height="14" fill="#2f6f73"/><text x="110" y="387" font-family="Arial" font-size="13">symmetric</text>')
    parts.append('<rect x="220" y="375" width="14" height="14" fill="#b65f3b"/><text x="240" y="387" font-family="Arial" font-size="13">asymmetric</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Phase 2 final result CSV files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="results_imbalance_selective/final_phase2")
    args = parser.parse_args(argv)

    results = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)
    diffs = paired_differences(results)
    results.to_csv(out / "all_phase2_results.csv", index=False)
    summary.to_csv(out / "summary_by_policy.csv", index=False)
    diffs.to_csv(out / "paired_differences.csv", index=False)
    write_figures(summary, out / "figures")
    write_key_tables(summary, diffs, out / "key_tables")
    print(f"wrote {len(results)} result rows, {len(summary)} summary rows, {len(diffs)} paired diffs")
    print(summary.loc[summary.groupby("dataset")["balanced_selective_risk_mean"].idxmin()][[
        "dataset", "model", "training_distribution", "calibration", "policy_label", "balanced_selective_risk_mean", "effective_minority_recall_mean", "coverage_mean"
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
