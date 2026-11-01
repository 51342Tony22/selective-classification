from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd


COLORS = {
    "logistic_regression / natural": "#2f6f73",
    "logistic_regression / controlled_1_to_3": "#82a6a8",
    "lightgbm / natural": "#b65f3b",
    "lightgbm / controlled_1_to_3": "#d7a07f",
    "mlp / natural": "#4a5f9e",
    "mlp / controlled_1_to_3": "#9aa6d6",
}


def make_figures(summary_path: str | Path, output_dir: str | Path) -> list[Path]:
    summary = pd.read_csv(summary_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dataset, subset in summary.groupby("dataset", sort=True):
        for metric in ["macro_f1", "minority_recall", "coverage"]:
            path = out_dir / f"{dataset}_{metric}.svg"
            path.write_text(_grouped_bar_svg(subset, metric=metric, title=f"{dataset}: {metric}"), encoding="utf-8")
            written.append(path)
    return written


def _grouped_bar_svg(data: pd.DataFrame, *, metric: str, title: str) -> str:
    policies = sorted(data["policy_label"].unique().tolist())
    series = sorted((data["model"] + " / " + data["experiment"]).unique().tolist())
    width = 1280
    height = 540
    margin_left = 84
    margin_right = 24
    margin_top = 58
    margin_bottom = 168
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    group_w = plot_w / max(len(policies), 1)
    bar_w = min(24, group_w / (len(series) + 1.4))
    max_value = max(1.0, float(data[f"{metric}_mean"].max()))

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" fill="#222">{escape(title)}</text>',
    ]
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        y = margin_top + plot_h - (tick / max_value) * plot_h
        elements.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        elements.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#555">{tick:.2f}</text>')
    elements.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>')
    elements.append(f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#333"/>')

    for policy_i, policy in enumerate(policies):
        group_x = margin_left + policy_i * group_w + group_w / 2
        for series_i, label in enumerate(series):
            model, experiment = label.split(" / ", maxsplit=1)
            row = data[
                (data["policy_label"] == policy)
                & (data["model"] == model)
                & (data["experiment"] == experiment)
            ]
            if row.empty:
                continue
            value = float(row.iloc[0][f"{metric}_mean"])
            bar_h = (value / max_value) * plot_h
            x = group_x - (len(series) * bar_w) / 2 + series_i * bar_w
            y = margin_top + plot_h - bar_h
            color = COLORS.get(label, "#777777")
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 4:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
            elements.append(f'<text x="{x + (bar_w - 4) / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-family="Arial" font-size="9" fill="#333">{value:.2f}</text>')
        label = escape(policy.replace("_", " "))
        elements.append(f'<text x="{group_x:.1f}" y="{margin_top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="11" fill="#333">{label}</text>')

    legend_x = margin_left
    legend_y = height - 74
    for i, label in enumerate(series):
        x = legend_x + (i % 3) * 360
        y = legend_y + (i // 3) * 24
        elements.append(f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{COLORS.get(label, "#777777")}"/>')
        elements.append(f'<text x="{x + 20}" y="{y}" font-family="Arial" font-size="13" fill="#333">{escape(label)}</text>')
    elements.append("</svg>")
    return "\n".join(elements)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create SVG figures from the main experiment summary.")
    parser.add_argument(
        "--summary",
        default="results_imbalance_selective/main_50k_three_models/summary_by_dataset_experiment_policy.csv",
    )
    parser.add_argument("--output-dir", default="results_imbalance_selective/main_50k_three_models/figures")
    args = parser.parse_args(argv)
    written = make_figures(args.summary, args.output_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
