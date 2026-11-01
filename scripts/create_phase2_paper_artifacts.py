from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DELTA_METRICS = [
    "delta_balanced_selective_risk",
    "delta_effective_minority_recall",
    "delta_coverage",
    "delta_validation_coverage",
]


def bootstrap_ci_table(
    diffs: pd.DataFrame,
    *,
    iterations: int = 5000,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    group_cols = ["dataset", "comparison", "policy_label"]
    for key, group in diffs.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key, strict=True))
        row["n_units"] = len(group)
        for metric in DELTA_METRICS:
            values = group[metric].to_numpy(dtype=float)
            values = values[~np.isnan(values)]
            if len(values) == 0:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_ci_low"] = np.nan
                row[f"{metric}_ci_high"] = np.nan
                continue
            boot = np.empty(iterations, dtype=float)
            for i in range(iterations):
                boot[i] = rng.choice(values, size=len(values), replace=True).mean()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci_low"] = float(np.quantile(boot, 0.025))
            row[f"{metric}_ci_high"] = float(np.quantile(boot, 0.975))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def threshold_stability(results: pd.DataFrame) -> pd.DataFrame:
    dual = results[results["policy_family"].isin(["symmetric_dual", "asymmetric_dual"])].copy()
    group_cols = ["dataset", "policy_family", "abstention_budget"]
    metrics = [
        "optimized_single_threshold",
        "lower_threshold",
        "upper_threshold",
        "abstention_width",
        "lower_side_abstention",
        "upper_side_abstention",
        "allocation_asymmetry",
        "coverage",
        "balanced_selective_risk",
        "effective_minority_recall",
    ]
    pieces = []
    grouped = dual.groupby(group_cols, dropna=False)
    for metric in metrics:
        pieces.append(grouped[metric].agg(["mean", "std"]).rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    return pd.concat(pieces, axis=1).reset_index().sort_values(group_cols).reset_index(drop=True)


def budget_curves(results: pd.DataFrame) -> pd.DataFrame:
    keep = results[results["policy_family"].isin(["optimized_single", "symmetric_dual", "asymmetric_dual"])].copy()
    group_cols = ["dataset", "policy_family", "abstention_budget"]
    return (
        keep.groupby(group_cols, dropna=False)
        .agg(
            balanced_selective_risk_mean=("balanced_selective_risk", "mean"),
            balanced_selective_risk_std=("balanced_selective_risk", "std"),
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            effective_minority_recall_mean=("effective_minority_recall", "mean"),
            effective_minority_recall_std=("effective_minority_recall", "std"),
            n=("balanced_selective_risk", "size"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def approximate_auc_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_cols = ["dataset", "model", "seed", "training_distribution", "calibration", "policy_family"]
    curve = results[results["policy_family"].isin(["optimized_single", "symmetric_dual", "asymmetric_dual"])].copy()
    for key, group in curve.groupby(key_cols, dropna=False):
        ordered = group.sort_values("coverage")
        coverage = ordered["coverage"].to_numpy(dtype=float)
        risk = ordered["balanced_selective_risk"].to_numpy(dtype=float)
        if len(np.unique(coverage)) < 2:
            auc = np.nan
        else:
            auc = _trapezoid_area(coverage, risk)
        row = dict(zip(key_cols, key, strict=True))
        row["approx_coverage_risk_auc"] = auc
        row["min_risk"] = float(np.nanmin(risk))
        row["max_coverage"] = float(np.nanmax(coverage))
        row["points"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(key_cols).reset_index(drop=True)


def when_diagnostic_units(results: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset", "model", "seed", "training_distribution", "calibration"]
    single = results[results["policy_label"] == "optimized_single"][key_cols + ["balanced_selective_risk"]].rename(
        columns={"balanced_selective_risk": "optimized_single_balanced_selective_risk"}
    )
    merged = results.merge(single, on=key_cols, how="left")
    merged = merged[merged["policy_label"] != "optimized_single"].copy()
    merged["delta_balanced_selective_risk_vs_optimized_single"] = (
        merged["balanced_selective_risk"] - merged["optimized_single_balanced_selective_risk"]
    )
    keep = [
        *key_cols,
        "policy_label",
        "policy_family",
        "abstention_budget",
        "coverage",
        "validation_coverage",
        "balanced_selective_risk",
        "delta_balanced_selective_risk_vs_optimized_single",
        "validation_reference_error_rate",
        "validation_boundary_error_rate",
        "validation_boundary_error_capture",
        "validation_boundary_error_enrichment",
        "validation_reject_error_rate",
        "validation_reject_error_capture",
        "validation_reject_error_enrichment",
    ]
    return merged[keep].sort_values(key_cols + ["policy_family", "policy_label"]).reset_index(drop=True)


def when_diagnostic_summary(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["dataset", "policy_family", "abstention_budget"]
    for key, group in units.groupby(group_cols, dropna=False):
        dataset, policy_family, abstention_budget = key
        rows.append(
            {
                "dataset": dataset,
                "policy_family": policy_family,
                "abstention_budget": abstention_budget,
                "n_units": len(group),
                "mean_delta_balanced_selective_risk_vs_optimized_single": float(
                    group["delta_balanced_selective_risk_vs_optimized_single"].mean()
                ),
                "mean_validation_boundary_error_enrichment": float(group["validation_boundary_error_enrichment"].mean()),
                "mean_validation_reject_error_enrichment": float(group["validation_reject_error_enrichment"].mean()),
                "mean_validation_reject_error_capture": float(group["validation_reject_error_capture"].mean()),
                "corr_boundary_enrichment_vs_delta_risk": _safe_corr(
                    group["validation_boundary_error_enrichment"].to_numpy(dtype=float),
                    group["delta_balanced_selective_risk_vs_optimized_single"].to_numpy(dtype=float),
                ),
                "corr_reject_enrichment_vs_delta_risk": _safe_corr(
                    group["validation_reject_error_enrichment"].to_numpy(dtype=float),
                    group["delta_balanced_selective_risk_vs_optimized_single"].to_numpy(dtype=float),
                ),
                "corr_reject_capture_vs_delta_risk": _safe_corr(
                    group["validation_reject_error_capture"].to_numpy(dtype=float),
                    group["delta_balanced_selective_risk_vs_optimized_single"].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_figures(ci: pd.DataFrame, curves: pd.DataFrame, stability: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    specs = [
        (
            "dual_vs_single_delta_risk.svg",
            _delta_risk_svg(
                ci[ci["comparison"] == "dual_minus_optimized_single"],
                "Dual minus Optimized Single: Delta Balanced Selective Risk",
            ),
        ),
        (
            "asymmetric_vs_symmetric_delta_risk.svg",
            _delta_risk_svg(
                ci[ci["comparison"] == "asymmetric_minus_symmetric"],
                "Asymmetric minus Symmetric: Delta Balanced Selective Risk",
            ),
        ),
        (
            "budget_coverage_risk_tradeoff.svg",
            _curve_svg(curves),
        ),
        (
            "allocation_asymmetry_by_budget.svg",
            _asymmetry_svg(stability),
        ),
    ]
    for filename, content in specs:
        path = out / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def _delta_risk_svg(data: pd.DataFrame, title: str) -> str:
    width, height = 1180, 520
    left, top, bottom, right = 90, 58, 150, 35
    plot_w, plot_h = width - left - right, height - top - bottom
    if data.empty:
        return _empty_svg(width, height, title)
    data = data.copy()
    data["budget"] = data["policy_label"].str.extract(r"_(0\.\d+|0\.1)$")[0].fillna("0")
    data = data.sort_values(["dataset", "policy_label"])
    labels = (data["dataset"] + " " + data["policy_label"]).tolist()
    values = data["delta_balanced_selective_risk_mean"].to_numpy(dtype=float)
    low = data["delta_balanced_selective_risk_ci_low"].to_numpy(dtype=float)
    high = data["delta_balanced_selective_risk_ci_high"].to_numpy(dtype=float)
    y_min = min(-0.04, float(np.nanmin(low)) - 0.005)
    y_max = max(0.02, float(np.nanmax(high)) + 0.005)
    parts = _svg_header(width, height, title)
    zero_y = _scale_y(0.0, y_min, y_max, top, plot_h)
    parts.append(f'<line x1="{left}" y1="{zero_y:.1f}" x2="{width-right}" y2="{zero_y:.1f}" stroke="#333" stroke-dasharray="4 4"/>')
    step = plot_w / max(len(values), 1)
    for i, value in enumerate(values):
        x = left + (i + 0.5) * step
        y = _scale_y(value, y_min, y_max, top, plot_h)
        y_low = _scale_y(low[i], y_min, y_max, top, plot_h)
        y_high = _scale_y(high[i], y_min, y_max, top, plot_h)
        color = "#2f6f73" if value < 0 else "#b65f3b"
        parts.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        label = labels[i].replace("_", " ")
        parts.append(f'<text x="{x:.1f}" y="{top+plot_h+18}" text-anchor="end" transform="rotate(-35 {x:.1f} {top+plot_h+18})" font-family="Arial" font-size="9">{label}</text>')
    parts.append(f'<text x="{left}" y="{height-18}" font-family="Arial" font-size="12">Negative values mean lower balanced selective risk.</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _curve_svg(curves: pd.DataFrame) -> str:
    width, height = 980, 520
    left, top, bottom, right = 90, 58, 80, 35
    plot_w, plot_h = width - left - right, height - top - bottom
    title = "Coverage-Risk Tradeoff by Policy Family"
    parts = _svg_header(width, height, title)
    data = curves[curves["policy_family"].isin(["optimized_single", "symmetric_dual", "asymmetric_dual"])]
    colors = {"optimized_single": "#555555", "symmetric_dual": "#2f6f73", "asymmetric_dual": "#b65f3b"}
    x_min, x_max = 0.85, 1.0
    y_min, y_max = 0.08, max(0.55, float(data["balanced_selective_risk_mean"].max()) + 0.03)
    for family, group in data.groupby("policy_family", sort=True):
        for dataset, subset in group.groupby("dataset", sort=True):
            ordered = subset.sort_values("coverage_mean")
            points = []
            for _, row in ordered.iterrows():
                x = left + (float(row["coverage_mean"]) - x_min) / (x_max - x_min) * plot_w
                y = _scale_y(float(row["balanced_selective_risk_mean"]), y_min, y_max, top, plot_h)
                points.append(f"{x:.1f},{y:.1f}")
            if len(points) >= 2:
                parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[family]}" stroke-width="2" opacity="0.65"/>')
            for point in points:
                x, y = point.split(",")
                parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{colors[family]}" opacity="0.8"/>')
    parts.append(f'<text x="{width/2}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="13">Coverage</text>')
    parts.append(f'<text x="18" y="{height/2}" text-anchor="middle" transform="rotate(-90 18 {height/2})" font-family="Arial" font-size="13">Balanced selective risk</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _asymmetry_svg(stability: pd.DataFrame) -> str:
    data = stability[stability["policy_family"].isin(["symmetric_dual", "asymmetric_dual"])].copy()
    width, height = 1080, 500
    left, top, bottom, right = 80, 58, 110, 35
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = _svg_header(width, height, "Allocation Asymmetry by Dataset and Budget")
    data = data.sort_values(["dataset", "policy_family", "abstention_budget"])
    labels = (data["dataset"] + " " + data["policy_family"].str.replace("_dual", "", regex=False) + " " + data["abstention_budget"].astype(str)).tolist()
    values = data["allocation_asymmetry_mean"].to_numpy(dtype=float)
    step = plot_w / max(len(values), 1)
    for i, value in enumerate(values):
        x = left + i * step + step * 0.25
        h = value * plot_h
        y = top + plot_h - h
        color = "#2f6f73" if "symmetric" in labels[i] else "#b65f3b"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{step*0.5:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x+step*0.25:.1f}" y="{top+plot_h+16}" text-anchor="end" transform="rotate(-35 {x+step*0.25:.1f} {top+plot_h+16})" font-family="Arial" font-size="9">{labels[i]}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
    ]


def _empty_svg(width: int, height: int, title: str) -> str:
    parts = _svg_header(width, height, title)
    parts.append(f'<text x="{width/2}" y="{height/2}" text-anchor="middle" font-family="Arial" font-size="14">No data</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _scale_y(value: float, y_min: float, y_max: float, top: int, plot_h: int) -> float:
    return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h


def _trapezoid_area(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    xs = x[order]
    ys = y[order]
    return float(np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1]) / 2.0))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return float("nan")
    xs = x[mask]
    ys = y[mask]
    if np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create paper-ready Phase 2 artifacts.")
    parser.add_argument("--results", default="results_imbalance_selective/final_phase2/all_phase2_results.csv")
    parser.add_argument("--diffs", default="results_imbalance_selective/final_phase2/paired_differences.csv")
    parser.add_argument("--output-dir", default="results_imbalance_selective/final_phase2/paper_artifacts")
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args(argv)

    results = pd.read_csv(args.results)
    diffs = pd.read_csv(args.diffs)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ci = bootstrap_ci_table(diffs, iterations=args.iterations)
    stability = threshold_stability(results)
    curves = budget_curves(results)
    auc = approximate_auc_table(results)
    when_units = when_diagnostic_units(results)
    when_summary = when_diagnostic_summary(when_units)

    ci.to_csv(out / "paired_unit_bootstrap_ci.csv", index=False)
    stability.to_csv(out / "threshold_stability.csv", index=False)
    curves.to_csv(out / "budget_curves.csv", index=False)
    auc.to_csv(out / "approx_coverage_risk_auc.csv", index=False)
    when_units.to_csv(out / "when_diagnostic_units.csv", index=False)
    when_summary.to_csv(out / "when_diagnostic_summary.csv", index=False)
    write_figures(ci, curves, stability, out / "figures")
    print(f"wrote paper artifacts to {out}")
    print(ci.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
