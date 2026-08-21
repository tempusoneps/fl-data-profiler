from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fldataprofiler.modules.base import ModuleResult
from fldataprofiler.modules.progress import ModuleProgress
from fldataprofiler.modules.statistics import DatasetShape
from fldataprofiler.utils import (
    _date_columns,
    _format_duration,
    _html_markdown_details,
    _markdown_table,
    _merge_inputs,
    _numeric_feature_columns,
    _numeric_series,
    _read_table_with_date_index,
    _round,
    _sample_rows,
    _select_targets,
    _write_csv,
    _write_json,
)

MAX_ROWS = 50_000
MAX_LABEL_CLASSES = 50
MIN_NON_NULL = 10
RANDOM_STATE = 42
DEFAULT_N_BINS = 20
DEFAULT_N_FOLDS = 5
EPSILON = 1e-6


@dataclass(frozen=True)
class ProbabilityDriftRunMetadata:
    module: str
    created_at: str
    execution_time: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    n_bins: int
    n_folds: int
    features: list[str]
    targets: list[str]
    model_rows: int


def _compute_quantile_bins(series: pd.Series, n_bins: int = DEFAULT_N_BINS) -> tuple[pd.Series, np.ndarray]:
    """Assign rank-based equal-frequency quantile bin indices (1 to n_bins) and bin edges."""
    if len(series) == 0:
        return pd.Series(dtype=int, index=series.index), np.array([])
    ranks = series.rank(method="first")
    actual_bins = min(n_bins, len(series))
    if actual_bins < 1:
        return pd.Series(1, index=series.index, dtype=int), np.array([float(series.min()), float(series.max())])
    bins = pd.qcut(ranks, q=actual_bins, labels=False) + 1
    return bins.astype(int), np.linspace(0, 1, actual_bins + 1)


def _compute_laplace_woe_iv(
    events_k: float,
    non_events_k: float,
    total_events: float,
    total_non_events: float,
    n_bins: int,
    alpha: float = 0.5,
) -> tuple[float, float]:
    """Calculate Laplace-smoothed Weight of Evidence and Information Value contribution."""
    p_e = (events_k + alpha) / (total_events + alpha * n_bins) if total_events > 0 else 1.0 / n_bins
    p_ne = (non_events_k + alpha) / (total_non_events + alpha * n_bins) if total_non_events > 0 else 1.0 / n_bins
    p_e = max(p_e, EPSILON)
    p_ne = max(p_ne, EPSILON)
    woe = float(np.log(p_e / p_ne))
    iv = float((p_e - p_ne) * woe)
    return woe, iv


def _compute_psi(actual_dist: np.ndarray, expected_dist: np.ndarray) -> float:
    """Calculate Population Stability Index (PSI) between two probability distributions."""
    if len(actual_dist) == 0 or len(expected_dist) == 0:
        return 0.0
    act = np.maximum(actual_dist, EPSILON)
    exp = np.maximum(expected_dist, EPSILON)
    act = act / act.sum()
    exp = exp / exp.sum()
    psi = float(np.sum((act - exp) * np.log(act / exp)))
    return max(psi, 0.0)


def _evaluate_feature_drift(
    feature_series: pd.Series,
    target_series: pd.Series,
    feature_name: str,
    target_name: str,
    n_bins: int = DEFAULT_N_BINS,
    n_folds: int = DEFAULT_N_FOLDS,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate quantile probability stability, PSI, and monotonicity drift across time folds."""
    clean_feature = _numeric_series(feature_series)
    valid_mask = clean_feature.notna() & target_series.notna()

    x = clean_feature[valid_mask]
    y = target_series[valid_mask]

    n_total = len(x)
    if n_total < (n_folds * 10) or y.nunique(dropna=True) < 2 or x.nunique(dropna=True) < 2:
        return [], [], []

    # Assign baseline 20-quantiles over full dataset
    global_bins, _ = _compute_quantile_bins(x, n_bins=n_bins)
    unique_classes = sorted(y.unique(), key=lambda v: str(v))
    bin_indices = np.arange(1, n_bins + 1)

    # Reference distribution of samples across bins
    expected_bin_dist = np.array([(global_bins == k).sum() / n_total for k in bin_indices], dtype=float)

    # Compute overall baseline conditional probabilities for each class
    overall_probs: dict[object, np.ndarray] = {}
    for c in unique_classes:
        probs = []
        for k in bin_indices:
            k_mask = global_bins == k
            n_k = k_mask.sum()
            prob_k = float((y[k_mask] == c).sum() / n_k) if n_k > 0 else 0.0
            probs.append(prob_k)
        overall_probs[c] = np.array(probs, dtype=float)

    # Time fold splitting (chronological partitions)
    fold_size = n_total // n_folds
    fold_ranges: list[tuple[int, int]] = []
    for f in range(n_folds):
        start_idx = f * fold_size
        end_idx = (f + 1) * fold_size if f < n_folds - 1 else n_total
        fold_ranges.append((start_idx, end_idx))

    feature_summary_scores: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []
    quantile_detail_rows: list[dict[str, object]] = []

    for c in unique_classes:
        fold_ivs: list[float] = []
        fold_spreads: list[float] = []
        fold_monos: list[float] = []
        fold_psis: list[float] = []
        fold_base_rates: list[float] = []
        fold_curve_diffs: list[float] = []

        for f_idx, (start_idx, end_idx) in enumerate(fold_ranges):
            x_fold = x.iloc[start_idx:end_idx]
            y_fold = y.iloc[start_idx:end_idx]
            bins_fold = global_bins.iloc[start_idx:end_idx]
            n_fold = len(x_fold)

            if n_fold < 10:
                continue

            # Actual bin distribution in this fold
            actual_bin_dist = np.array([(bins_fold == k).sum() / n_fold for k in bin_indices], dtype=float)
            psi_val = _compute_psi(actual_bin_dist, expected_bin_dist)
            fold_psis.append(psi_val)

            total_events_fold = float((y_fold == c).sum())
            total_non_events_fold = float(n_fold - total_events_fold)
            base_rate_fold = total_events_fold / n_fold if n_fold > 0 else 0.0
            fold_base_rates.append(base_rate_fold)

            fold_c_probs: list[float] = []
            fold_iv_total = 0.0

            for k in bin_indices:
                k_mask_fold = bins_fold == k
                n_k_fold = int(k_mask_fold.sum())
                events_k = int((y_fold[k_mask_fold] == c).sum())
                non_events_k = n_k_fold - events_k
                prob_k = events_k / n_k_fold if n_k_fold > 0 else 0.0
                fold_c_probs.append(prob_k)

                woe_k, iv_k = _compute_laplace_woe_iv(
                    events_k=events_k,
                    non_events_k=non_events_k,
                    total_events=total_events_fold,
                    total_non_events=total_non_events_fold,
                    n_bins=n_bins,
                )
                fold_iv_total += iv_k

                quantile_detail_rows.append(
                    {
                        "feature": feature_name,
                        "target": target_name,
                        "target_class": c,
                        "fold": f_idx + 1,
                        "bin_index": int(k),
                        "sample_count": n_k_fold,
                        "conditional_prob": _round(prob_k),
                        "overall_prob": _round(float(overall_probs[c][k - 1])),
                        "prob_diff": _round(float(prob_k - overall_probs[c][k - 1])),
                        "woe": _round(woe_k),
                        "iv_contribution": _round(iv_k),
                    }
                )

            prob_arr = np.array(fold_c_probs, dtype=float)
            spread_fold = float(prob_arr.max() - prob_arr.min()) if len(prob_arr) > 0 else 0.0
            fold_spreads.append(spread_fold)
            fold_ivs.append(fold_iv_total)

            # Monotonicity in fold
            if len(prob_arr) < 2 or np.all(np.isclose(prob_arr, prob_arr[0])):
                mono_fold = 0.0
            else:
                sp_res = stats.spearmanr(bin_indices, prob_arr)
                corr = getattr(sp_res, "statistic", getattr(sp_res, "correlation", 0.0))
                mono_fold = 0.0 if np.isnan(corr) else float(corr)
            fold_monos.append(mono_fold)

            # Difference from overall curve
            curve_diff = float(np.mean(np.abs(prob_arr - overall_probs[c])))
            fold_curve_diffs.append(curve_diff)

            fold_metric_rows.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "target_class": c,
                    "fold": f_idx + 1,
                    "sample_count": n_fold,
                    "iv": _round(fold_iv_total),
                    "prob_spread": _round(spread_fold),
                    "monotonicity": _round(mono_fold),
                    "psi": _round(psi_val),
                    "base_rate": _round(base_rate_fold),
                    "curve_drift": _round(curve_diff),
                }
            )

        if not fold_ivs:
            continue

        mean_iv = float(np.mean(fold_ivs))
        std_iv = float(np.std(fold_ivs))
        stability_ratio = float(mean_iv / (std_iv + 1e-4))

        mean_spread = float(np.mean(fold_spreads))
        mean_mono = float(np.mean(fold_monos))
        std_mono = float(np.std(fold_monos))

        # Count monotonicity sign flips between consecutive folds (e.g. positive to negative with magnitude > 0.15)
        flips = 0
        for i in range(len(fold_monos) - 1):
            m1, m2 = fold_monos[i], fold_monos[i + 1]
            if (m1 > 0.15 and m2 < -0.15) or (m1 < -0.15 and m2 > 0.15):
                flips += 1

        max_psi = float(np.max(fold_psis)) if fold_psis else 0.0
        mean_psi = float(np.mean(fold_psis)) if fold_psis else 0.0
        mean_curve_drift = float(np.mean(fold_curve_diffs)) if fold_curve_diffs else 0.0

        # IV Trend Slope across folds (positive = strengthening alpha, negative = alpha decay)
        if len(fold_ivs) >= 2:
            x_axis = np.arange(len(fold_ivs))
            slope, _, _, _, _ = stats.linregress(x_axis, fold_ivs)
            iv_trend_slope = float(slope)
        else:
            iv_trend_slope = 0.0

        # Assign Drift Status
        if max_psi < 0.10 and flips == 0 and stability_ratio >= 1.5:
            drift_status = "STABLE"
        elif max_psi <= 0.25 and flips <= 1 and stability_ratio >= 0.8:
            drift_status = "MODERATE_DRIFT"
        else:
            drift_status = "HIGH_DRIFT"

        feature_summary_scores.append(
            {
                "feature": feature_name,
                "target": target_name,
                "target_class": c,
                "drift_status": drift_status,
                "mean_iv": _round(mean_iv),
                "iv_std": _round(std_iv),
                "stability_ratio": _round(stability_ratio),
                "mean_prob_spread": _round(mean_spread),
                "monotonicity_mean": _round(mean_mono),
                "monotonicity_std": _round(std_mono),
                "monotonicity_flips": flips,
                "max_psi": _round(max_psi),
                "mean_psi": _round(mean_psi),
                "prob_curve_drift": _round(mean_curve_drift),
                "iv_trend_slope": _round(iv_trend_slope),
                "n_folds": len(fold_ivs),
                "sample_count": n_total,
            }
        )

    return feature_summary_scores, fold_metric_rows, quantile_detail_rows


def _plot_probability_drift_charts(
    summary_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    output_path: Path,
    top_n: int = 4,
) -> Path:
    """Plot multi-fold conditional probability curves to visualize stability vs drift."""
    if summary_df.empty or quantile_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No drift data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Pick top stable features and top drifting features
    stable_candidates = summary_df[summary_df["drift_status"] == "STABLE"].sort_values("mean_iv", ascending=False)
    drift_candidates = summary_df[summary_df["drift_status"] == "HIGH_DRIFT"].sort_values("max_psi", ascending=False)

    selected_rows = []
    if not stable_candidates.empty:
        selected_rows.extend(stable_candidates.head(2).to_dict(orient="records"))
    if not drift_candidates.empty:
        selected_rows.extend(drift_candidates.head(2).to_dict(orient="records"))
    if len(selected_rows) < top_n:
        remaining = summary_df.sort_values("mean_iv", ascending=False)
        for _, r in remaining.iterrows():
            if not any(s["feature"] == r["feature"] and s["target"] == r["target"] for s in selected_rows):
                selected_rows.append(r.to_dict())
            if len(selected_rows) >= top_n:
                break

    n_plots = len(selected_rows)
    if n_plots == 1:
        nrows, ncols = 1, 1
        figsize = (8, 4.5)
    elif n_plots == 2:
        nrows, ncols = 1, 2
        figsize = (14, 4.5)
    else:
        nrows, ncols = 2, 2
        figsize = (14, 9)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    fold_colors = ["#2563eb", "#059669", "#d97706", "#7c3aed", "#db2777", "#4b5563"]

    for i, item in enumerate(selected_rows):
        ax = flat_axes[i]
        feat = item["feature"]
        tgt = item["target"]
        cls_val = item["target_class"]
        status = item["drift_status"]

        subset = quantile_df[
            (quantile_df["feature"] == feat)
            & (quantile_df["target"] == tgt)
            & (quantile_df["target_class"] == cls_val)
        ]

        folds = sorted(subset["fold"].unique())
        for f_num in folds:
            f_data = subset[subset["fold"] == f_num].sort_values("bin_index")
            bins = f_data["bin_index"].to_numpy()
            probs = f_data["conditional_prob"].to_numpy()
            color = fold_colors[(f_num - 1) % len(fold_colors)]
            ax.plot(bins, probs, marker="o", markersize=4, label=f"Fold {f_num}", color=color, alpha=0.75, linewidth=1.5)

        # Plot overall curve as thick dashed line
        overall_data = subset[subset["fold"] == folds[0]].sort_values("bin_index")
        ax.plot(
            overall_data["bin_index"].to_numpy(),
            overall_data["overall_prob"].to_numpy(),
            color="#111827",
            linestyle="--",
            linewidth=2.5,
            label="Overall Ref",
        )

        status_color = "#16a34a" if status == "STABLE" else "#eab308" if status == "MODERATE_DRIFT" else "#dc2626"
        ax.set_title(
            f"{feat} vs {tgt} ({cls_val})\nStatus: {status} | Mean IV: {item['mean_iv']:.3f} | Max PSI: {item['max_psi']:.3f}",
            fontsize=10.5,
            fontweight="bold",
            color=status_color,
        )
        ax.set_xlabel("Quantile Bin (1 - 20)", fontsize=9.5)
        ax.set_ylabel("P(Class | Bin)", fontsize=9.5)
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, linestyle=":", alpha=0.6)

    for j in range(n_plots, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_markdown(
    metadata: ProbabilityDriftRunMetadata,
    summary_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
) -> str:
    metadata_table = _markdown_table(
        pd.DataFrame(
            [
                {"Metric": "Module", "Value": metadata.module},
                {"Metric": "Created At", "Value": metadata.created_at},
                {"Metric": "Execution Time", "Value": metadata.execution_time},
                {"Metric": "Feature File", "Value": metadata.feature_csv},
                {"Metric": "Label File", "Value": metadata.label_csv},
                {"Metric": "Join Strategy", "Value": metadata.join_strategy},
                {"Metric": "Quantile Bins", "Value": metadata.n_bins},
                {"Metric": "Chronological Folds", "Value": metadata.n_folds},
                {"Metric": "Features Analyzed", "Value": len(metadata.features)},
                {"Metric": "Targets Analyzed", "Value": ", ".join(metadata.targets)},
                {"Metric": "Rows Evaluated", "Value": metadata.model_rows},
            ]
        )
    )

    stable_count = int((summary_df["drift_status"] == "STABLE").sum()) if not summary_df.empty else 0
    mod_count = int((summary_df["drift_status"] == "MODERATE_DRIFT").sum()) if not summary_df.empty else 0
    drift_count = int((summary_df["drift_status"] == "HIGH_DRIFT").sum()) if not summary_df.empty else 0

    top_stable = (
        summary_df[summary_df["drift_status"] == "STABLE"].sort_values("stability_ratio", ascending=False).head(15)
        if not summary_df.empty
        else pd.DataFrame()
    )
    top_drifting = (
        summary_df[summary_df["drift_status"] == "HIGH_DRIFT"].sort_values("max_psi", ascending=False).head(15)
        if not summary_df.empty
        else pd.DataFrame()
    )

    stable_table = _markdown_table(top_stable) if not top_stable.empty else "No stable features found."
    drifting_table = _markdown_table(top_drifting) if not top_drifting.empty else "No high-drift features found."

    insights: list[str] = [
        f"- **Stability Breakdown**: `{stable_count}` features classified as 🟢 **STABLE**, `{mod_count}` as 🟡 **MODERATE_DRIFT**, and `{drift_count}` as 🔴 **HIGH_DRIFT**.",
    ]

    if not top_stable.empty:
        best_stable = top_stable.iloc[0]
        insights.append(
            f"- **Most Robust Feature**: `{best_stable['feature']}` for target `{best_stable['target']}` achieved Mean IV = `{best_stable['mean_iv']}` with Stability Ratio = `{best_stable['stability_ratio']}` and 0 regime flips."
        )

    if not top_drifting.empty:
        worst_drift = top_drifting.iloc[0]
        insights.append(
            f"- **Highest Population Drift (PSI)**: `{worst_drift['feature']}` showed Max PSI = `{worst_drift['max_psi']}` (drastic quantile distribution shift across market cycles)."
        )

    insights_text = "\n".join(insights)

    return f"""# Probability Drift & Alpha Stability Report

## Executive Summary & Key Insights

{insights_text}

## Run Metadata

{metadata_table}

## Top Stable Features (Robust Across All Time Folds)

{stable_table}

## High Drift / Regime-Inversion Features (Caution Advised)

{drifting_table}

## Visual Drift & Stability Curves

![Probability Drift Charts](probability_drift_charts.png)

## Artifacts

- `summary.json`
- `feature_drift_scores.csv`
- `fold_probability_metrics.csv`
- `quantile_drift_probabilities.csv`
- `probability_drift_charts.png`
- `report.html`
"""


def _render_html(
    metadata: ProbabilityDriftRunMetadata,
    markdown: str,
    summary_df: pd.DataFrame,
    fold_metrics_df: pd.DataFrame,
) -> str:
    stable_count = int((summary_df["drift_status"] == "STABLE").sum()) if not summary_df.empty else 0
    mod_count = int((summary_df["drift_status"] == "MODERATE_DRIFT").sum()) if not summary_df.empty else 0
    drift_count = int((summary_df["drift_status"] == "HIGH_DRIFT").sum()) if not summary_df.empty else 0

    display_cols = [
        "feature",
        "target",
        "target_class",
        "drift_status",
        "mean_iv",
        "iv_std",
        "stability_ratio",
        "mean_prob_spread",
        "monotonicity_mean",
        "monotonicity_flips",
        "max_psi",
    ]

    summary_html = (
        summary_df[display_cols].head(35).to_html(index=False, classes="data-table")
        if not summary_df.empty
        else "<p>No scores available.</p>"
    )

    details = _html_markdown_details(markdown)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Probability Drift & Alpha Stability Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            margin: 2rem auto;
            max-width: 1200px;
            padding: 0 1.5rem;
            color: #1e293b;
            background-color: #f8fafc;
        }}
        h1, h2, h3 {{ color: #0f172a; margin-top: 1.5rem; }}
        .card {{
            background: #ffffff;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
        }}
        .metrics-banner {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric-card {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        .metric-card.stable {{
            background: #f0fdf4;
            border-color: #bbf7d0;
        }}
        .metric-card.warning {{
            background: #fefce8;
            border-color: #fef08a;
        }}
        .metric-card.danger {{
            background: #fef2f2;
            border-color: #fecaca;
        }}
        .metric-title {{ font-size: 0.8rem; color: #64748b; text-transform: uppercase; font-weight: 700; }}
        .metric-value {{ font-size: 1.6rem; color: #0f172a; font-weight: 700; margin-top: 0.25rem; }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .meta-item {{
            background: #f1f5f9;
            padding: 0.75rem 1rem;
            border-radius: 6px;
        }}
        .meta-label {{ font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: 600; }}
        .meta-val {{ font-size: 1rem; color: #0f172a; font-weight: 500; margin-top: 0.25rem; }}
        .table-container {{
            overflow-x: auto;
            margin: 1rem 0;
        }}
        table.data-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.88rem;
        }}
        table.data-table th, table.data-table td {{
            border: 1px solid #e2e8f0;
            padding: 8px 12px;
            text-align: left;
        }}
        table.data-table th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: #334155;
        }}
        table.data-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            margin-top: 1rem;
        }}
        details.markdown-source {{
            margin-top: 2rem;
            padding: 1rem;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <h1>Probability Drift & Alpha Stability Report</h1>

    <div class="metrics-banner">
        <div class="metric-card stable">
            <div class="metric-title">🟢 Stable Features</div>
            <div class="metric-value">{stable_count}</div>
        </div>
        <div class="metric-card warning">
            <div class="metric-title">🟡 Moderate Drift</div>
            <div class="metric-value">{mod_count}</div>
        </div>
        <div class="metric-card danger">
            <div class="metric-title">🔴 High Drift / Flips</div>
            <div class="metric-value">{drift_count}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Time Folds</div>
            <div class="metric-value">{metadata.n_folds} Folds</div>
        </div>
    </div>

    <div class="card">
        <h2>Run Overview</h2>
        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-label">Module</div>
                <div class="meta-val">{metadata.module}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Execution Time</div>
                <div class="meta-val">{metadata.execution_time}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Quantile Bins</div>
                <div class="meta-val">{metadata.n_bins} Bins</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Evaluated Rows</div>
                <div class="meta-val">{metadata.model_rows}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Top Feature Stability & Drift Scores</h2>
        <div class="table-container">
            {summary_html}
        </div>
    </div>

    <div class="card">
        <h2>Multi-Fold Probability Drift Curves</h2>
        <img class="chart-img" src="probability_drift_charts.png" alt="Probability Drift Visualization"/>
    </div>

    {details}
</body>
</html>
"""


class ProbabilityDriftModule:
    name = "probability_drift"

    def __init__(
        self,
        progress: bool | None = None,
        n_bins: int = DEFAULT_N_BINS,
        n_folds: int = DEFAULT_N_FOLDS,
    ) -> None:
        self.progress = progress
        self.n_bins = n_bins
        self.n_folds = n_folds

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        start_time = time.perf_counter()
        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        with ModuleProgress(self.name, total=4, enabled=self.progress) as progress_bar:
            features = _read_table_with_date_index(feature_csv)
            labels = _read_table_with_date_index(label_csv)
            merged, feature_columns, label_columns, join_strategy = _merge_inputs(
                features, labels, join_key
            )

            ignored_columns = set(_date_columns([*feature_columns, *label_columns]))
            feature_columns = [col for col in feature_columns if col not in ignored_columns]
            label_columns = [col for col in label_columns if col not in ignored_columns]

            selected_targets = _select_targets(label_columns, targets)
            numeric_features = _numeric_feature_columns(
                merged, feature_columns, min_non_null=MIN_NON_NULL
            )

            valid_targets: list[str] = []
            for target_col in selected_targets:
                clean_target = merged[target_col].dropna()
                if 2 <= clean_target.nunique() <= MAX_LABEL_CLASSES:
                    valid_targets.append(target_col)

            model_frame = _sample_rows(
                merged[[*numeric_features, *valid_targets]],
                MAX_ROWS,
                RANDOM_STATE,
            ) if valid_targets and numeric_features else merged
            progress_bar.step("load")

            all_summary_scores: list[dict[str, object]] = []
            all_fold_metrics: list[dict[str, object]] = []
            all_quantile_details: list[dict[str, object]] = []

            for feature_col in numeric_features:
                for target_col in valid_targets:
                    scores, fold_m, q_details = _evaluate_feature_drift(
                        model_frame[feature_col],
                        model_frame[target_col],
                        feature_col,
                        target_col,
                        n_bins=self.n_bins,
                        n_folds=self.n_folds,
                    )
                    all_summary_scores.extend(scores)
                    all_fold_metrics.extend(fold_m)
                    all_quantile_details.extend(q_details)

            summary_df = pd.DataFrame(all_summary_scores)
            fold_metrics_df = pd.DataFrame(all_fold_metrics)
            quantile_df = pd.DataFrame(all_quantile_details)

            if not summary_df.empty:
                summary_df = summary_df.sort_values(
                    ["stability_ratio", "mean_iv"],
                    ascending=[False, False],
                ).reset_index(drop=True)

            progress_bar.step("drift_evaluation")

            plot_path = run_dir / "probability_drift_charts.png"
            _plot_probability_drift_charts(summary_df, quantile_df, plot_path)
            progress_bar.step("charts")

            metadata = ProbabilityDriftRunMetadata(
                module=self.name,
                created_at=datetime.now(UTC).isoformat(),
                execution_time=_format_duration(time.perf_counter() - start_time),
                feature_csv=str(feature_csv),
                label_csv=str(label_csv),
                join_strategy=join_strategy,
                feature_shape=DatasetShape(*features.shape),
                label_shape=DatasetShape(*labels.shape),
                merged_shape=DatasetShape(*merged.shape),
                n_bins=self.n_bins,
                n_folds=self.n_folds,
                features=numeric_features,
                targets=valid_targets,
                model_rows=len(model_frame),
            )

            scores_csv_path = _write_csv(run_dir / "feature_drift_scores.csv", summary_df)
            fold_csv_path = _write_csv(run_dir / "fold_probability_metrics.csv", fold_metrics_df)
            quantiles_csv_path = _write_csv(run_dir / "quantile_drift_probabilities.csv", quantile_df)

            summary_payload: dict[str, object] = {
                **asdict(metadata),
                "top_stable_features": summary_df[summary_df["drift_status"] == "STABLE"].head(10).to_dict(orient="records")
                if not summary_df.empty
                else [],
                "top_drifting_features": summary_df[summary_df["drift_status"] == "HIGH_DRIFT"].head(10).to_dict(orient="records")
                if not summary_df.empty
                else [],
                "summary_metrics": {
                    "features_evaluated": len(numeric_features),
                    "targets_evaluated": len(valid_targets),
                    "stable_count": int((summary_df["drift_status"] == "STABLE").sum()) if not summary_df.empty else 0,
                    "moderate_drift_count": int((summary_df["drift_status"] == "MODERATE_DRIFT").sum()) if not summary_df.empty else 0,
                    "high_drift_count": int((summary_df["drift_status"] == "HIGH_DRIFT").sum()) if not summary_df.empty else 0,
                },
            }
            summary_json_path = _write_json(run_dir / "summary.json", summary_payload)

            markdown_report = _render_markdown(metadata, summary_df, fold_metrics_df)
            report_md_path = run_dir / "report.md"
            report_md_path.write_text(markdown_report, encoding="utf-8")

            html_report = _render_html(metadata, markdown_report, summary_df, fold_metrics_df)
            report_html_path = run_dir / "report.html"
            report_html_path.write_text(html_report, encoding="utf-8")

            artifacts = [
                summary_json_path,
                scores_csv_path,
                fold_csv_path,
                quantiles_csv_path,
                plot_path,
                report_md_path,
                report_html_path,
            ]
            progress_bar.step("reports")

            return ModuleResult(report_dir=run_dir, artifacts=artifacts)
