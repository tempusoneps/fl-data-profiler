from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fldataprofiler.config import get_module_config
from fldataprofiler.modules.base import ModuleResult
from fldataprofiler.modules.progress import ModuleProgress, StatusTimer
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
DEFAULT_N_BINS = 10
DEFAULT_MAX_FEATURES = 12
DEFAULT_BASE_SCORE = 600
DEFAULT_BASE_ODDS = 1.0
DEFAULT_PDO = 20.0
DEFAULT_MIN_IV = 0.02
DEFAULT_SCORE_MIN_BOUND = 300
DEFAULT_SCORE_MAX_BOUND = 850
EPSILON = 1e-9


@dataclass
class ProbabilityScorecardConfig:
    base_score: int = DEFAULT_BASE_SCORE
    base_odds: float = DEFAULT_BASE_ODDS
    pdo: float = DEFAULT_PDO
    n_bins: int = DEFAULT_N_BINS
    max_features: int = DEFAULT_MAX_FEATURES
    min_iv: float = DEFAULT_MIN_IV
    score_min_bound: int = DEFAULT_SCORE_MIN_BOUND
    score_max_bound: int = DEFAULT_SCORE_MAX_BOUND


@dataclass(frozen=True)
class ProbabilityScorecardRunMetadata:
    module: str
    created_at: str
    execution_time: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    base_score: int
    base_odds: float
    pdo: float
    n_bins: int
    max_features: int
    min_iv: float
    features_selected: list[str]
    targets: list[str]
    model_rows: int
    auc: float
    ks_statistic: float


def _compute_quantile_bins_and_edges(
    series: pd.Series, n_bins: int = DEFAULT_N_BINS
) -> tuple[pd.Series, list[tuple[float, float]]]:
    """Assign quantile bins and return bin indices along with (min, max) edges."""
    if len(series) == 0:
        return pd.Series(dtype=int, index=series.index), []
    ranks = series.rank(method="first")
    actual_bins = min(n_bins, len(series))
    if actual_bins < 1:
        return pd.Series(1, index=series.index, dtype=int), [(float(series.min()), float(series.max()))]

    bin_labels = pd.qcut(ranks, q=actual_bins, labels=False) + 1
    edges: list[tuple[float, float]] = []
    for b in range(1, actual_bins + 1):
        vals = series[bin_labels == b]
        if len(vals) > 0:
            edges.append((float(vals.min()), float(vals.max())))
        else:
            edges.append((float("nan"), float("nan")))
    return bin_labels.astype(int), edges


def _compute_woe_and_iv(
    x: pd.Series,
    y_binary: pd.Series,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[dict[int, float], float, list[dict[str, Any]], pd.Series]:
    """Compute WoE mapping, Total IV, bin breakdown list, and WoE-transformed Series."""
    clean_x = _numeric_series(x)
    valid_mask = clean_x.notna() & y_binary.notna()
    x_val = clean_x[valid_mask]
    y_val = y_binary[valid_mask]

    if len(x_val) < 2 or y_val.nunique(dropna=True) < 2 or x_val.nunique(dropna=True) < 2:
        return {}, 0.0, [], pd.Series(0.0, index=x.index)

    bins, edges = _compute_quantile_bins_and_edges(x_val, n_bins=n_bins)
    df = pd.DataFrame({"x": x_val, "y": y_val, "bin": bins})
    total_events = int((df["y"] == 1).sum())
    total_non_events = int((df["y"] == 0).sum())

    if total_events == 0 or total_non_events == 0:
        return {}, 0.0, [], pd.Series(0.0, index=x.index)

    woe_map: dict[int, float] = {}
    iv_total = 0.0
    bin_details: list[dict[str, Any]] = []

    for b_idx in sorted(df["bin"].unique()):
        bin_df = df[df["bin"] == b_idx]
        n_k = len(bin_df)
        if n_k == 0:
            continue
        events_k = int((bin_df["y"] == 1).sum())
        non_events_k = n_k - events_k

        dist_e = max(events_k / total_events, EPSILON)
        dist_ne = max(non_events_k / total_non_events, EPSILON)

        woe_k = float(np.log(dist_e / dist_ne))
        iv_k = float((dist_e - dist_ne) * woe_k)
        woe_map[int(b_idx)] = woe_k
        iv_total += iv_k

        edge_min, edge_max = edges[int(b_idx) - 1] if int(b_idx) <= len(edges) else (float("nan"), float("nan"))
        bin_details.append(
            {
                "bin": int(b_idx),
                "min_val": _round(edge_min),
                "max_val": _round(edge_max),
                "sample_count": int(n_k),
                "events": int(events_k),
                "non_events": int(non_events_k),
                "event_rate": _round(events_k / n_k if n_k > 0 else 0.0),
                "woe": _round(woe_k),
                "iv": _round(iv_k),
            }
        )

    # Vectorized mapping to produce WoE series
    mapped_woe = bins.map(woe_map).fillna(0.0)
    full_woe_series = pd.Series(0.0, index=x.index)
    full_woe_series.loc[valid_mask] = mapped_woe

    return woe_map, iv_total, bin_details, full_woe_series


def _build_scorecard(
    model_frame: pd.DataFrame,
    feature_columns: list[str],
    target_series: pd.Series,
    base_score: int = DEFAULT_BASE_SCORE,
    base_odds: float = DEFAULT_BASE_ODDS,
    pdo: float = DEFAULT_PDO,
    n_bins: int = DEFAULT_N_BINS,
    max_features: int = DEFAULT_MAX_FEATURES,
    min_iv: float = DEFAULT_MIN_IV,
    progress_bar: ModuleProgress | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.Series, pd.Series]:
    """Train multivariate WoE Logistic Regression and scale coefficients into additive scorecard points."""
    y_binary = (target_series == target_series.iloc[0]).astype(int)
    # Check if target has 2 classes; if 0/1, ensure 1 is positive event
    unique_vals = sorted(target_series.dropna().unique())
    if 1 in unique_vals and 0 in unique_vals:
        y_binary = (target_series == 1).astype(int)

    # 1. Screen features by IV
    feature_ivs: dict[str, float] = {}
    feature_woe_data: dict[str, Any] = {}

    for f in feature_columns:
        woe_map, iv, bin_details, woe_series = _compute_woe_and_iv(
            model_frame[f], y_binary, n_bins=n_bins
        )
        feature_ivs[f] = iv
        feature_woe_data[f] = {
            "woe_map": woe_map,
            "iv": iv,
            "bin_details": bin_details,
            "woe_series": woe_series,
        }
        if progress_bar is not None:
            progress_bar.step(f"WoE:{f}")

    # Select top features with positive IV >= min_iv
    selected_features = [
        f for f, iv in sorted(feature_ivs.items(), key=lambda item: item[1], reverse=True)
        if iv >= min_iv
    ][:max_features]

    if not selected_features:
        selected_features = feature_columns[:max_features]

    # 2. Fit Logistic Regression on WoE features
    X_woe = pd.DataFrame({f: feature_woe_data[f]["woe_series"] for f in selected_features})
    valid_mask = y_binary.notna() & X_woe.notna().all(axis=1)

    X_train = X_woe[valid_mask]
    y_train = y_binary[valid_mask]

    clf = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    intercept = float(clf.intercept_[0])
    coefs = {f: float(c) for f, c in zip(selected_features, clf.coef_[0])}

    # 3. Scaling parameters
    factor = pdo / np.log(2.0)
    offset = base_score - (factor * np.log(base_odds))
    m = len(selected_features)

    # Scorecard points table
    points_rows: list[dict[str, Any]] = []
    base_point_offset = int(np.round(offset + factor * intercept))

    for f in selected_features:
        beta_f = coefs[f]
        bin_details = feature_woe_data[f]["bin_details"]
        for b in bin_details:
            woe_val = b["woe"]
            # Standard additive credit score points: higher positive WoE adds positive score points
            # Points = round(factor * beta_f * WoE)
            points_val = int(np.round(factor * beta_f * woe_val))
            points_rows.append(
                {
                    "feature": f,
                    "bin": b["bin"],
                    "range": f"[{b['min_val']}, {b['max_val']}]",
                    "min_val": b["min_val"],
                    "max_val": b["max_val"],
                    "sample_count": b["sample_count"],
                    "event_rate": b["event_rate"],
                    "woe": b["woe"],
                    "iv": b["iv"],
                    "feature_weight": _round(beta_f),
                    "points": points_val,
                }
            )

    points_df = pd.DataFrame(points_rows)

    # 4. Predict probabilities and total scores on dataset
    probs = clf.predict_proba(X_train)[:, 1]
    # Total Score = BaseOffset + sum(Points)
    # Exact logit-to-score scaling: Score = round(offset + factor * (intercept + sum(beta * WoE)))
    logits = intercept + X_train.dot(clf.coef_[0])
    total_scores = np.round(offset + factor * logits).astype(int)

    score_series = pd.Series(total_scores, index=X_train.index)
    prob_series = pd.Series(probs, index=X_train.index)

    # 5. Discrimination Metrics (KS Statistic & ROC AUC)
    auc_score = float(roc_auc_score(y_train, probs)) if len(np.unique(y_train)) > 1 else 0.5

    # KS-statistic
    df_eval = pd.DataFrame({"score": total_scores, "target": y_train}).sort_values("score")
    total_pos = int((df_eval["target"] == 1).sum())
    total_neg = int((df_eval["target"] == 0).sum())

    df_eval["cum_pos"] = (df_eval["target"] == 1).cumsum() / total_pos if total_pos > 0 else 0.0
    df_eval["cum_neg"] = (df_eval["target"] == 0).cumsum() / total_neg if total_neg > 0 else 0.0
    df_eval["ks_diff"] = np.abs(df_eval["cum_pos"] - df_eval["cum_neg"])
    ks_stat = float(df_eval["ks_diff"].max()) if not df_eval.empty else 0.0

    # 6. Score to Probability Calibration Table (Deciles)
    df_eval["score_decile"] = pd.qcut(df_eval["score"], q=10, labels=False, duplicates="drop") + 1
    calibration_rows: list[dict[str, Any]] = []

    for dec, grp in df_eval.groupby("score_decile", observed=False):
        n_dec = len(grp)
        e_dec = int((grp["target"] == 1).sum())
        emp_prob = e_dec / n_dec if n_dec > 0 else 0.0
        calibration_rows.append(
            {
                "decile": int(dec),
                "min_score": int(grp["score"].min()),
                "max_score": int(grp["score"].max()),
                "sample_count": int(n_dec),
                "events": int(e_dec),
                "empirical_win_rate": _round(emp_prob),
            }
        )

    calibration_df = pd.DataFrame(calibration_rows)

    metrics = {
        "auc": _round(auc_score),
        "ks_statistic": _round(ks_stat),
        "base_offset": base_point_offset,
        "intercept": _round(intercept),
        "features_count": len(selected_features),
        "selected_features": selected_features,
        "feature_weights": {f: _round(c) for f, c in coefs.items()},
    }

    return points_df, calibration_df, metrics, score_series, prob_series


def _plot_score_distributions(
    scores: pd.Series,
    y_true: pd.Series,
    calibration_df: pd.DataFrame,
    ks_stat: float,
    output_path: Path,
) -> Path:
    """Plot dual score distribution (Y=1 vs Y=0) and empirical calibration curve."""
    if scores.empty or y_true.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Scorecard data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Score Distribution Y=1 vs Y=0
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]

    bins = np.linspace(scores.min(), scores.max(), 25)
    ax1.hist(
        neg_scores,
        bins=bins,
        alpha=0.5,
        color="#64748b",
        density=True,
        label=f"Class 0 (N={len(neg_scores)})",
    )
    ax1.hist(
        pos_scores,
        bins=bins,
        alpha=0.65,
        color="#2563eb",
        density=True,
        label=f"Class 1 (N={len(pos_scores)})",
    )
    ax1.set_title(
        f"Scorecard Point Distribution (KS Statistic = {ks_stat:.1%})",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_xlabel("Total Score", fontsize=9)
    ax1.set_ylabel("Density", fontsize=9)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Plot 2: Empirical Win Rate vs Score Deciles
    if not calibration_df.empty:
        deciles = calibration_df["decile"].to_numpy()
        win_rates = calibration_df["empirical_win_rate"].to_numpy()
        score_labels = [
            f"D{d}\n({r['min_score']}-{r['max_score']})"
            for d, (_, r) in zip(deciles, calibration_df.iterrows())
        ]

        ax2.plot(deciles, win_rates, marker="o", color="#16a34a", linewidth=2.2, label="Empirical Win Rate")
        ax2.set_xticks(deciles)
        ax2.set_xticklabels(score_labels, fontsize=8, rotation=0)
        ax2.set_title("Score Deciles vs Actual Win Rate", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Score Deciles (Lowest to Highest)", fontsize=9)
        ax2.set_ylabel("Win Rate P(Y=1)", fontsize=9)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", fontsize=8)

        # Annotate points
        for x, y in zip(deciles, win_rates):
            ax2.annotate(
                f"{y:.1%}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_markdown(
    metadata: ProbabilityScorecardRunMetadata,
    points_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
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
                {"Metric": "Base Score", "Value": metadata.base_score},
                {"Metric": "Base Odds", "Value": metadata.base_odds},
                {"Metric": "Points to Double Odds (PDO)", "Value": metadata.pdo},
                {"Metric": "Bin Quantiles (n_bins)", "Value": metadata.n_bins},
                {"Metric": "Max Features", "Value": metadata.max_features},
                {"Metric": "Min Information Value (min_iv)", "Value": metadata.min_iv},
                {"Metric": "ROC AUC", "Value": f"{metadata.auc:.4f}"},
                {"Metric": "KS Statistic", "Value": f"{metadata.ks_statistic:.2%}"},
                {"Metric": "Features in Scorecard", "Value": ", ".join(metadata.features_selected)},
                {"Metric": "Targets Analyzed", "Value": ", ".join(metadata.targets)},
                {"Metric": "Evaluated Rows", "Value": metadata.model_rows},
            ]
        )
    )

    insights: list[str] = [
        f"- **Model Discrimination Power**: The scorecard achieved an **ROC AUC of {metadata.auc:.4f}** and a **Kolmogorov-Smirnov (KS) statistic of {metadata.ks_statistic:.2%}**, indicating {'excellent' if metadata.ks_statistic >= 0.3 else 'moderate'} separation between winning and losing events.",
    ]
    if not calibration_df.empty:
        top_dec = calibration_df.iloc[-1]
        bot_dec = calibration_df.iloc[0]
        insights.append(
            f"- **Decile Separation Spread**: Top Decile (Scores `{top_dec['min_score']}-{top_dec['max_score']}`) achieved a **Win Rate of {top_dec['empirical_win_rate']:.1%}** vs Bottom Decile (Scores `{bot_dec['min_score']}-{bot_dec['max_score']}`) with **{bot_dec['empirical_win_rate']:.1%}**."
        )

    insights_text = "\n".join(insights)

    points_display_cols = ["feature", "bin", "range", "sample_count", "event_rate", "woe", "iv", "points"]
    existing_p_cols = [c for c in points_display_cols if c in points_df.columns]
    points_table = _markdown_table(points_df[existing_p_cols].head(35)) if not points_df.empty else "No points available."

    calib_table = _markdown_table(calibration_df) if not calibration_df.empty else "No calibration data."

    return f"""# Multivariate WoE & Bayesian Log-Odds Scorecard Report

## Executive Summary & Key Insights

{insights_text}

## Run Metadata

{metadata_table}

## Scorecard Point Allocations

{points_table}

## Score Deciles & Win Rate Calibration

{calib_table}

## Visual Score Distribution & KS Calibration

![Score Distribution Plot](score_distribution_plot.png)

## Artifacts

- `summary.json`
- `scorecard_points.csv`
- `score_to_probability.csv`
- `score_distribution_plot.png`
- `report.html`
"""


def _render_html(
    metadata: ProbabilityScorecardRunMetadata,
    markdown: str,
    points_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> str:
    points_display_cols = ["feature", "bin", "range", "sample_count", "event_rate", "woe", "iv", "points"]
    existing_p_cols = [c for c in points_display_cols if c in points_df.columns]
    points_html = (
        points_df[existing_p_cols].to_html(index=False, classes="data-table")
        if not points_df.empty
        else "<p>No points data available.</p>"
    )

    calib_html = (
        calibration_df.to_html(index=False, classes="data-table")
        if not calibration_df.empty
        else "<p>No calibration data available.</p>"
    )

    details = _html_markdown_details(markdown)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Multivariate WoE Scorecard Report</title>
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
        .metric-card.accent {{
            background: #f0fdf4;
            border-color: #bbf7d0;
        }}
        .metric-card.warning {{
            background: #fefce8;
            border-color: #fef08a;
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
            max-height: 450px;
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
            position: sticky;
            top: 0;
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
    </style>
</head>
<body>
    <h1>Multivariate WoE & Bayesian Log-Odds Scorecard Report</h1>

    <div class="metrics-banner">
        <div class="metric-card accent">
            <div class="metric-title">ROC AUC</div>
            <div class="metric-value">{metadata.auc:.4f}</div>
        </div>
        <div class="metric-card warning">
            <div class="metric-title">KS Statistic</div>
            <div class="metric-value">{metadata.ks_statistic:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Base Score</div>
            <div class="metric-value">{metadata.base_score}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">PDO</div>
            <div class="metric-value">{metadata.pdo:.0f} pts</div>
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
                <div class="meta-label">Base Score & Odds</div>
                <div class="meta-val">{metadata.base_score} pts (Odds: {metadata.base_odds})</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">PDO</div>
                <div class="meta-val">{metadata.pdo:.0f} pts</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Bins & Min IV</div>
                <div class="meta-val">{metadata.n_bins} bins (Min IV: {metadata.min_iv})</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Features Included</div>
                <div class="meta-val">{len(metadata.features_selected)} features (Max: {metadata.max_features})</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Evaluated Rows</div>
                <div class="meta-val">{metadata.model_rows}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Score Deciles & Win Rate Calibration</h2>
        <div class="table-container">
            {calib_html}
        </div>
    </div>

    <div class="card">
        <h2>Scorecard Point Lookup Table</h2>
        <div class="table-container">
            {points_html}
        </div>
    </div>

    <div class="card">
        <h2>Visual Score Distribution & KS Calibration Curve</h2>
        <img class="chart-img" src="score_distribution_plot.png" alt="Scorecard Distribution"/>
    </div>

    {details}
</body>
</html>
"""


class ProbabilityScorecardModule:
    name = "probability_scorecard"

    def __init__(
        self,
        config: ProbabilityScorecardConfig | None = None,
        progress: bool | None = None,
        base_score: int | None = None,
        base_odds: float | None = None,
        pdo: float | None = None,
        n_bins: int | None = None,
        max_features: int | None = None,
        min_iv: float | None = None,
    ) -> None:
        self.progress = progress
        if config is not None:
            base_cfg = config
        else:
            mod_cfg = get_module_config("probability_scorecard")
            base_cfg = ProbabilityScorecardConfig(
                base_score=int(mod_cfg.get("base_score", DEFAULT_BASE_SCORE)),
                base_odds=float(mod_cfg.get("base_odds", DEFAULT_BASE_ODDS)),
                pdo=float(mod_cfg.get("pdo", DEFAULT_PDO)),
                n_bins=int(mod_cfg.get("n_bins", DEFAULT_N_BINS)),
                max_features=int(mod_cfg.get("max_features", DEFAULT_MAX_FEATURES)),
                min_iv=float(mod_cfg.get("min_iv", DEFAULT_MIN_IV)),
                score_min_bound=int(mod_cfg.get("score_min_bound", DEFAULT_SCORE_MIN_BOUND)),
                score_max_bound=int(mod_cfg.get("score_max_bound", DEFAULT_SCORE_MAX_BOUND)),
            )

        # Check environment variable overrides
        env_base_score = os.environ.get("SCORECARD_BASE_SCORE")
        env_base_odds = os.environ.get("SCORECARD_BASE_ODDS")
        env_pdo = os.environ.get("SCORECARD_PDO")
        env_n_bins = os.environ.get("SCORECARD_N_BINS")
        env_max_features = os.environ.get("SCORECARD_MAX_FEATURES")
        env_min_iv = os.environ.get("SCORECARD_MIN_IV")

        self.base_score = (
            base_score
            if base_score is not None
            else (int(env_base_score) if env_base_score else base_cfg.base_score)
        )
        self.base_odds = (
            base_odds
            if base_odds is not None
            else (float(env_base_odds) if env_base_odds else base_cfg.base_odds)
        )
        self.pdo = (
            pdo
            if pdo is not None
            else (float(env_pdo) if env_pdo else base_cfg.pdo)
        )
        self.n_bins = (
            n_bins
            if n_bins is not None
            else (int(env_n_bins) if env_n_bins else base_cfg.n_bins)
        )
        self.max_features = (
            max_features
            if max_features is not None
            else (int(env_max_features) if env_max_features else base_cfg.max_features)
        )
        self.min_iv = (
            min_iv
            if min_iv is not None
            else (float(env_min_iv) if env_min_iv else base_cfg.min_iv)
        )

    def run(
        self,
        feature_csv: str | Path,
        label_csv: str | Path,
        output_dir: str | Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        start_time = time.perf_counter()
        feature_path = Path(feature_csv)
        label_path = Path(label_csv)
        run_dir = Path(output_dir) / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        with StatusTimer(f"{self.name}: Loading & preparing", enabled=self.progress):
            features = _read_table_with_date_index(feature_path)
            labels = _read_table_with_date_index(label_path)
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

            model_frame = (
                _sample_rows(
                    merged[[*numeric_features, *valid_targets]],
                    MAX_ROWS,
                    RANDOM_STATE,
                )
                if valid_targets and numeric_features
                else merged
            )

            total_evals = len(numeric_features)
            # Scorecard training on primary target
            primary_target = valid_targets[0] if valid_targets else "target"
            target_series = model_frame[primary_target] if primary_target in model_frame else pd.Series(dtype=int)

        with ModuleProgress(
            self.name, total=max(1, total_evals), unit="feat", enabled=self.progress
        ) as progress_bar:
            points_df, calibration_df, metrics, score_series, prob_series = _build_scorecard(
                model_frame=model_frame,
                feature_columns=numeric_features,
                target_series=target_series,
                base_score=self.base_score,
                base_odds=self.base_odds,
                pdo=self.pdo,
                n_bins=self.n_bins,
                max_features=self.max_features,
                min_iv=self.min_iv,
                progress_bar=progress_bar,
            )

            if total_evals == 0:
                progress_bar.step("no_valid_features")

        # Generate Artifacts
        plot_path = run_dir / "score_distribution_plot.png"
        y_binary = (target_series == 1).astype(int) if (1 in target_series.values and 0 in target_series.values) else (target_series == target_series.iloc[0]).astype(int)
        _plot_score_distributions(score_series, y_binary.loc[score_series.index], calibration_df, metrics["ks_statistic"], plot_path)

        points_csv_path = _write_csv(
            run_dir / "scorecard_points.csv",
            points_df,
        )

        calibration_csv_path = _write_csv(
            run_dir / "score_to_probability.csv",
            calibration_df,
        )

        metadata = ProbabilityScorecardRunMetadata(
            module=self.name,
            created_at=datetime.now(UTC).isoformat(),
            execution_time=_format_duration(time.perf_counter() - start_time),
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=DatasetShape(*features.shape),
            label_shape=DatasetShape(*labels.shape),
            merged_shape=DatasetShape(*merged.shape),
            base_score=self.base_score,
            base_odds=self.base_odds,
            pdo=self.pdo,
            n_bins=self.n_bins,
            max_features=self.max_features,
            min_iv=self.min_iv,
            features_selected=metrics["selected_features"],
            targets=valid_targets,
            model_rows=len(model_frame),
            auc=metrics["auc"],
            ks_statistic=metrics["ks_statistic"],
        )

        summary_payload: dict[str, Any] = {
            **asdict(metadata),
            "model_metrics": metrics,
            "score_deciles": calibration_df.to_dict(orient="records"),
            "summary_metrics": {
                "roc_auc": metrics["auc"],
                "ks_statistic": metrics["ks_statistic"],
                "features_count": metrics["features_count"],
                "base_point_offset": metrics["base_offset"],
            },
        }
        summary_json_path = _write_json(run_dir / "summary.json", summary_payload)

        markdown_report = _render_markdown(metadata, points_df, calibration_df)
        report_md_path = run_dir / "report.md"
        report_md_path.write_text(markdown_report, encoding="utf-8")

        html_report = _render_html(metadata, markdown_report, points_df, calibration_df)
        report_html_path = run_dir / "report.html"
        report_html_path.write_text(html_report, encoding="utf-8")

        artifacts = [
            summary_json_path,
            points_csv_path,
            calibration_csv_path,
            plot_path,
            report_md_path,
            report_html_path,
        ]

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)
