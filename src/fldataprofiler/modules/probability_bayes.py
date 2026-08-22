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
from scipy import special, stats

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
DEFAULT_PRIOR_STRENGTH = 10.0
EPSILON = 1e-9


@dataclass(frozen=True)
class ProbabilityBayesRunMetadata:
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
    prior_strength: float
    features: list[str]
    targets: list[str]
    model_rows: int


def _compute_quantile_bins(series: pd.Series, n_bins: int = DEFAULT_N_BINS) -> pd.Series:
    """Assign rank-based equal-frequency quantile bin indices (1 to n_bins)."""
    if len(series) == 0:
        return pd.Series(dtype=int, index=series.index)
    ranks = series.rank(method="first")
    actual_bins = min(n_bins, len(series))
    if actual_bins < 1:
        return pd.Series(1, index=series.index, dtype=int)
    bins = pd.qcut(ranks, q=actual_bins, labels=False) + 1
    return bins.astype(int)


def _compute_log_bayes_factor(
    k_events: int,
    n_k: int,
    alpha_0: float,
    beta_0: float,
    p_0: float,
) -> float:
    """Compute natural log of Bayes Factor BF_10 (H1: bin-specific rate vs H0: global base rate)."""
    if n_k <= 0 or p_0 <= 0.0 or p_0 >= 1.0 or alpha_0 <= 0.0 or beta_0 <= 0.0:
        return 0.0

    k_non_events = n_k - k_events
    # Marginal log-likelihood under H1 (Beta-Binomial conjugate marginal)
    log_m_h1 = (
        special.betaln(k_events + alpha_0, k_non_events + beta_0)
        - special.betaln(alpha_0, beta_0)
    )

    # Log-likelihood under H0 (Binomial with parameter p_0)
    p_0_clamped = np.clip(p_0, EPSILON, 1.0 - EPSILON)
    log_m_h0 = (
        k_events * np.log(p_0_clamped)
        + k_non_events * np.log(1.0 - p_0_clamped)
    )

    log_bf = float(log_m_h1 - log_m_h0)
    return log_bf if np.isfinite(log_bf) else 0.0


def _compute_feature_target_bayes_probabilities(
    feature_series: pd.Series,
    target_series: pd.Series,
    feature_name: str,
    target_name: str,
    n_bins: int = DEFAULT_N_BINS,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Calculate Bayesian posterior probabilities, 95% Credible Intervals, Bayes WoE, Bayes IV, Monotonicity, and Entropy."""
    clean_feature = _numeric_series(feature_series)
    valid_mask = clean_feature.notna() & target_series.notna()

    x = clean_feature[valid_mask]
    y = target_series[valid_mask]

    if len(x) < 2 or y.nunique(dropna=True) < 2 or x.nunique(dropna=True) < 2:
        return [], []

    bins = _compute_quantile_bins(x, n_bins=n_bins)
    df = pd.DataFrame({"x": x, "y": y, "bin": bins})
    n_total = len(df)
    unique_classes = sorted(df["y"].unique(), key=lambda v: str(v))
    bin_indices = sorted(df["bin"].unique())

    # Precompute global base rates (Priors)
    class_global_counts = {c: int((df["y"] == c).sum()) for c in unique_classes}
    class_base_rates = {c: class_global_counts[c] / n_total for c in unique_classes}

    # Precalculate per-bin values
    bin_stats: dict[int, dict[str, object]] = {}
    for k in bin_indices:
        bin_df = df[df["bin"] == k]
        n_k = len(bin_df)
        if n_k == 0:
            continue

        val_min = float(bin_df["x"].min())
        val_max = float(bin_df["x"].max())
        val_mean = float(bin_df["x"].mean())

        class_counts: dict[object, int] = {}
        for c in unique_classes:
            class_counts[c] = int((bin_df["y"] == c).sum())

        # Multiclass Dirichlet posterior mean vector
        # P(Y=c | Bin_k) = (N_kc + m * P_0(c)) / (N_k + m)
        bayes_probs: dict[object, float] = {}
        bayes_entropy_k = 0.0
        for c in unique_classes:
            p_0 = class_base_rates[c]
            alpha_0 = prior_strength * p_0
            bayes_prob_c = (class_counts[c] + alpha_0) / (n_k + prior_strength)
            bayes_probs[c] = float(bayes_prob_c)
            if bayes_prob_c > 0:
                bayes_entropy_k -= float(bayes_prob_c * np.log2(bayes_prob_c))

        bin_stats[k] = {
            "n_k": n_k,
            "val_min": val_min,
            "val_max": val_max,
            "val_mean": val_mean,
            "class_counts": class_counts,
            "bayes_probs": bayes_probs,
            "bayes_entropy": bayes_entropy_k,
        }

    score_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []

    # For each class c, compute Bayesian WoE, IV, CI, BF, Spread, Monotonicity
    for c in unique_classes:
        p_0 = class_base_rates[c]
        alpha_0 = prior_strength * p_0
        beta_0 = prior_strength * (1.0 - p_0)

        bayes_probs_list: list[float] = []
        raw_probs_list: list[float] = []
        ci_widths_list: list[float] = []
        log_bfs_list: list[float] = []
        class_entropies: list[float] = []

        bayes_iv_total = 0.0

        for k in bin_indices:
            stats_k = bin_stats.get(k)
            if not stats_k:
                continue

            n_k = int(stats_k["n_k"])
            k_events = int(stats_k["class_counts"][c])
            raw_prob_k = k_events / n_k if n_k > 0 else 0.0
            bayes_prob_k = float(stats_k["bayes_probs"][c])

            # Posterior parameters for Beta distribution
            alpha_post = k_events + alpha_0
            beta_post = (n_k - k_events) + beta_0

            # 95% Bayesian Credible Interval
            ci_lower = float(stats.beta.ppf(0.025, alpha_post, beta_post))
            ci_upper = float(stats.beta.ppf(0.975, alpha_post, beta_post))
            ci_width = ci_upper - ci_lower

            # Bayes Factor
            log_bf = _compute_log_bayes_factor(k_events, n_k, alpha_0, beta_0, p_0)

            # Bayesian Likelihood Distributions for WoE & IV
            # P_bayes(Bin_k | Y=c) = P_bayes(Y=c | Bin_k) * (n_k / N_total) / P_0(c)
            p_bin = n_k / n_total
            dist_event_bayes = (bayes_prob_k * p_bin) / p_0 if p_0 > 0 else 0.0
            dist_non_event_bayes = (
                ((1.0 - bayes_prob_k) * p_bin) / (1.0 - p_0) if p_0 < 1.0 else 0.0
            )

            # Robust WoE without arbitrary epsilon hacks because bayes_prob_k is strictly in (0, 1)
            dist_e = max(dist_event_bayes, EPSILON)
            dist_ne = max(dist_non_event_bayes, EPSILON)
            bayes_woe_k = float(np.log(dist_e / dist_ne))
            bayes_iv_k = float((dist_event_bayes - dist_non_event_bayes) * bayes_woe_k)
            bayes_iv_total += bayes_iv_k

            bayes_entropy_k = float(stats_k["bayes_entropy"])

            bayes_probs_list.append(bayes_prob_k)
            raw_probs_list.append(raw_prob_k)
            ci_widths_list.append(ci_width)
            log_bfs_list.append(log_bf)
            class_entropies.append(bayes_entropy_k)

            quantile_rows.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "target_class": c,
                    "bin_index": int(k),
                    "val_min": _round(stats_k["val_min"]),
                    "val_max": _round(stats_k["val_max"]),
                    "val_mean": _round(stats_k["val_mean"]),
                    "sample_count": int(n_k),
                    "raw_prob": _round(raw_prob_k),
                    "bayes_prob": _round(bayes_prob_k),
                    "ci_lower_95": _round(ci_lower),
                    "ci_upper_95": _round(ci_upper),
                    "ci_width": _round(ci_width),
                    "log_bayes_factor": _round(log_bf),
                    "bayes_woe": _round(bayes_woe_k),
                    "bayes_iv_contribution": _round(bayes_iv_k),
                    "bayes_entropy": _round(bayes_entropy_k),
                }
            )

        bayes_arr = np.array(bayes_probs_list, dtype=float)
        raw_arr = np.array(raw_probs_list, dtype=float)

        bayes_spread = float(bayes_arr.max() - bayes_arr.min()) if len(bayes_arr) > 0 else 0.0
        raw_spread = float(raw_arr.max() - raw_arr.min()) if len(raw_arr) > 0 else 0.0

        # Monotonicity on Bayesian posterior probabilities
        if len(bayes_arr) < 2 or np.all(np.isclose(bayes_arr, bayes_arr[0])):
            bayes_mono = 0.0
        else:
            spearman_res = stats.spearmanr(np.arange(1, len(bayes_arr) + 1), bayes_arr)
            corr = getattr(spearman_res, "statistic", getattr(spearman_res, "correlation", 0.0))
            bayes_mono = 0.0 if np.isnan(corr) else float(corr)

        mean_ci_width = float(np.mean(ci_widths_list)) if ci_widths_list else 0.0
        mean_log_bf = float(np.mean(log_bfs_list)) if log_bfs_list else 0.0
        mean_entropy = float(np.mean(class_entropies)) if class_entropies else 0.0

        score_rows.append(
            {
                "feature": feature_name,
                "target": target_name,
                "target_class": c,
                "bayes_information_value": _round(bayes_iv_total),
                "bayes_prob_spread": _round(bayes_spread),
                "raw_prob_spread": _round(raw_spread),
                "max_bayes_prob": _round(float(bayes_arr.max())) if len(bayes_arr) > 0 else 0.0,
                "min_bayes_prob": _round(float(bayes_arr.min())) if len(bayes_arr) > 0 else 0.0,
                "base_rate": _round(p_0),
                "bayes_monotonicity": _round(bayes_mono),
                "mean_log_bayes_factor": _round(mean_log_bf),
                "mean_ci_width": _round(mean_ci_width),
                "mean_entropy": _round(mean_entropy),
                "n_bins": int(len(bayes_probs_list)),
                "prior_strength": float(prior_strength),
                "sample_count": int(n_total),
            }
        )

    return score_rows, quantile_rows


def _plot_bayes_probability_distribution(
    scores_df: pd.DataFrame,
    quantiles_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot Bayesian posterior probability distributions with 95% Credible Interval bands for top features."""
    if scores_df.empty or quantiles_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No probability data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Identify top unique feature-target relationships by bayes_information_value
    top_pairs_df = (
        scores_df.sort_values(
            ["bayes_information_value", "bayes_prob_spread"], ascending=[False, False]
        )
        .drop_duplicates(subset=["feature", "target"])
        .head(4)
    )

    n_plots = len(top_pairs_df)
    if n_plots == 1:
        nrows, ncols = 1, 1
        figsize = (8.5, 4.5)
    elif n_plots == 2:
        nrows, ncols = 1, 2
        figsize = (15, 4.5)
    else:
        nrows, ncols = 2, 2
        figsize = (15, 9.5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    for i, (_, row) in enumerate(top_pairs_df.iterrows()):
        ax = flat_axes[i]
        feature = str(row["feature"])
        target_name = str(row["target"])

        feat_scores = scores_df[
            (scores_df["feature"] == feature) & (scores_df["target"] == target_name)
        ]
        feat_quantiles = quantiles_df[
            (quantiles_df["feature"] == feature) & (quantiles_df["target"] == target_name)
        ]

        target_classes = sorted(feat_quantiles["target_class"].unique(), key=lambda v: str(v))

        if len(target_classes) == 2:
            # Binary: plot posterior curve with 95% Credible Interval error bars and raw ghost bars
            pos_classes = [
                c
                for c in target_classes
                if str(c).lower() in ("1", "1.0", "true", "buy", "up", "pos", "positive")
            ]
            if pos_classes:
                c = pos_classes[0]
                matching_rows = feat_scores[feat_scores["target_class"] == c]
                best_class_row = (
                    matching_rows.iloc[0] if not matching_rows.empty else feat_scores.iloc[0]
                )
            else:
                best_class_row = feat_scores.sort_values(
                    "bayes_prob_spread", ascending=False
                ).iloc[0]
                c = best_class_row["target_class"]

            c_quantiles = feat_quantiles[feat_quantiles["target_class"] == c].sort_values(
                "bin_index"
            )

            bins = c_quantiles["bin_index"].to_numpy()
            bayes_probs = c_quantiles["bayes_prob"].to_numpy()
            raw_probs = c_quantiles["raw_prob"].to_numpy()
            ci_lower = c_quantiles["ci_lower_95"].to_numpy()
            ci_upper = c_quantiles["ci_upper_95"].to_numpy()
            base_rate = float(best_class_row["base_rate"])
            iv = float(best_class_row["bayes_information_value"])
            spread = float(best_class_row["bayes_prob_spread"])

            # Ghost bars for raw empirical probability
            ax.bar(
                bins,
                raw_probs,
                color="#cbd5e1",
                alpha=0.45,
                width=0.65,
                label="Raw Empirical P(Y|Bin)",
            )

            # Posterior curve with error bars (95% Credible Interval)
            yerr = [bayes_probs - ci_lower, ci_upper - bayes_probs]
            ax.errorbar(
                bins,
                bayes_probs,
                yerr=yerr,
                fmt="-o",
                color="#2563eb",
                linewidth=2.2,
                markersize=5,
                capsize=3.5,
                capthick=1.2,
                label="Bayes Posterior (95% CI)",
            )

            # Fill shaded 95% Credible band
            ax.fill_between(bins, ci_lower, ci_upper, color="#3b82f6", alpha=0.15)

            # Prior Base rate reference
            base_label = (
                f"Prior Base Rate ({base_rate:.2%})"
                if base_rate < 0.05
                else f"Prior Base Rate ({base_rate:.2f})"
            )
            ax.axhline(
                base_rate,
                color="#ef4444",
                linestyle="--",
                linewidth=1.6,
                label=base_label,
            )

            ax.set_title(
                f"{feature} vs {target_name} ({c})\nBayes IV: {iv:.3f} | Bayes Spread: {spread:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(loc="upper left", fontsize=8.5)
            ax.set_ylabel(f"Bayesian P({c} | Bin)", fontsize=10)

            y_max = max(float(np.max(ci_upper)) if len(ci_upper) > 0 else 0.0, base_rate)
            upper_limit = min(1.05, max(0.01, y_max * 1.25))
            ax.set_ylim(0.0, upper_limit)
        else:
            # Multiclass: plot Bayesian curves with shaded credible bands
            for idx, c in enumerate(target_classes):
                c_quantiles = feat_quantiles[feat_quantiles["target_class"] == c].sort_values(
                    "bin_index"
                )
                bins = c_quantiles["bin_index"].to_numpy()
                bayes_probs = c_quantiles["bayes_prob"].to_numpy()
                ci_lower = c_quantiles["ci_lower_95"].to_numpy()
                ci_upper = c_quantiles["ci_upper_95"].to_numpy()
                color = colors[idx % len(colors)]

                ax.plot(
                    bins,
                    bayes_probs,
                    marker="o",
                    linewidth=2.0,
                    label=f"Class {c}",
                    color=color,
                )
                ax.fill_between(bins, ci_lower, ci_upper, color=color, alpha=0.12)

            max_iv = float(feat_scores["bayes_information_value"].max())
            max_spread = float(feat_scores["bayes_prob_spread"].max())
            ax.set_title(
                f"{feature} vs {target_name}\nMax Bayes IV: {max_iv:.3f} | Max Bayes Spread: {max_spread:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(loc="upper left", fontsize=8.5)
            ax.set_ylabel("Bayesian P(Class | Bin)", fontsize=10)

            all_uppers = feat_quantiles["ci_upper_95"].to_numpy()
            y_max = float(np.nanmax(all_uppers)) if len(all_uppers) > 0 else 0.0
            upper_limit = min(1.05, max(0.01, y_max * 1.25))
            ax.set_ylim(0.0, upper_limit)

        n_bins_plotted = len(bins) if len(target_classes) > 0 else DEFAULT_N_BINS
        ax.set_xlabel(f"Quantile Bin (1 - {n_bins_plotted})", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide unused subplots
    for j in range(n_plots, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_markdown(
    metadata: ProbabilityBayesRunMetadata,
    scores_df: pd.DataFrame,
    quantiles_df: pd.DataFrame,
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
                {"Metric": "Prior Strength (m)", "Value": metadata.prior_strength},
                {"Metric": "Features Analyzed", "Value": len(metadata.features)},
                {"Metric": "Targets Analyzed", "Value": ", ".join(metadata.targets)},
                {"Metric": "Rows Evaluated", "Value": metadata.model_rows},
            ]
        )
    )

    top_scores_table = (
        _markdown_table(scores_df.head(25)) if not scores_df.empty else "No feature scores."
    )

    insights: list[str] = []
    if not scores_df.empty:
        top_iv = scores_df.sort_values("bayes_information_value", ascending=False).iloc[0]
        insights.append(
            f"- **Top Predictive Feature by Bayes Information Value (IV)**: `{top_iv['feature']}` for target `{top_iv['target']}` (Class `{top_iv['target_class']}`) achieved Bayes IV = `{top_iv['bayes_information_value']}`."
        )
        top_spread = scores_df.sort_values("bayes_prob_spread", ascending=False).iloc[0]
        insights.append(
            f"- **Maximum Bayesian Probability Spread (ΔP)**: `{top_spread['feature']}` showed posterior shift from `{top_spread['min_bayes_prob']}` in low bins to `{top_spread['max_bayes_prob']}` in high bins (Bayes ΔP = `{top_spread['bayes_prob_spread']}`)."
        )
        top_bf = scores_df.sort_values("mean_log_bayes_factor", ascending=False).iloc[0]
        insights.append(
            f"- **Strongest Statistical Evidence (Mean ln BF₁₀)**: `{top_bf['feature']}` recorded mean ln(BF₁₀) = `{top_bf['mean_log_bayes_factor']}` across quantile bins."
        )
        top_mono = scores_df.sort_values("bayes_monotonicity", key=abs, ascending=False).iloc[0]
        insights.append(
            f"- **Strongest Monotonic Posterior Relationship**: `{top_mono['feature']}` showed Spearman rank correlation of `{top_mono['bayes_monotonicity']}` across bins."
        )
    else:
        insights.append(
            "- No valid numeric features or categorical targets available for Bayesian probability profiling."
        )

    insights_text = "\n".join(insights)

    return f"""# Bayesian Probability & Quantile Profiling Report

## Executive Summary & Key Insights

{insights_text}

## Run Metadata

{metadata_table}

## Top Feature Bayesian Probability Scores

{top_scores_table}

## Visual Bayesian Posterior Distribution & 95% Credible Intervals

![Bayesian Probability Distribution](bayes_probability_distribution.png)

## Artifacts

- `summary.json`
- `bayes_probability_scores.csv`
- `quantile_bayes_probabilities.csv`
- `bayes_probability_distribution.png`
- `report.html`
"""


def _render_html(
    metadata: ProbabilityBayesRunMetadata,
    markdown: str,
    scores_df: pd.DataFrame,
    quantiles_df: pd.DataFrame,
) -> str:
    scores_html = (
        scores_df.head(30).to_html(index=False, classes="data-table")
        if not scores_df.empty
        else "<p>No scores available.</p>"
    )
    details = _html_markdown_details(markdown)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Bayesian Probability & Quantile Profiling Report</title>
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
            font-size: 0.9rem;
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
    <h1>Bayesian Probability & Quantile Profiling Report</h1>

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
                <div class="meta-label">Prior Strength (m)</div>
                <div class="meta-val">{metadata.prior_strength}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Quantile Bins</div>
                <div class="meta-val">{metadata.n_bins}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Evaluated Rows</div>
                <div class="meta-val">{metadata.model_rows}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Top Feature Bayesian Probability Scores</h2>
        <div class="table-container">
            {scores_html}
        </div>
    </div>

    <div class="card">
        <h2>Bayesian Posterior & 95% Credible Interval Visualization</h2>
        <img class="chart-img" src="bayes_probability_distribution.png" alt="Bayesian Probability Distribution Visualization"/>
    </div>

    {details}
</body>
</html>
"""


class ProbabilityBayesModule:
    name = "probability_bayes"

    def __init__(
        self,
        progress: bool | None = None,
        n_bins: int = DEFAULT_N_BINS,
        prior_strength: float = DEFAULT_PRIOR_STRENGTH,
    ) -> None:
        self.progress = progress
        self.n_bins = n_bins
        self.prior_strength = prior_strength

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

            # Filter for discrete/categorical targets
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
            progress_bar.step("load")

            all_score_rows: list[dict[str, object]] = []
            all_quantile_rows: list[dict[str, object]] = []

            for feature_col in numeric_features:
                for target_col in valid_targets:
                    scores, quantiles = _compute_feature_target_bayes_probabilities(
                        model_frame[feature_col],
                        model_frame[target_col],
                        feature_col,
                        target_col,
                        n_bins=self.n_bins,
                        prior_strength=self.prior_strength,
                    )
                    all_score_rows.extend(scores)
                    all_quantile_rows.extend(quantiles)

            scores_df = pd.DataFrame(all_score_rows)
            quantiles_df = pd.DataFrame(all_quantile_rows)

            if not scores_df.empty:
                scores_df = scores_df.sort_values(
                    ["bayes_information_value", "bayes_prob_spread"],
                    ascending=[False, False],
                ).reset_index(drop=True)

            progress_bar.step("bayes_probabilities")

            plot_path = run_dir / "bayes_probability_distribution.png"
            _plot_bayes_probability_distribution(scores_df, quantiles_df, plot_path)
            progress_bar.step("plots")

            metadata = ProbabilityBayesRunMetadata(
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
                prior_strength=self.prior_strength,
                features=numeric_features,
                targets=valid_targets,
                model_rows=len(model_frame),
            )

            scores_csv_path = _write_csv(run_dir / "bayes_probability_scores.csv", scores_df)
            quantiles_csv_path = _write_csv(
                run_dir / "quantile_bayes_probabilities.csv", quantiles_df
            )

            summary_payload: dict[str, object] = {
                **asdict(metadata),
                "top_features": scores_df.head(10).to_dict(orient="records")
                if not scores_df.empty
                else [],
                "summary_metrics": {
                    "features_evaluated": len(numeric_features),
                    "targets_evaluated": len(valid_targets),
                    "max_bayes_information_value": _round(
                        float(scores_df["bayes_information_value"].max())
                    )
                    if not scores_df.empty
                    else 0.0,
                    "max_bayes_prob_spread": _round(
                        float(scores_df["bayes_prob_spread"].max())
                    )
                    if not scores_df.empty
                    else 0.0,
                    "mean_ci_width": _round(float(scores_df["mean_ci_width"].mean()))
                    if not scores_df.empty
                    else 0.0,
                },
            }
            summary_json_path = _write_json(run_dir / "summary.json", summary_payload)

            markdown_report = _render_markdown(metadata, scores_df, quantiles_df)
            report_md_path = run_dir / "report.md"
            report_md_path.write_text(markdown_report, encoding="utf-8")

            html_report = _render_html(metadata, markdown_report, scores_df, quantiles_df)
            report_html_path = run_dir / "report.html"
            report_html_path.write_text(html_report, encoding="utf-8")

            artifacts = [
                summary_json_path,
                scores_csv_path,
                quantiles_csv_path,
                plot_path,
                report_md_path,
                report_html_path,
            ]
            progress_bar.step("reports")

            return ModuleResult(report_dir=run_dir, artifacts=artifacts)
