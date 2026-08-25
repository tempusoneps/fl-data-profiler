from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fldataprofiler.config import get_module_config
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
DEFAULT_N_BINS = 10
DEFAULT_MAX_CANDIDATES = 16
DEFAULT_MIN_SUPPORT = 20
EPSILON = 1e-7


@dataclass
class Probability2DConfig:
    n_bins: int = DEFAULT_N_BINS
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    min_support: int = DEFAULT_MIN_SUPPORT


@dataclass(frozen=True)
class Probability2DRunMetadata:
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
    max_candidates: int
    min_support: int
    features_count: int
    candidate_features: list[str]
    pairs_evaluated: int
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


def _compute_1d_iv_and_spread(
    x: pd.Series,
    y: pd.Series,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[object, dict[str, float]]:
    """Compute 1D Information Value and Probability Spread for each class of target y."""
    clean_x = _numeric_series(x)
    valid_mask = clean_x.notna() & y.notna()
    x_val = clean_x[valid_mask]
    y_val = y[valid_mask]

    if len(x_val) < 2 or y_val.nunique(dropna=True) < 2 or x_val.nunique(dropna=True) < 2:
        return {}

    bins = _compute_quantile_bins(x_val, n_bins=n_bins)
    df = pd.DataFrame({"x": x_val, "y": y_val, "bin": bins})
    n_total = len(df)
    unique_classes = sorted(df["y"].unique(), key=lambda v: str(v))
    bin_indices = sorted(df["bin"].unique())

    # Pre-aggregate counts per bin
    bin_counts: dict[int, int] = {}
    bin_class_counts: dict[int, dict[object, int]] = {}
    for k in bin_indices:
        bin_df = df[df["bin"] == k]
        n_k = len(bin_df)
        bin_counts[k] = n_k
        bin_class_counts[k] = {c: int((bin_df["y"] == c).sum()) for c in unique_classes}

    result: dict[object, dict[str, float]] = {}
    for c in unique_classes:
        total_events = int((df["y"] == c).sum())
        total_non_events = n_total - total_events

        probs: list[float] = []
        iv_total = 0.0

        for k in bin_indices:
            n_k = bin_counts.get(k, 0)
            if n_k == 0:
                continue
            events_k = bin_class_counts[k].get(c, 0)
            non_events_k = n_k - events_k
            prob_k = events_k / n_k if n_k > 0 else 0.0
            probs.append(prob_k)

            dist_event = events_k / total_events if total_events > 0 else 0.0
            dist_non_event = non_events_k / total_non_events if total_non_events > 0 else 0.0

            p_e = max(dist_event, EPSILON)
            p_ne = max(dist_non_event, EPSILON)
            woe_k = float(np.log(p_e / p_ne))
            iv_k = float((dist_event - dist_non_event) * woe_k)
            iv_total += iv_k

        prob_arr = np.array(probs, dtype=float)
        spread = float(prob_arr.max() - prob_arr.min()) if len(prob_arr) > 0 else 0.0

        result[c] = {
            "iv": iv_total,
            "prob_spread": spread,
        }

    return result


def _prescreen_candidate_features(
    model_frame: pd.DataFrame,
    numeric_features: list[str],
    valid_targets: list[str],
    max_candidates: int,
    n_bins: int,
) -> tuple[list[str], dict[str, dict[str, dict[object, dict[str, float]]]]]:
    """Screen top features by 1D Information Value (IV) across targets."""
    feature_1d_stats: dict[str, dict[str, dict[object, dict[str, float]]]] = {}
    feature_max_iv: dict[str, float] = {}

    for f in numeric_features:
        feature_1d_stats[f] = {}
        max_iv_f = 0.0
        for t in valid_targets:
            stats_dict = _compute_1d_iv_and_spread(model_frame[f], model_frame[t], n_bins=n_bins)
            feature_1d_stats[f][t] = stats_dict
            for m in stats_dict.values():
                if m["iv"] > max_iv_f:
                    max_iv_f = m["iv"]
        feature_max_iv[f] = max_iv_f

    # Sort descending by max 1D IV
    sorted_features = sorted(numeric_features, key=lambda f: feature_max_iv.get(f, 0.0), reverse=True)
    candidate_features = sorted_features[:max_candidates]
    return candidate_features, feature_1d_stats


def _compute_pair_target_probabilities(
    f1_series: pd.Series,
    f2_series: pd.Series,
    target_series: pd.Series,
    f1_name: str,
    f2_name: str,
    target_name: str,
    n_bins: int,
    f1_1d_stats: dict[object, dict[str, float]],
    f2_1d_stats: dict[object, dict[str, float]],
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compute 2D joint quantile probabilities, 2D IV, synergy gain, and sweet spots for a feature pair."""
    clean_f1 = _numeric_series(f1_series)
    clean_f2 = _numeric_series(f2_series)
    valid_mask = clean_f1.notna() & clean_f2.notna() & target_series.notna()

    x1 = clean_f1[valid_mask]
    x2 = clean_f2[valid_mask]
    y = target_series[valid_mask]

    if (
        len(x1) < 2
        or y.nunique(dropna=True) < 2
        or x1.nunique(dropna=True) < 2
        or x2.nunique(dropna=True) < 2
    ):
        return [], []

    bin1 = _compute_quantile_bins(x1, n_bins=n_bins)
    bin2 = _compute_quantile_bins(x2, n_bins=n_bins)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y, "bin1": bin1, "bin2": bin2})
    n_total = len(df)
    unique_classes = sorted(df["y"].unique(), key=lambda v: str(v))

    # Precompute cell stats for all (bin1, bin2)
    cell_stats: dict[tuple[int, int], dict[str, object]] = {}
    for i in range(1, n_bins + 1):
        for j in range(1, n_bins + 1):
            cell_df = df[(df["bin1"] == i) & (df["bin2"] == j)]
            n_cell = len(cell_df)
            if n_cell == 0:
                continue

            x1_min = float(cell_df["x1"].min())
            x1_max = float(cell_df["x1"].max())
            x1_mean = float(cell_df["x1"].mean())
            x2_min = float(cell_df["x2"].min())
            x2_max = float(cell_df["x2"].max())
            x2_mean = float(cell_df["x2"].mean())

            # Cell Shannon entropy: H_cell = - sum p_c * log2(p_c)
            entropy_cell = 0.0
            class_counts: dict[object, int] = {}
            for c in unique_classes:
                cnt = int((cell_df["y"] == c).sum())
                class_counts[c] = cnt
                prob_c = cnt / n_cell
                if prob_c > 0:
                    entropy_cell -= prob_c * np.log2(prob_c)

            cell_stats[(i, j)] = {
                "n_cell": n_cell,
                "x1_min": x1_min,
                "x1_max": x1_max,
                "x1_mean": x1_mean,
                "x2_min": x2_min,
                "x2_max": x2_max,
                "x2_mean": x2_mean,
                "entropy": entropy_cell,
                "class_counts": class_counts,
            }

    pair_score_rows: list[dict[str, object]] = []
    cell_probability_rows: list[dict[str, object]] = []

    # For each class, calculate 2D IV, spread, synergy gain, and extract sweet spot cell
    for c in unique_classes:
        total_events = int((df["y"] == c).sum())
        total_non_events = n_total - total_events
        base_rate = total_events / n_total if n_total > 0 else 0.0

        f1_class_stats = f1_1d_stats.get(c, {"iv": 0.0, "prob_spread": 0.0})
        f2_class_stats = f2_1d_stats.get(c, {"iv": 0.0, "prob_spread": 0.0})

        iv_f1 = f1_class_stats.get("iv", 0.0)
        iv_f2 = f2_class_stats.get("iv", 0.0)
        prob_spread_f1 = f1_class_stats.get("prob_spread", 0.0)
        prob_spread_f2 = f2_class_stats.get("prob_spread", 0.0)
        max_1d_spread = max(prob_spread_f1, prob_spread_f2)

        iv_2d_total = 0.0
        cell_probs: list[float] = []
        cell_candidates: list[dict[str, object]] = []

        for (i, j), stats_ij in cell_stats.items():
            n_cell = stats_ij["n_cell"]
            events_cell = stats_ij["class_counts"][c]
            non_events_cell = n_cell - events_cell
            prob_cell = events_cell / n_cell if n_cell > 0 else 0.0
            lift_cell = prob_cell / base_rate if base_rate > 0 else 1.0

            dist_event = events_cell / total_events if total_events > 0 else 0.0
            dist_non_event = non_events_cell / total_non_events if total_non_events > 0 else 0.0

            p_e = max(dist_event, EPSILON)
            p_ne = max(dist_non_event, EPSILON)
            woe_cell = float(np.log(p_e / p_ne))
            iv_cell = float((dist_event - dist_non_event) * woe_cell)
            iv_2d_total += iv_cell
            cell_probs.append(prob_cell)

            cell_probability_rows.append(
                {
                    "feature_x": f1_name,
                    "feature_y": f2_name,
                    "target": target_name,
                    "target_class": c,
                    "bin_x": int(i),
                    "bin_y": int(j),
                    "x_min": _round(stats_ij["x1_min"]),
                    "x_max": _round(stats_ij["x1_max"]),
                    "x_mean": _round(stats_ij["x1_mean"]),
                    "y_min": _round(stats_ij["x2_min"]),
                    "y_max": _round(stats_ij["x2_max"]),
                    "y_mean": _round(stats_ij["x2_mean"]),
                    "sample_count": int(n_cell),
                    "conditional_prob": _round(prob_cell),
                    "lift": _round(lift_cell),
                    "woe": _round(woe_cell),
                    "iv_contribution": _round(iv_cell),
                    "entropy": _round(stats_ij["entropy"]),
                }
            )

            cell_candidates.append(
                {
                    "bin_x": i,
                    "bin_y": j,
                    "prob": prob_cell,
                    "lift": lift_cell,
                    "sample_count": n_cell,
                    "x_min": stats_ij["x1_min"],
                    "x_max": stats_ij["x1_max"],
                    "y_min": stats_ij["x2_min"],
                    "y_max": stats_ij["x2_max"],
                }
            )

        prob_arr = np.array(cell_probs, dtype=float)
        max_p_2d = float(prob_arr.max()) if len(prob_arr) > 0 else 0.0
        min_p_2d = float(prob_arr.min()) if len(prob_arr) > 0 else 0.0
        prob_spread_2d = max_p_2d - min_p_2d
        synergy_gain = prob_spread_2d - max_1d_spread
        iv_gain = iv_2d_total - max(iv_f1, iv_f2)

        # Sweet spot selection with adaptive minimum support fallback
        max_cell_samples = max((cand["sample_count"] for cand in cell_candidates), default=0)
        effective_support = min_support if max_cell_samples >= min_support else max(1, min(5, max_cell_samples))

        eligible = [cand for cand in cell_candidates if cand["sample_count"] >= effective_support]
        if not eligible:
            eligible = cell_candidates

        # Sort eligible candidates: highest probability first, then highest sample support
        eligible.sort(key=lambda item: (item["prob"], item["sample_count"]), reverse=True)
        best = eligible[0] if eligible else None

        def _format_clause(feat: str, v_min: float, v_max: float) -> str:
            if v_min == v_max:
                return f"{feat} == {v_min:.4g}"
            return f"{v_min:.4g} <= {feat} <= {v_max:.4g}"

        rule_str = (
            f"{_format_clause(f1_name, best['x_min'], best['x_max'])} AND "
            f"{_format_clause(f2_name, best['y_min'], best['y_max'])}"
            if best
            else ""
        )
        x_range_str = f"[{best['x_min']:.4g}, {best['x_max']:.4g}]" if best else ""
        y_range_str = f"[{best['y_min']:.4g}, {best['y_max']:.4g}]" if best else ""

        pair_score_rows.append(
            {
                "feature_x": f1_name,
                "feature_y": f2_name,
                "target": target_name,
                "target_class": c,
                "iv_2d": _round(iv_2d_total),
                "iv_f1": _round(iv_f1),
                "iv_f2": _round(iv_f2),
                "iv_gain": _round(iv_gain),
                "prob_spread_2d": _round(prob_spread_2d),
                "prob_spread_f1": _round(prob_spread_f1),
                "prob_spread_f2": _round(prob_spread_f2),
                "synergy_gain": _round(synergy_gain),
                "max_prob_2d": _round(max_p_2d),
                "min_prob_2d": _round(min_p_2d),
                "base_rate": _round(base_rate),
                "sweet_spot_rule": rule_str,
                "sweet_spot_prob": _round(best["prob"]) if best else 0.0,
                "sweet_spot_lift": _round(best["lift"]) if best else 1.0,
                "sweet_spot_samples": int(best["sample_count"]) if best else 0,
                "sweet_spot_bin_x": int(best["bin_x"]) if best else 0,
                "sweet_spot_bin_y": int(best["bin_y"]) if best else 0,
                "sweet_spot_x_range": x_range_str,
                "sweet_spot_y_range": y_range_str,
                "sample_count": int(n_total),
            }
        )

    return pair_score_rows, cell_probability_rows


def _plot_probability_2d_heatmaps(
    pair_scores_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    output_path: Path,
    n_bins: int = DEFAULT_N_BINS,
    top_n: int = 6,
) -> Path:
    """Plot 2D joint probability heatmaps with sweet spot bounding boxes for top synergistic pairs."""
    if pair_scores_df.empty or cell_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No 2D probability data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Select top pairs sorted by synergy gain descending
    top_pairs_df = (
        pair_scores_df.sort_values(["synergy_gain", "iv_2d"], ascending=[False, False])
        .drop_duplicates(subset=["feature_x", "feature_y", "target"])
        .head(top_n)
    )

    n_plots = len(top_pairs_df)
    if n_plots == 1:
        nrows, ncols = 1, 1
        figsize = (7, 6)
    elif n_plots == 2:
        nrows, ncols = 1, 2
        figsize = (14, 6)
    elif n_plots in (3, 4):
        nrows, ncols = 2, 2
        figsize = (14, 12)
    else:
        nrows, ncols = 2, 3
        figsize = (20, 12)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for idx, (_, pair_row) in enumerate(top_pairs_df.iterrows()):
        ax = flat_axes[idx]
        fx = pair_row["feature_x"]
        fy = pair_row["feature_y"]
        target = pair_row["target"]
        target_class = pair_row["target_class"]

        pair_cells = cell_df[
            (cell_df["feature_x"] == fx)
            & (cell_df["feature_y"] == fy)
            & (cell_df["target"] == target)
            & (cell_df["target_class"] == target_class)
        ]

        grid = np.full((n_bins, n_bins), np.nan)
        for _, cell in pair_cells.iterrows():
            bx = int(cell["bin_x"]) - 1
            by = int(cell["bin_y"]) - 1
            if 0 <= bx < n_bins and 0 <= by < n_bins:
                grid[by, bx] = float(cell["conditional_prob"])

        # Heatmap plot
        im = ax.imshow(grid, origin="lower", cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=1.0)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("P(Y=class | Bin)", fontsize=8)

        # Annotate cells with conditional probability
        for by in range(n_bins):
            for bx in range(n_bins):
                val = grid[by, bx]
                if not np.isnan(val):
                    text_color = "white" if val > 0.6 else "black"
                    ax.text(
                        bx,
                        by,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7,
                        fontweight="semibold",
                    )

        # Highlight sweet spot cell with a red border
        sbx = int(pair_row["sweet_spot_bin_x"]) - 1
        sby = int(pair_row["sweet_spot_bin_y"]) - 1
        if 0 <= sbx < n_bins and 0 <= sby < n_bins:
            rect = patches.Rectangle(
                (sbx - 0.5, sby - 0.5),
                1.0,
                1.0,
                linewidth=2.5,
                edgecolor="#dc2626",
                facecolor="none",
                label="Sweet Spot",
            )
            ax.add_patch(rect)

        synergy = float(pair_row["synergy_gain"])
        iv_2d = float(pair_row["iv_2d"])
        best_prob = float(pair_row["sweet_spot_prob"])
        lift = float(pair_row["sweet_spot_lift"])

        ax.set_title(
            f"{fx} × {fy}\nTarget: {target} ({target_class}) | Synergy: {synergy:+.3f} | IV: {iv_2d:.3f}\n"
            f"Sweet Spot P={best_prob:.1%} (Lift {lift:.2f}x)",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xlabel(f"{fx} (Quantile 1-{n_bins})", fontsize=8)
        ax.set_ylabel(f"{fy} (Quantile 1-{n_bins})", fontsize=8)
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels(range(1, n_bins + 1), fontsize=8)
        ax.set_yticks(range(n_bins))
        ax.set_yticklabels(range(1, n_bins + 1), fontsize=8)

    # Hide unused subplots
    for j in range(n_plots, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_markdown(
    metadata: Probability2DRunMetadata,
    pair_scores_df: pd.DataFrame,
    cell_df: pd.DataFrame,
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
                {"Metric": "Quantile Bins per Axis", "Value": f"{metadata.n_bins}x{metadata.n_bins} (100 cells)"},
                {"Metric": "Candidate Features Screened", "Value": len(metadata.candidate_features)},
                {"Metric": "Minimum Cell Support", "Value": metadata.min_support},
                {"Metric": "Feature Pairs Evaluated", "Value": metadata.pairs_evaluated},
                {"Metric": "Targets Analyzed", "Value": ", ".join(metadata.targets)},
                {"Metric": "Rows Evaluated", "Value": metadata.model_rows},
            ]
        )
    )

    insights: list[str] = []
    if not pair_scores_df.empty:
        top_synergy = pair_scores_df.sort_values("synergy_gain", ascending=False).iloc[0]
        insights.append(
            f"- **Highest Synergy Interaction**: Feature pair `{top_synergy['feature_x']}` × `{top_synergy['feature_y']}` for target `{top_synergy['target']}` (Class `{top_synergy['target_class']}`) achieved a Synergy Gain of `+{top_synergy['synergy_gain']}` (2D Spread `{top_synergy['prob_spread_2d']}` vs Best 1D Spread `{max(top_synergy['prob_spread_f1'], top_synergy['prob_spread_f2'])}`)."
        )

        top_iv = pair_scores_df.sort_values("iv_2d", ascending=False).iloc[0]
        insights.append(
            f"- **Top Joint Predictive Power (2D IV)**: `{top_iv['feature_x']}` × `{top_iv['feature_y']}` achieved 2D Information Value of `{top_iv['iv_2d']}` (Gain of `+{top_iv['iv_gain']}` over single features)."
        )

        top_sweet = pair_scores_df.sort_values("sweet_spot_prob", ascending=False).iloc[0]
        insights.append(
            f"- **Optimal Sweet Spot Decision Rule**: `{top_sweet['sweet_spot_rule']}` delivers win rate / conditional probability of `{top_sweet['sweet_spot_prob']:.1%}` ({top_sweet['sweet_spot_lift']:.2f}x lift over base rate `{top_sweet['base_rate']:.1%}`) with `{top_sweet['sweet_spot_samples']}` samples support."
        )
    else:
        insights.append("- No valid feature pairs or categorical targets available for 2D probability profiling.")

    insights_text = "\n".join(insights)

    # Top 20 pairs table
    display_pairs_cols = [
        "feature_x",
        "feature_y",
        "target",
        "target_class",
        "synergy_gain",
        "iv_2d",
        "prob_spread_2d",
        "prob_spread_f1",
        "prob_spread_f2",
        "sweet_spot_prob",
        "sweet_spot_lift",
    ]
    top_pairs_table = (
        _markdown_table(pair_scores_df[display_pairs_cols].head(20))
        if not pair_scores_df.empty
        else "No pairs evaluated."
    )

    # Sweet spots table
    sweet_spots_cols = [
        "feature_x",
        "feature_y",
        "target",
        "target_class",
        "sweet_spot_rule",
        "sweet_spot_prob",
        "sweet_spot_lift",
        "sweet_spot_samples",
    ]
    sweet_spots_table = (
        _markdown_table(
            pair_scores_df.sort_values("sweet_spot_prob", ascending=False)[sweet_spots_cols].head(15)
        )
        if not pair_scores_df.empty
        else "No sweet spots extracted."
    )

    return f"""# 2D Joint Probability & Synergy Profiling Report

## Executive Summary & Key Insights

{insights_text}

## Run Metadata

{metadata_table}

## Top 20 Synergistic Feature Pairs

{top_pairs_table}

## Optimal Sweet Spot Decision Rules

{sweet_spots_table}

## 2D Probability Heatmaps

![2D Probability Heatmaps](probability_2d_heatmaps.png)

## Artifacts

- `summary.json`
- `pair_probability_scores.csv`
- `cell_conditional_probabilities.csv`
- `probability_2d_heatmaps.png`
- `report.html`
"""


def _render_html(
    metadata: Probability2DRunMetadata,
    markdown: str,
    pair_scores_df: pd.DataFrame,
    cell_df: pd.DataFrame,
) -> str:
    display_pairs_cols = [
        "feature_x",
        "feature_y",
        "target",
        "target_class",
        "synergy_gain",
        "iv_2d",
        "prob_spread_2d",
        "prob_spread_f1",
        "prob_spread_f2",
        "sweet_spot_prob",
        "sweet_spot_lift",
    ]
    pairs_html = (
        pair_scores_df[display_pairs_cols].head(25).to_html(index=False, classes="data-table")
        if not pair_scores_df.empty
        else "<p>No feature pairs evaluated.</p>"
    )

    sweet_spots_cols = [
        "feature_x",
        "feature_y",
        "target",
        "target_class",
        "sweet_spot_rule",
        "sweet_spot_prob",
        "sweet_spot_lift",
        "sweet_spot_samples",
    ]
    sweet_spots_html = (
        pair_scores_df.sort_values("sweet_spot_prob", ascending=False)[sweet_spots_cols]
        .head(15)
        .to_html(index=False, classes="data-table")
        if not pair_scores_df.empty
        else "<p>No sweet spots extracted.</p>"
    )

    max_synergy = (
        f"+{pair_scores_df['synergy_gain'].max():.3f}"
        if not pair_scores_df.empty and pair_scores_df["synergy_gain"].max() is not None
        else "N/A"
    )
    max_iv = (
        f"{pair_scores_df['iv_2d'].max():.3f}"
        if not pair_scores_df.empty and pair_scores_df["iv_2d"].max() is not None
        else "N/A"
    )
    max_prob = (
        f"{pair_scores_df['sweet_spot_prob'].max():.1%}"
        if not pair_scores_df.empty and pair_scores_df["sweet_spot_prob"].max() is not None
        else "N/A"
    )
    max_lift = (
        f"{pair_scores_df['sweet_spot_lift'].max():.2f}x"
        if not pair_scores_df.empty and pair_scores_df["sweet_spot_lift"].max() is not None
        else "N/A"
    )

    details = _html_markdown_details(markdown)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>2D Joint Probability & Synergy Profiling Report</title>
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
    <h1>2D Joint Probability & Synergy Profiling Report</h1>

    <div class="metrics-banner">
        <div class="metric-card">
            <div class="metric-title">Max Synergy Gain</div>
            <div class="metric-value">{max_synergy}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Max 2D IV</div>
            <div class="metric-value">{max_iv}</div>
        </div>
        <div class="metric-card accent">
            <div class="metric-title">Peak Sweet Spot Win Rate</div>
            <div class="metric-value">{max_prob}</div>
        </div>
        <div class="metric-card warning">
            <div class="metric-title">Max Base Rate Lift</div>
            <div class="metric-value">{max_lift}</div>
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
                <div class="meta-label">Grid Resolution</div>
                <div class="meta-val">{metadata.n_bins} × {metadata.n_bins} Quantile Grid</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Pairs Evaluated</div>
                <div class="meta-val">{metadata.pairs_evaluated} pairs</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Candidate Features</div>
                <div class="meta-val">{len(metadata.candidate_features)} screened</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Evaluated Rows</div>
                <div class="meta-val">{metadata.model_rows}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Top Synergistic Feature Pairs</h2>
        <div class="table-container">
            {pairs_html}
        </div>
    </div>

    <div class="card">
        <h2>Optimal Sweet Spot Decision Rules</h2>
        <div class="table-container">
            {sweet_spots_html}
        </div>
    </div>

    <div class="card">
        <h2>2D Joint Probability Heatmaps</h2>
        <img class="chart-img" src="probability_2d_heatmaps.png" alt="2D Probability Heatmaps"/>
    </div>

    {details}
</body>
</html>
"""


class Probability2DModule:
    name = "probability_2d"

    def __init__(
        self,
        config: Probability2DConfig | None = None,
        progress: bool | None = None,
        n_bins: int | None = None,
        max_candidates: int | None = None,
        min_support: int | None = None,
    ) -> None:
        self.progress = progress
        if config is not None:
            base_cfg = config
        else:
            mod_cfg = get_module_config("probability_2d")
            base_cfg = Probability2DConfig(
                n_bins=int(mod_cfg.get("n_bins", DEFAULT_N_BINS)),
                max_candidates=int(mod_cfg.get("max_candidates", DEFAULT_MAX_CANDIDATES)),
                min_support=int(
                    mod_cfg.get("min_support", mod_cfg.get("min_samples", DEFAULT_MIN_SUPPORT))
                ),
            )
        self.config = Probability2DConfig(
            n_bins=n_bins if n_bins is not None else base_cfg.n_bins,
            max_candidates=max_candidates if max_candidates is not None else base_cfg.max_candidates,
            min_support=min_support if min_support is not None else base_cfg.min_support,
        )
        self.n_bins = self.config.n_bins
        self.max_candidates = self.config.max_candidates
        self.min_support = self.config.min_support

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

            # Filter for discrete / categorical targets with 2..MAX_LABEL_CLASSES unique values
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

            # 1D Pre-screening to select top max_candidates features by 1D IV
            candidate_features, feature_1d_stats = _prescreen_candidate_features(
                model_frame,
                numeric_features,
                valid_targets,
                max_candidates=self.max_candidates,
                n_bins=self.n_bins,
            )

            # Evaluate all pairs of candidate features
            feature_pairs = list(itertools.combinations(candidate_features, 2))
            all_pair_scores: list[dict[str, object]] = []
            all_cell_rows: list[dict[str, object]] = []

            for f1, f2 in feature_pairs:
                for target_col in valid_targets:
                    f1_stats = feature_1d_stats.get(f1, {}).get(target_col, {})
                    f2_stats = feature_1d_stats.get(f2, {}).get(target_col, {})

                    pair_scores, cell_rows = _compute_pair_target_probabilities(
                        model_frame[f1],
                        model_frame[f2],
                        model_frame[target_col],
                        f1_name=f1,
                        f2_name=f2,
                        target_name=target_col,
                        n_bins=self.n_bins,
                        f1_1d_stats=f1_stats,
                        f2_1d_stats=f2_stats,
                        min_support=self.min_support,
                    )
                    all_pair_scores.extend(pair_scores)
                    all_cell_rows.extend(cell_rows)

            pair_scores_df = pd.DataFrame(all_pair_scores)
            cell_df = pd.DataFrame(all_cell_rows)

            if not pair_scores_df.empty:
                pair_scores_df = pair_scores_df.sort_values(
                    ["synergy_gain", "iv_2d"],
                    ascending=[False, False],
                ).reset_index(drop=True)

            progress_bar.step("probabilities_2d")

            plot_path = run_dir / "probability_2d_heatmaps.png"
            _plot_probability_2d_heatmaps(
                pair_scores_df,
                cell_df,
                plot_path,
                n_bins=self.n_bins,
                top_n=6,
            )
            progress_bar.step("heatmaps")

            metadata = Probability2DRunMetadata(
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
                max_candidates=self.max_candidates,
                min_support=self.min_support,
                features_count=len(numeric_features),
                candidate_features=candidate_features,
                pairs_evaluated=len(feature_pairs),
                targets=valid_targets,
                model_rows=len(model_frame),
            )

            pair_scores_csv_path = _write_csv(run_dir / "pair_probability_scores.csv", pair_scores_df)
            cell_csv_path = _write_csv(run_dir / "cell_conditional_probabilities.csv", cell_df)

            summary_payload: dict[str, object] = {
                **asdict(metadata),
                "top_pairs": pair_scores_df.head(10).to_dict(orient="records")
                if not pair_scores_df.empty
                else [],
                "top_sweet_spots": pair_scores_df.sort_values("sweet_spot_prob", ascending=False)
                .head(10)
                .to_dict(orient="records")
                if not pair_scores_df.empty
                else [],
                "summary_metrics": {
                    "features_evaluated": len(numeric_features),
                    "candidate_features": len(candidate_features),
                    "pairs_evaluated": len(feature_pairs),
                    "targets_evaluated": len(valid_targets),
                    "max_synergy_gain": _round(float(pair_scores_df["synergy_gain"].max()))
                    if not pair_scores_df.empty
                    else 0.0,
                    "max_iv_2d": _round(float(pair_scores_df["iv_2d"].max()))
                    if not pair_scores_df.empty
                    else 0.0,
                    "max_sweet_spot_prob": _round(float(pair_scores_df["sweet_spot_prob"].max()))
                    if not pair_scores_df.empty
                    else 0.0,
                    "max_lift": _round(float(pair_scores_df["sweet_spot_lift"].max()))
                    if not pair_scores_df.empty
                    else 0.0,
                },
            }
            summary_json_path = _write_json(run_dir / "summary.json", summary_payload)

            markdown_report = _render_markdown(metadata, pair_scores_df, cell_df)
            report_md_path = run_dir / "report.md"
            report_md_path.write_text(markdown_report, encoding="utf-8")

            html_report = _render_html(metadata, markdown_report, pair_scores_df, cell_df)
            report_html_path = run_dir / "report.html"
            report_html_path.write_text(html_report, encoding="utf-8")

            artifacts = [
                summary_json_path,
                pair_scores_csv_path,
                cell_csv_path,
                plot_path,
                report_md_path,
                report_html_path,
            ]
            progress_bar.step("reports")

            return ModuleResult(report_dir=run_dir, artifacts=artifacts)
