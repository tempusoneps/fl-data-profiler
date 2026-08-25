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
DEFAULT_N_BINS = 5
DEFAULT_MAX_CANDIDATES = 12
DEFAULT_MIN_SUPPORT = 20
EPSILON = 1e-7


@dataclass
class Probability3DConfig:
    n_bins: int = DEFAULT_N_BINS
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    min_support: int = DEFAULT_MIN_SUPPORT


@dataclass(frozen=True)
class Probability3DRunMetadata:
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
    triplets_evaluated: int
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

    sorted_features = sorted(numeric_features, key=lambda f: feature_max_iv.get(f, 0.0), reverse=True)
    candidate_features = sorted_features[:max_candidates]
    return candidate_features, feature_1d_stats


def _compute_triplet_target_probabilities(
    f1_series: pd.Series,
    f2_series: pd.Series,
    f3_series: pd.Series,
    target_series: pd.Series,
    f1_name: str,
    f2_name: str,
    f3_name: str,
    target_name: str,
    n_bins: int,
    f1_1d_stats: dict[object, dict[str, float]],
    f2_1d_stats: dict[object, dict[str, float]],
    f3_1d_stats: dict[object, dict[str, float]],
    min_support: int = DEFAULT_MIN_SUPPORT,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Compute 3D joint quantile probabilities, 3D IV, synergy gain, and sweet spots for a feature triplet."""
    clean_f1 = _numeric_series(f1_series)
    clean_f2 = _numeric_series(f2_series)
    clean_f3 = _numeric_series(f3_series)
    valid_mask = clean_f1.notna() & clean_f2.notna() & clean_f3.notna() & target_series.notna()

    x1 = clean_f1[valid_mask]
    x2 = clean_f2[valid_mask]
    x3 = clean_f3[valid_mask]
    y = target_series[valid_mask]

    if (
        len(x1) < 3
        or y.nunique(dropna=True) < 2
        or x1.nunique(dropna=True) < 2
        or x2.nunique(dropna=True) < 2
        or x3.nunique(dropna=True) < 2
    ):
        return [], []

    bin1 = _compute_quantile_bins(x1, n_bins=n_bins)
    bin2 = _compute_quantile_bins(x2, n_bins=n_bins)
    bin3 = _compute_quantile_bins(x3, n_bins=n_bins)
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y, "bin1": bin1, "bin2": bin2, "bin3": bin3})
    n_total = len(df)
    unique_classes = sorted(df["y"].unique(), key=lambda v: str(v))

    voxel_stats: dict[tuple[int, int, int], dict[str, object]] = {}
    for i in range(1, n_bins + 1):
        for j in range(1, n_bins + 1):
            for k in range(1, n_bins + 1):
                cell_df = df[(df["bin1"] == i) & (df["bin2"] == j) & (df["bin3"] == k)]
                n_cell = len(cell_df)
                if n_cell == 0:
                    continue

                x1_min = float(cell_df["x1"].min())
                x1_max = float(cell_df["x1"].max())
                x1_mean = float(cell_df["x1"].mean())
                x2_min = float(cell_df["x2"].min())
                x2_max = float(cell_df["x2"].max())
                x2_mean = float(cell_df["x2"].mean())
                x3_min = float(cell_df["x3"].min())
                x3_max = float(cell_df["x3"].max())
                x3_mean = float(cell_df["x3"].mean())

                entropy_cell = 0.0
                class_counts: dict[object, int] = {}
                for c in unique_classes:
                    cnt = int((cell_df["y"] == c).sum())
                    class_counts[c] = cnt
                    prob_c = cnt / n_cell
                    if prob_c > 0:
                        entropy_cell -= prob_c * np.log2(prob_c)

                voxel_stats[(i, j, k)] = {
                    "n_cell": n_cell,
                    "x1_min": x1_min,
                    "x1_max": x1_max,
                    "x1_mean": x1_mean,
                    "x2_min": x2_min,
                    "x2_max": x2_max,
                    "x2_mean": x2_mean,
                    "x3_min": x3_min,
                    "x3_max": x3_max,
                    "x3_mean": x3_mean,
                    "entropy": entropy_cell,
                    "class_counts": class_counts,
                }

    triplet_score_rows: list[dict[str, object]] = []
    voxel_probability_rows: list[dict[str, object]] = []

    for c in unique_classes:
        total_events = int((df["y"] == c).sum())
        total_non_events = n_total - total_events
        base_rate = total_events / n_total if n_total > 0 else 0.0

        f1_class_stats = f1_1d_stats.get(c, {"iv": 0.0, "prob_spread": 0.0})
        f2_class_stats = f2_1d_stats.get(c, {"iv": 0.0, "prob_spread": 0.0})
        f3_class_stats = f3_1d_stats.get(c, {"iv": 0.0, "prob_spread": 0.0})

        iv_f1 = f1_class_stats.get("iv", 0.0)
        iv_f2 = f2_class_stats.get("iv", 0.0)
        iv_f3 = f3_class_stats.get("iv", 0.0)
        prob_spread_f1 = f1_class_stats.get("prob_spread", 0.0)
        prob_spread_f2 = f2_class_stats.get("prob_spread", 0.0)
        prob_spread_f3 = f3_class_stats.get("prob_spread", 0.0)
        max_1d_spread = max(prob_spread_f1, prob_spread_f2, prob_spread_f3)
        max_1d_iv = max(iv_f1, iv_f2, iv_f3)

        iv_3d_total = 0.0
        voxel_probs: list[float] = []
        voxel_candidates: list[dict[str, object]] = []

        for (i, j, k), stats_ijk in voxel_stats.items():
            n_cell = stats_ijk["n_cell"]
            events_cell = stats_ijk["class_counts"][c]
            non_events_cell = n_cell - events_cell
            prob_cell = events_cell / n_cell if n_cell > 0 else 0.0
            lift_cell = prob_cell / base_rate if base_rate > 0 else 1.0

            dist_event = events_cell / total_events if total_events > 0 else 0.0
            dist_non_event = non_events_cell / total_non_events if total_non_events > 0 else 0.0

            p_e = max(dist_event, EPSILON)
            p_ne = max(dist_non_event, EPSILON)
            woe_cell = float(np.log(p_e / p_ne))
            iv_cell = float((dist_event - dist_non_event) * woe_cell)
            iv_3d_total += iv_cell
            voxel_probs.append(prob_cell)

            voxel_probability_rows.append(
                {
                    "feature_x": f1_name,
                    "feature_y": f2_name,
                    "feature_z": f3_name,
                    "target": target_name,
                    "target_class": c,
                    "bin_x": int(i),
                    "bin_y": int(j),
                    "bin_z": int(k),
                    "x_min": _round(stats_ijk["x1_min"]),
                    "x_max": _round(stats_ijk["x1_max"]),
                    "x_mean": _round(stats_ijk["x1_mean"]),
                    "y_min": _round(stats_ijk["x2_min"]),
                    "y_max": _round(stats_ijk["x2_max"]),
                    "y_mean": _round(stats_ijk["x2_mean"]),
                    "z_min": _round(stats_ijk["x3_min"]),
                    "z_max": _round(stats_ijk["x3_max"]),
                    "z_mean": _round(stats_ijk["x3_mean"]),
                    "sample_count": int(n_cell),
                    "conditional_prob": _round(prob_cell),
                    "lift": _round(lift_cell),
                    "shannon_entropy": _round(stats_ijk["entropy"]),
                }
            )

            voxel_candidates.append(
                {
                    "bin_x": i,
                    "bin_y": j,
                    "bin_z": k,
                    "x_min": stats_ijk["x1_min"],
                    "x_max": stats_ijk["x1_max"],
                    "y_min": stats_ijk["x2_min"],
                    "y_max": stats_ijk["x2_max"],
                    "z_min": stats_ijk["x3_min"],
                    "z_max": stats_ijk["x3_max"],
                    "prob": prob_cell,
                    "lift": lift_cell,
                    "sample_count": n_cell,
                }
            )

        prob_arr = np.array(voxel_probs, dtype=float)
        max_prob_3d = float(prob_arr.max()) if len(prob_arr) > 0 else 0.0
        min_prob_3d = float(prob_arr.min()) if len(prob_arr) > 0 else 0.0
        prob_spread_3d = float(max_prob_3d - min_prob_3d)
        synergy_gain = float(prob_spread_3d - max_1d_spread)
        iv_gain = float(iv_3d_total - max_1d_iv)

        filtered_candidates = [cand for cand in voxel_candidates if cand["sample_count"] >= min_support]
        if not filtered_candidates:
            filtered_candidates = voxel_candidates

        if filtered_candidates:
            best_voxel = max(filtered_candidates, key=lambda d: (d["prob"], d["lift"], d["sample_count"]))
            def _format_clause(feat: str, v_min: float, v_max: float) -> str:
                if v_min == v_max:
                    return f"{feat} == {_round(v_min)}"
                return f"{_round(v_min)} <= {feat} <= {_round(v_max)}"

            sweet_spot_rule = (
                f"{_format_clause(f1_name, best_voxel['x_min'], best_voxel['x_max'])} AND "
                f"{_format_clause(f2_name, best_voxel['y_min'], best_voxel['y_max'])} AND "
                f"{_format_clause(f3_name, best_voxel['z_min'], best_voxel['z_max'])}"
            )
            sweet_spot_prob = best_voxel["prob"]
            sweet_spot_lift = best_voxel["lift"]
            sweet_spot_samples = best_voxel["sample_count"]
            sweet_spot_bin_x = best_voxel["bin_x"]
            sweet_spot_bin_y = best_voxel["bin_y"]
            sweet_spot_bin_z = best_voxel["bin_z"]
            sweet_spot_x_range = f"[{_round(best_voxel['x_min'])}, {_round(best_voxel['x_max'])}]"
            sweet_spot_y_range = f"[{_round(best_voxel['y_min'])}, {_round(best_voxel['y_max'])}]"
            sweet_spot_z_range = f"[{_round(best_voxel['z_min'])}, {_round(best_voxel['z_max'])}]"
        else:
            sweet_spot_rule = "N/A"
            sweet_spot_prob = 0.0
            sweet_spot_lift = 1.0
            sweet_spot_samples = 0
            sweet_spot_bin_x = 0
            sweet_spot_bin_y = 0
            sweet_spot_bin_z = 0
            sweet_spot_x_range = "N/A"
            sweet_spot_y_range = "N/A"
            sweet_spot_z_range = "N/A"

        triplet_score_rows.append(
            {
                "feature_x": f1_name,
                "feature_y": f2_name,
                "feature_z": f3_name,
                "target": target_name,
                "target_class": c,
                "iv_3d": _round(iv_3d_total),
                "iv_f1": _round(iv_f1),
                "iv_f2": _round(iv_f2),
                "iv_f3": _round(iv_f3),
                "iv_gain": _round(iv_gain),
                "prob_spread_3d": _round(prob_spread_3d),
                "prob_spread_f1": _round(prob_spread_f1),
                "prob_spread_f2": _round(prob_spread_f2),
                "prob_spread_f3": _round(prob_spread_f3),
                "synergy_gain": _round(synergy_gain),
                "max_prob_3d": _round(max_prob_3d),
                "min_prob_3d": _round(min_prob_3d),
                "base_rate": _round(base_rate),
                "sweet_spot_rule": sweet_spot_rule,
                "sweet_spot_prob": _round(sweet_spot_prob),
                "sweet_spot_lift": _round(sweet_spot_lift),
                "sweet_spot_samples": int(sweet_spot_samples),
                "sweet_spot_bin_x": int(sweet_spot_bin_x),
                "sweet_spot_bin_y": int(sweet_spot_bin_y),
                "sweet_spot_bin_z": int(sweet_spot_bin_z),
                "sweet_spot_x_range": sweet_spot_x_range,
                "sweet_spot_y_range": sweet_spot_y_range,
                "sweet_spot_z_range": sweet_spot_z_range,
                "sample_count": int(n_total),
            }
        )

    return triplet_score_rows, voxel_probability_rows


def _generate_probability_3d_heatmaps(
    triplet_scores: list[dict[str, object]],
    voxel_probabilities: list[dict[str, object]],
    output_path: Path,
    n_bins: int,
) -> None:
    """Generate sliced 2D heatmaps across Z-bins for top 3D synergistic triplets."""
    if not triplet_scores:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No 3D Triplet Probabilities Available", ha="center", va="center", fontsize=12)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return

    top_triplets_df = pd.DataFrame(triplet_scores).sort_values("synergy_gain", ascending=False)
    unique_triplets = top_triplets_df.drop_duplicates(subset=["feature_x", "feature_y", "feature_z", "target", "target_class"]).head(3)
    n_triplets = len(unique_triplets)

    if n_triplets == 0:
        return

    voxel_df = pd.DataFrame(voxel_probabilities)
    fig, axes = plt.subplots(n_triplets, n_bins, figsize=(n_bins * 3.2, n_triplets * 3.2), squeeze=False)

    for row_idx, (_, r) in enumerate(unique_triplets.iterrows()):
        fx = r["feature_x"]
        fy = r["feature_y"]
        fz = r["feature_z"]
        t = r["target"]
        tc = r["target_class"]

        sub_df = voxel_df[
            (voxel_df["feature_x"] == fx)
            & (voxel_df["feature_y"] == fy)
            & (voxel_df["feature_z"] == fz)
            & (voxel_df["target"] == t)
            & (voxel_df["target_class"] == tc)
        ]

        for z_bin in range(1, n_bins + 1):
            ax = axes[row_idx, z_bin - 1]
            grid = np.full((n_bins, n_bins), np.nan)
            z_df = sub_df[sub_df["bin_z"] == z_bin]

            for _, cell in z_df.iterrows():
                bx = int(cell["bin_x"]) - 1
                by = int(cell["bin_y"]) - 1
                if 0 <= bx < n_bins and 0 <= by < n_bins:
                    grid[by, bx] = float(cell["conditional_prob"])

            ax.imshow(grid, origin="lower", cmap="viridis", vmin=0.0, vmax=max(1.0, float(r["max_prob_3d"])))
            ax.set_title(f"Slice {fz} [Bin {z_bin}]", fontsize=9, fontweight="bold")
            ax.set_xlabel(f"{fx} (bin)", fontsize=8)
            ax.set_ylabel(f"{fy} (bin)", fontsize=8)
            ax.set_xticks(range(n_bins))
            ax.set_xticklabels(range(1, n_bins + 1), fontsize=7)
            ax.set_yticks(range(n_bins))
            ax.set_yticklabels(range(1, n_bins + 1), fontsize=7)

            if r["sweet_spot_bin_z"] == z_bin:
                ss_x = r["sweet_spot_bin_x"] - 1
                ss_y = r["sweet_spot_bin_y"] - 1
                rect = patches.Rectangle(
                    (ss_x - 0.45, ss_y - 0.45),
                    0.9,
                    0.9,
                    linewidth=2.5,
                    edgecolor="red",
                    facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    ss_x,
                    ss_y,
                    f"{r['sweet_spot_prob']:.2f}",
                    color="white",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    fontsize=8,
                )

    fig.suptitle("3D Quantile Joint Probability Sliced Heatmaps (Red Box = Sweet Spot)", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _render_report_html(
    metadata: Probability3DRunMetadata,
    triplet_scores: list[dict[str, object]],
    voxel_probabilities: list[dict[str, object]],
    summary_metrics: dict[str, object],
    output_dir: Path,
) -> None:
    """Render interactive HTML report for 3D Probability Profiling."""
    top_triplets_html = ""
    for r in triplet_scores[:20]:
        synergy_badge = (
            '<span class="badge bg-success">Strong Synergy</span>'
            if r["synergy_gain"] >= 0.3
            else '<span class="badge bg-primary">Moderate</span>'
            if r["synergy_gain"] >= 0.1
            else '<span class="badge bg-secondary">Low</span>'
        )
        top_triplets_html += f"""
        <tr>
            <td><code>{r['feature_x']}</code> &times; <code>{r['feature_y']}</code> &times; <code>{r['feature_z']}</code></td>
            <td><span class="badge bg-dark">{r['target']}</span></td>
            <td><strong>{r['target_class']}</strong></td>
            <td>{r['iv_3d']:.4f}</td>
            <td>+{r['iv_gain']:.4f}</td>
            <td><strong>{r['prob_spread_3d']:.4f}</strong></td>
            <td>+{r['synergy_gain']:.4f} {synergy_badge}</td>
            <td>{r['base_rate']:.4f}</td>
            <td><span class="badge bg-warning text-dark">{r['sweet_spot_prob']:.2%} (Lift: {r['sweet_spot_lift']:.2f}x)</span></td>
            <td><code>{r['sweet_spot_rule']}</code></td>
            <td>{r['sweet_spot_samples']}</td>
        </tr>
        """

    sweet_spots_html = ""
    sorted_sweet_spots = sorted(triplet_scores, key=lambda x: (x["sweet_spot_lift"], x["sweet_spot_prob"]), reverse=True)[:15]
    for r in sorted_sweet_spots:
        sweet_spots_html += f"""
        <tr>
            <td><code>{r['feature_x']}</code> &times; <code>{r['feature_y']}</code> &times; <code>{r['feature_z']}</code></td>
            <td>{r['target']} (<em>{r['target_class']}</em>)</td>
            <td><span class="badge bg-success fs-6">{r['sweet_spot_prob']:.1%}</span></td>
            <td><strong class="text-danger">{r['sweet_spot_lift']:.2f}x</strong></td>
            <td>{r['base_rate']:.1%}</td>
            <td><code>{r['sweet_spot_rule']}</code></td>
            <td>{r['sweet_spot_samples']}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Joint Probability & Sweet Spots Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        .card {{ margin-bottom: 20px; border-radius: 8px; }}
        .badge {{ font-size: 0.85rem; }}
        table {{ font-size: 0.9rem; }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1 class="mb-3">📊 3D Joint Probability & Sweet Spots Report</h1>
        <p class="text-muted">High-confluence 3-way interactions, Synergy Gains, and Quantile Hyper-Voxel Sweet Spot Extraction.</p>

        <!-- KPI Cards -->
        <div class="row">
            <div class="col-md-3">
                <div class="card text-center bg-primary text-white">
                    <div class="card-body">
                        <h5>Triplets Evaluated</h5>
                        <h2>{metadata.triplets_evaluated}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center bg-success text-white">
                    <div class="card-body">
                        <h5>Max Synergy Gain</h5>
                        <h2>+{summary_metrics.get('max_synergy_gain', 0.0):.4f}</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center bg-warning text-dark">
                    <div class="card-body">
                        <h5>Max Sweet Spot Lift</h5>
                        <h2>{summary_metrics.get('max_lift', 1.0):.2f}x</h2>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card text-center bg-info text-dark">
                    <div class="card-body">
                        <h5>Peak 3D Probability</h5>
                        <h2>{summary_metrics.get('max_sweet_spot_prob', 0.0):.1%}</h2>
                    </div>
                </div>
            </div>
        </div>

        <!-- Heatmap Visualization -->
        <div class="card">
            <div class="card-header">
                <h5>🔥 3D Sliced Quantile Heatmaps (Top Triplets)</h5>
            </div>
            <div class="card-body text-center">
                <img src="probability_3d_heatmaps.png" class="img-fluid rounded border" alt="3D Probability Heatmaps">
            </div>
        </div>

        <!-- Top Sweet Spot Rules Table -->
        <div class="card">
            <div class="card-header bg-dark text-white">
                <h5>🎯 Top 3D Sweet Spot Trading Rules (Actionable Confluence)</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover table-striped align-middle">
                        <thead>
                            <tr>
                                <th>Triplet (F1 &times; F2 &times; F3)</th>
                                <th>Target & Class</th>
                                <th>Win-Rate</th>
                                <th>Lift</th>
                                <th>Base Rate</th>
                                <th>Executable 3D Rule</th>
                                <th>Support (Bars)</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sweet_spots_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Top 3D Triplets Table -->
        <div class="card">
            <div class="card-header">
                <h5>⭐ Top 3D Interacting Feature Triplets by Synergy Gain</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover table-sm align-middle">
                        <thead>
                            <tr>
                                <th>Triplet</th>
                                <th>Target</th>
                                <th>Class</th>
                                <th>IV 3D</th>
                                <th>IV Gain</th>
                                <th>Spread 3D</th>
                                <th>Synergy Gain</th>
                                <th>Base Rate</th>
                                <th>Sweet Spot</th>
                                <th>Rule</th>
                                <th>Support</th>
                            </tr>
                        </thead>
                        <tbody>
                            {top_triplets_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""
    (output_dir / "report.html").write_text(html_content, encoding="utf-8")


class Probability3DModule:
    """3D Joint Quantile Conditional Probability, Synergy Gain & Sweet Spot Rule Profiling."""

    name = "probability_3d"

    def __init__(
        self,
        config: Probability3DConfig | None = None,
        progress: bool | None = None,
        n_bins: int | None = None,
        max_candidates: int | None = None,
        min_support: int | None = None,
    ) -> None:
        self.progress = progress
        if config is not None:
            base_cfg = config
        else:
            mod_cfg = get_module_config("probability_3d")
            base_cfg = Probability3DConfig(
                n_bins=int(mod_cfg.get("n_bins", DEFAULT_N_BINS)),
                max_candidates=int(mod_cfg.get("max_candidates", DEFAULT_MAX_CANDIDATES)),
                min_support=int(
                    mod_cfg.get("min_support", mod_cfg.get("min_samples", DEFAULT_MIN_SUPPORT))
                ),
            )
        self.config = Probability3DConfig(
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

            candidate_features, feature_1d_stats = _prescreen_candidate_features(
                model_frame,
                numeric_features,
                valid_targets,
                max_candidates=self.max_candidates,
                n_bins=self.n_bins,
            )

            feature_triplets = list(itertools.combinations(candidate_features, 3))
            all_triplet_scores: list[dict[str, object]] = []
            all_voxel_rows: list[dict[str, object]] = []

            for f1, f2, f3 in feature_triplets:
                for target_col in valid_targets:
                    f1_stats = feature_1d_stats.get(f1, {}).get(target_col, {})
                    f2_stats = feature_1d_stats.get(f2, {}).get(target_col, {})
                    f3_stats = feature_1d_stats.get(f3, {}).get(target_col, {})

                    triplet_scores, voxel_rows = _compute_triplet_target_probabilities(
                        model_frame[f1],
                        model_frame[f2],
                        model_frame[f3],
                        model_frame[target_col],
                        f1_name=f1,
                        f2_name=f2,
                        f3_name=f3,
                        target_name=target_col,
                        n_bins=self.n_bins,
                        f1_1d_stats=f1_stats,
                        f2_1d_stats=f2_stats,
                        f3_1d_stats=f3_stats,
                        min_support=self.min_support,
                    )
                    all_triplet_scores.extend(triplet_scores)
                    all_voxel_rows.extend(voxel_rows)

            triplet_scores_df = pd.DataFrame(all_triplet_scores)
            voxel_df = pd.DataFrame(all_voxel_rows)

            if not triplet_scores_df.empty:
                triplet_scores_df = triplet_scores_df.sort_values(
                    ["synergy_gain", "iv_3d"],
                    ascending=[False, False],
                ).reset_index(drop=True)

            progress_bar.step("probabilities_3d")

            plot_path = run_dir / "probability_3d_heatmaps.png"
            _generate_probability_3d_heatmaps(
                all_triplet_scores,
                all_voxel_rows,
                plot_path,
                n_bins=self.n_bins,
            )
            progress_bar.step("heatmaps")

            metadata = Probability3DRunMetadata(
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
                triplets_evaluated=len(feature_triplets),
                targets=valid_targets,
                model_rows=len(model_frame),
            )

            _write_csv(run_dir / "triplet_probability_scores.csv", triplet_scores_df)
            _write_csv(run_dir / "voxel_conditional_probabilities.csv", voxel_df)

            max_synergy = float(triplet_scores_df["synergy_gain"].max()) if not triplet_scores_df.empty else 0.0
            max_iv_3d = float(triplet_scores_df["iv_3d"].max()) if not triplet_scores_df.empty else 0.0
            max_prob = float(triplet_scores_df["sweet_spot_prob"].max()) if not triplet_scores_df.empty else 0.0
            max_lift = float(triplet_scores_df["sweet_spot_lift"].max()) if not triplet_scores_df.empty else 1.0

            summary_metrics = {
                "features_evaluated": len(numeric_features),
                "candidate_features": len(candidate_features),
                "triplets_evaluated": len(feature_triplets),
                "targets_evaluated": len(valid_targets),
                "max_synergy_gain": _round(max_synergy),
                "max_iv_3d": _round(max_iv_3d),
                "max_sweet_spot_prob": _round(max_prob),
                "max_lift": _round(max_lift),
            }

            summary_payload: dict[str, object] = {
                **asdict(metadata),
                "top_triplets": triplet_scores_df.head(10).to_dict(orient="records")
                if not triplet_scores_df.empty
                else [],
                "top_sweet_spots": triplet_scores_df.sort_values("sweet_spot_prob", ascending=False)
                .head(10)
                .to_dict(orient="records")
                if not triplet_scores_df.empty
                else [],
                "summary_metrics": summary_metrics,
            }
            _write_json(run_dir / "summary.json", summary_payload)

            # Generate report.md
            sorted_sweet_spots = (
                triplet_scores_df.sort_values(["sweet_spot_lift", "sweet_spot_prob"], ascending=[False, False])
                .head(10)
                .to_dict(orient="records")
                if not triplet_scores_df.empty
                else []
            )

            report_md_lines = [
                "# 3D Joint Probability & Sweet Spots Report",
                "",
                "## Executive Summary",
                "",
                f"- **Triplets Evaluated**: `{len(feature_triplets)}` combinations across `{len(candidate_features)}` pre-screened candidate features.",
                f"- **Max Synergy Gain**: `+{max_synergy:.4f}` (3-way interaction probability lift over single feature baselines).",
                f"- **Peak Sweet Spot Probability**: `{max_prob:.1%}` with maximum lift `{max_lift:.2f}x`.",
                "",
                "## Run Metadata",
                "",
                _markdown_table(
                    pd.DataFrame(
                        [
                            {"Metric": "Module", "Value": metadata.module},
                            {"Metric": "Execution Time", "Value": metadata.execution_time},
                            {"Metric": "Quantile Bins (per dimension)", "Value": metadata.n_bins},
                            {"Metric": "Total Voxels per Triplet", "Value": metadata.n_bins ** 3},
                            {"Metric": "Candidate Features", "Value": metadata.max_candidates},
                            {"Metric": "Minimum Voxel Support", "Value": metadata.min_support},
                            {"Metric": "Triplets Evaluated", "Value": metadata.triplets_evaluated},
                            {"Metric": "Rows Evaluated", "Value": metadata.model_rows},
                        ]
                    )
                ),
                "",
                "## Top 3D Synergistic Triplets (F1 x F2 x F3)",
                "",
                _markdown_table(triplet_scores_df.head(15)) if not triplet_scores_df.empty else "No triplets evaluated.",
                "",
                "## Top Actionable 3D Sweet Spot Rules",
                "",
                _markdown_table(pd.DataFrame(sorted_sweet_spots)) if sorted_sweet_spots else "No sweet spots extracted.",
                "",
                "## Sliced 3D Quantile Heatmaps",
                "",
                "![3D Probability Heatmaps](probability_3d_heatmaps.png)",
                "",
                "## Artifacts",
                "",
                "- `summary.json`",
                "- `triplet_probability_scores.csv`",
                "- `voxel_conditional_probabilities.csv`",
                "- `probability_3d_heatmaps.png`",
                "- `report.html`",
                "",
            ]

            (run_dir / "report.md").write_text("\n".join(report_md_lines), encoding="utf-8")
            _render_report_html(metadata, all_triplet_scores, all_voxel_rows, summary_metrics, run_dir)
            progress_bar.step("reports")

        return ModuleResult(
            report_dir=run_dir,
            artifacts=[
                run_dir / "summary.json",
                run_dir / "triplet_probability_scores.csv",
                run_dir / "voxel_conditional_probabilities.csv",
                run_dir / "probability_3d_heatmaps.png",
                run_dir / "report.md",
                run_dir / "report.html",
            ],
        )
