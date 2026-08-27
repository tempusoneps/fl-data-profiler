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

from fldataprofiler.config import get_global_config, get_module_config
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
DEFAULT_N_QUANTILES = 20
DEFAULT_MIN_PROBABILITY = 0.55
DEFAULT_MIN_SUPPORT = 20
DEFAULT_MIN_LIFT = 1.0
DEFAULT_TOP_FEATURES = 25
DEFAULT_MIN_FEATURE_UNIQUE_VALUES = 10
EPSILON = 1e-7


@dataclass
class ProbabilityCoverageConfig:
    n_quantiles: int = DEFAULT_N_QUANTILES
    min_probability: float = DEFAULT_MIN_PROBABILITY
    min_support: int = DEFAULT_MIN_SUPPORT
    min_lift: float = DEFAULT_MIN_LIFT
    top_features: int = DEFAULT_TOP_FEATURES
    max_label_classes: int = MAX_LABEL_CLASSES
    min_feature_unique_values: int = DEFAULT_MIN_FEATURE_UNIQUE_VALUES


@dataclass(frozen=True)
class ProbabilityCoverageRunMetadata:
    module: str
    created_at: str
    execution_time: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    n_quantiles: int
    min_probability: float
    min_support: int
    min_lift: float
    max_label_classes: int
    min_feature_unique_values: int
    features_count: int
    features_evaluated: int
    targets: list[str]
    model_rows: int


def _compute_quantile_bins(series: pd.Series, n_bins: int = DEFAULT_N_QUANTILES) -> pd.Series:
    """Assign rank-based equal-frequency quantile bin indices (1 to n_bins)."""
    valid_series = series.dropna()
    if len(valid_series) == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    ranks = valid_series.rank(method="first")
    actual_bins = min(n_bins, len(valid_series))
    if actual_bins < 1:
        return pd.Series(1.0, index=series.index, dtype=float)
    bins = pd.qcut(ranks, q=actual_bins, labels=False) + 1
    result = pd.Series(np.nan, index=series.index, dtype=float)
    result.loc[valid_series.index] = bins.astype(float)
    return result


def _evaluate_feature_crosstab_coverage(
    feature_series: pd.Series,
    target_series: pd.Series,
    feature_name: str,
    target_name: str,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    min_support: int = DEFAULT_MIN_SUPPORT,
    min_lift: float = DEFAULT_MIN_LIFT,
    max_label_classes: int = MAX_LABEL_CLASSES,
    min_feature_unique_values: int = DEFAULT_MIN_FEATURE_UNIQUE_VALUES,
    precomputed_bins: pd.Series | None = None,
    precomputed_ranges: dict[int, tuple[float, float]] | None = None,
) -> tuple[dict[str, object] | None, list[dict[str, object]], list[dict[str, object]]]:
    """
    Perform 1 Feature x 1 Label crosstab, convert count -> percent across bins,
    and evaluate coverage of bins exceeding min_probability threshold.
    """
    # Guard: Skip non-numeric or boolean features
    if not pd.api.types.is_numeric_dtype(feature_series) or pd.api.types.is_bool_dtype(feature_series):
        return None, [], []

    clean_x = _numeric_series(feature_series)
    valid_mask = clean_x.notna() & target_series.notna()

    x_val = clean_x[valid_mask]
    t_val = target_series[valid_mask]

    n_samples = len(x_val)
    if n_samples < min_support * 2:
        return None, [], []

    # Guard: Target must be discrete categorical (2 to max_label_classes)
    n_unique_targets = t_val.nunique(dropna=True)
    if not (2 <= n_unique_targets <= max_label_classes):
        return None, [], []

    # Guard: Skip categorical / low-cardinality discrete features (must have sufficient unique values for quantile binning)
    if x_val.nunique(dropna=True) < max(2, min_feature_unique_values):
        return None, [], []

    if precomputed_bins is not None and precomputed_ranges is not None:
        bins = precomputed_bins.loc[x_val.index].dropna().astype(int)
        bin_ranges = precomputed_ranges
    else:
        bins_series = _compute_quantile_bins(x_val, n_bins=n_quantiles)
        bins = bins_series.dropna().astype(int)
        bin_ranges = {}
        actual_n_bins = int(bins.max()) if len(bins) > 0 else n_quantiles
        for b in range(1, actual_n_bins + 1):
            vals = x_val[bins == b]
            if len(vals) > 0:
                bin_ranges[b] = (float(vals.min()), float(vals.max()))
            else:
                bin_ranges[b] = (np.nan, np.nan)

    actual_n_bins = max(bin_ranges.keys()) if bin_ranges else n_quantiles

    # Fast bincount crosstab
    t_codes, unique_classes = pd.factorize(t_val, sort=True)
    n_classes = len(unique_classes)
    bins_arr = bins.to_numpy(dtype=np.int64) - 1
    valid_idx = (bins_arr >= 0) & (bins_arr < actual_n_bins)

    combined = bins_arr[valid_idx] * n_classes + t_codes[valid_idx]
    counts = np.bincount(combined, minlength=actual_n_bins * n_classes).reshape(actual_n_bins, n_classes)
    row_sums = counts.sum(axis=1)

    # Convert count -> percent (row-normalized %)
    row_sums_safe = np.where(row_sums > 0, row_sums, 1)[:, None]
    probs = counts / row_sums_safe

    class_totals = counts.sum(axis=0)
    base_rates = class_totals / n_samples if n_samples > 0 else np.zeros(n_classes)

    feature_class_summaries: list[dict[str, object]] = []
    cell_details: list[dict[str, object]] = []
    crosstab_matrix_rows: list[dict[str, object]] = []

    matrix_qualified_cells = 0
    matrix_qualified_bin_set: set[int] = set()
    matrix_peak_prob = 0.0
    dominant_class_set: set[str] = set()
    extracted_rules: list[str] = []

    # 1. Evaluate individual classes
    for c_idx, target_class in enumerate(unique_classes):
        n_class_total = int(class_totals[c_idx])
        base_rate = float(base_rates[c_idx])

        if n_class_total == 0 or n_class_total == n_samples:
            continue

        q_bins_count = 0
        q_samples_count = 0
        q_events_count = 0
        q_probs: list[float] = []
        q_bin_indices: list[int] = []
        max_prob_c = 0.0
        best_bin_c = np.nan

        for b_idx in range(actual_n_bins):
            b = b_idx + 1
            n_bin = int(row_sums[b_idx])
            if n_bin == 0:
                continue

            events = int(counts[b_idx, c_idx])
            prob = float(probs[b_idx, c_idx])
            lift = (prob / base_rate) if base_rate > 0 else 0.0

            is_qualified = (
                prob >= min_probability
                and n_bin >= min_support
                and lift >= min_lift
            )

            if is_qualified:
                q_bins_count += 1
                q_samples_count += n_bin
                q_events_count += events
                q_probs.append(prob)
                q_bin_indices.append(b)
                matrix_qualified_cells += 1
                matrix_qualified_bin_set.add(b)
                dominant_class_set.add(str(target_class))

            if prob > matrix_peak_prob:
                matrix_peak_prob = prob

            if n_bin >= min_support and prob > max_prob_c:
                max_prob_c = prob
                best_bin_c = b

            val_min, val_max = bin_ranges.get(b, (np.nan, np.nan))

            cell_details.append({
                "feature": feature_name,
                "target": target_name,
                "target_class": target_class,
                "bin": b,
                "val_min": val_min,
                "val_max": val_max,
                "samples": n_bin,
                "events": events,
                "conditional_prob": prob,
                "conditional_prob_pct": prob * 100.0,
                "base_rate": base_rate,
                "base_rate_pct": base_rate * 100.0,
                "lift": lift,
                "is_qualified": bool(is_qualified),
            })

        bin_coverage_pct = (q_bins_count / actual_n_bins) * 100.0 if actual_n_bins > 0 else 0.0
        sample_coverage_pct = (q_samples_count / n_samples) * 100.0 if n_samples > 0 else 0.0
        weighted_qualified_prob = (
            (q_events_count / q_samples_count)
            if q_samples_count > 0
            else 0.0
        )
        mean_qualified_prob = float(np.mean(q_probs)) if q_probs else 0.0
        min_qualified_prob = float(np.min(q_probs)) if q_probs else 0.0
        mean_lift = (weighted_qualified_prob / base_rate) if base_rate > 0 else 0.0

        composite_score = (
            q_bins_count
            * (mean_lift if mean_lift > 0 else 1.0)
            * np.sqrt(sample_coverage_pct / 100.0)
        )

        if q_bins_count > 0:
            min_b, max_b = min(q_bin_indices), max(q_bin_indices)
            overall_low = bin_ranges.get(min_b, (0.0, 0.0))[0]
            overall_high = bin_ranges.get(max_b, (0.0, 0.0))[1]

            rule_str = (
                f"IF {feature_name} in [{overall_low:.3f}, {overall_high:.3f}] (Bins Q{min_b:02d}..Q{max_b:02d}) "
                f"THEN P({target_class})={weighted_qualified_prob*100:.1f}% ({q_bins_count}/{actual_n_bins} bins, {sample_coverage_pct:.1f}% samples)"
            )
            extracted_rules.append(rule_str)
        else:
            rule_str = f"No bins met threshold P >= {min_probability*100:.0f}%"

        feature_class_summaries.append({
            "feature": feature_name,
            "target": target_name,
            "target_class": target_class,
            "samples": n_samples,
            "base_rate": base_rate,
            "min_probability_threshold": min_probability,
            "qualified_bins": q_bins_count,
            "total_bins": actual_n_bins,
            "bin_coverage_pct": bin_coverage_pct,
            "qualified_samples": q_samples_count,
            "sample_coverage_pct": sample_coverage_pct,
            "weighted_qualified_prob": weighted_qualified_prob,
            "mean_qualified_prob": mean_qualified_prob,
            "min_qualified_prob": min_qualified_prob,
            "max_bin_prob": max_prob_c,
            "best_bin": best_bin_c,
            "mean_lift": mean_lift,
            "composite_coverage_score": float(composite_score),
            "coverage_rule": rule_str,
        })

    # 2. Build full Crosstab Matrix Table rows (for report.md & report.html)
    for b_idx in range(actual_n_bins):
        b = b_idx + 1
        n_bin = int(row_sums[b_idx])
        val_min, val_max = bin_ranges.get(b, (np.nan, np.nan))

        row_item: dict[str, object] = {
            "Bin": f"Q{b:02d}",
            "Range": f"[{val_min:.3f}, {val_max:.3f}]",
            "Samples": f"{n_bin:,}",
        }

        qual_cell_labels: list[str] = []
        for c_idx, target_class in enumerate(unique_classes):
            prob = float(probs[b_idx, c_idx])
            base_rate = float(base_rates[c_idx])
            lift = (prob / base_rate) if base_rate > 0 else 0.0
            is_qualified = (
                prob >= min_probability
                and n_bin >= min_support
                and lift >= min_lift
            )

            cell_str = f"{prob * 100:.1f}%"
            if is_qualified:
                cell_str += " ★"
                qual_cell_labels.append(f"{target_class} ({prob * 100:.1f}%)")

            row_item[f"{target_class} (%)"] = cell_str

        row_item["Qualified (>min_x)"] = ", ".join(qual_cell_labels) if qual_cell_labels else "-"
        crosstab_matrix_rows.append(row_item)

    # 3. Formulate overall Matrix Summary
    matrix_qualified_bins = len(matrix_qualified_bin_set)
    matrix_bin_cov_pct = (matrix_qualified_bins / actual_n_bins) * 100.0 if actual_n_bins > 0 else 0.0
    matrix_qual_samples = int(sum(row_sums[b - 1] for b in matrix_qualified_bin_set))
    matrix_sample_cov_pct = (matrix_qual_samples / n_samples) * 100.0 if n_samples > 0 else 0.0

    matrix_summary: dict[str, object] = {
        "feature": feature_name,
        "target": target_name,
        "total_matrix_cells": actual_n_bins * n_classes,
        "qualified_cells": matrix_qualified_cells,
        "qualified_bins": matrix_qualified_bins,
        "total_bins": actual_n_bins,
        "bin_coverage_pct": matrix_bin_cov_pct,
        "qualified_samples": matrix_qual_samples,
        "sample_coverage_pct": matrix_sample_cov_pct,
        "peak_probability": matrix_peak_prob,
        "target_classes_count": n_classes,
        "dominant_classes": ", ".join(sorted(dominant_class_set)) if dominant_class_set else "None",
        "rules": extracted_rules if extracted_rules else ["No bins met threshold"],
        "crosstab_rows": crosstab_matrix_rows,
    }

    return matrix_summary, feature_class_summaries, cell_details


def _plot_feature_coverage_charts(
    matrix_summaries: list[dict[str, object]],
    cell_df: pd.DataFrame,
    output_path: Path,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    top_n: int = 6,
) -> Path:
    """Plot quantile probability bar charts for top matrices, highlighting bins >= min_probability."""
    if not matrix_summaries or cell_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No probability coverage data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    top_matrices = matrix_summaries[:top_n]
    n_plots = len(top_matrices)
    if n_plots == 1:
        nrows, ncols = 1, 1
        figsize = (9, 5)
    elif n_plots == 2:
        nrows, ncols = 1, 2
        figsize = (16, 5)
    elif n_plots in (3, 4):
        nrows, ncols = 2, 2
        figsize = (16, 10)
    else:
        nrows, ncols = 2, 3
        figsize = (22, 10)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for idx, mat in enumerate(top_matrices):
        ax = flat_axes[idx]
        feat = str(mat["feature"])
        tgt = str(mat["target"])
        q_cells = int(mat["qualified_cells"])
        q_bins = int(mat["qualified_bins"])
        tot_b = int(mat["total_bins"])

        sub_cells = cell_df[
            (cell_df["feature"] == feat)
            & (cell_df["target"] == tgt)
        ].copy()

        if sub_cells.empty:
            continue

        unique_classes = sorted(sub_cells["target_class"].unique(), key=lambda v: str(v))
        bins = sorted(sub_cells["bin"].unique())
        bin_labels = [f"Q{b:02d}" for b in bins]
        x = np.arange(len(bins))
        width = 0.8 / max(1, len(unique_classes))

        colors = ["#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

        for c_idx, t_cls in enumerate(unique_classes):
            cls_cells = sub_cells[sub_cells["target_class"] == t_cls].sort_values("bin")
            probs_pct = cls_cells["conditional_prob_pct"].values
            is_quals = cls_cells["is_qualified"].values

            color = colors[c_idx % len(colors)]
            pos = x - 0.4 + (c_idx + 0.5) * width
            bars = ax.bar(pos, probs_pct, width=width, label=str(t_cls), color=color, alpha=0.85, edgecolor="black", linewidth=0.5)

            for bar, prob, q in zip(bars, probs_pct, is_quals):
                if q:
                    bar.set_edgecolor("#eab308")
                    bar.set_linewidth(1.8)

        ax.axhline(
            min_probability * 100.0,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label=f"Min Threshold ({min_probability*100:.0f}%)",
        )

        ax.set_title(
            f"{feat} × {tgt}\n"
            f"★ Qualified Cells: {q_cells} | Qualified Bins: {q_bins}/{tot_b} ({mat['bin_coverage_pct']:.1f}%) | Cov: {mat['sample_coverage_pct']:.1f}%",
            fontsize=9,
            fontweight="bold",
            pad=8,
        )
        ax.set_xlabel("Quantile Bins", fontsize=8)
        ax.set_ylabel("Conditional Prob P(Y=c | Bin) %", fontsize=8)
        ax.set_ylim(0, 105.0)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, rotation=45, fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(loc="upper left", fontsize=7)

    for ax in flat_axes[n_plots:]:
        ax.axis("off")

    fig.suptitle(
        f"Top Feature × Label Matrices Ranked by High-Probability Coverage (Cells P ≥ {min_probability*100:.0f}% Highlighted)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _generate_markdown_report(
    metadata: ProbabilityCoverageRunMetadata,
    matrix_summaries: list[dict[str, object]],
    image_rel_path: str,
) -> str:
    md: list[str] = [
        f"# Probability Coverage & Quantile Matrix Ranking (`{metadata.module}`)",
        "",
        f"- **Generated At**: {metadata.created_at}",
        f"- **Execution Time**: {metadata.execution_time}",
        f"- **Merged Rows**: {metadata.model_rows:,}",
        f"- **Quantile Discretization**: {metadata.n_quantiles} Quantiles (qcut rank-based)",
        f"- **Min Probability Threshold (min_x)**: **{metadata.min_probability * 100:.1f}%**",
        f"- **Min Bin Sample Support**: {metadata.min_support} samples",
        f"- **Max Label Classes**: {metadata.max_label_classes}",
        f"- **Features Evaluated**: {metadata.features_evaluated}",
        f"- **Discrete Targets Evaluated**: {len(metadata.targets)} ({', '.join(metadata.targets)})",
        "",
        "---",
        "",
        "## 1. Executive Summary & Master Matrix Rankings",
        "",
    ]

    if not matrix_summaries:
        md.append("No features evaluated or no discrete classification targets found (targets must have 2..max_label_classes classes).")
        return "\n".join(md)

    top_m = matrix_summaries[0]
    md.extend([
        f"- **Top Coverage Matrix**: `{top_m['feature']}` × `{top_m['target']}`.",
        f"  - **Qualified Cells Count ($P \\ge {metadata.min_probability*100:.0f}\\%$)**: **{top_m['qualified_cells']} ô** (trên tổng số {top_m['total_matrix_cells']} ô).",
        f"  - **Qualified Bins**: **{top_m['qualified_bins']} / {top_m['total_bins']} bins ({top_m['bin_coverage_pct']:.1f}%)**.",
        f"  - **Sample Coverage**: **{top_m['sample_coverage_pct']:.2f}%** ({top_m['qualified_samples']:,} samples).",
        f"  - **Peak Probability**: **{top_m['peak_probability']*100:.2f}%**.",
        f"  - **Dominant Classes**: `{top_m['dominant_classes']}`.",
        "",
        f"Bảng xếp hạng toàn bộ các **Ma trận Phân vị × Nhãn (1 Feature × 1 Label)** được sắp xếp theo **tổng số ô đạt chuẩn $P \\ge {metadata.min_probability*100:.0f}\\%$** giảm dần:",
        "",
    ])

    rank_rows = []
    for rank_idx, mat in enumerate(matrix_summaries, 1):
        rank_rows.append({
            "Rank": f"#{rank_idx}",
            "Feature": mat["feature"],
            "Target": mat["target"],
            f"Qualified Cells (≥{metadata.min_probability*100:.0f}%)": f"{mat['qualified_cells']} / {mat['total_matrix_cells']}",
            "Qualified Bins": f"{mat['qualified_bins']} / {mat['total_bins']} ({mat['bin_coverage_pct']:.1f}%)",
            "Sample Coverage": f"{mat['sample_coverage_pct']:.1f}%",
            "Peak Prob (%)": f"{mat['peak_probability']*100:.1f}%",
            "Dominant Class": mat["dominant_classes"],
        })

    rank_df = pd.DataFrame(rank_rows)
    md.append(_markdown_table(rank_df.head(30)))

    md.extend([
        "",
        "---",
        "",
        "## 2. Chi Tiết Ma Trận Phân Vị × Nhãn (% Crosstab Matrices)",
        "",
        f"Dưới đây là **ma trận bảng chéo xác suất (%)** chi tiết cho các đặc trưng hàng đầu. Các ô có xác suất **$P \\ge {metadata.min_probability*100:.0f}\\%$** được đánh dấu nổi bật với dấu **★**:",
        "",
    ])

    # Render top 12 matrices in full detail
    for rank_idx, mat in enumerate(matrix_summaries[:12], 1):
        feat = mat["feature"]
        tgt = mat["target"]
        q_cells = mat["qualified_cells"]
        tot_cells = mat["total_matrix_cells"]
        q_bins = mat["qualified_bins"]
        tot_bins = mat["total_bins"]
        samp_cov = mat["sample_coverage_pct"]
        peak_p = mat["peak_probability"] * 100

        md.extend([
            f"### #{rank_idx}. Ma trận: `{feat}` × `{tgt}`",
            "",
            f"- **Tổng số ô đạt chuẩn ($P \\ge {metadata.min_probability*100:.0f}\\%$)**: **{q_cells} / {tot_cells} ô** ({mat['bin_coverage_pct']:.1f}% số phân vị).",
            f"- **Độ phủ mẫu (Sample Coverage)**: **{samp_cov:.1f}%** ({mat['qualified_samples']:,} mẫu) | **Xác suất đỉnh (Peak)**: **{peak_p:.1f}%**.",
            "- **Quy tắc trích xuất (Decision Rules)**:",
        ])
        for rule in mat["rules"]:
            md.append(f"  - `{rule}`")
        md.append("")

        # Matrix Table
        matrix_table_df = pd.DataFrame(mat["crosstab_rows"])
        md.append(_markdown_table(matrix_table_df))
        md.append("")

    md.extend([
        "---",
        "",
        "## 3. Visual Distribution Charts",
        "",
        f"![Probability Coverage Distribution]({image_rel_path})",
        "",
    ])

    return "\n".join(md)


def _generate_html_report(
    metadata: ProbabilityCoverageRunMetadata,
    matrix_summaries: list[dict[str, object]],
    image_rel_path: str,
) -> str:
    summary_rows = []
    for rank_idx, mat in enumerate(matrix_summaries, 1):
        feat = mat["feature"]
        tgt = mat["target"]
        q_cells = mat["qualified_cells"]
        tot_cells = mat["total_matrix_cells"]
        q_bins = mat["qualified_bins"]
        tot_bins = mat["total_bins"]
        cov_pct = mat["bin_coverage_pct"]
        samp_pct = mat["sample_coverage_pct"]
        peak_p = mat["peak_probability"] * 100
        dom_cls = mat["dominant_classes"]

        badge_class = "badge-success" if q_cells >= 5 else ("badge-warning" if q_cells >= 2 else "badge-secondary")

        summary_rows.append(f"""
        <tr>
            <td><strong>#{rank_idx}</strong></td>
            <td><strong>{feat}</strong></td>
            <td><span class="badge badge-info">{tgt}</span></td>
            <td style="text-align: center;"><span class="badge {badge_class}">{q_cells} / {tot_cells} ô</span></td>
            <td style="text-align: center;">{q_bins} / {tot_bins} ({cov_pct:.1f}%)</td>
            <td style="text-align: right;">{samp_pct:.1f}%</td>
            <td style="text-align: right; font-weight: bold; color: #16a34a;">{peak_p:.1f}%</td>
            <td><span class="badge badge-primary">{dom_cls}</span></td>
        </tr>
        """)

    summary_tbody = "\n".join(summary_rows)

    # Detailed Matrices HTML
    detailed_matrices_html = []
    for rank_idx, mat in enumerate(matrix_summaries[:15], 1):
        feat = mat["feature"]
        tgt = mat["target"]
        q_cells = mat["qualified_cells"]
        tot_cells = mat["total_matrix_cells"]
        samp_cov = mat["sample_coverage_pct"]
        peak_p = mat["peak_probability"] * 100

        crosstab_df = pd.DataFrame(mat["crosstab_rows"])
        headers = crosstab_df.columns.tolist()

        th_html = "".join(f"<th>{h}</th>" for h in headers)
        tr_list = []
        for _, r in crosstab_df.iterrows():
            td_list = []
            for h in headers:
                val = str(r[h])
                if "★" in val:
                    td_list.append(f'<td style="background-color: #fef9c3; font-weight: bold; color: #854d0e;">{val}</td>')
                elif h == "Bin":
                    td_list.append(f'<td><strong>{val}</strong></td>')
                elif h == "Qualified (>min_x)":
                    td_list.append(f'<td style="font-size: 11px; font-weight: 600; color: #166534;">{val}</td>')
                else:
                    td_list.append(f'<td>{val}</td>')
            tr_list.append(f"<tr>{''.join(td_list)}</tr>")

        rules_list_html = "".join(f"<li><code>{rule}</code></li>" for rule in mat["rules"])

        detailed_matrices_html.append(f"""
        <div class="card" style="margin-bottom: 24px;">
            <h3 style="margin-top: 0;">#{rank_idx}. Ma trận: <code>{feat}</code> × <code>{tgt}</code></h3>
            <p style="font-size: 13px; color: #475569;">
                <strong>Số ô đạt chuẩn (P ≥ {metadata.min_probability*100:.0f}%):</strong> <span class="badge badge-success">{q_cells} / {tot_cells} ô</span> | 
                <strong>Độ phủ mẫu:</strong> {samp_cov:.1f}% ({mat['qualified_samples']:,} mẫu) | 
                <strong>Xác suất đỉnh:</strong> {peak_p:.1f}%
            </p>
            <ul style="font-size: 12px; margin-bottom: 16px;">
                {rules_list_html}
            </ul>
            <div style="overflow-x: auto;">
                <table>
                    <thead><tr>{th_html}</tr></thead>
                    <tbody>{''.join(tr_list)}</tbody>
                </table>
            </div>
        </div>
        """)

    detailed_html_section = "\n".join(detailed_matrices_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Probability Coverage Report - {metadata.module}</title>
    <style>
        :root {{
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text: #1e293b;
            --primary: #2563eb;
            --border: #e2e8f0;
            --header-bg: #0f172a;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            line-height: 1.5;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e293b, #0f172a);
            color: white;
            padding: 28px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; color: #94a3b8; font-size: 14px; }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px 0 rgb(0 0 0 / 0.05);
        }}
        .card .title {{ font-size: 12px; color: #64748b; text-transform: uppercase; font-weight: 600; }}
        .card .value {{ font-size: 22px; font-weight: bold; margin-top: 4px; color: #0f172a; }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-primary {{ background: #dbeafe; color: #1e40af; }}
        .badge-info {{ background: #e0e7ff; color: #3730a3; }}
        .badge-success {{ background: #dcfce7; color: #166534; }}
        .badge-warning {{ background: #fef9c3; color: #854d0e; }}
        .badge-secondary {{ background: #f1f5f9; color: #475569; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--border);
            text-align: left;
        }}
        th {{
            background: #f1f5f9;
            font-weight: 600;
            color: #334155;
            position: sticky;
            top: 0;
        }}
        tr:hover {{ background: #f8fafc; }}
        .img-container {{
            background: white;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            text-align: center;
            margin-top: 24px;
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Probability Coverage & Feature Quantile Ranking</h1>
            <p>Profiling 1 Feature × 1 Label crosstab (count → percent) and ranking matrices by coverage of cells exceeding threshold (P ≥ {metadata.min_probability*100:.1f}%)</p>
        </div>

        <div class="grid">
            <div class="card">
                <div class="title">Merged Samples</div>
                <div class="value">{metadata.model_rows:,}</div>
            </div>
            <div class="card">
                <div class="title">Quantile Bins</div>
                <div class="value">{metadata.n_quantiles} Quantiles</div>
            </div>
            <div class="card">
                <div class="title">Min Threshold (min_x)</div>
                <div class="value">{metadata.min_probability*100:.1f}%</div>
            </div>
            <div class="card">
                <div class="title">Min Bin Support</div>
                <div class="value">{metadata.min_support} samples</div>
            </div>
            <div class="card">
                <div class="title">Features Evaluated</div>
                <div class="value">{metadata.features_evaluated} features</div>
            </div>
        </div>

        <h2 style="margin-top: 32px; margin-bottom: 16px;">1. Master Matrix Rankings (Sorted by Qualified Cells Count)</h2>
        <div class="card" style="padding: 0; overflow-x: auto; margin-bottom: 32px;">
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Target</th>
                        <th style="text-align: center;">Qualified Cells (P ≥ {metadata.min_probability*100:.0f}%)</th>
                        <th style="text-align: center;">Qualified Bins</th>
                        <th style="text-align: right;">Sample Cov %</th>
                        <th style="text-align: right;">Peak Prob %</th>
                        <th>Dominant Class</th>
                    </tr>
                </thead>
                <tbody>
                    {summary_tbody}
                </tbody>
            </table>
        </div>

        <h2 style="margin-top: 32px; margin-bottom: 16px;">2. Detailed % Crosstab Matrices for Top Features</h2>
        {detailed_html_section}

        <div class="img-container">
            <h3 style="margin-top: 0;">Top Feature Coverage Charts (Cells ≥ {metadata.min_probability*100:.0f}% in Gold)</h3>
            <img src="{image_rel_path}" alt="Probability Coverage Distribution">
        </div>
    </div>
</body>
</html>
"""
    return html_content


class ProbabilityCoverageModule:
    name: str = "probability_coverage"
    description: str = "1 Feature x 1 Label Quantile Crosstab Coverage & Threshold Ranking"

    def __init__(self, progress: bool | None = None) -> None:
        self.progress = progress

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        start_time = time.time()
        now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        features_df = _read_table_with_date_index(feature_csv)
        labels_df = _read_table_with_date_index(label_csv)

        feature_shape = DatasetShape(rows=len(features_df), columns=len(features_df.columns))
        label_shape = DatasetShape(rows=len(labels_df), columns=len(labels_df.columns))

        merged, feature_cols, label_cols, join_strategy = _merge_inputs(
            features_df, labels_df, join_key
        )
        merged_shape = DatasetShape(rows=len(merged), columns=len(merged.columns))

        sampled_df = _sample_rows(merged, max_rows=MAX_ROWS, random_state=RANDOM_STATE)
        selected_targets = _select_targets(label_cols, targets)

        raw_cfg = get_module_config("probability_coverage")
        global_cfg = get_global_config()
        cfg = ProbabilityCoverageConfig(
            n_quantiles=int(raw_cfg.get("n_quantiles", DEFAULT_N_QUANTILES)),
            min_probability=float(raw_cfg.get("min_probability", DEFAULT_MIN_PROBABILITY)),
            min_support=int(raw_cfg.get("min_support", DEFAULT_MIN_SUPPORT)),
            min_lift=float(raw_cfg.get("min_lift", DEFAULT_MIN_LIFT)),
            top_features=int(raw_cfg.get("top_features", DEFAULT_TOP_FEATURES)),
            max_label_classes=int(
                raw_cfg.get(
                    "max_label_classes",
                    global_cfg.get("max_label_classes", MAX_LABEL_CLASSES),
                )
            ),
            min_feature_unique_values=int(
                raw_cfg.get(
                    "min_feature_unique_values",
                    DEFAULT_MIN_FEATURE_UNIQUE_VALUES,
                )
            ),
        )

        with StatusTimer(f"{self.name}: Loading & pre-binning", enabled=self.progress):
            # Filter valid discrete targets (2 to max_label_classes unique values)
            valid_discrete_targets: list[str] = []
            for tgt in selected_targets:
                target_series = sampled_df[tgt].dropna()
                if 2 <= target_series.nunique() <= cfg.max_label_classes:
                    valid_discrete_targets.append(tgt)

            numeric_feature_cols = _numeric_feature_columns(sampled_df, feature_cols)

            binned_features: dict[str, pd.Series] = {}
            feature_bin_ranges: dict[str, dict[int, tuple[float, float]]] = {}

            for col in numeric_feature_cols:
                series = sampled_df[col]
                # Skip categorical / boolean / non-numeric features
                if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                    continue

                clean_x = _numeric_series(series)
                valid_mask = clean_x.notna()
                x_val = clean_x[valid_mask]

                # Only include continuous numeric features with sufficient unique values for quantile binning
                if len(x_val) >= cfg.min_support * 2 and x_val.nunique() >= cfg.min_feature_unique_values:
                    bins_series = _compute_quantile_bins(clean_x, n_bins=cfg.n_quantiles)
                    binned_features[col] = bins_series
                    ranges: dict[int, tuple[float, float]] = {}
                    bins_int = bins_series.dropna().astype(int)
                    actual_n_bins = int(bins_int.max()) if len(bins_int) > 0 else cfg.n_quantiles
                    for b in range(1, actual_n_bins + 1):
                        vals = clean_x[bins_series == b]
                        if len(vals) > 0:
                            ranges[b] = (float(vals.min()), float(vals.max()))
                        else:
                            ranges[b] = (np.nan, np.nan)
                    feature_bin_ranges[col] = ranges

            total_evals = len(valid_discrete_targets) * sum(1 for col in numeric_feature_cols if col in binned_features)

        matrix_summaries: list[dict[str, object]] = []
        all_feature_class_summaries: list[dict[str, object]] = []
        all_cell_details: list[dict[str, object]] = []

        with ModuleProgress(
            self.name, total=max(1, total_evals), unit="matrix", enabled=self.progress
        ) as progress_bar:
            for target in valid_discrete_targets:
                target_series = sampled_df[target]

                for col in numeric_feature_cols:
                    if col not in binned_features:
                        continue

                    m_sum, f_cls_sum, c_det = _evaluate_feature_crosstab_coverage(
                        feature_series=sampled_df[col],
                        target_series=target_series,
                        feature_name=col,
                        target_name=target,
                        n_quantiles=cfg.n_quantiles,
                        min_probability=cfg.min_probability,
                        min_support=cfg.min_support,
                        min_lift=cfg.min_lift,
                        max_label_classes=cfg.max_label_classes,
                        min_feature_unique_values=cfg.min_feature_unique_values,
                        precomputed_bins=binned_features.get(col),
                        precomputed_ranges=feature_bin_ranges.get(col),
                    )
                    if m_sum is not None:
                        matrix_summaries.append(m_sum)
                    all_feature_class_summaries.extend(f_cls_sum)
                    all_cell_details.extend(c_det)
                    progress_bar.step(f"{col}->{target}")

            if total_evals == 0:
                progress_bar.step("no_valid_crosstabs")

        # Sort primary by qualified_cells descending, secondary by sample_coverage_pct, tertiary by peak_probability
        matrix_summaries.sort(
            key=lambda m: (m["qualified_cells"], m["sample_coverage_pct"], m["peak_probability"]),
            reverse=True,
        )

        feature_scores_df = (
            pd.DataFrame(all_feature_class_summaries).sort_values(
                ["qualified_bins", "sample_coverage_pct", "weighted_qualified_prob"],
                ascending=[False, False, False],
            ).reset_index(drop=True)
            if all_feature_class_summaries
            else pd.DataFrame(columns=[
                "feature", "target", "target_class", "samples", "base_rate",
                "min_probability_threshold", "qualified_bins", "total_bins",
                "bin_coverage_pct", "qualified_samples", "sample_coverage_pct",
                "weighted_qualified_prob", "mean_qualified_prob", "min_qualified_prob",
                "max_bin_prob", "best_bin", "mean_lift", "composite_coverage_score",
                "coverage_rule",
            ])
        )

        cell_df = pd.DataFrame(all_cell_details) if all_cell_details else pd.DataFrame(columns=[
            "feature", "target", "target_class", "bin", "val_min", "val_max",
            "samples", "events", "conditional_prob", "conditional_prob_pct",
            "base_rate", "base_rate_pct", "lift", "is_qualified",
        ])

        img_path = run_dir / "probability_coverage_distribution.png"
        _plot_feature_coverage_charts(
            matrix_summaries=matrix_summaries,
            cell_df=cell_df,
            output_path=img_path,
            min_probability=cfg.min_probability,
            top_n=6,
        )

        feature_scores_path = _write_csv(run_dir / "feature_coverage_scores.csv", feature_scores_df)
        cell_details_path = _write_csv(run_dir / "quantile_crosstab_probabilities.csv", cell_df)

        matrix_export_rows = [
            {
                "rank": rank_i,
                "feature": m["feature"],
                "target": m["target"],
                "qualified_cells": m["qualified_cells"],
                "total_matrix_cells": m["total_matrix_cells"],
                "qualified_bins": m["qualified_bins"],
                "total_bins": m["total_bins"],
                "bin_coverage_pct": m["bin_coverage_pct"],
                "qualified_samples": m["qualified_samples"],
                "sample_coverage_pct": m["sample_coverage_pct"],
                "peak_probability": m["peak_probability"],
                "dominant_classes": m["dominant_classes"],
            }
            for rank_i, m in enumerate(matrix_summaries, 1)
        ]
        matrix_scores_path = _write_csv(run_dir / "matrix_coverage_rankings.csv", pd.DataFrame(matrix_export_rows))

        # Export clean wide-format crosstab matrices per Target
        per_target_csv_paths: list[Path] = []
        for target_name in valid_discrete_targets:
            target_matrices = [m for m in matrix_summaries if m["target"] == target_name]
            if not target_matrices:
                continue

            target_crosstab_rows = []
            for m in target_matrices:
                feat = m["feature"]
                for row_dict in m["crosstab_rows"]:
                    target_crosstab_rows.append({
                        "feature": feat,
                        **row_dict,
                    })

            if target_crosstab_rows:
                target_df = pd.DataFrame(target_crosstab_rows)
                safe_target_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in target_name)
                target_csv_path = _write_csv(run_dir / f"crosstab_matrices_{safe_target_name}.csv", target_df)
                per_target_csv_paths.append(target_csv_path)

        duration = _format_duration(time.time() - start_time)

        metadata = ProbabilityCoverageRunMetadata(
            module=self.name,
            created_at=now,
            execution_time=duration,
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=feature_shape,
            label_shape=label_shape,
            merged_shape=merged_shape,
            n_quantiles=cfg.n_quantiles,
            min_probability=cfg.min_probability,
            min_support=cfg.min_support,
            min_lift=cfg.min_lift,
            max_label_classes=cfg.max_label_classes,
            min_feature_unique_values=cfg.min_feature_unique_values,
            features_count=len(numeric_feature_cols),
            features_evaluated=len(binned_features),
            targets=valid_discrete_targets,
            model_rows=len(sampled_df),
        )

        top_matrices_json = matrix_export_rows[:30] if matrix_export_rows else []
        summary_payload = {
            "metadata": asdict(metadata),
            "top_coverage_matrices": top_matrices_json,
        }
        summary_path = _write_json(run_dir / "summary.json", summary_payload)

        report_md_content = _generate_markdown_report(
            metadata=metadata,
            matrix_summaries=matrix_summaries,
            image_rel_path="probability_coverage_distribution.png",
        )
        report_md_path = run_dir / "report.md"
        report_md_path.write_text(report_md_content, encoding="utf-8")

        report_html_content = _generate_html_report(
            metadata=metadata,
            matrix_summaries=matrix_summaries,
            image_rel_path="probability_coverage_distribution.png",
        )
        report_html_path = run_dir / "report.html"
        report_html_path.write_text(report_html_content, encoding="utf-8")

        artifacts = [
            report_md_path,
            report_html_path,
            summary_path,
            feature_scores_path,
            matrix_scores_path,
            *per_target_csv_paths,
            cell_details_path,
            img_path,
        ]

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)
