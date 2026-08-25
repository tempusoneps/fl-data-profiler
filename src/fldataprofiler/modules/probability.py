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
DEFAULT_N_BINS = 20
DEFAULT_MIN_SAMPLES = 10
EPSILON = 1e-7


@dataclass
class ProbabilityConfig:
    n_bins: int = DEFAULT_N_BINS
    min_samples: int = DEFAULT_MIN_SAMPLES


@dataclass(frozen=True)
class ProbabilityRunMetadata:
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
    min_samples: int
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


def _compute_feature_target_probabilities(
    feature_series: pd.Series,
    target_series: pd.Series,
    feature_name: str,
    target_name: str,
    n_bins: int = DEFAULT_N_BINS,
    min_samples: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Calculate quantile conditional probabilities, WoE, IV, Spread, Monotonicity, and Entropy."""
    clean_feature = _numeric_series(feature_series)
    valid_mask = clean_feature.notna() & target_series.notna()

    x = clean_feature[valid_mask]
    y = target_series[valid_mask]

    if len(x) < max(2, min_samples) or y.nunique(dropna=True) < 2 or x.nunique(dropna=True) < 2:
        return [], []

    bins = _compute_quantile_bins(x, n_bins=n_bins)
    df = pd.DataFrame({"x": x, "y": y, "bin": bins})
    n_total = len(df)
    unique_classes = sorted(df["y"].unique(), key=lambda v: str(v))

    bin_indices = sorted(df["bin"].unique())
    score_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []

    # Calculate per-bin metrics
    bin_stats: dict[int, dict[str, object]] = {}
    for k in bin_indices:
        bin_df = df[df["bin"] == k]
        n_k = len(bin_df)
        if n_k == 0:
            continue

        val_min = float(bin_df["x"].min())
        val_max = float(bin_df["x"].max())
        val_mean = float(bin_df["x"].mean())

        # Shannon Entropy of bin k: H_k = - sum p_c * log2(p_c)
        entropy_k = 0.0
        class_counts: dict[object, int] = {}
        for c in unique_classes:
            cnt = int((bin_df["y"] == c).sum())
            class_counts[c] = cnt
            prob_c = cnt / n_k
            if prob_c > 0:
                entropy_k -= prob_c * np.log2(prob_c)

        bin_stats[k] = {
            "n_k": n_k,
            "val_min": val_min,
            "val_max": val_max,
            "val_mean": val_mean,
            "entropy": entropy_k,
            "class_counts": class_counts,
        }

    # For each class, calculate WoE, IV, Spread, Monotonicity, and KL Divergence
    for c in unique_classes:
        total_events = int((df["y"] == c).sum())
        total_non_events = n_total - total_events
        base_rate = total_events / n_total if n_total > 0 else 0.0

        probs: list[float] = []
        iv_total = 0.0
        kl_div_total = 0.0
        class_entropies: list[float] = []

        for k in bin_indices:
            stats_k = bin_stats.get(k)
            if not stats_k:
                continue

            n_k = stats_k["n_k"]
            events_k = stats_k["class_counts"][c]
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

            if dist_event > 0:
                kl_div_total += float(dist_event * np.log(p_e / p_ne))

            entropy_k = stats_k["entropy"]
            class_entropies.append(entropy_k)

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
                    "conditional_prob": _round(prob_k),
                    "woe": _round(woe_k),
                    "iv_contribution": _round(iv_k),
                    "entropy": _round(entropy_k),
                }
            )

        prob_arr = np.array(probs, dtype=float)
        max_p = float(prob_arr.max()) if len(prob_arr) > 0 else 0.0
        min_p = float(prob_arr.min()) if len(prob_arr) > 0 else 0.0
        prob_spread = max_p - min_p

        # Monotonicity: Spearman correlation between bin indices (1..K) and probabilities
        if len(prob_arr) < 2 or np.all(np.isclose(prob_arr, prob_arr[0])):
            monotonicity = 0.0
        else:
            spearman_res = stats.spearmanr(np.arange(1, len(prob_arr) + 1), prob_arr)
            corr = getattr(spearman_res, "statistic", getattr(spearman_res, "correlation", 0.0))
            monotonicity = 0.0 if np.isnan(corr) else float(corr)

        mean_entropy = float(np.mean(class_entropies)) if class_entropies else 0.0

        score_rows.append(
            {
                "feature": feature_name,
                "target": target_name,
                "target_class": c,
                "information_value": _round(iv_total),
                "prob_spread": _round(prob_spread),
                "max_prob": _round(max_p),
                "min_prob": _round(min_p),
                "base_rate": _round(base_rate),
                "monotonicity": _round(monotonicity),
                "kl_divergence": _round(kl_div_total),
                "mean_entropy": _round(mean_entropy),
                "n_bins": int(len(probs)),
                "sample_count": int(n_total),
            }
        )

    return score_rows, quantile_rows


def _plot_probability_distribution(
    scores_df: pd.DataFrame,
    quantiles_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot quantile conditional probability distributions for top features."""
    if scores_df.empty or quantiles_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No probability data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Identify top unique feature-target relationships by information_value / prob_spread
    top_pairs_df = (
        scores_df.sort_values(["information_value", "prob_spread"], ascending=[False, False])
        .drop_duplicates(subset=["feature", "target"])
        .head(4)
    )

    n_plots = len(top_pairs_df)
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
            # Binary: plot positive or max-spread class
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
                best_class_row = feat_scores.sort_values("prob_spread", ascending=False).iloc[0]
                c = best_class_row["target_class"]

            c_quantiles = feat_quantiles[feat_quantiles["target_class"] == c].sort_values("bin_index")

            bins = c_quantiles["bin_index"].to_numpy()
            probs = c_quantiles["conditional_prob"].to_numpy()
            base_rate = float(best_class_row["base_rate"])
            iv = float(best_class_row["information_value"])
            spread = float(best_class_row["prob_spread"])

            ax.bar(bins, probs, color="#3b82f6", alpha=0.85, edgecolor="#1d4ed8", width=0.8)
            base_label = (
                f"Base Rate ({base_rate:.2%})"
                if base_rate < 0.05
                else f"Base Rate ({base_rate:.2f})"
            )
            ax.axhline(
                base_rate,
                color="#ef4444",
                linestyle="--",
                linewidth=1.5,
                label=base_label,
            )
            ax.set_title(
                f"{feature} vs {target_name} ({c})\nIV: {iv:.3f} | Spread: {spread:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(loc="upper left", fontsize=9)
            ax.set_ylabel(f"P({c} | Bin)", fontsize=10)

            y_max = max(float(np.max(probs)) if len(probs) > 0 else 0.0, base_rate)
            upper_limit = min(1.05, max(0.01, y_max * 1.25))
            ax.set_ylim(0.0, upper_limit)
        else:
            # Multiclass: plot lines for each class
            for idx, c in enumerate(target_classes):
                c_quantiles = feat_quantiles[feat_quantiles["target_class"] == c].sort_values(
                    "bin_index"
                )
                bins = c_quantiles["bin_index"].to_numpy()
                probs = c_quantiles["conditional_prob"].to_numpy()
                color = colors[idx % len(colors)]
                ax.plot(
                    bins,
                    probs,
                    marker="o",
                    linewidth=2,
                    label=f"Class {c}",
                    color=color,
                )

            max_iv = float(feat_scores["information_value"].max())
            max_spread = float(feat_scores["prob_spread"].max())
            ax.set_title(
                f"{feature} vs {target_name}\nMax IV: {max_iv:.3f} | Max Spread: {max_spread:.3f}",
                fontsize=11,
                fontweight="bold",
            )
            ax.legend(loc="upper left", fontsize=9)
            ax.set_ylabel("P(Class | Bin)", fontsize=10)

            all_probs = feat_quantiles["conditional_prob"].to_numpy()
            y_max = float(np.nanmax(all_probs)) if len(all_probs) > 0 else 0.0
            upper_limit = min(1.05, max(0.01, y_max * 1.25))
            ax.set_ylim(0.0, upper_limit)

        n_bins_plotted = len(bins) if len(target_classes) > 0 else DEFAULT_N_BINS
        ax.set_xlabel(f"Quantile Bin (1 - {n_bins_plotted})", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide unused subplots if any
    for j in range(n_plots, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_markdown(
    metadata: ProbabilityRunMetadata,
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
                {"Metric": "Minimum Samples", "Value": metadata.min_samples},
                {"Metric": "Features Analyzed", "Value": len(metadata.features)},
                {"Metric": "Targets Analyzed", "Value": ", ".join(metadata.targets)},
                {"Metric": "Rows Evaluated", "Value": metadata.model_rows},
            ]
        )
    )

    top_scores_table = _markdown_table(scores_df.head(25)) if not scores_df.empty else "No feature scores."

    insights: list[str] = []
    if not scores_df.empty:
        top_iv = scores_df.sort_values("information_value", ascending=False).iloc[0]
        insights.append(
            f"- **Top Predictive Feature by Information Value (IV)**: `{top_iv['feature']}` for target `{top_iv['target']}` (Class `{top_iv['target_class']}`) achieved IV = `{top_iv['information_value']}`."
        )
        top_spread = scores_df.sort_values("prob_spread", ascending=False).iloc[0]
        insights.append(
            f"- **Maximum Probability Spread (ΔP)**: `{top_spread['feature']}` showed probability shift from `{top_spread['min_prob']}` in low bins to `{top_spread['max_prob']}` in high bins (ΔP = `{top_spread['prob_spread']}`)."
        )
        top_mono = scores_df.sort_values("monotonicity", key=abs, ascending=False).iloc[0]
        insights.append(
            f"- **Strongest Monotonic Relationship**: `{top_mono['feature']}` showed Spearman rank correlation of `{top_mono['monotonicity']}` across the 20 quantile bins."
        )
    else:
        insights.append("- No valid numeric features or categorical targets available for probability profiling.")

    insights_text = "\n".join(insights)

    return f"""# Probability & Quantile Profiling Report

## Executive Summary & Key Insights

{insights_text}

## Run Metadata

{metadata_table}

## Top Feature Probability Scores

{top_scores_table}

## Visual Distribution

![Probability Distribution](probability_distribution.png)

## Artifacts

- `summary.json`
- `feature_probability_scores.csv`
- `quantile_conditional_probabilities.csv`
- `probability_distribution.png`
- `report.html`
"""


def _render_html(
    metadata: ProbabilityRunMetadata,
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
    <title>Probability & Quantile Profiling Report</title>
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
    <h1>Probability & Quantile Profiling Report</h1>

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
                <div class="meta-val">{metadata.n_bins}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Evaluated Rows</div>
                <div class="meta-val">{metadata.model_rows}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Top Feature Probability Scores</h2>
        <div class="table-container">
            {scores_html}
        </div>
    </div>

    <div class="card">
        <h2>Probability Distribution Visualization</h2>
        <img class="chart-img" src="probability_distribution.png" alt="Probability Distribution Visualization"/>
    </div>

    {details}
</body>
</html>
"""


class ProbabilityModule:
    name = "probability"

    def __init__(
        self,
        config: ProbabilityConfig | None = None,
        progress: bool | None = None,
        n_bins: int | None = None,
        min_samples: int | None = None,
    ) -> None:
        self.progress = progress
        if config is not None:
            base_cfg = config
        else:
            mod_cfg = get_module_config("probability")
            base_cfg = ProbabilityConfig(
                n_bins=int(mod_cfg.get("n_bins", mod_cfg.get("n_quantiles", DEFAULT_N_BINS))),
                min_samples=int(
                    mod_cfg.get("min_samples", mod_cfg.get("min_bin_samples", DEFAULT_MIN_SAMPLES))
                ),
            )
        self.config = ProbabilityConfig(
            n_bins=n_bins if n_bins is not None else base_cfg.n_bins,
            min_samples=min_samples if min_samples is not None else base_cfg.min_samples,
        )
        self.n_bins = self.config.n_bins
        self.min_samples = self.config.min_samples

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

            # Filter for categorical / discrete targets with 2..MAX_LABEL_CLASSES unique values
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

            all_score_rows: list[dict[str, object]] = []
            all_quantile_rows: list[dict[str, object]] = []
            effective_min_samples = min(self.min_samples, len(model_frame)) if len(model_frame) > 0 else self.min_samples

            for feature_col in numeric_features:
                for target_col in valid_targets:
                    scores, quantiles = _compute_feature_target_probabilities(
                        model_frame[feature_col],
                        model_frame[target_col],
                        feature_col,
                        target_col,
                        n_bins=self.n_bins,
                        min_samples=effective_min_samples,
                    )
                    all_score_rows.extend(scores)
                    all_quantile_rows.extend(quantiles)

            scores_df = pd.DataFrame(all_score_rows)
            quantiles_df = pd.DataFrame(all_quantile_rows)

            if not scores_df.empty:
                scores_df = scores_df.sort_values(
                    ["information_value", "prob_spread"],
                    ascending=[False, False],
                ).reset_index(drop=True)

            progress_bar.step("probabilities")

            plot_path = run_dir / "probability_distribution.png"
            _plot_probability_distribution(scores_df, quantiles_df, plot_path)
            progress_bar.step("plots")

            metadata = ProbabilityRunMetadata(
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
                min_samples=self.min_samples,
                features=numeric_features,
                targets=valid_targets,
                model_rows=len(model_frame),
            )

            scores_csv_path = _write_csv(run_dir / "feature_probability_scores.csv", scores_df)
            quantiles_csv_path = _write_csv(
                run_dir / "quantile_conditional_probabilities.csv", quantiles_df
            )

            summary_payload: dict[str, object] = {
                **asdict(metadata),
                "top_features": scores_df.head(10).to_dict(orient="records")
                if not scores_df.empty
                else [],
                "summary_metrics": {
                    "features_evaluated": len(numeric_features),
                    "targets_evaluated": len(valid_targets),
                    "max_information_value": _round(float(scores_df["information_value"].max()))
                    if not scores_df.empty
                    else 0.0,
                    "max_prob_spread": _round(float(scores_df["prob_spread"].max()))
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
