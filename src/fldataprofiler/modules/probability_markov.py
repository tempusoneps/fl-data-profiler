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
DEFAULT_N_BINS = 5  # Quintiles by default for clean 5x5 transitions
DEFAULT_MIN_PATTERN_SAMPLES = 100
DEFAULT_MIN_SUPPORT = 0.002  # At least 0.2% of dataset (100 samples / 50k rows)
DEFAULT_MIN_EXCESS_PROB = 0.05  # At least +5% edge over static probability
DEFAULT_MIN_LIFT = 1.10
DEFAULT_OBJECTIVE = "support_weighted"
EPSILON = 1e-9


@dataclass
class ProbabilityMarkovConfig:
    n_bins: int = DEFAULT_N_BINS
    min_pattern_samples: int = DEFAULT_MIN_PATTERN_SAMPLES
    min_support: float = DEFAULT_MIN_SUPPORT
    min_excess_probability: float = DEFAULT_MIN_EXCESS_PROB
    min_lift: float = DEFAULT_MIN_LIFT
    objective: str = DEFAULT_OBJECTIVE


@dataclass(frozen=True)
class ProbabilityMarkovRunMetadata:
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
    min_pattern_samples: int
    min_support: float
    objective: str
    features_analyzed: list[str]
    targets: list[str]
    model_rows: int
    transitions_count: int
    top_patterns_count: int


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


def _compute_markov_transitions_for_feature(
    feature_series: pd.Series,
    target_series: pd.Series,
    feature_name: str,
    target_name: str,
    n_bins: int = DEFAULT_N_BINS,
    min_samples: int = DEFAULT_MIN_PATTERN_SAMPLES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute 1st-order sequential Markov transition probabilities and conditional label win rates."""
    clean_f = _numeric_series(feature_series)
    valid_mask = clean_f.notna() & target_series.notna()
    f_clean = clean_f[valid_mask]
    y_clean = target_series[valid_mask]

    n_total = len(f_clean)
    if n_total < (min_samples * 2):
        return [], {}

    # Assign quantile states
    states = _compute_quantile_bins(f_clean, n_bins=n_bins)
    df = pd.DataFrame(
        {
            "state_t": states,
            "state_prev": states.shift(1),
            "target": y_clean,
            "target_next": y_clean.shift(-1),  # Next bar target outcome
        }
    ).dropna()

    if df.empty:
        return [], {}

    total_valid = len(df)
    unique_classes = sorted(df["target"].unique(), key=lambda v: str(v))

    transitions: list[dict[str, Any]] = []
    feature_entropies: list[float] = []

    for c in unique_classes:
        clean_c = (
            int(c)
            if isinstance(c, (int, np.integer))
            else (float(c) if isinstance(c, (float, np.floating)) else str(c))
        )
        is_event = (df["target"] == c).astype(int)
        baseline_rate = float(is_event.mean())

        # Static conditional probabilities P(Y=c | state_t = Q_i)
        static_p = df.groupby("state_t").apply(
            lambda g: float((g["target"] == c).mean()), include_groups=False
        ).to_dict()

        # Group by (state_prev, state_t)
        for (q_prev, q_curr), group in df.groupby(["state_prev", "state_t"], observed=False):
            count_ij = len(group)
            if count_ij == 0:
                continue

            events_ij = int((group["target"] == c).sum())
            win_rate = events_ij / count_ij if count_ij > 0 else 0.0
            static_rate = static_p.get(q_curr, baseline_rate)
            excess_prob = win_rate - static_rate
            lift = win_rate / baseline_rate if baseline_rate > 0 else 1.0
            support = count_ij / total_valid

            # Fisher exact test (greater)
            contingency = [
                [events_ij, count_ij - events_ij],
                [int(is_event.sum()) - events_ij, (total_valid - int(is_event.sum())) - (count_ij - events_ij)],
            ]
            try:
                contingency[1][0] = max(0, contingency[1][0])
                contingency[1][1] = max(0, contingency[1][1])
                _, p_val = stats.fisher_exact(contingency, alternative="greater")
            except Exception:
                p_val = 1.0

            # Bayesian 95% Credible Interval
            alpha_post = events_ij + 0.5
            beta_post = (count_ij - events_ij) + 0.5
            ci_low = float(stats.beta.ppf(0.025, alpha_post, beta_post))
            ci_high = float(stats.beta.ppf(0.975, alpha_post, beta_post))

            transitions.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "target_class": clean_c,
                    "state_prev": int(q_prev),
                    "state_curr": int(q_curr),
                    "transition_label": f"Q{int(q_prev)} -> Q{int(q_curr)}",
                    "sample_count": int(count_ij),
                    "support": _round(support),
                    "target_positive_count": int(events_ij),
                    "win_rate": _round(win_rate),
                    "static_win_rate": _round(static_rate),
                    "excess_probability": _round(excess_prob),
                    "baseline_rate": _round(baseline_rate),
                    "lift": _round(lift),
                    "p_value_fisher": _round(p_val) if p_val is not None else 1.0,
                    "credible_interval_low_95": _round(ci_low),
                    "credible_interval_high_95": _round(ci_high),
                }
            )

        # Transition matrix entropy for state transition predictability
        trans_counts = pd.crosstab(df["state_prev"], df["state_t"], normalize="index").fillna(0)
        row_entropies = [
            stats.entropy(row + EPSILON) for _, row in trans_counts.iterrows() if np.sum(row) > 0
        ]
        if row_entropies:
            feature_entropies.append(float(np.mean(row_entropies)))

    meta_feature = {
        "feature": feature_name,
        "mean_transition_entropy": _round(float(np.mean(feature_entropies)))
        if feature_entropies
        else 0.0,
        "total_samples": total_valid,
    }

    return transitions, meta_feature


def _plot_markov_heatmaps(
    transitions_df: pd.DataFrame,
    output_path: Path,
    max_features: int = 4,
) -> Path:
    """Plot 2D heatmaps of state transition win rates (Q_prev x Q_curr)."""
    if transitions_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Markov transition data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Select top features with highest excess probability
    top_feats = (
        transitions_df.groupby("feature")["excess_probability"]
        .max()
        .sort_values(ascending=False)
        .head(max_features)
        .index.tolist()
    )

    n_plots = len(top_feats)
    if n_plots == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Markov features available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    nrows = 1 if n_plots <= 2 else 2
    ncols = min(n_plots, 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for idx, feat in enumerate(top_feats):
        ax = flat_axes[idx]
        sub = transitions_df[transitions_df["feature"] == feat]
        best_class = sub.groupby("target_class")["excess_probability"].max().idxmax()
        class_sub = sub[sub["target_class"] == best_class]

        pivot_wr = class_sub.pivot(index="state_prev", columns="state_curr", values="win_rate")
        pivot_cnt = class_sub.pivot(index="state_prev", columns="state_curr", values="sample_count")

        matrix_vals = pivot_wr.to_numpy()
        cax = ax.imshow(matrix_vals, cmap="YlGnBu", aspect="auto", origin="lower")
        fig.colorbar(cax, ax=ax, label="Win Rate")

        # Annotate cells
        for r in range(len(pivot_wr.index)):
            for c in range(len(pivot_wr.columns)):
                wr_val = pivot_wr.iloc[r, c] if r < len(pivot_wr) and c < len(pivot_wr.columns) else np.nan
                cnt_val = pivot_cnt.iloc[r, c] if r < len(pivot_cnt) and c < len(pivot_cnt.columns) else 0
                if not np.isnan(wr_val):
                    color = "white" if wr_val > 0.55 else "black"
                    ax.text(
                        c,
                        r,
                        f"{wr_val:.1%}\n(N={int(cnt_val)})",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=8,
                        fontweight="bold",
                    )

        ax.set_xticks(range(len(pivot_wr.columns)))
        ax.set_xticklabels([f"Q{col}" for col in pivot_wr.columns])
        ax.set_yticks(range(len(pivot_wr.index)))
        ax.set_yticklabels([f"Q{idx_val}" for idx_val in pivot_wr.index])

        target_name = class_sub["target"].iloc[0]
        ax.set_title(
            f"{feat}\nTarget: {target_name}={best_class} Transition Win Rate",
            fontsize=10,
            fontweight="bold",
        )
        ax.set_xlabel("State at t (Current Quantile)", fontsize=9)
        ax.set_ylabel("State at t-1 (Previous Quantile)", fontsize=9)

    for j in range(n_plots, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_markdown(
    metadata: ProbabilityMarkovRunMetadata,
    top_patterns_df: pd.DataFrame,
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
                {"Metric": "Quantile States (n_bins)", "Value": metadata.n_bins},
                {"Metric": "Min Pattern Samples", "Value": metadata.min_pattern_samples},
                {"Metric": "Min Support", "Value": f"{metadata.min_support:.2%}"},
                {"Metric": "Optimization Objective", "Value": metadata.objective},
                {"Metric": "Features Analyzed", "Value": len(metadata.features_analyzed)},
                {"Metric": "Targets Analyzed", "Value": ", ".join(metadata.targets)},
                {"Metric": "Evaluated Rows", "Value": metadata.model_rows},
                {"Metric": "Total Transitions Evaluated", "Value": metadata.transitions_count},
                {"Metric": "Top Alpha Patterns Discovered", "Value": metadata.top_patterns_count},
            ]
        )
    )

    insights: list[str] = []
    if not top_patterns_df.empty:
        best_p = top_patterns_df.sort_values("excess_probability", ascending=False).iloc[0]
        insights.append(
            f"- **Top Sequential Alpha Trigger**: `{best_p['feature']}` transitioning `{best_p['transition_label']}` produced a **Win Rate of {best_p['win_rate']:.1%}** (Excess Alpha: **+{best_p['excess_probability']:.1%}** over static {best_p['static_win_rate']:.1%}, Lift: `{best_p['lift']:.2f}x`) across `{best_p['sample_count']}` occurrences with Fisher p-value = `{best_p['p_value_fisher']}`."
        )
        insights.append(
            f"- **Target Class**: `{best_p['target']} = {best_p['target_class']}` with 95% Credible Interval `[{best_p['credible_interval_low_95']:.1%}, {best_p['credible_interval_high_95']:.1%}]`."
        )
    else:
        insights.append("- No sequential transition patterns met the minimum excess alpha threshold.")

    insights_text = "\n".join(insights)

    display_cols = [
        "feature",
        "transition_label",
        "target",
        "target_class",
        "win_rate",
        "static_win_rate",
        "excess_probability",
        "lift",
        "support",
        "sample_count",
        "credible_interval_low_95",
        "credible_interval_high_95",
        "p_value_fisher",
    ]
    existing_cols = [c for c in display_cols if c in top_patterns_df.columns]
    patterns_table = (
        _markdown_table(top_patterns_df[existing_cols].head(25))
        if not top_patterns_df.empty
        else "No significant sequential patterns discovered."
    )

    return f"""# Sequential State-Transition & Markov Probability Report

## Executive Summary & Key Insights

{insights_text}

## Run Metadata

{metadata_table}

## Top High-Alpha Sequential State-Transition Patterns

{patterns_table}

## Visual Markov State Transition Heatmap

![Markov Heatmap](markov_heatmap.png)

## Artifacts

- `summary.json`
- `markov_transitions.csv`
- `top_sequential_patterns.csv`
- `markov_heatmap.png`
- `report.html`
"""


def _render_html(
    metadata: ProbabilityMarkovRunMetadata,
    markdown: str,
    top_patterns_df: pd.DataFrame,
) -> str:
    display_cols = [
        "feature",
        "transition_label",
        "target",
        "target_class",
        "win_rate",
        "static_win_rate",
        "excess_probability",
        "lift",
        "support",
        "sample_count",
        "credible_interval_low_95",
        "credible_interval_high_95",
        "p_value_fisher",
    ]
    existing_cols = [c for c in display_cols if c in top_patterns_df.columns]
    patterns_html = (
        top_patterns_df[existing_cols].head(30).to_html(index=False, classes="data-table")
        if not top_patterns_df.empty
        else "<p>No significant sequential patterns discovered.</p>"
    )

    max_excess = (
        f"+{top_patterns_df['excess_probability'].max():.1%}"
        if not top_patterns_df.empty and top_patterns_df["excess_probability"].max() is not None
        else "N/A"
    )
    max_win_rate = (
        f"{top_patterns_df['win_rate'].max():.1%}"
        if not top_patterns_df.empty and top_patterns_df["win_rate"].max() is not None
        else "N/A"
    )
    max_lift = (
        f"{top_patterns_df['lift'].max():.2f}x"
        if not top_patterns_df.empty and top_patterns_df["lift"].max() is not None
        else "N/A"
    )
    total_patterns = str(len(top_patterns_df))

    details = _html_markdown_details(markdown)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Markov State-Transition & Sequential Probability Report</title>
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
    </style>
</head>
<body>
    <h1>Sequential State-Transition & Markov Probability Report</h1>

    <div class="metrics-banner">
        <div class="metric-card accent">
            <div class="metric-title">Max Excess Alpha (ΔP)</div>
            <div class="metric-value">{max_excess}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Peak Pattern Win Rate</div>
            <div class="metric-value">{max_win_rate}</div>
        </div>
        <div class="metric-card warning">
            <div class="metric-title">Max Pattern Lift</div>
            <div class="metric-value">{max_lift}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Patterns Discovered</div>
            <div class="metric-value">{total_patterns}</div>
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
                <div class="meta-label">Quantile Bins (States)</div>
                <div class="meta-val">{metadata.n_bins} (5x5 Transition Matrix)</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Min Pattern Samples</div>
                <div class="meta-val">{metadata.min_pattern_samples} samples ({metadata.min_support:.1%})</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Optimization Objective</div>
                <div class="meta-val">{metadata.objective}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Evaluated Rows</div>
                <div class="meta-val">{metadata.model_rows}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Top High-Alpha Sequential State-Transition Patterns</h2>
        <div class="table-container">
            {patterns_html}
        </div>
    </div>

    <div class="card">
        <h2>Visual Markov State Transition Heatmap</h2>
        <img class="chart-img" src="markov_heatmap.png" alt="Markov Heatmap Visualization"/>
    </div>

    {details}
</body>
</html>
"""


class ProbabilityMarkovModule:
    name = "probability_markov"

    def __init__(
        self,
        config: ProbabilityMarkovConfig | None = None,
        progress: bool | None = None,
        n_bins: int | None = None,
        min_pattern_samples: int | None = None,
        min_support: float | None = None,
        min_excess_probability: float | None = None,
        min_lift: float | None = None,
        objective: str | None = None,
    ) -> None:
        self.progress = progress
        if config is not None:
            base_cfg = config
        else:
            mod_cfg = get_module_config("probability_markov")
            base_cfg = ProbabilityMarkovConfig(
                n_bins=int(mod_cfg.get("n_bins", mod_cfg.get("n_quantiles", DEFAULT_N_BINS))),
                min_pattern_samples=int(
                    mod_cfg.get(
                        "min_pattern_samples",
                        mod_cfg.get("min_transition_samples", DEFAULT_MIN_PATTERN_SAMPLES),
                    )
                ),
                min_support=float(mod_cfg.get("min_support", DEFAULT_MIN_SUPPORT)),
                min_excess_probability=float(
                    mod_cfg.get("min_excess_probability", DEFAULT_MIN_EXCESS_PROB)
                ),
                min_lift=float(mod_cfg.get("min_lift", DEFAULT_MIN_LIFT)),
                objective=str(mod_cfg.get("objective", DEFAULT_OBJECTIVE)),
            )

        # Check environment variable overrides
        env_min_samples = os.environ.get("MARKOV_MIN_SAMPLES")
        env_min_support = os.environ.get("MARKOV_MIN_SUPPORT")
        env_n_bins = os.environ.get("MARKOV_N_BINS")
        env_excess_prob = os.environ.get("MARKOV_MIN_EXCESS_PROB")
        env_min_lift = os.environ.get("MARKOV_MIN_LIFT")
        env_objective = os.environ.get("MARKOV_OBJECTIVE")

        self.n_bins = (
            n_bins
            if n_bins is not None
            else (int(env_n_bins) if env_n_bins else base_cfg.n_bins)
        )
        self.min_pattern_samples = (
            min_pattern_samples
            if min_pattern_samples is not None
            else (int(env_min_samples) if env_min_samples else base_cfg.min_pattern_samples)
        )
        self.min_support = (
            min_support
            if min_support is not None
            else (float(env_min_support) if env_min_support else base_cfg.min_support)
        )
        self.min_excess_probability = (
            min_excess_probability
            if min_excess_probability is not None
            else (float(env_excess_prob) if env_excess_prob else base_cfg.min_excess_probability)
        )
        self.min_lift = (
            min_lift
            if min_lift is not None
            else (float(env_min_lift) if env_min_lift else base_cfg.min_lift)
        )
        self.objective = (
            objective
            if objective is not None
            else (env_objective if env_objective else base_cfg.objective)
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

        with ModuleProgress(self.name, total=4, enabled=self.progress) as progress_bar:
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

            # Filter valid discrete targets
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

            all_transitions: list[dict[str, Any]] = []
            all_meta_features: list[dict[str, Any]] = []

            # Adjust min samples adaptively based on min_pattern_samples and min_support
            eff_min_samples = max(
                10,
                max(self.min_pattern_samples, int(self.min_support * len(model_frame))),
            )
            eff_min_samples = min(eff_min_samples, max(10, int(len(model_frame) * 0.15)))

            for target_col in valid_targets:
                for feat in numeric_features:
                    t_list, meta_f = _compute_markov_transitions_for_feature(
                        feature_series=model_frame[feat],
                        target_series=model_frame[target_col],
                        feature_name=feat,
                        target_name=target_col,
                        n_bins=self.n_bins,
                        min_samples=eff_min_samples,
                    )
                    all_transitions.extend(t_list)
                    if meta_f:
                        all_meta_features.append(meta_f)

            transitions_df = pd.DataFrame(all_transitions)
            progress_bar.step("markov_transitions")

            # Extract top patterns
            if not transitions_df.empty:
                top_patterns = transitions_df[
                    (transitions_df["excess_probability"] >= self.min_excess_probability)
                    & (transitions_df["sample_count"] >= eff_min_samples)
                    & (transitions_df["lift"] >= self.min_lift)
                ].copy()

                if self.objective == "support_weighted":
                    top_patterns["utility_score"] = top_patterns["excess_probability"] * np.sqrt(
                        top_patterns["sample_count"]
                    )
                    top_patterns.sort_values(
                        by=["utility_score", "excess_probability", "lift", "sample_count"],
                        ascending=[False, False, False, False],
                        inplace=True,
                    )
                else:
                    top_patterns.sort_values(
                        by=["excess_probability", "lift", "sample_count"],
                        ascending=[False, False, False],
                        inplace=True,
                    )
            else:
                top_patterns = pd.DataFrame()

            # Generate Artifacts
            plot_path = run_dir / "markov_heatmap.png"
            _plot_markov_heatmaps(transitions_df, plot_path)

            transitions_csv_path = _write_csv(
                run_dir / "markov_transitions.csv",
                transitions_df,
            )

            top_patterns_csv_path = _write_csv(
                run_dir / "top_sequential_patterns.csv",
                top_patterns,
            )

            metadata = ProbabilityMarkovRunMetadata(
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
                min_pattern_samples=eff_min_samples,
                min_support=self.min_support,
                objective=self.objective,
                features_analyzed=numeric_features,
                targets=valid_targets,
                model_rows=len(model_frame),
                transitions_count=len(transitions_df),
                top_patterns_count=len(top_patterns),
            )

            def _clean_pattern_for_json(r: dict[str, Any]) -> dict[str, Any]:
                return {
                    "feature": str(r.get("feature", "")),
                    "transition_label": str(r.get("transition_label", "")),
                    "target": str(r.get("target", "")),
                    "target_class": r.get("target_class"),
                    "sample_count": int(r.get("sample_count", 0)),
                    "support": float(r.get("support", 0.0)),
                    "win_rate": float(r.get("win_rate", 0.0)),
                    "static_win_rate": float(r.get("static_win_rate", 0.0)),
                    "excess_probability": float(r.get("excess_probability", 0.0)),
                    "lift": float(r.get("lift", 1.0)),
                    "p_value_fisher": float(r.get("p_value_fisher", 1.0)),
                    "credible_interval_low_95": float(r.get("credible_interval_low_95", 0.0)),
                    "credible_interval_high_95": float(r.get("credible_interval_high_95", 1.0)),
                }

            summary_payload: dict[str, Any] = {
                **asdict(metadata),
                "top_patterns": [
                    _clean_pattern_for_json(r)
                    for r in top_patterns.head(10).to_dict(orient="records")
                ]
                if not top_patterns.empty
                else [],
                "summary_metrics": {
                    "total_transitions_evaluated": len(transitions_df),
                    "alpha_patterns_count": len(top_patterns),
                    "max_excess_probability": _round(
                        float(top_patterns["excess_probability"].max())
                    )
                    if not top_patterns.empty
                    else 0.0,
                    "max_win_rate": _round(float(top_patterns["win_rate"].max()))
                    if not top_patterns.empty
                    else 0.0,
                    "max_lift": _round(float(top_patterns["lift"].max()))
                    if not top_patterns.empty
                    else 1.0,
                },
            }
            summary_json_path = _write_json(run_dir / "summary.json", summary_payload)

            markdown_report = _render_markdown(metadata, top_patterns)
            report_md_path = run_dir / "report.md"
            report_md_path.write_text(markdown_report, encoding="utf-8")

            html_report = _render_html(metadata, markdown_report, top_patterns)
            report_html_path = run_dir / "report.html"
            report_html_path.write_text(html_report, encoding="utf-8")

            progress_bar.step("reports")

            artifacts = [
                summary_json_path,
                transitions_csv_path,
                top_patterns_csv_path,
                plot_path,
                report_md_path,
                report_html_path,
            ]

            return ModuleResult(report_dir=run_dir, artifacts=artifacts)
