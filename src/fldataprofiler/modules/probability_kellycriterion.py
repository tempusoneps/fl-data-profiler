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
DEFAULT_PAYOFF_RATIO = 1.5  # Win/Loss Payoff Ratio (R:R = 1.5)
EPSILON = 1e-9


@dataclass(frozen=True)
class ProbabilityKellyCriterionRunMetadata:
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
    payoff_ratio_b: float
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


def _compute_kelly_metrics(
    p_win: float,
    payoff_b: float,
) -> tuple[float, float, float, float, float, str]:
    """Calculate Kelly Fraction, Half-Kelly, Quarter-Kelly, Expected Value (EV), and Growth Rate."""
    p = float(np.clip(p_win, 0.0, 1.0))
    q = 1.0 - p
    b = max(float(payoff_b), 1e-4)

    # Expected Value (EV in units of 1R risk)
    ev = p * b - q * 1.0

    # Kelly Fraction: f* = (p*b - q) / b = p - (1-p)/b
    raw_kelly = ev / b

    # Fractional Kelly sizing (clipped at 0 if no edge)
    half_kelly = max(0.0, 0.5 * raw_kelly)
    quarter_kelly = max(0.0, 0.25 * raw_kelly)

    # Expected compound growth rate g = p*ln(1 + b*f) + q*ln(1 - f)
    if raw_kelly > 0.0:
        # Cap f at 0.99 for numerical stability
        f_safe = min(raw_kelly, 0.99)
        growth_rate = float(
            p * np.log(1.0 + b * f_safe) + q * np.log(max(1.0 - f_safe, EPSILON))
        )
    else:
        growth_rate = 0.0

    # Action recommendation
    if raw_kelly >= 0.15:
        recommendation = "STRONG_BET"
    elif raw_kelly >= 0.05:
        recommendation = "MODERATE_BET"
    elif raw_kelly > 0.0:
        recommendation = "SMALL_BET"
    else:
        recommendation = "AVOID_NO_BET"

    return raw_kelly, half_kelly, quarter_kelly, ev, growth_rate, recommendation


def _compute_feature_target_kelly_profiles(
    feature_series: pd.Series,
    target_series: pd.Series,
    feature_name: str,
    target_name: str,
    n_bins: int = DEFAULT_N_BINS,
    payoff_ratio_b: float = DEFAULT_PAYOFF_RATIO,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Calculate Kelly Criterion fraction, Half-Kelly, EV, and Growth Rate across quantile bins."""
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

    # Precompute per-bin values
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

        bin_stats[k] = {
            "n_k": n_k,
            "val_min": val_min,
            "val_max": val_max,
            "val_mean": val_mean,
            "class_counts": class_counts,
        }

    score_rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []

    breakeven_p = 1.0 / (1.0 + payoff_ratio_b)

    for c in unique_classes:
        total_events = int((df["y"] == c).sum())
        base_win_rate = total_events / n_total if n_total > 0 else 0.0

        kelly_fractions: list[float] = []
        half_kellys: list[float] = []
        ev_list: list[float] = []
        win_probs: list[float] = []

        favorable_bins = 0

        for k in bin_indices:
            stats_k = bin_stats.get(k)
            if not stats_k:
                continue

            n_k = int(stats_k["n_k"])
            events_k = int(stats_k["class_counts"][c])
            win_prob_k = events_k / n_k if n_k > 0 else 0.0

            (
                raw_kelly,
                half_kelly,
                quarter_kelly,
                ev,
                growth_rate,
                recommendation,
            ) = _compute_kelly_metrics(win_prob_k, payoff_ratio_b)

            if raw_kelly > 0.0:
                favorable_bins += 1

            kelly_fractions.append(raw_kelly)
            half_kellys.append(half_kelly)
            ev_list.append(ev)
            win_probs.append(win_prob_k)

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
                    "win_prob": _round(win_prob_k),
                    "breakeven_prob": _round(breakeven_p),
                    "payoff_ratio_b": _round(payoff_ratio_b),
                    "expected_value_ev": _round(ev),
                    "kelly_fraction_f": _round(raw_kelly),
                    "half_kelly": _round(half_kelly),
                    "quarter_kelly": _round(quarter_kelly),
                    "expected_growth_rate": _round(growth_rate),
                    "action_recommendation": recommendation,
                }
            )

        kelly_arr = np.array(kelly_fractions, dtype=float)
        prob_arr = np.array(win_probs, dtype=float)
        ev_arr = np.array(ev_list, dtype=float)

        max_kelly = float(kelly_arr.max()) if len(kelly_arr) > 0 else 0.0
        min_kelly = float(kelly_arr.min()) if len(kelly_arr) > 0 else 0.0
        kelly_spread = max_kelly - min_kelly

        max_half_kelly = float(np.max(half_kellys)) if half_kellys else 0.0
        max_ev = float(ev_arr.max()) if len(ev_arr) > 0 else 0.0
        ev_spread = float(ev_arr.max() - ev_arr.min()) if len(ev_arr) > 0 else 0.0
        prob_spread = float(prob_arr.max() - prob_arr.min()) if len(prob_arr) > 0 else 0.0

        # Monotonicity on Kelly fraction
        if len(kelly_arr) < 2 or np.all(np.isclose(kelly_arr, kelly_arr[0])):
            kelly_mono = 0.0
        else:
            spearman_res = stats.spearmanr(np.arange(1, len(kelly_arr) + 1), kelly_arr)
            corr = getattr(spearman_res, "statistic", getattr(spearman_res, "correlation", 0.0))
            kelly_mono = 0.0 if np.isnan(corr) else float(corr)

        # Composite Kelly Ranking Score = max(0, max_kelly) * (1 + |mono|) * ev_spread
        kelly_rank_score = (
            max(0.0, max_kelly) * (1.0 + abs(kelly_mono)) * (1.0 + max(0.0, ev_spread))
        )

        score_rows.append(
            {
                "feature": feature_name,
                "target": target_name,
                "target_class": c,
                "kelly_rank_score": _round(kelly_rank_score),
                "max_kelly_fraction": _round(max_kelly),
                "max_half_kelly": _round(max_half_kelly),
                "max_expected_value": _round(max_ev),
                "kelly_spread": _round(kelly_spread),
                "win_prob_spread": _round(prob_spread),
                "base_win_rate": _round(base_win_rate),
                "breakeven_prob": _round(breakeven_p),
                "favorable_bins_count": int(favorable_bins),
                "kelly_monotonicity": _round(kelly_mono),
                "payoff_ratio_b": _round(payoff_ratio_b),
                "n_bins": int(len(kelly_fractions)),
                "sample_count": int(n_total),
            }
        )

    return score_rows, quantile_rows


def _plot_kelly_distribution(
    scores_df: pd.DataFrame,
    quantiles_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot Kelly Criterion bet sizing and Win Probabilities against Breakeven Rate for top features."""
    if scores_df.empty or quantiles_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Kelly data available", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Identify top unique feature-target relationships by max kelly_rank_score
    top_pairs_df = (
        scores_df.sort_values(
            ["kelly_rank_score", "max_expected_value"], ascending=[False, False]
        )
        .drop_duplicates(subset=["feature", "target"])
        .head(4)
    )

    n_plots = len(top_pairs_df)
    if n_plots == 1:
        nrows, ncols = 1, 1
        figsize = (9, 5)
    elif n_plots == 2:
        nrows, ncols = 1, 2
        figsize = (16, 5)
    else:
        nrows, ncols = 2, 2
        figsize = (16, 10)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

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

        best_score_row = feat_scores.sort_values("kelly_rank_score", ascending=False).iloc[0]
        c = best_score_row["target_class"]
        payoff_b = best_score_row["payoff_ratio_b"]
        breakeven_p = best_score_row["breakeven_prob"]

        c_quantiles = feat_quantiles[feat_quantiles["target_class"] == c].sort_values("bin_index")
        bins = c_quantiles["bin_index"].to_numpy()
        kellys = c_quantiles["kelly_fraction_f"].to_numpy()
        half_kellys = c_quantiles["half_kelly"].to_numpy()
        win_probs = c_quantiles["win_prob"].to_numpy()

        # Bar colors: Green for positive Kelly edge, Slate for negative/zero edge
        colors = ["#16a34a" if k > 0 else "#94a3b8" for k in kellys]

        # Primary axis: Kelly Bet Size
        bars = ax.bar(
            bins,
            kellys,
            color=colors,
            alpha=0.75,
            edgecolor="#0f172a",
            width=0.7,
            label="Full Kelly f*",
        )
        ax.plot(
            bins,
            half_kellys,
            color="#2563eb",
            marker="o",
            linewidth=1.8,
            markersize=4,
            label="Half-Kelly (Position Size)",
        )

        ax.axhline(0.0, color="#334155", linestyle="-", linewidth=1.0, alpha=0.7)
        ax.set_ylabel("Kelly Fraction (Position Size %)", color="#0f172a", fontsize=10)
        ax.set_xlabel(f"Quantile Bin (1 - {len(bins)})", fontsize=10)

        # Secondary axis: Win Probability & Breakeven Line
        ax2 = ax.twinx()
        ax2.plot(
            bins,
            win_probs,
            color="#dc2626",
            linestyle="--",
            marker="s",
            markersize=3.5,
            linewidth=1.5,
            label="P(Win | Bin)",
        )
        ax2.axhline(
            breakeven_p,
            color="#d97706",
            linestyle=":",
            linewidth=1.6,
            label=f"Breakeven ({breakeven_p:.2f})",
        )
        ax2.set_ylabel("Win Probability", color="#dc2626", fontsize=10)
        ax2.set_ylim(0.0, 1.05)

        max_k = float(best_score_row["max_kelly_fraction"])
        max_ev = float(best_score_row["max_expected_value"])
        ax.set_title(
            f"{feature} vs {target_name} ({c}) [R:R = {payoff_b}:1]\nMax Kelly: {max_k*100:.1f}% | Max EV: +{max_ev:.2f}R",
            fontsize=11,
            fontweight="bold",
        )

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    for j in range(n_plots, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_markdown(
    metadata: ProbabilityKellyCriterionRunMetadata,
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
                {"Metric": "Assumed Payoff Ratio (b)", "Value": f"{metadata.payoff_ratio_b}:1"},
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
        top_kelly = scores_df.sort_values("max_kelly_fraction", ascending=False).iloc[0]
        insights.append(
            f"- **Highest Capital Allocation Edge**: `{top_kelly['feature']}` for target `{top_kelly['target']}` (Class `{top_kelly['target_class']}`) allows peak Full Kelly sizing of `{_round(float(top_kelly['max_kelly_fraction'])*100)}%` (Half-Kelly: `{_round(float(top_kelly['max_half_kelly'])*100)}%`)."
        )
        top_ev = scores_df.sort_values("max_expected_value", ascending=False).iloc[0]
        insights.append(
            f"- **Maximum Expected Value (EV)**: `{top_ev['feature']}` reached peak expected return of `+{top_ev['max_expected_value']}R` per trade at favorable bins."
        )
        top_fav = scores_df.sort_values("favorable_bins_count", ascending=False).iloc[0]
        insights.append(
            f"- **Broadest Positive Edge Area**: `{top_fav['feature']}` exhibited positive Kelly edge ($f^* > 0$) across `{top_fav['favorable_bins_count']}/20` quantile bins."
        )
    else:
        insights.append(
            "- No valid numeric features or categorical targets available for Kelly Criterion profiling."
        )

    insights_text = "\n".join(insights)

    return f"""# Kelly Criterion & Optimal Sizing Profiling Report

## Executive Summary & Key Insights

{insights_text}

## Run Metadata

{metadata_table}

## Top Feature Kelly Criterion & Expected Value Scores

{top_scores_table}

## Visual Kelly Position Sizing & Win Probability

![Kelly Criterion Distribution](kelly_distribution.png)

## Artifacts

- `summary.json`
- `kelly_probability_scores.csv`
- `quantile_kelly_probabilities.csv`
- `kelly_distribution.png`
- `report.html`
"""


def _render_html(
    metadata: ProbabilityKellyCriterionRunMetadata,
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
    <title>Kelly Criterion & Position Sizing Profiling Report</title>
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
    <h1>Kelly Criterion & Position Sizing Profiling Report</h1>

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
                <div class="meta-label">Payoff Ratio (b)</div>
                <div class="meta-val">{metadata.payoff_ratio_b}:1</div>
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
        <h2>Top Feature Kelly & Expected Value Scores</h2>
        <div class="table-container">
            {scores_html}
        </div>
    </div>

    <div class="card">
        <h2>Kelly Position Sizing & Win Probability Visualization</h2>
        <img class="chart-img" src="kelly_distribution.png" alt="Kelly Criterion Distribution Visualization"/>
    </div>

    {details}
</body>
</html>
"""


class ProbabilityKellyCriterionModule:
    name = "probability_kellycriterion"

    def __init__(
        self,
        progress: bool | None = None,
        n_bins: int = DEFAULT_N_BINS,
        payoff_ratio_b: float = DEFAULT_PAYOFF_RATIO,
    ) -> None:
        self.progress = progress
        self.n_bins = n_bins
        self.payoff_ratio_b = payoff_ratio_b

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
                    scores, quantiles = _compute_feature_target_kelly_profiles(
                        model_frame[feature_col],
                        model_frame[target_col],
                        feature_col,
                        target_col,
                        n_bins=self.n_bins,
                        payoff_ratio_b=self.payoff_ratio_b,
                    )
                    all_score_rows.extend(scores)
                    all_quantile_rows.extend(quantiles)

            scores_df = pd.DataFrame(all_score_rows)
            quantiles_df = pd.DataFrame(all_quantile_rows)

            if not scores_df.empty:
                scores_df = scores_df.sort_values(
                    ["kelly_rank_score", "max_kelly_fraction"],
                    ascending=[False, False],
                ).reset_index(drop=True)

            progress_bar.step("kelly_profiles")

            plot_path = run_dir / "kelly_distribution.png"
            _plot_kelly_distribution(scores_df, quantiles_df, plot_path)
            progress_bar.step("plots")

            metadata = ProbabilityKellyCriterionRunMetadata(
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
                payoff_ratio_b=self.payoff_ratio_b,
                features=numeric_features,
                targets=valid_targets,
                model_rows=len(model_frame),
            )

            scores_csv_path = _write_csv(run_dir / "kelly_probability_scores.csv", scores_df)
            quantiles_csv_path = _write_csv(
                run_dir / "quantile_kelly_probabilities.csv", quantiles_df
            )

            summary_payload: dict[str, object] = {
                **asdict(metadata),
                "top_features": scores_df.head(10).to_dict(orient="records")
                if not scores_df.empty
                else [],
                "summary_metrics": {
                    "features_evaluated": len(numeric_features),
                    "targets_evaluated": len(valid_targets),
                    "max_kelly_fraction": _round(float(scores_df["max_kelly_fraction"].max()))
                    if not scores_df.empty
                    else 0.0,
                    "max_half_kelly": _round(float(scores_df["max_half_kelly"].max()))
                    if not scores_df.empty
                    else 0.0,
                    "max_expected_value": _round(float(scores_df["max_expected_value"].max()))
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
