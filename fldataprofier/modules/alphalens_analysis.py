from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _date_columns,
    _markdown_table,
    _merge_inputs,
    _numeric_series,
    _read_table_with_date_index,
    _round,
    _sample_rows,
    _write_csv,
    _write_json,
)

DEFAULT_HORIZONS = [1, 5, 15, 60]
DEFAULT_QUANTILES = 5
MAX_SAMPLED_ROWS = 50_000
MAX_FEATURES_TO_SCORE = 150


@dataclass(frozen=True)
class AlphalensRunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    evaluated_rows: int
    evaluated_features_count: int
    horizons: list[int]
    quantiles_count: int
    primary_return_col: str


def _find_price_or_return_columns(
    merged: pd.DataFrame,
    label_columns: list[str],
    targets: list[str] | None,
) -> tuple[pd.Series | None, dict[str, pd.Series]]:
    """
    Identifies or computes forward returns from Close price or target return columns.
    Returns: (price_series, dict_of_forward_returns)
    """
    price_col = None
    for col in merged.columns:
        if str(col).lower() in {"close", "close_price", "price"}:
            price_col = col
            break

    fwd_returns: dict[str, pd.Series] = {}

    if price_col is not None:
        price_series = _numeric_series(merged[price_col])
        for h in DEFAULT_HORIZONS:
            # Forward return from t to t+h
            ret = price_series.pct_change(periods=h).shift(-h)
            fwd_returns[f"fwd_ret_{h}"] = ret
    else:
        price_series = None

    # Check for existing continuous return columns in labels or explicit targets
    candidate_targets = targets if targets else label_columns
    for col in candidate_targets:
        if col in merged.columns and col != price_col:
            numeric_col = _numeric_series(merged[col])
            # Check if numeric with sufficient unique values (continuous-like)
            if numeric_col.dropna().nunique() > 10:
                fwd_returns[col] = numeric_col

    return price_series, fwd_returns


def _compute_rolling_or_fold_ic(
    factor: pd.Series,
    target_ret: pd.Series,
    n_folds: int = 10,
) -> tuple[float, float, float, float, float]:
    """
    Computes time-series IC statistics: Mean IC, IC Std, Information Ratio (IR), p-value, and positive ratio.
    """
    valid = pd.DataFrame({"f": factor, "r": target_ret}).dropna()
    if len(valid) < 30:
        return 0.0, 0.0, 0.0, 1.0, 0.0

    fold_size = len(valid) // n_folds
    if fold_size < 10:
        # Fallback to single global rank IC
        rho, p_val = stats.spearmanr(valid["f"], valid["r"])
        rho = 0.0 if np.isnan(rho) else float(rho)
        return rho, 0.0, rho, float(p_val) if not np.isnan(p_val) else 1.0, 1.0 if rho > 0 else 0.0

    fold_ics: list[float] = []
    for i in range(n_folds):
        chunk = valid.iloc[i * fold_size : (i + 1) * fold_size]
        if chunk["f"].nunique() > 1 and chunk["r"].nunique() > 1:
            rho, _ = stats.spearmanr(chunk["f"], chunk["r"])
            if not np.isnan(rho):
                fold_ics.append(float(rho))

    if not fold_ics:
        return 0.0, 0.0, 0.0, 1.0, 0.0

    mean_ic = float(np.mean(fold_ics))
    std_ic = float(np.std(fold_ics, ddof=1)) if len(fold_ics) > 1 else 0.0
    ir = mean_ic / (std_ic + 1e-9)
    pos_ratio = float(np.mean([1.0 if x > 0 else 0.0 for x in fold_ics]))

    # t-test for mean IC != 0
    if std_ic > 1e-9 and len(fold_ics) > 1:
        t_stat, p_val = stats.ttest_1samp(fold_ics, 0.0)
    else:
        p_val = 1.0

    return mean_ic, std_ic, ir, float(p_val) if not np.isnan(p_val) else 1.0, pos_ratio


def _evaluate_feature_factor(
    feature_name: str,
    feature_data: pd.Series,
    fwd_returns: dict[str, pd.Series],
    primary_ret_name: str,
    n_quantiles: int = DEFAULT_QUANTILES,
) -> tuple[dict[str, object], dict[int, float], pd.DataFrame]:
    """
    Evaluates a single factor: Quantiles, Forward Returns, IC across horizons, Spread, Monotonicity.
    """
    f = _numeric_series(feature_data)
    if f.dropna().nunique() < 5:
        return {}, {}, pd.DataFrame()

    primary_ret = fwd_returns[primary_ret_name]
    valid_mask = f.notna() & primary_ret.notna()
    if valid_mask.sum() < 30:
        return {}, {}, pd.DataFrame()

    f_valid = f[valid_mask]
    r_valid = primary_ret[valid_mask]

    # Quantile grouping
    try:
        quantiles = pd.qcut(f_valid, q=n_quantiles, labels=False, duplicates="drop")
        n_actual_quantiles = quantiles.nunique()
    except Exception:
        return {}, {}, pd.DataFrame()

    if n_actual_quantiles < 2:
        return {}, {}, pd.DataFrame()

    # Calculate Mean Return per Quantile for primary return
    grouped = r_valid.groupby(quantiles)
    q_mean_returns: dict[int, float] = {int(k) + 1: float(v) for k, v in grouped.mean().items()}

    q_min = min(q_mean_returns.keys())
    q_max = max(q_mean_returns.keys())
    spread = q_mean_returns[q_max] - q_mean_returns[q_min]

    # Quantile Monotonicity (Spearman correlation between quantile ID and mean return)
    q_indices = list(q_mean_returns.keys())
    q_returns = [q_mean_returns[k] for k in q_indices]
    if len(q_indices) > 2:
        monotonicity, _ = stats.spearmanr(q_indices, q_returns)
        monotonicity = 0.0 if np.isnan(monotonicity) else float(monotonicity)
    else:
        monotonicity = 1.0 if spread > 0 else -1.0

    # IC stats on primary return
    mean_ic, std_ic, ir, p_val, pos_ratio = _compute_rolling_or_fold_ic(f_valid, r_valid)

    # Global Rank IC and Pearson IC
    rank_ic, _ = stats.spearmanr(f_valid, r_valid)
    pearson_ic, _ = stats.pearsonr(f_valid, r_valid)
    rank_ic = 0.0 if np.isnan(rank_ic) else float(rank_ic)
    pearson_ic = 0.0 if np.isnan(pearson_ic) else float(pearson_ic)

    # Multi-horizon IC
    horizon_metrics: dict[str, float] = {}
    for ret_name, ret_series in fwd_returns.items():
        v_mask = f.notna() & ret_series.notna()
        if v_mask.sum() >= 20:
            h_ic, _ = stats.spearmanr(f[v_mask], ret_series[v_mask])
            horizon_metrics[f"ic_{ret_name}"] = 0.0 if np.isnan(h_ic) else float(h_ic)

    metrics: dict[str, object] = {
        "feature": feature_name,
        "primary_return": primary_ret_name,
        "samples": int(valid_mask.sum()),
        "rank_ic": _round(rank_ic),
        "pearson_ic": _round(pearson_ic),
        "mean_ic": _round(mean_ic),
        "ic_std": _round(std_ic),
        "ir": _round(ir),
        "ic_p_value": _round(p_val),
        "positive_ic_ratio": _round(pos_ratio),
        "long_short_spread": _round(spread),
        "monotonicity_score": _round(monotonicity),
        "q_bottom_mean_return": _round(q_mean_returns.get(q_min, 0.0)),
        "q_top_mean_return": _round(q_mean_returns.get(q_max, 0.0)),
        **{k: _round(v) for k, v in horizon_metrics.items()},
    }

    # Quantile DataFrame summary
    q_rows = []
    for q_idx in sorted(q_mean_returns.keys()):
        q_rows.append({
            "feature": feature_name,
            "quantile": q_idx,
            "mean_return": _round(q_mean_returns[q_idx]),
            "samples": int((quantiles == (q_idx - 1)).sum()),
        })
    q_df = pd.DataFrame(q_rows)

    return metrics, q_mean_returns, q_df


def _write_ic_decay_chart(
    path: Path,
    metrics_df: pd.DataFrame,
    horizon_names: list[str],
) -> Path | None:
    if metrics_df.empty or not horizon_names:
        return None

    top_features = metrics_df.head(5)
    ic_cols = [f"ic_{h}" for h in horizon_names if f"ic_{h}" in top_features.columns]
    if not ic_cols:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(ic_cols))
    width = 0.15

    for i, (_, row) in enumerate(top_features.iterrows()):
        values = [row.get(col, 0.0) or 0.0 for col in ic_cols]
        ax.bar(x + i * width, values, width, label=f"{row['feature']} (IR={row.get('ir', 0.0)})")

    ax.set_xlabel("Forward Return Horizons")
    ax.set_ylabel("Rank Information Coefficient (IC)")
    ax.set_title("Multi-Horizon IC Decay: Top 5 Factors")
    ax.set_xticks(x + width * (len(top_features) - 1) / 2)
    clean_labels = [col.replace("ic_fwd_ret_", "Horizon ").replace("ic_", "") for col in ic_cols]
    ax.set_xticklabels(clean_labels)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_quantile_returns_chart(
    path: Path,
    top_quantile_dfs: list[tuple[str, pd.DataFrame]],
) -> Path | None:
    if not top_quantile_dfs:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    n_groups = len(top_quantile_dfs)
    width = 0.8 / n_groups

    for i, (feat_name, q_df) in enumerate(top_quantile_dfs):
        if q_df.empty:
            continue
        x = np.arange(len(q_df))
        ax.bar(
            x + i * width,
            q_df["mean_return"] * 100,  # in percentage
            width,
            label=feat_name,
        )

    ax.set_xlabel("Factor Quantile (Q1 = Lowest Value, Q5 = Highest Value)")
    ax.set_ylabel("Mean Forward Return (%)")
    ax.set_title("Quantile Mean Forward Returns (Top Alpha Factors)")
    first_df = top_quantile_dfs[0][1]
    if not first_df.empty:
        ax.set_xticks(np.arange(len(first_df)) + width * (n_groups - 1) / 2)
        ax.set_xticklabels([f"Q{int(q)}" for q in first_df["quantile"]])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_cumulative_spread_chart(
    path: Path,
    merged: pd.DataFrame,
    top_feature: str,
    primary_ret_name: str,
    fwd_returns: dict[str, pd.Series],
) -> Path | None:
    if top_feature not in merged.columns or primary_ret_name not in fwd_returns:
        return None

    f = _numeric_series(merged[top_feature])
    ret = fwd_returns[primary_ret_name]
    valid = pd.DataFrame({"f": f, "r": ret}).dropna()
    if len(valid) < 50:
        return None

    try:
        quantiles = pd.qcut(valid["f"], q=DEFAULT_QUANTILES, labels=False, duplicates="drop")
    except Exception:
        return None

    q_min = quantiles.min()
    q_max = quantiles.max()

    r_top = np.where(quantiles == q_max, valid["r"], 0.0)
    r_bottom = np.where(quantiles == q_min, valid["r"], 0.0)
    r_spread = r_top - r_bottom

    cum_top = np.cumprod(1 + r_top) - 1
    cum_bottom = np.cumprod(1 + r_bottom) - 1
    cum_spread = np.cumprod(1 + r_spread) - 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cum_top * 100, label=f"Top Quantile (Q{q_max+1}) Long", color="#2ca02c", linewidth=1.5)
    ax.plot(cum_bottom * 100, label=f"Bottom Quantile (Q{q_min+1}) Short-Target", color="#d62728", linewidth=1.5)
    ax.plot(cum_spread * 100, label="Long-Short Spread Portfolio (Q_top - Q_bot)", color="#1f77b4", linewidth=2.0, linestyle="--")

    ax.set_xlabel("Time Index / Bar Sequence")
    ax.set_ylabel("Cumulative Compounded Return (%)")
    ax.set_title(f"Cumulative Return Simulation: Top Factor `{top_feature}`")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _generate_alphalens_insights(
    metrics_df: pd.DataFrame,
    primary_ret_name: str,
) -> list[str]:
    insights: list[str] = []
    if metrics_df.empty:
        return ["No valid features could be evaluated for factor analysis."]

    top_by_ir = metrics_df.sort_values(by="ir", key=abs, ascending=False).iloc[0]
    top_by_spread = metrics_df.sort_values(by="long_short_spread", key=abs, ascending=False).iloc[0]
    top_monotonic = metrics_df.sort_values(by="monotonicity_score", ascending=False).iloc[0]

    # Best overall
    insights.append(
        f"⭐ **Top Alpha Factor (Highest Information Ratio)**: `{top_by_ir['feature']}` with **IR = {top_by_ir['ir']}** (Mean IC = {top_by_ir['mean_ic']}, Rank IC = {top_by_ir['rank_ic']}, p-value = {top_by_ir['ic_p_value']})."
    )

    # Long-Short Spread
    spread_direction = "Positive (Long High / Short Low)" if (top_by_spread['long_short_spread'] or 0) > 0 else "Inverted (Short High / Long Low)"
    insights.append(
        f"💰 **Widest Long-Short Spread**: `{top_by_spread['feature']}` with **Spread = {top_by_spread['long_short_spread']}** ({spread_direction})."
    )

    # Monotonicity
    if (top_monotonic['monotonicity_score'] or 0) >= 0.8:
        insights.append(
            f"📈 **Strong Monotonic Factor**: `{top_monotonic['feature']}` has Monotonicity Score = **{top_monotonic['monotonicity_score']}** (Quantiles scale predictably with returns)."
        )

    # Negative Alpha Alert
    strong_negative = metrics_df[metrics_df["mean_ic"] <= -0.05]
    if not strong_negative.empty:
        top_neg = strong_negative.iloc[0]
        insights.append(
            f"🔄 **Top Inverted / Contrarian Alpha**: `{top_neg['feature']}` (Mean IC = {top_neg['mean_ic']}). Ideal for trigger Short or contrarian reversion strategy."
        )

    return insights


def _render_markdown(
    metadata: AlphalensRunMetadata,
    insights: list[str],
    metrics_df: pd.DataFrame,
    quantile_df: pd.DataFrame,
    chart_artifacts: list[Path],
) -> str:
    insights_text = "\n".join([f"- {insight}" for insight in insights]) if insights else "- No specific insights generated."
    
    top_table_df = metrics_df.head(25)[[
        "feature",
        "rank_ic",
        "mean_ic",
        "ic_std",
        "ir",
        "ic_p_value",
        "positive_ic_ratio",
        "long_short_spread",
        "monotonicity_score",
    ]] if not metrics_df.empty else pd.DataFrame()

    top_table = _markdown_table(top_table_df) if not top_table_df.empty else "No feature metrics evaluated."
    quantile_table = _markdown_table(quantile_df.head(20)) if not quantile_df.empty else "No quantile details available."

    images_text = ""
    if chart_artifacts:
        images_list = [f"![{path.stem}]({path.name})" for path in chart_artifacts]
        images_text = "\n\n## Visual Factor Tearsheet Charts\n\n" + "\n\n".join(images_list)

    return f"""# Alphalens Factor Tearsheet Analysis Report

## Executive Summary & Key Alpha Insights

{insights_text}

## Run Metadata

- Module: `{metadata.module}`
- Created at: `{metadata.created_at}`
- Feature CSV: `{metadata.feature_csv}`
- Label CSV: `{metadata.label_csv}`
- Join strategy: {metadata.join_strategy}
- Evaluated features: {metadata.evaluated_features_count}
- Evaluated samples: {metadata.evaluated_rows}
- Primary return target: `{metadata.primary_return_col}`
- Forward horizons evaluated: {metadata.horizons}
- Quantiles: {metadata.quantiles_count}

## 1. Top Alpha Factor Performance Ranking

Factors are ranked by Information Ratio (IR = Mean IC / Std IC) and Long-Short Return Spread:

{top_table}

## 2. Quantile Return Breakdown (Top Factors)

{quantile_table}
{images_text}
"""


class AlphalensAnalysisModule:
    name = "alphalens_analysis"

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
        features = _read_table_with_date_index(feature_csv)
        labels = _read_table_with_date_index(label_csv)
        merged, feature_columns, label_columns, join_strategy = _merge_inputs(
            features, labels, join_key
        )

        ignored_columns = _date_columns([*feature_columns, *label_columns])
        feature_columns = [col for col in feature_columns if col not in ignored_columns]

        # Extract Price & Forward Returns
        price_series, fwd_returns = _find_price_or_return_columns(merged, label_columns, targets)
        if not fwd_returns:
            raise ValueError(
                "Could not find a price column (e.g. 'Close') or continuous numeric return targets in label/feature datasets."
            )

        # Primary return is 5-period forward return or first available return target
        primary_ret_name = (
            "fwd_ret_5"
            if "fwd_ret_5" in fwd_returns
            else ("fwd_ret_1" if "fwd_ret_1" in fwd_returns else list(fwd_returns.keys())[0])
        )

        # Sample rows if dataset is too large
        if len(merged) > MAX_SAMPLED_ROWS:
            merged = _sample_rows(merged, MAX_SAMPLED_ROWS, random_state=42)

        # Select features to score (cap to avoid extreme latency on 800+ features)
        score_features = feature_columns[:MAX_FEATURES_TO_SCORE]

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        feature_metrics_list: list[dict[str, object]] = []
        all_quantile_dfs: list[pd.DataFrame] = []
        top_quantile_charts_data: list[tuple[str, pd.DataFrame]] = []

        with ModuleProgress(self.name, total=4, enabled=self.progress) as progress_bar:
            # 1. Evaluate Features
            for feat in score_features:
                metrics, q_returns, q_df = _evaluate_feature_factor(
                    feat,
                    merged[feat],
                    fwd_returns,
                    primary_ret_name,
                    n_quantiles=DEFAULT_QUANTILES,
                )
                if metrics:
                    feature_metrics_list.append(metrics)
                    if not q_df.empty:
                        all_quantile_dfs.append(q_df)
            progress_bar.step("evaluate_factors")

            # 2. Rank & Aggregate
            metrics_df = pd.DataFrame(feature_metrics_list)
            if not metrics_df.empty:
                # Sort by absolute IR descending
                metrics_df["abs_ir"] = metrics_df["ir"].abs()
                metrics_df = metrics_df.sort_values(by="abs_ir", ascending=False).drop(columns=["abs_ir"])
            progress_bar.step("rank_factors")

            # 3. Prepare Visuals
            chart_artifacts: list[Path] = []
            if not metrics_df.empty:
                # Quantile chart for top 3
                top_3_names = metrics_df.head(3)["feature"].tolist()
                for name in top_3_names:
                    matching_q = [df for df in all_quantile_dfs if not df.empty and df.iloc[0]["feature"] == name]
                    if matching_q:
                        top_quantile_charts_data.append((name, matching_q[0]))

                # 3.1. Quantile Returns Chart
                q_chart_path = run_dir / "quantile_returns.png"
                if _write_quantile_returns_chart(q_chart_path, top_quantile_charts_data):
                    chart_artifacts.append(q_chart_path)

                # 3.2. IC Decay Chart
                ic_chart_path = run_dir / "ic_decay.png"
                horizons_to_plot = [f"fwd_ret_{h}" for h in DEFAULT_HORIZONS if f"fwd_ret_{h}" in fwd_returns]
                if not horizons_to_plot:
                    horizons_to_plot = list(fwd_returns.keys())[:4]
                if _write_ic_decay_chart(ic_chart_path, metrics_df, horizons_to_plot):
                    chart_artifacts.append(ic_chart_path)

                # 3.3. Cumulative Spread Chart for #1 Top Feature
                cum_chart_path = run_dir / "cumulative_spread.png"
                best_feature = metrics_df.iloc[0]["feature"]
                if _write_cumulative_spread_chart(cum_chart_path, merged, best_feature, primary_ret_name, fwd_returns):
                    chart_artifacts.append(cum_chart_path)
            progress_bar.step("generate_charts")

            # 4. Generate Reports & Artifacts
            quantile_full_df = pd.concat(all_quantile_dfs, ignore_index=True) if all_quantile_dfs else pd.DataFrame()

            metadata = AlphalensRunMetadata(
                module=self.name,
                created_at=datetime.now(timezone.utc).isoformat(),
                feature_csv=str(feature_csv),
                label_csv=str(label_csv),
                join_strategy=join_strategy,
                feature_shape=DatasetShape(*features.shape),
                label_shape=DatasetShape(*labels.shape),
                merged_shape=DatasetShape(*merged.shape),
                evaluated_rows=len(merged),
                evaluated_features_count=len(feature_metrics_list),
                horizons=DEFAULT_HORIZONS,
                quantiles_count=DEFAULT_QUANTILES,
                primary_return_col=primary_ret_name,
            )

            insights = _generate_alphalens_insights(metrics_df, primary_ret_name)
            markdown = _render_markdown(metadata, insights, metrics_df, quantile_full_df, chart_artifacts)

            report_md_path = run_dir / "report.md"
            report_md_path.write_text(markdown, encoding="utf-8")

            summary_json_path = _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "insights": insights,
                    "top_factors": metrics_df.head(20).to_dict(orient="records") if not metrics_df.empty else [],
                },
            )

            factor_metrics_csv_path = _write_csv(run_dir / "factor_metrics.csv", metrics_df)
            quantile_csv_path = _write_csv(run_dir / "quantile_returns.csv", quantile_full_df)
            progress_bar.step("save_artifacts")

        artifacts = [
            report_md_path,
            summary_json_path,
            factor_metrics_csv_path,
            quantile_csv_path,
            *chart_artifacts,
        ]

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)
