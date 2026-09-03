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
from scipy.stats import beta

from fldataprofiler.modules.base import ModuleResult
from fldataprofiler.modules.progress import ModuleProgress
from fldataprofiler.modules.statistics import DatasetShape
from fldataprofiler.utils import (
    _date_columns,
    _format_duration,
    _markdown_table,
    _merge_inputs,
    _read_table_with_date_index,
    _round,
    _sample_rows,
    _write_csv,
    _write_json,
)

MAX_ROWS = 50_000
RANDOM_STATE = 42
TARGET_LABEL = "allow_entry"


@dataclass(frozen=True)
class SignalRunMetadata:
    module: str
    created_at: str
    execution_time: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    model_rows: int
    signal_features_count: int
    target_label: str


def _normalize_signal_state(val: object) -> str:
    """Normalize any signal value into one of four canonical states: buy, sell, hold, none."""
    if pd.isna(val):
        return "none"
    s = str(val).strip().lower()
    if s in ["buy", "long", "b", "1", "1.0"]:
        return "buy"
    if s in ["sell", "short", "s", "-1", "-1.0"]:
        return "sell"
    if s in ["hold", "0", "0.0", "neutral"]:
        return "hold"
    return "none"


def _classify_outcome(signal_state: str, target_val: str) -> str:
    """Classify outcome into: 'true_alpha', 'sideway_trap', 'reversal_trap', 'vol_trap', or 'other_trap'."""
    s = signal_state.lower().strip()
    t = str(target_val).lower().strip()

    is_bull = any(w in t for w in ["buy", "long", "bull", "1"])
    is_bear = any(w in t for w in ["sell", "short", "bear", "-1"])
    is_sideway = any(w in t for w in ["sideway", "neutral", "narrow", "flat", "0", "range"])
    is_none = any(w in t for w in ["none", "no - none", "lockout", "skip"])

    if s == "buy":
        if is_bull:
            return "true_alpha"
        if is_bear:
            return "reversal_trap"
        if is_sideway:
            return "sideway_trap"
        if is_none:
            return "vol_trap"
        return "other_trap"

    if s == "sell":
        if is_bear:
            return "true_alpha"
        if is_bull:
            return "reversal_trap"
        if is_sideway:
            return "sideway_trap"
        if is_none:
            return "vol_trap"
        return "other_trap"

    return "inactive"


class SignalAnalysisModule:
    name = "signal_analysis"

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
        start_time = time.perf_counter()
        features = _read_table_with_date_index(feature_csv)
        labels = _read_table_with_date_index(label_csv)
        merged, feature_columns, label_columns, join_strategy = _merge_inputs(
            features, labels, join_key
        )

        ignored_columns = _date_columns([*feature_columns, *label_columns])
        feature_columns = [col for col in feature_columns if col not in ignored_columns]

        signal_columns = [col for col in feature_columns if "signal" in col.lower()]
        if not signal_columns:
            raise ValueError("No columns matching '*signal*' were found in the feature dataset.")

        target_col = (
            TARGET_LABEL
            if TARGET_LABEL in merged.columns
            else (targets[0] if targets and targets[0] in merged.columns else label_columns[0])
        )

        model_frame = _sample_rows(merged[[*signal_columns, target_col]], MAX_ROWS, RANDOM_STATE)

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        with ModuleProgress(self.name, total=2, enabled=self.progress) as progress_bar:
            # 1. Discrete Conditional Probability & Bayesian Lift Matrix
            prob_matrix_df = _compute_discrete_probability_matrix(
                model_frame, signal_columns, target_col
            )
            progress_bar.step("probability_matrix")

            # 2. Whipsaw & Trap Diagnosis
            trap_df, top_clean_df = _compute_trap_diagnosis(
                model_frame, signal_columns, target_col
            )
            progress_bar.step("trap_diagnosis")

        # Generate Visual Charts
        chart_artifacts: list[Path] = []
        trap_chart_path = run_dir / "signal_trap_distribution.png"
        if _write_trap_distribution_chart(trap_chart_path, top_clean_df):
            chart_artifacts.append(trap_chart_path)

        prob_chart_path = run_dir / "top_signal_probabilities.png"
        if _write_top_probabilities_chart(prob_chart_path, prob_matrix_df):
            chart_artifacts.append(prob_chart_path)

        metadata = SignalRunMetadata(
            module=self.name,
            created_at=datetime.now(UTC).isoformat(),
            execution_time=_format_duration(time.perf_counter() - start_time),
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=DatasetShape(*features.shape),
            label_shape=DatasetShape(*labels.shape),
            merged_shape=DatasetShape(*merged.shape),
            model_rows=len(model_frame),
            signal_features_count=len(signal_columns),
            target_label=target_col,
        )

        insights = _generate_signal_insights(prob_matrix_df, trap_df, top_clean_df)
        markdown = _render_markdown(
            metadata,
            insights,
            top_clean_df,
            trap_df,
            prob_matrix_df,
            chart_artifacts,
        )

        report_md_path = run_dir / "report.md"
        report_md_path.write_text(markdown, encoding="utf-8")

        html_path = run_dir / "report.html"
        html_path.write_text(
            _render_html(markdown, top_clean_df, prob_matrix_df),
            encoding="utf-8",
        )

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "insights": insights,
                    "top_clean_signals": top_clean_df.head(30).to_dict(orient="records"),
                    "top_whipsaw_signals": trap_df.sort_values("sideway_trap_pct", ascending=False)
                    .head(20)
                    .to_dict(orient="records"),
                    "highest_reversal_risk": trap_df.sort_values("adverse_risk_ratio", ascending=False)
                    .head(20)
                    .to_dict(orient="records"),
                    "top_probabilities": prob_matrix_df.head(30).to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "signal_probability_matrix.csv", prob_matrix_df),
            _write_csv(run_dir / "signal_trap_diagnosis.csv", trap_df),
            _write_csv(run_dir / "top_clean_signals.csv", top_clean_df),
            report_md_path,
            html_path,
            *chart_artifacts,
        ]

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _compute_discrete_probability_matrix(
    df: pd.DataFrame, signal_columns: list[str], target_col: str
) -> pd.DataFrame:
    """Compute discrete conditional probabilities, lift, Bayesian credible interval, and WoE/IV."""
    valid_df = df.dropna(subset=[target_col]).copy()
    total_samples = len(valid_df)
    if total_samples == 0:
        return pd.DataFrame()

    unique_classes = sorted(valid_df[target_col].astype(str).unique())
    base_rates = {
        cls: (valid_df[target_col].astype(str) == cls).sum() / total_samples
        for cls in unique_classes
    }
    class_totals = {
        cls: (valid_df[target_col].astype(str) == cls).sum()
        for cls in unique_classes
    }

    rows: list[dict[str, object]] = []
    canonical_states = ["buy", "sell", "hold", "none"]

    for col in signal_columns:
        norm_series = valid_df[col].apply(_normalize_signal_state)
        for state in canonical_states:
            state_mask = norm_series == state
            state_count = int(state_mask.sum())
            if state_count == 0:
                continue

            state_pct = _round((state_count / total_samples) * 100)

            for cls in unique_classes:
                class_mask = valid_df[target_col].astype(str) == cls
                joint_count = int((state_mask & class_mask).sum())

                cond_prob = joint_count / state_count if state_count > 0 else 0.0
                base_prob = base_rates[cls]
                lift = cond_prob / base_prob if base_prob > 0 else 0.0

                # 95% Bayesian Credible Interval with Beta(0.5, 0.5) Jeffreys Prior
                ci_lower = float(beta.ppf(0.025, joint_count + 0.5, state_count - joint_count + 0.5))
                ci_upper = float(beta.ppf(0.975, joint_count + 0.5, state_count - joint_count + 0.5))

                # Weight of Evidence (WoE) & IV contribution
                target_total = class_totals[cls]
                non_target_total = total_samples - target_total
                good_rate = (joint_count + 0.5) / (target_total + 1.0)
                bad_rate = ((state_count - joint_count) + 0.5) / (non_target_total + 1.0)
                woe = float(np.log(good_rate / bad_rate))
                iv_contrib = float((good_rate - bad_rate) * woe)

                rows.append(
                    {
                        "signal_name": col,
                        "signal_state": state,
                        "target": target_col,
                        "target_class": cls,
                        "count": state_count,
                        "state_pct": state_pct,
                        "class_count": joint_count,
                        "conditional_prob": _round(cond_prob),
                        "base_rate": _round(base_prob),
                        "lift": _round(lift),
                        "ci_lower_95": _round(ci_lower),
                        "ci_upper_95": _round(ci_upper),
                        "woe": _round(woe),
                        "iv_contrib": _round(iv_contrib),
                    }
                )

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values(["lift", "count"], ascending=[False, False]).reset_index(
            drop=True
        )
    return res_df


def _compute_trap_diagnosis(
    df: pd.DataFrame, signal_columns: list[str], target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnose Whipsaw & Trap patterns: True Alpha % vs Sideway Trap % vs Reversal Trap %."""
    valid_df = df.dropna(subset=[target_col]).copy()
    total_samples = len(valid_df)
    if total_samples == 0:
        return pd.DataFrame(), pd.DataFrame()

    unique_classes = [str(c) for c in valid_df[target_col].unique()]
    buy_map = {cls: _classify_outcome("buy", cls) for cls in unique_classes}
    sell_map = {cls: _classify_outcome("sell", cls) for cls in unique_classes}

    trap_rows: list[dict[str, object]] = []

    for col in signal_columns:
        norm_series = valid_df[col].apply(_normalize_signal_state)
        for state in ["buy", "sell"]:
            mask = norm_series == state
            count = int(mask.sum())
            if count < 5:
                continue

            trigger_pct = _round((count / total_samples) * 100)
            target_slice = valid_df.loc[mask, target_col].astype(str)
            val_counts = target_slice.value_counts().to_dict()
            outcome_map = buy_map if state == "buy" else sell_map

            alpha_cnt = sum(cnt for cls, cnt in val_counts.items() if outcome_map.get(cls) == "true_alpha")
            sideway_cnt = sum(cnt for cls, cnt in val_counts.items() if outcome_map.get(cls) == "sideway_trap")
            reversal_cnt = sum(cnt for cls, cnt in val_counts.items() if outcome_map.get(cls) == "reversal_trap")
            vol_cnt = sum(cnt for cls, cnt in val_counts.items() if outcome_map.get(cls) in ["vol_trap", "other_trap"])

            alpha_pct = (alpha_cnt / count) * 100
            sideway_pct = (sideway_cnt / count) * 100
            reversal_pct = (reversal_cnt / count) * 100
            vol_pct = (vol_cnt / count) * 100

            clean_edge = alpha_pct - reversal_pct
            adverse_risk = reversal_pct / (alpha_pct + 1e-6)

            trap_rows.append(
                {
                    "signal_name": col,
                    "signal_state": state,
                    "trigger_count": count,
                    "trigger_pct": trigger_pct,
                    "true_alpha_pct": _round(alpha_pct),
                    "sideway_trap_pct": _round(sideway_pct),
                    "reversal_trap_pct": _round(reversal_pct),
                    "vol_trap_pct": _round(vol_pct),
                    "clean_edge": _round(clean_edge),
                    "adverse_risk_ratio": _round(adverse_risk),
                }
            )

    trap_df = pd.DataFrame(trap_rows)
    if trap_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    trap_df = trap_df.sort_values("clean_edge", ascending=False).reset_index(drop=True)

    # Top clean signals: active count >= 20 (or >= 5 if dataset is small)
    min_triggers = 20 if total_samples >= 500 else 5
    top_clean_df = (
        trap_df[trap_df["trigger_count"] >= min_triggers]
        .sort_values("clean_edge", ascending=False)
        .reset_index(drop=True)
    )

    return trap_df, top_clean_df


def _write_trap_distribution_chart(path: Path, top_clean_df: pd.DataFrame) -> Path | None:
    if top_clean_df.empty:
        return None

    plot_df = top_clean_df.head(15).copy()
    plot_df["label"] = plot_df["signal_name"] + " (" + plot_df["signal_state"].str.upper() + ")"
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df["true_alpha_pct"], color="#2ecc71", label="True Alpha (Win)")
    ax.barh(
        y_pos,
        plot_df["sideway_trap_pct"],
        left=plot_df["true_alpha_pct"],
        color="#f39c12",
        label="Sideway Trap (Whipsaw)",
    )
    ax.barh(
        y_pos,
        plot_df["reversal_trap_pct"],
        left=plot_df["true_alpha_pct"] + plot_df["sideway_trap_pct"],
        color="#e74c3c",
        label="Reversal Trap (Counter-trend)",
    )
    ax.barh(
        y_pos,
        plot_df["vol_trap_pct"],
        left=plot_df["true_alpha_pct"] + plot_df["sideway_trap_pct"] + plot_df["reversal_trap_pct"],
        color="#95a5a6",
        label="Volatility / Lockout Trap",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["label"], fontsize=9)
    ax.set_xlabel("Percentage of Triggers (%)")
    ax.set_title("Top 15 Signals: True Alpha vs Whipsaw & Reversal Traps", fontsize=12, pad=15)
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9)

    for i, row in plot_df.iterrows():
        edge_text = (
            f"Edge: +{row['clean_edge']:.1f}%"
            if row["clean_edge"] >= 0
            else f"Edge: {row['clean_edge']:.1f}%"
        )
        ax.text(101, i, edge_text, va="center", fontsize=8, fontweight="bold", color="#2c3e50")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_top_probabilities_chart(path: Path, prob_df: pd.DataFrame) -> Path | None:
    if prob_df.empty:
        return None

    # Filter directional states with count >= 20 and positive target
    filtered = (
        prob_df[
            prob_df["signal_state"].isin(["buy", "sell"])
            & (prob_df["count"] >= 20)
            & (~prob_df["target_class"].astype(str).str.lower().str.contains("sideway|none"))
        ]
        .sort_values("conditional_prob", ascending=False)
        .head(15)
        .copy()
    )

    if filtered.empty:
        return None

    filtered["label"] = (
        filtered["signal_name"]
        + " ["
        + filtered["signal_state"].str.upper()
        + "] -> "
        + filtered["target_class"].astype(str)
    )
    filtered = filtered.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(filtered))

    probs = filtered["conditional_prob"]
    xerr_lower = np.maximum(0, probs - filtered["ci_lower_95"])
    xerr_upper = np.maximum(0, filtered["ci_upper_95"] - probs)

    ax.barh(
        y_pos,
        probs * 100,
        xerr=[xerr_lower * 100, xerr_upper * 100],
        color="#3498db",
        capsize=4,
        alpha=0.85,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(filtered["label"], fontsize=9)
    ax.set_xlabel("Conditional Probability (%) with 95% Bayesian Credible Interval")
    ax.set_title("Top 15 Directional Signals: Conditional Probability & 95% CI", fontsize=12, pad=15)

    for i, row in filtered.iterrows():
        lift_str = f"Lift: {row['lift']:.2f}x (N={row['count']})"
        ax.text(row["conditional_prob"] * 100 + 2, i, lift_str, va="center", fontsize=8, color="#1a365d")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _generate_signal_insights(
    prob_df: pd.DataFrame, trap_df: pd.DataFrame, top_clean_df: pd.DataFrame
) -> list[str]:
    insights: list[str] = []

    if not top_clean_df.empty:
        best_clean = top_clean_df.iloc[0]
        insights.append(
            f"⭐ **Best Clean Alpha Signal**: `{best_clean['signal_name']}` ({best_clean['signal_state'].upper()}) achieved Clean Edge = **+{best_clean['clean_edge']:.2f}%** (True Alpha: {best_clean['true_alpha_pct']}%, Reversal Trap: {best_clean['reversal_trap_pct']}%, Active Triggers: {best_clean['trigger_count']})."
        )

    if not prob_df.empty:
        top_lift = (
            prob_df[prob_df["count"] >= 30].sort_values("lift", ascending=False).head(1)
        )
        if not top_lift.empty:
            best_lift = top_lift.iloc[0]
            insights.append(
                f"🚀 **Highest Probability Lift**: `{best_lift['signal_name']}` [{best_lift['signal_state']}] drives `{best_lift['target_class']}` probability to **{best_lift['conditional_prob']*100:.2f}%** (Lift = **{best_lift['lift']:.2f}x**, 95% CI: [{best_lift['ci_lower_95']*100:.1f}%, {best_lift['ci_upper_95']*100:.1f}%])."
            )

    if not trap_df.empty:
        toxic_reversal = (
            trap_df[trap_df["trigger_count"] >= 30]
            .sort_values("adverse_risk_ratio", ascending=False)
            .head(1)
        )
        if not toxic_reversal.empty:
            worst_rev = toxic_reversal.iloc[0]
            insights.append(
                f"⚠️ **High Reversal Trap Warning**: `{worst_rev['signal_name']}` ({worst_rev['signal_state'].upper()}) has an Adverse Risk Ratio of **{worst_rev['adverse_risk_ratio']:.2f}x** (Reversal Trap: {worst_rev['reversal_trap_pct']}% vs True Alpha: {worst_rev['true_alpha_pct']}%). Buying or selling on this signal frequently triggers opposite counter-trend moves."
            )

        worst_whipsaw = (
            trap_df[trap_df["trigger_count"] >= 30]
            .sort_values("sideway_trap_pct", ascending=False)
            .head(1)
        )
        if not worst_whipsaw.empty:
            worst_whip = worst_whipsaw.iloc[0]
            insights.append(
                f"🌪️ **Most Sideway-Prone Signal**: `{worst_whip['signal_name']}` ({worst_whip['signal_state'].upper()}) gets caught in Sideway Whipsaw **{worst_whip['sideway_trap_pct']:.2f}%** of the time. Requires an ADX or Choppiness trend filter before execution."
            )

    return insights


def _render_markdown(
    metadata: SignalRunMetadata,
    insights: list[str],
    top_clean_df: pd.DataFrame,
    trap_df: pd.DataFrame,
    prob_df: pd.DataFrame,
    chart_artifacts: list[Path],
) -> str:
    insights_text = (
        "\n".join([f"- {insight}" for insight in insights])
        if insights
        else "- No specific warnings."
    )

    clean_table = (
        _markdown_table(top_clean_df.head(20))
        if not top_clean_df.empty
        else "No clean alpha signals found."
    )

    reversal_table = (
        _markdown_table(
            trap_df[trap_df["trigger_count"] >= 30]
            .sort_values("adverse_risk_ratio", ascending=False)
            .head(10)
        )
        if not trap_df.empty
        else "No reversal trap data available."
    )

    prob_table = (
        _markdown_table(
            prob_df[prob_df["count"] >= 30]
            .sort_values(["lift", "conditional_prob"], ascending=[False, False])
            .head(20)
        )
        if not prob_df.empty
        else "No probability matrix data available."
    )

    images_text = ""
    if chart_artifacts:
        images_list = [f"![{path.stem}]({path.name})" for path in chart_artifacts]
        images_text = "\n\n## Visual Charts\n\n" + "\n\n".join(images_list)

    return f"""# Discrete Signal Probability & Trap Diagnosis Report (`{metadata.target_label}`)

## Executive Summary & Insights

{insights_text}

## Run Metadata

- Module: `{metadata.module}`
- Created at: `{metadata.created_at}`
- Execution time: `{metadata.execution_time}`
- Feature CSV: `{metadata.feature_csv}`
- Label CSV: `{metadata.label_csv}`
- Join strategy: {metadata.join_strategy}
- Signal features evaluated: {metadata.signal_features_count}
- Target label: `{metadata.target_label}`
- Model rows sampled: {metadata.model_rows}

## 1. Top Clean Alpha Signals (Ranked by Clean Directional Edge)

Clean Edge = `True Alpha %` - `Reversal Trap %`. A positive clean edge indicates that the signal's directional wins reliably outpace disastrous counter-trend reversals.

{clean_table}

## 2. High-Risk Reversal Trap Diagnosis (Toxic Signals to Avoid)

Adverse Risk Ratio = `Reversal Trap %` / `True Alpha %`. A ratio > 1.0 indicates that when this signal triggers, it is more likely to cause a counter-trend liquidation than a profitable trade.

{reversal_table}

## 3. Discrete Conditional Probability & Bayesian Lift Highlights

Full discrete conditional distribution $P(\\text{{Target}} = \\text{{Class}} \\mid \\text{{Signal}} = \\text{{State}})$ with 95% Bayesian Credible Intervals (Beta-Binomial Jeffreys Prior).

{prob_table}
{images_text}

## Artifacts

- `summary.json`
- `signal_probability_matrix.csv`
- `signal_trap_diagnosis.csv`
- `top_clean_signals.csv`
"""


def _render_html(
    markdown: str,
    top_clean_df: pd.DataFrame,
    prob_df: pd.DataFrame,
) -> str:
    clean_html = (
        top_clean_df.head(20).to_html(index=False, classes="data-table")
        if not top_clean_df.empty
        else ""
    )
    prob_html = (
        prob_df.head(25).to_html(index=False, classes="data-table")
        if not prob_df.empty
        else ""
    )
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Discrete Signal Probability & Trap Diagnosis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; margin: 2rem; color: #2d3748; }}
        h1, h2, h3 {{ color: #1a202c; }}
        table.data-table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        table.data-table th, table.data-table td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
        table.data-table th {{ background-color: #f7fafc; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f8fafc; }}
    </style>
</head>
<body>
    <h1>Discrete Signal Probability & Trap Diagnosis Report</h1>
    <h2>Top Clean Alpha Signals (True Alpha vs Reversal Edge)</h2>
    {clean_html}
    <h2>Discrete Conditional Probability & Bayesian Lift</h2>
    {prob_html}
</body>
</html>
"""
