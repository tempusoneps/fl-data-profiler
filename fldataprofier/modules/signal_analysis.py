from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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

MAX_ROWS = 20_000
RANDOM_STATE = 42
TARGET_LABEL = "allow_entry"


@dataclass(frozen=True)
class SignalRunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    model_rows: int
    signal_features_count: int
    target_label: str


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

        target_col = TARGET_LABEL if TARGET_LABEL in merged.columns else (targets[0] if targets and targets[0] in merged.columns else label_columns[0])

        model_frame = _sample_rows(merged[[*signal_columns, target_col]], MAX_ROWS, RANDOM_STATE)

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        with ModuleProgress(self.name, total=3, enabled=self.progress) as progress_bar:
            # 1. Single Signal Analysis
            single_signal_df = _evaluate_single_signals(model_frame, signal_columns, target_col)
            progress_bar.step("single_signals")

            # 2. Combined Signal Model
            combined_model_res, combined_importances_df = _fit_combined_signal_model(
                model_frame, signal_columns, target_col
            )
            progress_bar.step("combined_model")

            # 3. Correlation / Redundancy Matrix
            redundancy_df = _calculate_signal_redundancy(model_frame, signal_columns)
            progress_bar.step("redundancy")

        # Generate PNG Charts
        chart_artifacts: list[Path] = []
        top_single_chart_path = run_dir / "top_single_signals.png"
        if _write_top_single_signals_chart(top_single_chart_path, single_signal_df):
            chart_artifacts.append(top_single_chart_path)

        combined_imp_chart_path = run_dir / "combined_signal_importance.png"
        if _write_combined_importance_chart(combined_imp_chart_path, combined_importances_df):
            chart_artifacts.append(combined_imp_chart_path)

        heatmap_path = run_dir / "signal_redundancy_heatmap.png"
        if _write_signal_correlation_heatmap(heatmap_path, model_frame, combined_importances_df):
            chart_artifacts.append(heatmap_path)

        metadata = SignalRunMetadata(
            module=self.name,
            created_at=datetime.now(timezone.utc).isoformat(),
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

        insights = _generate_signal_insights(single_signal_df, combined_model_res, combined_importances_df, redundancy_df)
        markdown = _render_markdown(metadata, insights, single_signal_df, combined_model_res, combined_importances_df, redundancy_df, chart_artifacts)

        report_md_path = run_dir / "report.md"
        report_md_path.write_text(markdown, encoding="utf-8")

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, single_signal_df, combined_model_res, combined_importances_df), encoding="utf-8")

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "insights": insights,
                    "combined_model": combined_model_res,
                    "top_single_signals": single_signal_df.head(50).to_dict(orient="records"),
                    "combined_importances": combined_importances_df.head(50).to_dict(orient="records"),
                    "redundant_pairs": redundancy_df.head(30).to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "single_signal_scores.csv", single_signal_df),
            _write_csv(run_dir / "combined_signal_importance.csv", combined_importances_df),
            _write_csv(run_dir / "signal_redundancy.csv", redundancy_df),
            report_md_path,
            html_path,
            *chart_artifacts,
        ]

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _evaluate_single_signals(
    df: pd.DataFrame, signal_columns: list[str], target_col: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid_df = df.dropna(subset=[target_col]).copy()
    total_samples = len(valid_df)

    for col in signal_columns:
        s = _numeric_series(valid_df[col]).fillna(0)
        active_mask = s != 0
        active_count = int(active_mask.sum())
        active_pct = _round((active_count / total_samples) * 100 if total_samples > 0 else 0)

        buy_precision, buy_recall, buy_f1, buy_support = 0.0, 0.0, 0.0, 0
        sell_precision, sell_recall, sell_f1, sell_support = 0.0, 0.0, 0.0, 0

        if active_count >= 5:
            # Check Buy signal alignment (signal > 0 or 1 vs Yes - Buy)
            buy_mask = s > 0
            if buy_mask.sum() > 0:
                true_buy = (valid_df[target_col] == "Yes - Buy")
                tp_buy = (buy_mask & true_buy).sum()
                fp_buy = (buy_mask & ~true_buy).sum()
                fn_buy = (~buy_mask & true_buy).sum()
                buy_support = int(true_buy.sum())
                buy_precision = float(tp_buy / (tp_buy + fp_buy)) if (tp_buy + fp_buy) > 0 else 0.0
                buy_recall = float(tp_buy / (tp_buy + fn_buy)) if (tp_buy + fn_buy) > 0 else 0.0
                buy_f1 = (2 * buy_precision * buy_recall / (buy_precision + buy_recall)) if (buy_precision + buy_recall) > 0 else 0.0

            # Check Sell signal alignment (signal < 0 or -1 vs Yes - Sell)
            sell_mask = s < 0
            if sell_mask.sum() > 0:
                true_sell = (valid_df[target_col] == "Yes - Sell")
                tp_sell = (sell_mask & true_sell).sum()
                fp_sell = (sell_mask & ~true_sell).sum()
                fn_sell = (~sell_mask & true_sell).sum()
                sell_support = int(true_sell.sum())
                sell_precision = float(tp_sell / (tp_sell + fp_sell)) if (tp_sell + fp_sell) > 0 else 0.0
                sell_recall = float(tp_sell / (tp_sell + fn_sell)) if (tp_sell + fn_sell) > 0 else 0.0
                sell_f1 = (2 * sell_precision * sell_recall / (sell_precision + sell_recall)) if (sell_precision + sell_recall) > 0 else 0.0

        max_precision = max(buy_precision, sell_precision)
        max_f1 = max(buy_f1, sell_f1)

        rows.append(
            {
                "signal_name": col,
                "active_count": active_count,
                "active_pct": active_pct,
                "buy_precision": _round(buy_precision),
                "buy_recall": _round(buy_recall),
                "buy_f1": _round(buy_f1),
                "sell_precision": _round(sell_precision),
                "sell_recall": _round(sell_recall),
                "sell_f1": _round(sell_f1),
                "max_precision": _round(max_precision),
                "max_f1": _round(max_f1),
            }
        )

    res_df = pd.DataFrame(rows)
    if not res_df.empty:
        res_df = res_df.sort_values("max_precision", ascending=False).reset_index(drop=True)
    return res_df


def _fit_combined_signal_model(
    df: pd.DataFrame, signal_columns: list[str], target_col: str
) -> tuple[dict[str, object], pd.DataFrame]:
    valid_df = df.dropna(subset=[target_col]).copy()
    if len(valid_df) < 30:
        return {}, pd.DataFrame()

    x = valid_df[signal_columns].apply(_numeric_series).fillna(0)
    encoder = LabelEncoder()
    y = encoder.fit_transform(valid_df[target_col].astype(str))
    class_names = list(encoder.classes_)

    split_idx = int(len(valid_df) * 0.8)
    x_train = x.iloc[:split_idx]
    x_test = x.iloc[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    model = XGBClassifier(
        objective="multi:softprob" if len(class_names) > 2 else "binary:logistic",
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    train_bal_acc = float(balanced_accuracy_score(y_train, train_preds))
    test_bal_acc = float(balanced_accuracy_score(y_test, test_preds))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_f1 = float(f1_score(y_test, test_preds, average="weighted"))

    imp_values = model.feature_importances_
    imp_rows = [
        {"signal_name": str(col), "importance": _round(float(val))}
        for col, val in zip(signal_columns, imp_values)
    ]
    imp_df = pd.DataFrame(imp_rows).sort_values("importance", ascending=False).reset_index(drop=True)

    result_dict = {
        "target": target_col,
        "model": "XGBClassifier (Signals Only)",
        "samples": len(valid_df),
        "signal_features": len(signal_columns),
        "score_train": _round(train_bal_acc),
        "score_primary": _round(test_bal_acc),
        "overfit_gap": _round(train_bal_acc - test_bal_acc),
        "accuracy": _round(test_acc),
        "balanced_accuracy": _round(test_bal_acc),
        "f1_weighted": _round(test_f1),
    }

    return result_dict, imp_df


def _calculate_signal_redundancy(df: pd.DataFrame, signal_columns: list[str]) -> pd.DataFrame:
    x = df[signal_columns].apply(_numeric_series).fillna(0)
    corr = x.corr().abs()

    pairs: list[dict[str, object]] = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = corr.loc[c1, c2]
            if val >= 0.70:
                pairs.append({"signal_1": c1, "signal_2": c2, "correlation": _round(float(val))})

    res_df = pd.DataFrame(pairs)
    if not res_df.empty:
        res_df = res_df.sort_values("correlation", ascending=False).reset_index(drop=True)
    else:
        res_df = pd.DataFrame(columns=["signal_1", "signal_2", "correlation"])
    return res_df


def _write_top_single_signals_chart(path: Path, single_df: pd.DataFrame) -> Path | None:
    if single_df.empty:
        return None
    top = single_df.head(15).sort_values("max_precision", ascending=True)
    if top.empty or top["max_precision"].sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["signal_name"], top["max_precision"], color="#2b6cb0")
    ax.set_title("Top 15 Single Signals by Precision for allow_entry")
    ax.set_xlabel("Max Precision (Buy or Sell)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_combined_importance_chart(path: Path, imp_df: pd.DataFrame) -> Path | None:
    if imp_df.empty:
        return None
    top = imp_df.head(15).sort_values("importance", ascending=True)
    if top.empty or top["importance"].sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["signal_name"], top["importance"], color="#2f855a")
    ax.set_title("Top 15 Signals by Combined XGBoost Feature Importance")
    ax.set_xlabel("Gain Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_signal_correlation_heatmap(
    path: Path, df: pd.DataFrame, imp_df: pd.DataFrame
) -> Path | None:
    top_signals = imp_df.head(15)["signal_name"].tolist() if not imp_df.empty else []
    if len(top_signals) < 2:
        return None
    x = df[top_signals].apply(_numeric_series).fillna(0)
    corr = x.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(top_signals)))
    ax.set_yticks(np.arange(len(top_signals)))
    ax.set_xticklabels(top_signals, rotation=90, ha="left", fontsize=8)
    ax.set_yticklabels(top_signals, fontsize=8)
    ax.set_title("Correlation Heatmap: Top 15 Signals", pad=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _generate_signal_insights(
    single_df: pd.DataFrame,
    combined_res: dict[str, object],
    imp_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
) -> list[str]:
    insights: list[str] = []
    if not single_df.empty:
        top_single = single_df.iloc[0]
        insights.append(
            f"⭐ **Best Single Signal**: `{top_single['signal_name']}` achieved Max Precision = **{top_single['max_precision']}** (Buy Precision: {top_single['buy_precision']}, Sell Precision: {top_single['sell_precision']}, Active: {top_single['active_pct']}%)."
        )

    if combined_res:
        insights.append(
            f"📊 **Combined Signal Model (`allow_entry`)**: Balanced Accuracy = **{combined_res.get('balanced_accuracy')}** (Train Score: {combined_res.get('score_train')}, Overfit Gap: {combined_res.get('overfit_gap')})."
        )

    if not imp_df.empty:
        top_3_imp = ", ".join([f"`{row['signal_name']}` ({row['importance']:.4f})" for _, row in imp_df.head(3).iterrows()])
        insights.append(f"🔥 **Top 3 Combined Signals**: {top_3_imp}.")

    if not redundancy_df.empty:
        high_corr_count = len(redundancy_df[redundancy_df["correlation"] >= 0.85])
        if high_corr_count > 0:
            top_pair = redundancy_df.iloc[0]
            insights.append(
                f"⚠️ **Signal Redundancy**: Found {high_corr_count} highly correlated signal pairs (correlation >= 0.85). Highest redundancy: `{top_pair['signal_1']}` & `{top_pair['signal_2']}` (corr = {top_pair['correlation']})."
            )

    return insights


def _render_markdown(
    metadata: SignalRunMetadata,
    insights: list[str],
    single_df: pd.DataFrame,
    combined_res: dict[str, object],
    imp_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    chart_artifacts: list[Path],
) -> str:
    insights_text = "\n".join([f"- {insight}" for insight in insights]) if insights else "- No specific warnings."
    single_table = _markdown_table(single_df.head(20)) if not single_df.empty else "No single signals evaluated."
    imp_table = _markdown_table(imp_df.head(20)) if not imp_df.empty else "No combined importances available."
    redundancy_table = _markdown_table(redundancy_df.head(15)) if not redundancy_df.empty else "No highly correlated signal pairs found."

    images_text = ""
    if chart_artifacts:
        images_list = [f"![{path.stem}]({path.name})" for path in chart_artifacts]
        images_text = "\n\n## Visual Charts\n\n" + "\n\n".join(images_list)

    combined_summary = (
        f"- Target: `{combined_res.get('target')}`\n"
        f"- Model: `{combined_res.get('model')}`\n"
        f"- Test Balanced Accuracy: **{combined_res.get('balanced_accuracy')}**\n"
        f"- Test Accuracy: **{combined_res.get('accuracy')}**\n"
        f"- F1 Weighted: **{combined_res.get('f1_weighted')}**\n"
        f"- Train Score: {combined_res.get('score_train')}\n"
        f"- Overfit Gap: {combined_res.get('overfit_gap')}"
    ) if combined_res else "Combined model unavailable."

    return f"""# Signal Feature Analysis Report (`allow_entry`)

## Executive Summary & Insights

{insights_text}

## Run Metadata

- Module: `{metadata.module}`
- Created at: `{metadata.created_at}`
- Feature CSV: `{metadata.feature_csv}`
- Label CSV: `{metadata.label_csv}`
- Join strategy: {metadata.join_strategy}
- Signal features evaluated: {metadata.signal_features_count}
- Target label: `{metadata.target_label}`
- Model rows sampled: {metadata.model_rows}

## 1. Single Signal Performance Ranking (`allow_entry`)

{single_table}

## 2. Combined Signals Machine Learning Model

{combined_summary}

### Top Combined Signal Importances

{imp_table}

## 3. Signal Redundancy & Correlation Analysis

{redundancy_table}
{images_text}

## Artifacts

- `summary.json`
- `single_signal_scores.csv`
- `combined_signal_importance.csv`
- `signal_redundancy.csv`
"""


def _render_html(
    markdown: str,
    single_df: pd.DataFrame,
    combined_res: dict[str, object],
    imp_df: pd.DataFrame,
) -> str:
    single_html = single_df.head(20).to_html(index=False, classes="data-table") if not single_df.empty else ""
    imp_html = imp_df.head(20).to_html(index=False, classes="data-table") if not imp_df.empty else ""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Signal Feature Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; margin: 2rem; color: #2d3748; }}
        h1, h2, h3 {{ color: #1a202c; }}
        table.data-table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
        table.data-table th, table.data-table td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
        table.data-table th {{ background-color: #f7fafc; }}
    </style>
</head>
<body>
    <h1>Signal Feature Analysis Report (allow_entry)</h1>
    <h2>Top Single Signals</h2>
    {single_html}
    <h2>Top Combined Signal Importances</h2>
    {imp_html}
</body>
</html>
"""
