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
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _date_columns,
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

MAX_ROWS = 20_000
MAX_CLASS_COUNT = 50
RANDOM_STATE = 42

XGBOOST_MODEL_RESULT_COLUMNS = [
    "label",
    "task",
    "model",
    "samples",
    "features",
    "score_train",
    "score_primary",
    "overfit_gap",
    "score_primary_name",
    "mae",
    "rmse",
    "accuracy",
    "balanced_accuracy",
    "f1_weighted",
    "note",
]

PER_CLASS_COLUMNS = ["label", "class_name", "precision", "recall", "f1_score", "support"]


@dataclass(frozen=True)
class XGBoostRunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    model_rows: int
    features: list[str]
    targets: list[str]
    ignored_columns: list[str]


class XGBoostRelationshipsModule:
    name = "xgboost"

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
        feature_columns = [column for column in feature_columns if column not in ignored_columns]
        label_columns = [column for column in label_columns if column not in ignored_columns]
        selected_targets = _select_targets(label_columns, targets)
        numeric_features = _numeric_feature_columns(merged, feature_columns)

        model_frame = _sample_rows(merged[[*numeric_features, *selected_targets]], MAX_ROWS, RANDOM_STATE)
        with ModuleProgress(self.name, total=len(selected_targets), enabled=self.progress) as progress_bar:
            model_results, importances, per_class_df, cm_dict, reg_preds_dict = _fit_target_models(
                model_frame,
                numeric_features,
                selected_targets,
                progress_bar,
            )

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate PNG Charts
        chart_artifacts: list[Path] = []
        top_chart_path = run_dir / "top_feature_importance.png"
        if _write_top_features_chart(top_chart_path, importances):
            chart_artifacts.append(top_chart_path)

        for label, (class_names, cm) in cm_dict.items():
            cm_path = run_dir / f"cm_{label}.png"
            if _write_confusion_matrix_chart(cm_path, label, class_names, cm):
                chart_artifacts.append(cm_path)

        if reg_preds_dict:
            best_reg_label = max(
                reg_preds_dict.keys(),
                key=lambda l: model_results.loc[model_results["label"] == l, "score_primary"].values[0]
                if not model_results.loc[model_results["label"] == l, "score_primary"].empty
                else -999,
            )
            y_true, y_pred = reg_preds_dict[best_reg_label]
            reg_chart_path = run_dir / "regression_pred_vs_actual.png"
            if _write_regression_pred_chart(reg_chart_path, best_reg_label, y_true, y_pred):
                chart_artifacts.append(reg_chart_path)

        metadata = XGBoostRunMetadata(
            module=self.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=DatasetShape(*features.shape),
            label_shape=DatasetShape(*labels.shape),
            merged_shape=DatasetShape(*merged.shape),
            model_rows=len(model_frame),
            features=numeric_features,
            targets=selected_targets,
            ignored_columns=ignored_columns,
        )

        insights = _generate_executive_insights(model_results, importances, per_class_df)

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "insights": insights,
                    "model_results": model_results.to_dict(orient="records"),
                    "top_feature_importance": importances.head(100).to_dict(orient="records"),
                    "per_class_metrics": per_class_df.to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "scores.csv", model_results),
            _write_csv(run_dir / "importance.csv", importances),
            _write_csv(run_dir / "per_class_metrics.csv", per_class_df),
            *chart_artifacts,
        ]

        markdown = _render_markdown(metadata, insights, model_results, per_class_df, importances, chart_artifacts)
        md_path = run_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")
        artifacts.append(md_path)

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, model_results, per_class_df, importances, chart_artifacts), encoding="utf-8")
        artifacts.append(html_path)

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _fit_target_models(
    merged: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    progress_bar: ModuleProgress | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, tuple[list[str], np.ndarray]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    result_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    cm_dict: dict[str, tuple[list[str], np.ndarray]] = {}
    reg_preds_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if not feature_columns:
        return (
            _model_results_frame(result_rows),
            _importance_frame(importance_rows),
            _per_class_frame(per_class_rows),
            cm_dict,
            reg_preds_dict,
        )

    x = merged[feature_columns].apply(_numeric_series)

    for label in label_columns:
        y_raw = merged[label]
        y_numeric = _numeric_series(y_raw)
        is_numeric_target = (
            y_numeric.notna().sum() >= 10
            and y_numeric.nunique(dropna=True) > MAX_CLASS_COUNT
        )
        if is_numeric_target:
            result, importance, (y_test, predictions) = _fit_regression(label, x, y_numeric)
            if result is not None:
                result_rows.append(result)
                importance_rows.extend(importance)
                reg_preds_dict[label] = (y_test, predictions)
        else:
            result, importance, class_rows, (class_names, cm) = _fit_classification(label, x, y_raw)
            if result is not None:
                result_rows.append(result)
                importance_rows.extend(importance)
                per_class_rows.extend(class_rows)
                if class_names and cm is not None:
                    cm_dict[label] = (class_names, cm)

        if progress_bar is not None:
            progress_bar.step(label)

    return (
        _model_results_frame(result_rows),
        _importance_frame(importance_rows),
        _per_class_frame(per_class_rows),
        cm_dict,
        reg_preds_dict,
    )


def _fit_regression(
    label: str, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, object] | None, list[dict[str, object]], tuple[np.ndarray, np.ndarray]]:
    frame = pd.concat([features, target.rename(label)], axis=1).dropna(subset=[label])
    if len(frame) < 30 or frame[label].nunique() < 2:
        return None, [], (np.array([]), np.array([]))

    x_train, x_test, y_train, y_test = train_test_split(
        frame[features.columns],
        frame[label],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        missing=np.nan,
    )
    model.fit(x_train, y_train)

    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    train_r2 = float(r2_score(y_train, train_preds))
    test_r2 = float(r2_score(y_test, test_preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    importance = _feature_importance(label, features.columns, model.feature_importances_)

    return (
        {
            "label": label,
            "task": "regression",
            "model": "XGBRegressor",
            "samples": int(len(frame)),
            "features": int(len(features.columns)),
            "score_train": _round(train_r2),
            "score_primary": _round(test_r2),
            "overfit_gap": _round(train_r2 - test_r2),
            "score_primary_name": "r2",
            "mae": _round(float(mean_absolute_error(y_test, test_preds))),
            "rmse": _round(rmse),
            "accuracy": None,
            "balanced_accuracy": None,
            "f1_weighted": None,
            "note": "",
        },
        importance,
        (y_test.to_numpy(), test_preds),
    )


def _fit_classification(
    label: str, features: pd.DataFrame, target: pd.Series
) -> tuple[
    dict[str, object] | None,
    list[dict[str, object]],
    list[dict[str, object]],
    tuple[list[str], np.ndarray | None],
]:
    frame = pd.concat([features, target.rename(label)], axis=1).dropna(subset=[label])
    class_count = int(frame[label].nunique(dropna=True))
    if len(frame) < 30 or class_count < 2 or class_count > MAX_CLASS_COUNT:
        return None, [], [], ([], None)

    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[label].astype(str))
    class_names = list(encoder.classes_)
    class_sizes = np.bincount(y)
    if np.min(class_sizes) < 2:
        return None, [], [], ([], None)

    x_train, x_test, y_train, y_test = train_test_split(
        frame[features.columns],
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    objective = "binary:logistic" if class_count == 2 else "multi:softprob"
    model = XGBClassifier(
        objective=objective,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        missing=np.nan,
        eval_metric="logloss" if class_count == 2 else "mlogloss",
    )
    model.fit(x_train, y_train)

    train_preds = model.predict(x_train)
    test_preds = model.predict(x_test)

    train_bal_acc = float(balanced_accuracy_score(y_train, train_preds))
    test_bal_acc = float(balanced_accuracy_score(y_test, test_preds))
    test_acc = float(accuracy_score(y_test, test_preds))
    test_f1 = float(f1_score(y_test, test_preds, average="weighted"))

    importance = _feature_importance(label, features.columns, model.feature_importances_)

    # Per-class metrics
    report = classification_report(y_test, test_preds, target_names=class_names, output_dict=True, zero_division=0)
    per_class_rows: list[dict[str, object]] = []
    for c_name in class_names:
        if c_name in report:
            metrics = report[c_name]
            per_class_rows.append(
                {
                    "label": label,
                    "class_name": c_name,
                    "precision": _round(float(metrics["precision"])),
                    "recall": _round(float(metrics["recall"])),
                    "f1_score": _round(float(metrics["f1-score"])),
                    "support": int(metrics["support"]),
                }
            )

    cm = confusion_matrix(y_test, test_preds)

    return (
        {
            "label": label,
            "task": "classification",
            "model": "XGBClassifier",
            "samples": int(len(frame)),
            "features": int(len(features.columns)),
            "score_train": _round(train_bal_acc),
            "score_primary": _round(test_bal_acc),
            "overfit_gap": _round(train_bal_acc - test_bal_acc),
            "score_primary_name": "balanced_accuracy",
            "mae": None,
            "rmse": None,
            "accuracy": _round(test_acc),
            "balanced_accuracy": _round(test_bal_acc),
            "f1_weighted": _round(test_f1),
            "note": f"classes={class_count}",
        },
        importance,
        per_class_rows,
        (class_names, cm),
    )


def _feature_importance(
    label: str, feature_names: pd.Index, importances: np.ndarray
) -> list[dict[str, object]]:
    values = np.asarray(importances, dtype=float).reshape(-1)
    rows = [
        {
            "label": label,
            "feature": str(feature),
            "importance": _round(float(value)),
            "importance_name": "gain_importance",
        }
        for feature, value in zip(feature_names, values, strict=False)
    ]
    return sorted(rows, key=lambda row: row["importance"] or 0, reverse=True)


def _model_results_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=XGBOOST_MODEL_RESULT_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["score_primary", "samples"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def _per_class_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=PER_CLASS_COLUMNS)
    if frame.empty:
        return frame
    return frame.reset_index(drop=True)


def _importance_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = ["label", "feature", "importance", "importance_name"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["label", "importance"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def _write_top_features_chart(path: Path, importances: pd.DataFrame) -> Path | None:
    if importances.empty:
        return None
    top = (
        importances.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .sort_values(ascending=True)
    )
    if top.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top.values, color="#3182ce")
    ax.set_title("Top 15 Feature Importances (Mean Gain across Targets)")
    ax.set_xlabel("Mean Gain Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_confusion_matrix_chart(
    path: Path, label: str, class_names: list[str], cm: np.ndarray
) -> Path | None:
    if cm is None or len(class_names) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="left", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=9)

    ax.set_title(f"Confusion Matrix: {label}", pad=20)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _write_regression_pred_chart(
    path: Path, label: str, y_true: np.ndarray, y_pred: np.ndarray
) -> Path | None:
    if len(y_true) == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, color="#2b6cb0", edgecolors="none", s=15)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal Perfect Fit")
    ax.set_title(f"Actual vs Predicted: {label}")
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _generate_executive_insights(
    model_results: pd.DataFrame,
    importances: pd.DataFrame,
    per_class_df: pd.DataFrame,
) -> list[str]:
    insights: list[str] = []
    if model_results.empty:
        return insights

    reg_df = model_results[model_results["task"] == "regression"]
    if not reg_df.empty:
        top_reg = reg_df.sort_values("score_primary", ascending=False).iloc[0]
        if top_reg["score_primary"] is not None and top_reg["score_primary"] > 0.5:
            insights.append(
                f"**Best Regression Target**: `{top_reg['label']}` achieved R² = **{top_reg['score_primary']}** (MAE = {top_reg['mae']}). Range/volatility modeling is strong."
            )

    cls_df = model_results[model_results["task"] == "classification"]
    if not cls_df.empty:
        top_cls = cls_df.sort_values("score_primary", ascending=False).iloc[0]
        if top_cls["score_primary"] is not None:
            insights.append(
                f"**Best Classification Target**: `{top_cls['label']}` achieved Balanced Accuracy = **{top_cls['score_primary']}** (Accuracy = {top_cls['accuracy']})."
            )

    overfitted = model_results[model_results["overfit_gap"].notna() & (model_results["overfit_gap"] > 0.15)]
    if not overfitted.empty:
        labels_str = ", ".join([f"`{row['label']}` (Gap: {row['overfit_gap']})" for _, row in overfitted.iterrows()])
        insights.append(
            f"⚠️ **Overfitting Warning**: The following targets have Train vs Test gap > 15%: {labels_str}. Consider reducing `max_depth` or increasing `subsample`."
        )

    if not importances.empty:
        top_feats = (
            importances.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .head(5)
        )
        feats_str = ", ".join([f"`{feat}` ({imp:.4f})" for feat, imp in top_feats.items()])
        insights.append(f"⭐ **Top 5 Global Features**: {feats_str}.")

    if not per_class_df.empty:
        min_support_class = per_class_df.sort_values("support").iloc[0]
        if min_support_class["support"] < 100:
            insights.append(
                f"⚠️ **Class Imbalance**: Class `{min_support_class['class_name']}` for `{min_support_class['label']}` has few samples ({min_support_class['support']}). Consider resampling."
            )

    return insights


def _render_markdown(
    metadata: XGBoostRunMetadata,
    insights: list[str],
    model_results: pd.DataFrame,
    per_class_df: pd.DataFrame,
    importances: pd.DataFrame,
    chart_artifacts: list[Path],
) -> str:
    scores = _markdown_table(model_results) if not model_results.empty else "No XGBoost models were available."
    per_class_table = _markdown_table(per_class_df) if not per_class_df.empty else "No per-class metrics available."
    top_importance = (
        _markdown_table(importances.groupby("label", group_keys=False).head(10))
        if not importances.empty
        else "No feature importance was available."
    )
    ignored = ", ".join(metadata.ignored_columns) if metadata.ignored_columns else "none"

    insights_text = "\n".join([f"- {insight}" for insight in insights]) if insights else "- No specific warnings."

    images_text = ""
    if chart_artifacts:
        images_list = [f"![{path.stem}]({path.name})" for path in chart_artifacts]
        images_text = "\n\n## Visual Charts\n\n" + "\n\n".join(images_list)

    return f"""# XGBoost Feature/Label Relationship Report

## Executive Summary & Insights

{insights_text}

## Run

- Module: `{metadata.module}`
- Created at: `{metadata.created_at}`
- Feature CSV: `{metadata.feature_csv}`
- Label CSV: `{metadata.label_csv}`
- Join strategy: {metadata.join_strategy}
- Feature shape: {metadata.feature_shape.rows} rows x {metadata.feature_shape.columns} columns
- Label shape: {metadata.label_shape.rows} rows x {metadata.label_shape.columns} columns
- Merged shape: {metadata.merged_shape.rows} rows x {metadata.merged_shape.columns} columns
- Model rows: {metadata.model_rows}
- Ignored columns: {ignored}
- Numeric features: {len(metadata.features)}
- Targets: {", ".join(metadata.targets)}

## Model Scores (Train vs Test Performance)

{scores}

## Classification Per-Class Metrics

{per_class_table}

## Top Feature Importance

{top_importance}
{images_text}

## Artifacts

- `summary.json`
- `scores.csv`
- `importance.csv`
- `per_class_metrics.csv`
"""


def _render_html(
    markdown: str,
    model_results: pd.DataFrame,
    per_class_df: pd.DataFrame,
    importances: pd.DataFrame,
    chart_artifacts: list[Path],
) -> str:
    scores = model_results.to_html(index=False, classes="data-table") if not model_results.empty else ""
    per_class = per_class_df.to_html(index=False, classes="data-table") if not per_class_df.empty else ""
    top_importance = (
        importances.groupby("label", group_keys=False).head(20).to_html(index=False, classes="data-table")
        if not importances.empty
        else ""
    )
    charts_html = "".join([f'<div style="margin-top: 20px;"><img src="{p.name}" style="max-width: 100%; border-radius: 8px;"></div>' for p in chart_artifacts])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>XGBoost Feature/Label Relationship Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    pre {{ white-space: pre-wrap; background: #f5f7fa; padding: 16px; border-radius: 6px; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-top: 24px; }}
    .data-table th, .data-table td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    .data-table th {{ background: #edf2f7; }}
  </style>
</head>
<body>
  {_html_markdown_details(markdown)}
  <h2>Model Scores</h2>
  {scores}
  <h2>Classification Per-Class Metrics</h2>
  {per_class}
  <h2>Top Feature Importance</h2>
  {top_importance}
  <h2>Visual Charts</h2>
  {charts_html}
</body>
</html>
"""
