from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.sklearn import _date_columns
from fldataprofier.modules.statistics import (
    DatasetShape,
    _markdown_table,
    _merge_inputs,
    _numeric_series,
    _round,
    _select_targets,
)


MAX_ROWS = 10_000
MAX_EXPLAIN_ROWS = 1_000
MAX_CLASS_COUNT = 50
RANDOM_STATE = 42


@dataclass(frozen=True)
class ShapRunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    model_rows: int
    explain_rows: int
    features: list[str]
    targets: list[str]
    ignored_columns: list[str]


class ShapRelationshipsModule:
    name = "shap"

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        features = pd.read_csv(feature_csv, parse_dates=["Date"], index_col="Date")
        labels = pd.read_csv(label_csv, parse_dates=["Date"], index_col="Date")
        merged, feature_columns, label_columns, join_strategy = _merge_inputs(
            features, labels, join_key
        )

        ignored_columns = _date_columns([*feature_columns, *label_columns])
        feature_columns = [column for column in feature_columns if column not in ignored_columns]
        label_columns = [column for column in label_columns if column not in ignored_columns]
        selected_targets = _select_targets(label_columns, targets)
        numeric_features = _numeric_feature_columns(merged, feature_columns)

        model_frame = _sample_rows(merged[[*numeric_features, *selected_targets]], MAX_ROWS)
        model_results, shap_importance = _fit_target_models(
            model_frame,
            numeric_features,
            selected_targets,
        )

        explain_rows = 0
        if not shap_importance.empty:
            explain_rows = int(shap_importance["explain_rows"].max())

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = ShapRunMetadata(
            module=self.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=DatasetShape(*features.shape),
            label_shape=DatasetShape(*labels.shape),
            merged_shape=DatasetShape(*merged.shape),
            model_rows=len(model_frame),
            explain_rows=explain_rows,
            features=numeric_features,
            targets=selected_targets,
            ignored_columns=ignored_columns,
        )

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "model_results": model_results.to_dict(orient="records"),
                    "top_shap_importance": shap_importance.head(100).to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "scores.csv", model_results),
            _write_csv(run_dir / "shap_importance.csv", shap_importance),
        ]

        markdown = _render_markdown(metadata, model_results, shap_importance)
        md_path = run_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")
        artifacts.append(md_path)

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, model_results, shap_importance), encoding="utf-8")
        artifacts.append(html_path)

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _numeric_feature_columns(merged: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    columns: list[str] = []
    for column in feature_columns:
        values = _numeric_series(merged[column])
        if values.notna().sum() >= 10 and values.nunique(dropna=True) >= 2:
            columns.append(column)
    return columns


def _sample_rows(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=RANDOM_STATE).sort_index()


def _fit_target_models(
    merged: pd.DataFrame, feature_columns: list[str], label_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    if not feature_columns:
        return _model_results_frame(result_rows), _importance_frame(importance_rows)

    x = merged[feature_columns].apply(_numeric_series)

    for label in label_columns:
        y_raw = merged[label]
        y_numeric = _numeric_series(y_raw)
        is_numeric_target = (
            y_numeric.notna().sum() >= 10
            and y_numeric.nunique(dropna=True) > MAX_CLASS_COUNT
        )
        if is_numeric_target:
            result, importance = _fit_regression(label, x, y_numeric)
        else:
            result, importance = _fit_classification(label, x, y_raw)

        if result is not None:
            result_rows.append(result)
        importance_rows.extend(importance)

    return _model_results_frame(result_rows), _importance_frame(importance_rows)


def _fit_regression(
    label: str, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    frame = pd.concat([features, target.rename(label)], axis=1).dropna(subset=[label])
    if len(frame) < 30 or frame[label].nunique() < 2:
        return None, []

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
    predictions = model.predict(x_test)
    explain_x = _sample_rows(frame[features.columns], MAX_EXPLAIN_ROWS)
    importance = _shap_importance(label, "regression", model, explain_x)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return (
        {
            "label": label,
            "task": "regression",
            "model": "XGBRegressor + TreeExplainer",
            "samples": int(len(frame)),
            "features": int(len(features.columns)),
            "score_primary": _round(float(r2_score(y_test, predictions))),
            "score_primary_name": "r2",
            "mae": _round(float(mean_absolute_error(y_test, predictions))),
            "rmse": _round(rmse),
            "accuracy": None,
            "balanced_accuracy": None,
            "f1_weighted": None,
            "note": "",
        },
        importance,
    )


def _fit_classification(
    label: str, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    frame = pd.concat([features, target.rename(label)], axis=1).dropna(subset=[label])
    class_count = int(frame[label].nunique(dropna=True))
    if len(frame) < 30 or class_count < 2 or class_count > MAX_CLASS_COUNT:
        return None, []

    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[label].astype(str))
    if np.min(np.bincount(y)) < 2:
        return None, []

    x_train, x_test, y_train, y_test = train_test_split(
        frame[features.columns],
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    class_count = len(encoder.classes_)
    model = XGBClassifier(
        objective="binary:logistic" if class_count == 2 else "multi:softprob",
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
    predictions = model.predict(x_test)
    explain_x = _sample_rows(frame[features.columns], MAX_EXPLAIN_ROWS)
    importance = _shap_importance(label, "classification", model, explain_x)
    return (
        {
            "label": label,
            "task": "classification",
            "model": "XGBClassifier + TreeExplainer",
            "samples": int(len(frame)),
            "features": int(len(features.columns)),
            "score_primary": _round(float(balanced_accuracy_score(y_test, predictions))),
            "score_primary_name": "balanced_accuracy",
            "mae": None,
            "rmse": None,
            "accuracy": _round(float(accuracy_score(y_test, predictions))),
            "balanced_accuracy": _round(float(balanced_accuracy_score(y_test, predictions))),
            "f1_weighted": _round(float(f1_score(y_test, predictions, average="weighted"))),
            "note": f"classes={class_count}",
        },
        importance,
    )


def _shap_importance(
    label: str, task: str, model: XGBClassifier | XGBRegressor, explain_x: pd.DataFrame
) -> list[dict[str, object]]:
    import shap

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(explain_x)
    array = np.asarray(values, dtype=float)
    if isinstance(values, list):
        array = np.stack([np.asarray(value, dtype=float) for value in values], axis=-1)
    if array.ndim == 3:
        mean_abs = np.mean(np.abs(array), axis=(0, 2))
    else:
        mean_abs = np.mean(np.abs(array), axis=0)

    rows = [
        {
            "label": label,
            "task": task,
            "feature": str(feature),
            "mean_abs_shap": _round(float(value)),
            "explain_rows": int(len(explain_x)),
        }
        for feature, value in zip(explain_x.columns, mean_abs, strict=False)
    ]
    return sorted(rows, key=lambda row: row["mean_abs_shap"] or 0, reverse=True)


def _model_results_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "label",
        "task",
        "model",
        "samples",
        "features",
        "score_primary",
        "score_primary_name",
        "mae",
        "rmse",
        "accuracy",
        "balanced_accuracy",
        "f1_weighted",
        "note",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["score_primary", "samples"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def _importance_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = ["label", "task", "feature", "mean_abs_shap", "explain_rows"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["label", "mean_abs_shap"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _render_markdown(
    metadata: ShapRunMetadata, model_results: pd.DataFrame, shap_importance: pd.DataFrame
) -> str:
    scores = _markdown_table(model_results) if not model_results.empty else "No SHAP models were available."
    top_importance = (
        _markdown_table(shap_importance.groupby("label", group_keys=False).head(10))
        if not shap_importance.empty
        else "No SHAP importance was available."
    )
    ignored = ", ".join(metadata.ignored_columns) if metadata.ignored_columns else "none"
    return f"""# SHAP Feature/Label Explanation Report

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
- Explain rows: {metadata.explain_rows}
- Ignored columns: {ignored}
- Numeric features: {len(metadata.features)}
- Targets: {", ".join(metadata.targets)}

## Model Scores

{scores}

## Top Mean Absolute SHAP Values

{top_importance}

## Artifacts

- `summary.json`
- `scores.csv`
- `shap_importance.csv`
"""


def _render_html(markdown: str, model_results: pd.DataFrame, shap_importance: pd.DataFrame) -> str:
    escaped_markdown = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    scores = model_results.to_html(index=False, classes="data-table") if not model_results.empty else ""
    top_importance = (
        shap_importance.groupby("label", group_keys=False).head(20).to_html(index=False, classes="data-table")
        if not shap_importance.empty
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SHAP Feature/Label Explanation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    pre {{ white-space: pre-wrap; background: #f5f7fa; padding: 16px; border-radius: 6px; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-top: 24px; }}
    .data-table th, .data-table td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    .data-table th {{ background: #edf2f7; }}
  </style>
</head>
<body>
  <pre>{escaped_markdown}</pre>
  <h2>Model Scores</h2>
  {scores}
  <h2>Top Mean Absolute SHAP Values</h2>
  {top_importance}
</body>
</html>
"""
