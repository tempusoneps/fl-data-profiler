from __future__ import annotations

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

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _html_markdown_details,
    _read_table_with_date_index,
    _date_columns,
    _markdown_table,
    _merge_inputs,
    _model_results_frame,
    _numeric_feature_columns,
    _numeric_series,
    _round,
    _sample_rows,
    _select_targets,
    _write_csv,
    _write_json,
)

MAX_ROWS = 20_000
MAX_CLASS_COUNT = 50
RANDOM_STATE = 42
TIME_BUDGET_SECONDS = 60  # Default time budget per target


@dataclass(frozen=True)
class PyCaretRunMetadata:
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
    time_budget: int


class PyCaretRelationshipsModule:
    name = "pycaret"

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
        try:
            import pycaret
        except ImportError:
            raise ImportError(
                "PyCaret is not installed. Please run `uv pip install pycaret` or `pip install pycaret` to install it."
            )

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
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with ModuleProgress(self.name, total=len(selected_targets), enabled=self.progress) as progress_bar:
                model_results, importances = _fit_target_models(
                    model_frame,
                    numeric_features,
                    selected_targets,
                    progress_bar,
                )

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = PyCaretRunMetadata(
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
            time_budget=TIME_BUDGET_SECONDS,
        )

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "model_results": model_results.to_dict(orient="records"),
                    "top_feature_importance": importances.head(100).to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "scores.csv", model_results),
            _write_csv(run_dir / "importance.csv", importances),
        ]

        markdown = _render_markdown(metadata, model_results, importances)
        md_path = run_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")
        artifacts.append(md_path)

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, model_results, importances), encoding="utf-8")
        artifacts.append(html_path)

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _fit_target_models(
    merged: pd.DataFrame,
    feature_columns: list[str],
    label_columns: list[str],
    progress_bar: ModuleProgress | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    if not feature_columns:
        return _model_results_frame(result_rows), _importance_frame(importance_rows)

    x = merged[feature_columns].apply(_numeric_series)

    for label in label_columns:
        y_raw = merged[label]
        y_numeric = _numeric_series(y_raw)
        is_numeric_target = y_numeric.notna().sum() >= 10 and y_numeric.nunique(dropna=True) > MAX_CLASS_COUNT
        if is_numeric_target:
            result, importance = _fit_regression(label, x, y_numeric)
        else:
            result, importance = _fit_classification(label, x, y_raw)

        if result is not None:
            result_rows.append(result)
        importance_rows.extend(importance)

        if progress_bar is not None:
            progress_bar.step(label)

    return _model_results_frame(result_rows), _importance_frame(importance_rows)


def _fit_regression(
    label: str, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    from pycaret.regression import setup, compare_models, predict_model

    frame = pd.concat([features, target.rename(label)], axis=1).dropna(subset=[label])
    if len(frame) < 30 or frame[label].nunique() < 2:
        return None, []

    x_train, x_test, y_train, y_test = train_test_split(
        frame[features.columns],
        frame[label],
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_data = pd.concat([x_train, y_train], axis=1)

    # Initialize PyCaret setup
    setup(
        data=train_data,
        target=label,
        session_id=RANDOM_STATE,
        verbose=False,
        html=False,
    )

    # Compare models with a time budget
    best_model = compare_models(budget_time=TIME_BUDGET_SECONDS, verbose=False)
    if best_model is None:
        return None, []

    # Evaluate on holdout/test set
    preds_df = predict_model(best_model, data=x_test)
    # PyCaret regression output typically adds a 'prediction_label' column
    predictions = preds_df["prediction_label"].values

    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

    # Extract feature importance dynamically
    feature_imp_vals = None
    try:
        # Check standard properties
        if hasattr(best_model, "feature_importances_") and best_model.feature_importances_ is not None:
            feature_imp_vals = best_model.feature_importances_
        elif hasattr(best_model, "coef_") and best_model.coef_ is not None:
            feature_imp_vals = np.abs(best_model.coef_)
        elif hasattr(best_model, "named_steps") and "model" in best_model.named_steps:
            inner_model = best_model.named_steps["model"]
            if hasattr(inner_model, "feature_importances_"):
                feature_imp_vals = inner_model.feature_importances_
            elif hasattr(inner_model, "coef_"):
                feature_imp_vals = np.abs(inner_model.coef_)
    except Exception:
        pass

    if feature_imp_vals is None or len(feature_imp_vals) != len(features.columns):
        feature_imp_vals = np.zeros(len(features.columns))

    importance = _feature_importance(label, features.columns, feature_imp_vals)
    model_name = type(best_model).__name__

    return (
        {
            "label": label,
            "task": "regression",
            "model": f"PyCaret_{model_name}",
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
    from pycaret.classification import setup, compare_models, predict_model

    frame = pd.concat([features, target.rename(label)], axis=1).dropna(subset=[label])
    class_count = int(frame[label].nunique(dropna=True))
    if len(frame) < 30 or class_count < 2 or class_count > MAX_CLASS_COUNT:
        return None, []

    encoder = LabelEncoder()
    # PyCaret can handle string/categorical target, but mapping to encoded is safer
    y_encoded = encoder.fit_transform(frame[label].astype(str))
    class_sizes = np.bincount(y_encoded)
    if np.min(class_sizes) < 2:
        return None, []

    x_train, x_test, y_train, y_test = train_test_split(
        frame[features.columns],
        y_encoded,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    train_data = pd.concat([x_train, pd.Series(y_train, name=label)], axis=1)

    # Initialize PyCaret setup
    setup(
        data=train_data,
        target=label,
        session_id=RANDOM_STATE,
        verbose=False,
        html=False,
    )

    # Compare models with a time budget
    best_model = compare_models(budget_time=TIME_BUDGET_SECONDS, verbose=False)
    if best_model is None:
        return None, []

    # Evaluate on test set
    preds_df = predict_model(best_model, data=x_test)
    # PyCaret classification output typically adds 'prediction_label' column
    predictions = preds_df["prediction_label"].values

    # Extract feature importance dynamically
    feature_imp_vals = None
    try:
        # Check standard properties
        if hasattr(best_model, "feature_importances_") and best_model.feature_importances_ is not None:
            feature_imp_vals = best_model.feature_importances_
        elif hasattr(best_model, "coef_") and best_model.coef_ is not None:
            feature_imp_vals = np.abs(best_model.coef_)
        elif hasattr(best_model, "named_steps") and "model" in best_model.named_steps:
            inner_model = best_model.named_steps["model"]
            if hasattr(inner_model, "feature_importances_"):
                feature_imp_vals = inner_model.feature_importances_
            elif hasattr(inner_model, "coef_"):
                feature_imp_vals = np.abs(inner_model.coef_)
    except Exception:
        pass

    if feature_imp_vals is None or len(feature_imp_vals) != len(features.columns):
        feature_imp_vals = np.zeros(len(features.columns))

    importance = _feature_importance(label, features.columns, feature_imp_vals)
    model_name = type(best_model).__name__

    return (
        {
            "label": label,
            "task": "classification",
            "model": f"PyCaret_{model_name}",
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


def _feature_importance(
    label: str, feature_names: pd.Index, importances: np.ndarray
) -> list[dict[str, object]]:
    values = np.asarray(importances, dtype=float).reshape(-1)
    rows = [
        {
            "label": label,
            "feature": str(feature),
            "importance": _round(float(value)),
            "importance_name": "pycaret_feature_importance",
        }
        for feature, value in zip(feature_names, values, strict=False)
    ]
    return sorted(rows, key=lambda row: row["importance"] or 0, reverse=True)


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


def _render_markdown(
    metadata: PyCaretRunMetadata, model_results: pd.DataFrame, importances: pd.DataFrame
) -> str:
    scores = _markdown_table(model_results) if not model_results.empty else "No PyCaret models were available."
    top_importance = (
        _markdown_table(importances.groupby("label", group_keys=False).head(10))
        if not importances.empty
        else "No feature importance was available."
    )
    ignored = ", ".join(metadata.ignored_columns) if metadata.ignored_columns else "none"
    return f"""# PyCaret AutoML Feature/Label Relationship Report

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
- Time budget per target: {metadata.time_budget} seconds

## Model Scores

{scores}

## Top Feature Importance

{top_importance}

## Artifacts

- `summary.json`
- `scores.csv`
- `importance.csv`
"""


def _render_html(markdown: str, model_results: pd.DataFrame, importances: pd.DataFrame) -> str:
    scores = model_results.to_html(index=False, classes="data-table") if not model_results.empty else ""
    top_importance = (
        importances.groupby("label", group_keys=False).head(20).to_html(index=False, classes="data-table")
        if not importances.empty
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PyCaret AutoML Feature/Label Relationship Report</title>
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
  <h2>Top Feature Importance</h2>
  {top_importance}
</body>
</html>
"""
