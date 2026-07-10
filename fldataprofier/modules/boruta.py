from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
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
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
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


MAX_ROWS = 10_000
MAX_CLASS_COUNT = 50
RANDOM_STATE = 42
BORUTA_ITERATIONS = 30
P_VALUE = 0.05


@dataclass(frozen=True)
class BorutaRunMetadata:
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
    iterations: int
    p_value: float


class BorutaRelationshipsModule:
    name = "boruta"

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
        model_results, selections = _fit_target_models(
            model_frame,
            numeric_features,
            selected_targets,
        )

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = BorutaRunMetadata(
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
            iterations=BORUTA_ITERATIONS,
            p_value=P_VALUE,
        )

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "model_results": model_results.to_dict(orient="records"),
                    "top_selected_features": selections.head(100).to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "scores.csv", model_results),
            _write_csv(run_dir / "boruta_features.csv", selections),
        ]

        markdown = _render_markdown(metadata, model_results, selections)
        md_path = run_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")
        artifacts.append(md_path)

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, model_results, selections), encoding="utf-8")
        artifacts.append(html_path)

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _fit_target_models(
    merged: pd.DataFrame, feature_columns: list[str], label_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    if not feature_columns:
        return _model_results_frame(result_rows), _selection_frame(selection_rows)

    x = merged[feature_columns].apply(_numeric_series)

    for label in label_columns:
        y_raw = merged[label]
        y_numeric = _numeric_series(y_raw)
        is_numeric_target = (
            y_numeric.notna().sum() >= 10
            and y_numeric.nunique(dropna=True) > MAX_CLASS_COUNT
        )
        if is_numeric_target:
            result, selections = _fit_regression(label, x, y_numeric)
        else:
            result, selections = _fit_classification(label, x, y_raw)

        if result is not None:
            result_rows.append(result)
        selection_rows.extend(selections)

    return _model_results_frame(result_rows), _selection_frame(selection_rows)


def _fit_regression(
    label: str, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    frame = pd.concat([features, target.rename(label)], axis=1).dropna(subset=[label])
    if len(frame) < 30 or frame[label].nunique() < 2:
        return None, []

    x = frame[features.columns]
    y = frame[label]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    imputer = SimpleImputer(strategy="median")
    x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=x.columns)
    x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=x.columns)
    model.fit(x_train_imputed, y_train)
    predictions = model.predict(x_test_imputed)
    selections = _boruta_select(label, "regression", x, y)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    return (
        {
            "label": label,
            "task": "regression",
            "model": "RandomForestRegressor + Boruta shadow features",
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
        selections,
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
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    imputer = SimpleImputer(strategy="median")
    x_train_imputed = pd.DataFrame(imputer.fit_transform(x_train), columns=features.columns)
    x_test_imputed = pd.DataFrame(imputer.transform(x_test), columns=features.columns)
    model.fit(x_train_imputed, y_train)
    predictions = model.predict(x_test_imputed)
    selections = _boruta_select(label, "classification", frame[features.columns], y)
    return (
        {
            "label": label,
            "task": "classification",
            "model": "RandomForestClassifier + Boruta shadow features",
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
        selections,
    )


def _boruta_select(
    label: str, task: str, features: pd.DataFrame, target: pd.Series | np.ndarray
) -> list[dict[str, object]]:
    rng = np.random.default_rng(RANDOM_STATE)
    imputer = SimpleImputer(strategy="median")
    x = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
    hit_counts = np.zeros(len(features.columns), dtype=int)
    mean_importance = np.zeros(len(features.columns), dtype=float)
    mean_shadow_threshold = 0.0

    for iteration in range(BORUTA_ITERATIONS):
        shadow = x.apply(lambda column: rng.permutation(column.to_numpy()))
        shadow.columns = [f"shadow_{column}" for column in x.columns]
        train_x = pd.concat([x, shadow], axis=1)
        model = _boruta_forest(task, iteration)
        model.fit(train_x, target)
        importances = np.asarray(model.feature_importances_, dtype=float)
        real_importances = importances[: len(features.columns)]
        shadow_threshold = float(np.max(importances[len(features.columns) :]))
        hit_counts += real_importances > shadow_threshold
        mean_importance += real_importances
        mean_shadow_threshold += shadow_threshold

    mean_importance /= BORUTA_ITERATIONS
    mean_shadow_threshold /= BORUTA_ITERATIONS

    rows = []
    for feature, hits, importance in zip(features.columns, hit_counts, mean_importance, strict=False):
        accepted_p = binomtest(int(hits), BORUTA_ITERATIONS, 0.5, alternative="greater").pvalue
        rejected_p = binomtest(int(hits), BORUTA_ITERATIONS, 0.5, alternative="less").pvalue
        if accepted_p < P_VALUE:
            decision = "confirmed"
        elif rejected_p < P_VALUE:
            decision = "rejected"
        else:
            decision = "tentative"
        rows.append(
            {
                "label": label,
                "task": task,
                "feature": str(feature),
                "decision": decision,
                "hits": int(hits),
                "iterations": BORUTA_ITERATIONS,
                "hit_rate": _round(float(hits / BORUTA_ITERATIONS)),
                "mean_importance": _round(float(importance)),
                "mean_shadow_threshold": _round(mean_shadow_threshold),
                "accepted_p_value": _round(float(accepted_p)),
                "rejected_p_value": _round(float(rejected_p)),
            }
        )
    return sorted(rows, key=lambda row: (row["decision"] != "confirmed", -(row["hit_rate"] or 0)))


def _boruta_forest(task: str, iteration: int) -> RandomForestClassifier | RandomForestRegressor:
    params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": RANDOM_STATE + iteration,
    }
    if task == "classification":
        return RandomForestClassifier(class_weight="balanced", **params)
    return RandomForestRegressor(**params)


def _selection_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "label",
        "task",
        "feature",
        "decision",
        "hits",
        "iterations",
        "hit_rate",
        "mean_importance",
        "mean_shadow_threshold",
        "accepted_p_value",
        "rejected_p_value",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    decision_rank = {"confirmed": 0, "tentative": 1, "rejected": 2}
    return (
        frame.assign(_decision_rank=frame["decision"].map(decision_rank))
        .sort_values(
            ["label", "_decision_rank", "hit_rate", "mean_importance"],
            ascending=[True, True, False, False],
            na_position="last",
        )
        .drop(columns=["_decision_rank"])
        .reset_index(drop=True)
    )


def _render_markdown(
    metadata: BorutaRunMetadata, model_results: pd.DataFrame, selections: pd.DataFrame
) -> str:
    scores = _markdown_table(model_results) if not model_results.empty else "No Boruta models were available."
    top_features = (
        _markdown_table(selections.groupby("label", group_keys=False).head(15))
        if not selections.empty
        else "No Boruta feature decisions were available."
    )
    ignored = ", ".join(metadata.ignored_columns) if metadata.ignored_columns else "none"
    return f"""# Boruta Feature Selection Report

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
- Iterations: {metadata.iterations}
- P-value threshold: {metadata.p_value}
- Ignored columns: {ignored}
- Numeric features: {len(metadata.features)}
- Targets: {", ".join(metadata.targets)}

## Model Scores

{scores}

## Feature Decisions

{top_features}

## Artifacts

- `summary.json`
- `scores.csv`
- `boruta_features.csv`
"""


def _render_html(markdown: str, model_results: pd.DataFrame, selections: pd.DataFrame) -> str:
    escaped_markdown = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    scores = model_results.to_html(index=False, classes="data-table") if not model_results.empty else ""
    top_features = (
        selections.groupby("label", group_keys=False).head(30).to_html(index=False, classes="data-table")
        if not selections.empty
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Boruta Feature Selection Report</title>
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
  <h2>Feature Decisions</h2>
  {top_features}
</body>
</html>
"""
