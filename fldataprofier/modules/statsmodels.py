from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.statistics import (
    DatasetShape,
    _markdown_table,
    _merge_inputs,
    _numeric_series,
    _round,
    _select_targets,
)


MAX_ROWS = 50_000
MAX_FEATURES_PER_LABEL = 25
RANDOM_STATE = 42


@dataclass(frozen=True)
class StatsmodelsRunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    model_rows: int
    max_features_per_label: int
    targets: list[str]
    ignored_columns: list[str]


class StatsmodelsRelationshipsModule:
    name = "statsmodels"

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

        model_frame = _sample_rows(merged[[*feature_columns, *selected_targets]])
        model_results, coefficients = _fit_ols_models(model_frame, feature_columns, selected_targets)

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = StatsmodelsRunMetadata(
            module=self.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=DatasetShape(*features.shape),
            label_shape=DatasetShape(*labels.shape),
            merged_shape=DatasetShape(*merged.shape),
            model_rows=len(model_frame),
            max_features_per_label=MAX_FEATURES_PER_LABEL,
            targets=selected_targets,
            ignored_columns=ignored_columns,
        )

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "model_results": model_results.to_dict(orient="records"),
                    "top_coefficients": coefficients.head(100).to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "scores.csv", model_results),
            _write_csv(run_dir / "coefficients.csv", coefficients),
        ]

        markdown = _render_markdown(metadata, model_results, coefficients)
        md_path = run_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")
        artifacts.append(md_path)

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, model_results, coefficients), encoding="utf-8")
        artifacts.append(html_path)

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _date_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if str(column).lower() == "date"]


def _sample_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= MAX_ROWS:
        return frame
    return frame.sample(n=MAX_ROWS, random_state=RANDOM_STATE).sort_index()


def _fit_ols_models(
    merged: pd.DataFrame, feature_columns: list[str], label_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, object]] = []
    coefficient_rows: list[dict[str, object]] = []

    numeric_features = [
        column
        for column in feature_columns
        if _numeric_series(merged[column]).notna().sum() >= 30
        and _numeric_series(merged[column]).nunique(dropna=True) >= 2
    ]
    if not numeric_features:
        return _model_frame(model_rows), _coefficient_frame(coefficient_rows)

    for label in label_columns:
        y = _numeric_series(merged[label])
        if y.notna().sum() < 30 or y.nunique(dropna=True) < 2:
            continue
        selected_features = _select_features_by_correlation(merged, numeric_features, y)
        if not selected_features:
            continue
        result, coefficients = _fit_single_ols(label, merged[selected_features], y)
        if result is None:
            continue
        model_rows.append(result)
        coefficient_rows.extend(coefficients)

    return _model_frame(model_rows), _coefficient_frame(coefficient_rows)


def _select_features_by_correlation(
    merged: pd.DataFrame, feature_columns: list[str], target: pd.Series
) -> list[str]:
    rows: list[tuple[str, float]] = []
    for feature in feature_columns:
        x = _numeric_series(merged[feature])
        pair = pd.concat([x, target], axis=1).dropna()
        if len(pair) < 30 or pair.iloc[:, 0].nunique() < 2:
            continue
        correlation = pair.iloc[:, 0].corr(pair.iloc[:, 1])
        if pd.notna(correlation):
            rows.append((feature, abs(float(correlation))))
    rows.sort(key=lambda item: item[1], reverse=True)
    return [feature for feature, _ in rows[:MAX_FEATURES_PER_LABEL]]


def _fit_single_ols(
    label: str, features: pd.DataFrame, target: pd.Series
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    frame = pd.concat([features.apply(_numeric_series), target.rename(label)], axis=1).dropna()
    if len(frame) < 30:
        return None, []

    x = sm.add_constant(frame[features.columns], has_constant="add")
    y = frame[label]
    try:
        model = sm.OLS(y, x).fit()
    except (np.linalg.LinAlgError, ValueError):
        return None, []

    coefficients: list[dict[str, object]] = []
    conf_int = model.conf_int()
    for feature in features.columns:
        coefficients.append(
            {
                "label": label,
                "feature": feature,
                "coefficient": _round(float(model.params.get(feature, np.nan))),
                "abs_coefficient": _round(abs(float(model.params.get(feature, np.nan)))),
                "p_value": _round(float(model.pvalues.get(feature, np.nan))),
                "t_value": _round(float(model.tvalues.get(feature, np.nan))),
                "std_error": _round(float(model.bse.get(feature, np.nan))),
                "ci_low": _round(float(conf_int.loc[feature, 0])),
                "ci_high": _round(float(conf_int.loc[feature, 1])),
                "samples": int(model.nobs),
            }
        )

    return (
        {
            "label": label,
            "model": "OLS",
            "samples": int(model.nobs),
            "features": int(len(features.columns)),
            "r_squared": _round(float(model.rsquared)),
            "adjusted_r_squared": _round(float(model.rsquared_adj)),
            "f_statistic": _round(float(model.fvalue)),
            "f_p_value": _round(float(model.f_pvalue)),
            "aic": _round(float(model.aic)),
            "bic": _round(float(model.bic)),
            "note": "Numeric labels only; top features selected by absolute Pearson correlation before OLS.",
        },
        coefficients,
    )


def _model_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "label",
        "model",
        "samples",
        "features",
        "r_squared",
        "adjusted_r_squared",
        "f_statistic",
        "f_p_value",
        "aic",
        "bic",
        "note",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(["adjusted_r_squared", "samples"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _coefficient_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "label",
        "feature",
        "coefficient",
        "abs_coefficient",
        "p_value",
        "t_value",
        "std_error",
        "ci_low",
        "ci_high",
        "samples",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    return frame.sort_values(["label", "p_value", "abs_coefficient"], ascending=[True, True, False], na_position="last").reset_index(drop=True)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _render_markdown(
    metadata: StatsmodelsRunMetadata, model_results: pd.DataFrame, coefficients: pd.DataFrame
) -> str:
    scores = _markdown_table(model_results) if not model_results.empty else "No statsmodels OLS models were available."
    top_coefficients = (
        _markdown_table(coefficients.groupby("label", group_keys=False).head(10))
        if not coefficients.empty
        else "No coefficient table was available."
    )
    ignored = ", ".join(metadata.ignored_columns) if metadata.ignored_columns else "none"
    return f"""# Statsmodels Feature/Label Relationship Report

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
- Max features per label: {metadata.max_features_per_label}
- Ignored columns: {ignored}
- Targets: {", ".join(metadata.targets)}

## Model Scores

{scores}

## Top Coefficients

{top_coefficients}

## Artifacts

- `summary.json`
- `scores.csv`
- `coefficients.csv`
"""


def _render_html(markdown: str, model_results: pd.DataFrame, coefficients: pd.DataFrame) -> str:
    escaped_markdown = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    scores = model_results.to_html(index=False, classes="data-table") if not model_results.empty else ""
    top_coefficients = (
        coefficients.groupby("label", group_keys=False).head(20).to_html(index=False, classes="data-table")
        if not coefficients.empty
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Statsmodels Feature/Label Relationship Report</title>
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
  <h2>Top Coefficients</h2>
  {top_coefficients}
</body>
</html>
"""
