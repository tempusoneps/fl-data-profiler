from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _html_markdown_details,
    _read_table,
    _markdown_table,
    _numeric_series,
    _round,
    _write_csv,
    _write_json,
)


MAX_CORRELATION_HEATMAP_COLUMNS = 80


@dataclass(frozen=True)
class EdaRunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    feature_shape: DatasetShape
    label_shape: DatasetShape


class EdaModule:
    name = "eda"

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
        with ModuleProgress(self.name, total=7, enabled=self.progress) as progress_bar:
            features = _read_table(feature_csv)
            labels = _read_table(label_csv)
            progress_bar.step("load")

            selected_labels = _select_label_columns(labels, targets)
            progress_bar.step("select")

            run_dir = output_dir / self.name
            run_dir.mkdir(parents=True, exist_ok=True)

            metadata = EdaRunMetadata(
                module=self.name,
                created_at=datetime.now(timezone.utc).isoformat(),
                feature_csv=str(feature_csv),
                label_csv=str(label_csv),
                feature_shape=DatasetShape(*features.shape),
                label_shape=DatasetShape(*labels.shape),
            )

            overview = _dataset_overview({"feature": features, "label": selected_labels})
            column_profile = _column_profile({"feature": features, "label": selected_labels})
            missingness = _missingness(column_profile)
            progress_bar.step("profile")

            numeric_summary = _numeric_summary({"feature": features, "label": selected_labels})
            categorical_summary = _categorical_summary({"feature": features, "label": selected_labels})
            progress_bar.step("summaries")

            artifacts = [
                _write_json(
                    run_dir / "summary.json",
                    {
                        "metadata": asdict(metadata),
                        "overview": overview.to_dict(orient="records"),
                        "top_missing_columns": missingness.head(25).to_dict(orient="records"),
                        "numeric_summary": numeric_summary.to_dict(orient="records"),
                        "categorical_summary": categorical_summary.to_dict(orient="records"),
                        "notes": _run_notes(join_key, targets),
                    },
                ),
                _write_csv(run_dir / "dataset_overview.csv", overview),
                _write_csv(run_dir / "columns_profile.csv", column_profile),
                _write_csv(run_dir / "missingness.csv", missingness),
                _write_csv(run_dir / "numeric_summary.csv", numeric_summary),
                _write_csv(run_dir / "categorical_summary.csv", categorical_summary),
            ]
            progress_bar.step("write_tables")

            feature_heatmap = run_dir / "feature_correlation_heatmap.png"
            label_heatmap = run_dir / "label_correlation_heatmap.png"
            _write_correlation_heatmap(feature_heatmap, features, "Feature numeric correlation")
            _write_correlation_heatmap(label_heatmap, selected_labels, "Label numeric correlation")
            artifacts.extend([feature_heatmap, label_heatmap])
            progress_bar.step("heatmaps")

            markdown = _render_markdown(metadata, overview, column_profile, missingness, numeric_summary)
            md_path = run_dir / "report.md"
            md_path.write_text(markdown, encoding="utf-8")
            artifacts.append(md_path)

            html_path = run_dir / "report.html"
            html_path.write_text(_render_html(markdown, overview, missingness), encoding="utf-8")
            artifacts.append(html_path)
            progress_bar.step("report")

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _select_label_columns(labels: pd.DataFrame, targets: list[str] | None) -> pd.DataFrame:
    if not targets:
        return labels
    missing = sorted(set(targets) - set(labels.columns))
    if missing:
        available = ", ".join(labels.columns)
        raise ValueError(f"Unknown target column(s): {missing}. Available labels: {available}")
    return labels[targets]


def _dataset_overview(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, frame in frames.items():
        missing = int(frame.isna().sum().sum())
        cells = int(frame.shape[0] * frame.shape[1])
        rows.append(
            {
                "dataset": dataset,
                "rows": int(frame.shape[0]),
                "columns": int(frame.shape[1]),
                "duplicate_rows": int(frame.duplicated().sum()),
                "total_missing": missing,
                "total_missing_pct": _round(missing / cells * 100) if cells else None,
                "numeric_columns": int(len(_numeric_columns(frame))),
                "categorical_columns": int(len(_categorical_columns(frame))),
                "datetime_columns": int(len(_datetime_columns(frame))),
            }
        )
    return pd.DataFrame(rows)


def _column_profile(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, frame in frames.items():
        for column in frame.columns:
            series = frame[column]
            numeric = _numeric_series(series)
            non_null = int(series.notna().sum())
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "dtype": str(series.dtype),
                    "inferred_type": _inferred_type(series),
                    "rows": int(len(series)),
                    "non_null": non_null,
                    "missing": int(series.isna().sum()),
                    "missing_pct": _round(float(series.isna().mean() * 100)),
                    "unique": int(series.nunique(dropna=True)),
                    "unique_pct": _round(series.nunique(dropna=True) / non_null * 100)
                    if non_null
                    else None,
                    "zero_count": int((numeric == 0).sum()) if numeric.notna().sum() else None,
                    "negative_count": int((numeric < 0).sum()) if numeric.notna().sum() else None,
                }
            )
    return pd.DataFrame(rows)


def _missingness(column_profile: pd.DataFrame) -> pd.DataFrame:
    result = column_profile[
        ["dataset", "column", "missing", "missing_pct", "non_null", "rows"]
    ].copy()
    return result.sort_values(["missing_pct", "missing"], ascending=[False, False]).reset_index(drop=True)


def _numeric_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, frame in frames.items():
        for column in _numeric_columns(frame):
            series = _numeric_series(frame[column]).dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "count": int(series.count()),
                    "mean": _round(float(series.mean())),
                    "std": _round(float(series.std())),
                    "min": _round(float(series.min())),
                    "q25": _round(float(series.quantile(0.25))),
                    "median": _round(float(series.median())),
                    "q75": _round(float(series.quantile(0.75))),
                    "max": _round(float(series.max())),
                    "skew": _round(float(series.skew())) if len(series) >= 3 else None,
                    "kurtosis": _round(float(series.kurtosis())) if len(series) >= 4 else None,
                }
            )
    return pd.DataFrame(rows)


def _categorical_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, frame in frames.items():
        for column in _categorical_columns(frame):
            counts = frame[column].value_counts(dropna=False).head(10)
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "unique": int(frame[column].nunique(dropna=True)),
                    "top_values": "; ".join(f"{value}={count}" for value, count in counts.items()),
                }
            )
    return pd.DataFrame(rows)


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if _numeric_series(frame[column]).notna().sum() > 0 and not _is_datetime(frame[column])
    ]


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    numeric = set(_numeric_columns(frame))
    datetimes = set(_datetime_columns(frame))
    return [column for column in frame.columns if column not in numeric and column not in datetimes]


def _datetime_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if _is_datetime(frame[column])]


def _inferred_type(series: pd.Series) -> str:
    if _is_datetime(series):
        return "datetime"
    numeric = _numeric_series(series)
    if numeric.notna().sum() > 0:
        return "numeric"
    return "categorical"


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _write_correlation_heatmap(path: Path, frame: pd.DataFrame, title: str) -> None:
    numeric_columns = _numeric_columns(frame)
    original_column_count = len(numeric_columns)
    if len(numeric_columns) > MAX_CORRELATION_HEATMAP_COLUMNS:
        numeric_columns = numeric_columns[:MAX_CORRELATION_HEATMAP_COLUMNS]
    if len(numeric_columns) < 2:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "Need at least 2 numeric columns", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    numeric_frame = frame[numeric_columns].apply(_numeric_series)
    corr = numeric_frame.corr().fillna(0)
    height = max(4, min(14, 0.45 * len(corr.index) + 2))
    width = max(5, min(14, 0.45 * len(corr.columns) + 3))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(corr.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(corr.index)), labels=corr.index)
    display_title = title
    if original_column_count > len(numeric_columns):
        display_title = f"{title} (first {len(numeric_columns)} of {original_column_count} numeric columns)"
    ax.set_title(display_title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _run_notes(join_key: str | None, targets: list[str] | None) -> list[str]:
    notes = [
        "EDA analyzes feature.csv and label.csv independently; it does not merge rows.",
    ]
    if join_key:
        notes.append("--join-key is accepted for CLI compatibility but is not used by EDA.")
    if targets:
        notes.append("Label EDA is limited to the selected --target columns.")
    return notes


def _render_markdown(
    metadata: EdaRunMetadata,
    overview: pd.DataFrame,
    column_profile: pd.DataFrame,
    missingness: pd.DataFrame,
    numeric_summary: pd.DataFrame,
) -> str:
    return f"""# Exploratory Data Analysis Report

## Run

- Module: `{metadata.module}`
- Created at: `{metadata.created_at}`
- Feature CSV: `{metadata.feature_csv}`
- Label CSV: `{metadata.label_csv}`
- Feature shape: {metadata.feature_shape.rows} rows x {metadata.feature_shape.columns} columns
- Label shape: {metadata.label_shape.rows} rows x {metadata.label_shape.columns} columns

## Dataset Overview

{_markdown_table(overview)}

## Top Missing Columns

{_markdown_table(missingness.head(20))}

## Numeric Summary

{_markdown_table(numeric_summary.head(30))}

## Column Profile

{_markdown_table(column_profile.head(50))}

## Artifacts

- `summary.json`
- `dataset_overview.csv`
- `columns_profile.csv`
- `missingness.csv`
- `numeric_summary.csv`
- `categorical_summary.csv`
- `feature_correlation_heatmap.png`
- `label_correlation_heatmap.png`
"""


def _render_html(markdown: str, overview: pd.DataFrame, missingness: pd.DataFrame) -> str:
    overview_table = overview.to_html(index=False, classes="data-table")
    missing_table = missingness.head(25).to_html(index=False, classes="data-table")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exploratory Data Analysis Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    pre {{ white-space: pre-wrap; background: #f5f7fa; padding: 16px; border-radius: 6px; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-top: 24px; }}
    .data-table th, .data-table td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    .data-table th {{ background: #edf2f7; }}
    img {{ max-width: 100%; margin-top: 24px; border: 1px solid #d9e2ec; }}
  </style>
</head>
<body>
  {_html_markdown_details(markdown)}
  <h2>Dataset Overview</h2>
  {overview_table}
  <h2>Top Missing Columns</h2>
  {missing_table}
  <h2>Feature Correlation</h2>
  <img src="feature_correlation_heatmap.png" alt="Feature numeric correlation heatmap">
  <h2>Label Correlation</h2>
  <img src="label_correlation_heatmap.png" alt="Label numeric correlation heatmap">
</body>
</html>
"""
