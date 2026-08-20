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

from fldataprofiler.modules.base import ModuleResult
from fldataprofiler.modules.progress import ModuleProgress
from fldataprofiler.utils import (
    _format_duration,
    _html_markdown_details,
    _markdown_table,
    _merge_inputs,
    _numeric_series,
    _read_table_with_date_index,
    _round,
    _select_targets,
    _write_csv,
    _write_json,
)


@dataclass(frozen=True)
class DatasetShape:
    rows: int
    columns: int


@dataclass(frozen=True)
class RunMetadata:
    module: str
    created_at: str
    execution_time: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    targets: list[str]


class StatisticsModule:
    name = "statistics"

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
        with ModuleProgress(self.name, total=4, enabled=self.progress) as progress_bar:
            features = _read_table_with_date_index(feature_csv)
            labels = _read_table_with_date_index(label_csv)
            merged, feature_columns, label_columns, join_strategy = _merge_inputs(
                features, labels, join_key
            )

            selected_targets = _select_targets(label_columns, targets)
            run_dir = output_dir / self.name
            run_dir.mkdir(parents=True, exist_ok=True)
            progress_bar.step("load")

            feature_profile = _profile_frame(merged[feature_columns])
            label_profile = _profile_frame(merged[selected_targets])
            correlations = _feature_label_correlations(merged, feature_columns, selected_targets)
            target_summary = _target_summary(merged, feature_columns, selected_targets)
            progress_bar.step("profile")

            heatmap_path = run_dir / "feature_label_correlation_heatmap.png"
            _write_heatmap(heatmap_path, correlations)
            progress_bar.step("heatmap")

            metadata = RunMetadata(
                module=self.name,
                created_at=datetime.now(UTC).isoformat(),
                execution_time=_format_duration(time.perf_counter() - start_time),
                feature_csv=str(feature_csv),
                label_csv=str(label_csv),
                join_strategy=join_strategy,
                feature_shape=DatasetShape(*features.shape),
                label_shape=DatasetShape(*labels.shape),
                merged_shape=DatasetShape(*merged.shape),
                targets=selected_targets,
            )

            artifacts = [
                _write_json(
                    run_dir / "statistics_summary.json",
                    {
                        "metadata": asdict(metadata),
                        "feature_profile": feature_profile,
                        "label_profile": label_profile,
                        "target_summary": target_summary,
                        "top_relationships": correlations.head(25).to_dict(orient="records"),
                    },
                ),
                _write_csv(run_dir / "feature_profile.csv", pd.DataFrame(feature_profile)),
                _write_csv(run_dir / "label_profile.csv", pd.DataFrame(label_profile)),
                _write_csv(run_dir / "feature_label_correlations.csv", correlations),
                heatmap_path,
            ]

            markdown = _render_markdown(metadata, feature_profile, label_profile, correlations)
            md_path = run_dir / "report.md"
            md_path.write_text(markdown, encoding="utf-8")
            artifacts.append(md_path)

            html_path = run_dir / "report.html"
            html_path.write_text(_render_html(markdown, correlations), encoding="utf-8")
            artifacts.append(html_path)
            progress_bar.step("artifacts")

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _profile_frame(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column]
        numeric = _numeric_series(series)
        row: dict[str, object] = {
            "column": column,
            "dtype": str(series.dtype),
            "rows": len(series),
            "missing": int(series.isna().sum()),
            "missing_pct": _round(float(series.isna().mean() * 100)),
            "unique": int(series.nunique(dropna=True)),
        }
        if numeric.notna().sum() > 0:
            row.update(
                {
                    "mean": _round(float(numeric.mean())),
                    "std": _round(float(numeric.std())),
                    "min": _round(float(numeric.min())),
                    "median": _round(float(numeric.median())),
                    "max": _round(float(numeric.max())),
                }
            )
        rows.append(row)
    return rows


def _feature_label_correlations(
    merged: pd.DataFrame, feature_columns: list[str], label_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        feature_values = _numeric_series(merged[feature])
        if feature_values.notna().sum() < 2:
            continue
        for label in label_columns:
            label_values = _numeric_series(merged[label])
            pair = pd.concat([feature_values, label_values], axis=1).dropna()
            if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                continue
            corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            if pd.isna(corr):
                continue
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "pearson_correlation": _round(float(corr)),
                    "abs_correlation": _round(float(abs(corr))),
                    "samples": len(pair),
                }
            )
    columns = ["feature", "label", "pearson_correlation", "abs_correlation", "samples"]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(["abs_correlation", "samples"], ascending=[False, False]).reset_index(
        drop=True
    )


def _target_summary(
    merged: pd.DataFrame, feature_columns: list[str], label_columns: list[str]
) -> dict[str, object]:
    summary: dict[str, object] = {}
    numeric_features = [
        column for column in feature_columns if _numeric_series(merged[column]).notna().sum() > 0
    ]
    for label in label_columns:
        values = merged[label]
        entry: dict[str, object] = {
            "unique": int(values.nunique(dropna=True)),
            "missing": int(values.isna().sum()),
        }
        if values.nunique(dropna=True) <= 20:
            entry["distribution"] = values.value_counts(dropna=False).head(20).to_dict()
        if numeric_features:
            numeric_label = _numeric_series(values)
            if numeric_label.notna().sum() > 0:
                entry["numeric_feature_means_by_label_quantile"] = _means_by_label_quantile(
                    merged, numeric_features, numeric_label
                )
        summary[label] = entry
    return summary


def _means_by_label_quantile(
    merged: pd.DataFrame, numeric_features: list[str], numeric_label: pd.Series
) -> dict[str, dict[str, float | None]]:
    try:
        buckets = pd.qcut(numeric_label, q=min(4, numeric_label.nunique()), duplicates="drop")
    except ValueError:
        return {}
    result: dict[str, dict[str, float | None]] = {}
    frame = merged[numeric_features].apply(_numeric_series)
    for bucket, group in frame.groupby(buckets, observed=False):
        result[str(bucket)] = {
            column: _round(float(value))
            for column, value in group.mean(numeric_only=True).dropna().to_dict().items()
        }
    return result


MAX_HEATMAP_FEATURES_PER_LABEL = 10
MAX_TOTAL_HEATMAP_FEATURES = 30


def _select_top_heatmap_features(
    correlations: pd.DataFrame,
    top_k_per_label: int = MAX_HEATMAP_FEATURES_PER_LABEL,
    max_total: int = MAX_TOTAL_HEATMAP_FEATURES,
) -> list[str]:
    if correlations.empty:
        return []

    unique_features = correlations["feature"].unique()
    if len(unique_features) <= max_total:
        max_abs = correlations.groupby("feature")["abs_correlation"].max()
        return list(max_abs.sort_values(ascending=False).index)

    # Union of Top-K features per label
    selected_features: set[str] = set()
    for _, group in correlations.groupby("label"):
        top_in_group = group.sort_values("abs_correlation", ascending=False).head(top_k_per_label)
        selected_features.update(top_in_group["feature"].tolist())

    # Rank the selected union by maximum absolute correlation across all labels
    filtered_corrs = correlations[correlations["feature"].isin(selected_features)]
    max_abs = filtered_corrs.groupby("feature")["abs_correlation"].max()
    ranked_features = list(max_abs.sort_values(ascending=False).index)

    return ranked_features[:max_total]


def _write_heatmap(
    path: Path,
    correlations: pd.DataFrame,
    top_k_per_label: int = MAX_HEATMAP_FEATURES_PER_LABEL,
    max_total: int = MAX_TOTAL_HEATMAP_FEATURES,
) -> None:
    if correlations.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No numeric feature/label correlations", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    top_features = _select_top_heatmap_features(correlations, top_k_per_label, max_total)
    plot_df = correlations[correlations["feature"].isin(top_features)]

    pivot = (
        plot_df.pivot(index="feature", columns="label", values="pearson_correlation")
        .reindex(index=top_features)
        .fillna(0)
    )

    height = max(3.5, min(14.0, 0.38 * len(pivot.index) + 1.8))
    width = max(5.5, min(14.0, 1.2 * len(pivot.columns) + 4.5))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)

    total_features_count = correlations["feature"].nunique()
    if len(top_features) < total_features_count:
        ax.set_title(
            f"Feature / Label Pearson Correlation (Top {len(top_features)} of {total_features_count} Features)"
        )
    else:
        ax.set_title("Feature / Label Pearson Correlation")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_markdown(
    metadata: RunMetadata,
    feature_profile: list[dict[str, object]],
    label_profile: list[dict[str, object]],
    correlations: pd.DataFrame,
) -> str:
    top_rows = correlations.head(10)
    top_table = (
        _markdown_table(top_rows)
        if not top_rows.empty
        else "No numeric feature/label correlations were available."
    )
    return f"""# Feature/Label Statistics Report

## Run

- Module: `{metadata.module}`
- Created at: `{metadata.created_at}`
- Execution time: `{metadata.execution_time}`
- Feature CSV: `{metadata.feature_csv}`
- Label CSV: `{metadata.label_csv}`
- Join strategy: {metadata.join_strategy}
- Feature shape: {metadata.feature_shape.rows} rows x {metadata.feature_shape.columns} columns
- Label shape: {metadata.label_shape.rows} rows x {metadata.label_shape.columns} columns
- Merged shape: {metadata.merged_shape.rows} rows x {metadata.merged_shape.columns} columns
- Targets: {", ".join(metadata.targets)}

## Top Relationships

{top_table}

## Feature Columns

{_markdown_table(pd.DataFrame(feature_profile))}

## Label Columns

{_markdown_table(pd.DataFrame(label_profile))}

## Artifacts

- `statistics_summary.json`
- `feature_profile.csv`
- `label_profile.csv`
- `feature_label_correlations.csv`
- `feature_label_correlation_heatmap.png`
"""


def _render_html(markdown: str, correlations: pd.DataFrame) -> str:
    table = (
        correlations.head(25).to_html(index=False, classes="data-table")
        if not correlations.empty
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Feature/Label Statistics Report</title>
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
  <h2>Top Correlations</h2>
  {table}
  <img src="feature_label_correlation_heatmap.png" alt="Feature label correlation heatmap">
</body>
</html>
"""
