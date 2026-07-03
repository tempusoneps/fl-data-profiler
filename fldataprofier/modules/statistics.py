from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fldataprofier.modules.base import ModuleResult


@dataclass(frozen=True)
class DatasetShape:
    rows: int
    columns: int


@dataclass(frozen=True)
class RunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    targets: list[str]


class StatisticsModule:
    name = "statistics"

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        features = pd.read_csv(feature_csv, parse_dates=['Date'], index_col='Date')
        labels = pd.read_csv(label_csv, parse_dates=['Date'], index_col='Date')
        merged, feature_columns, label_columns, join_strategy = _merge_inputs(
            features, labels, join_key
        )

        selected_targets = _select_targets(label_columns, targets)
        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        feature_profile = _profile_frame(merged[feature_columns])
        label_profile = _profile_frame(merged[selected_targets])
        correlations = _feature_label_correlations(
            merged, feature_columns, selected_targets
        )
        target_summary = _target_summary(merged, feature_columns, selected_targets)

        metadata = RunMetadata(
            module=self.name,
            created_at=datetime.now(timezone.utc).isoformat(),
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
        ]

        heatmap_path = run_dir / "feature_label_correlation_heatmap.png"
        _write_heatmap(heatmap_path, correlations)
        artifacts.append(heatmap_path)

        markdown = _render_markdown(metadata, feature_profile, label_profile, correlations)
        md_path = run_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")
        artifacts.append(md_path)

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, correlations), encoding="utf-8")
        artifacts.append(html_path)

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _merge_inputs(
    features: pd.DataFrame, labels: pd.DataFrame, join_key: str | None
) -> tuple[pd.DataFrame, list[str], list[str], str]:
    if join_key:
        if _has_join_key(features, join_key) and _has_join_key(labels, join_key):
            left = _frame_with_join_key(features, join_key)
            right = _frame_with_join_key(labels, join_key)
            merged = left.merge(right, on=join_key, how="inner", suffixes=("", "__label"))
            feature_columns = [column for column in features.columns if column != join_key]
            label_columns = [
                _label_output_name(column, features.columns) for column in labels.columns if column != join_key
            ]
            return merged, feature_columns, label_columns, f"inner join on {join_key}"
        else:
            raise ValueError(f"--join-key {join_key!r} must exist in both CSV files")

    if features.index.name and features.index.name == labels.index.name:
        label_frame = labels.rename(columns=lambda column: _label_output_name(column, features.columns))
        merged = features.join(label_frame, how="inner")
        return (
            merged,
            list(features.columns),
            [_label_output_name(column, features.columns) for column in labels.columns],
            f"inner join on common index {features.index.name}",
        )

    common_columns = [column for column in features.columns if column in labels.columns]
    if common_columns:
        key = common_columns[0]
        merged = features.merge(labels, on=key, how="inner", suffixes=("", "__label"))
        feature_columns = [column for column in features.columns if column != key]
        label_columns = [
            _label_output_name(column, features.columns) for column in labels.columns if column != key
        ]
        return merged, feature_columns, label_columns, f"inner join on common column {key}"

    row_count = min(len(features), len(labels))
    if len(features) != len(labels):
        raise ValueError(
            "feature.csv and label.csv have different row counts and no common join column. "
            "Pass --join-key to align rows explicitly."
        )
    merged = pd.concat(
        [
            features.reset_index(drop=True),
            labels.reset_index(drop=True).rename(columns=lambda column: _label_output_name(column, features.columns)),
        ],
        axis=1,
    )
    return (
        merged,
        list(features.columns),
        [_label_output_name(column, features.columns) for column in labels.columns],
        f"row index alignment for {row_count} rows",
    )


def _has_join_key(frame: pd.DataFrame, join_key: str) -> bool:
    return join_key in frame.columns or frame.index.name == join_key


def _frame_with_join_key(frame: pd.DataFrame, join_key: str) -> pd.DataFrame:
    if join_key in frame.columns:
        return frame
    return frame.reset_index()


def _label_output_name(column: str, feature_columns: pd.Index) -> str:
    return f"{column}__label" if column in feature_columns else column


def _select_targets(label_columns: list[str], targets: list[str] | None) -> list[str]:
    if not targets:
        return label_columns

    missing = sorted(set(targets) - set(label_columns))
    if missing:
        available = ", ".join(label_columns)
        raise ValueError(f"Unknown target column(s): {missing}. Available labels: {available}")
    return targets


def _profile_frame(frame: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        series = frame[column]
        numeric = _numeric_series(series)
        row: dict[str, object] = {
            "column": column,
            "dtype": str(series.dtype),
            "rows": int(len(series)),
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
                    "samples": int(len(pair)),
                }
            )
    columns = ["feature", "label", "pearson_correlation", "abs_correlation", "samples"]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(["abs_correlation", "samples"], ascending=[False, False]).reset_index(drop=True)


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


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _write_heatmap(path: Path, correlations: pd.DataFrame) -> None:
    if correlations.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No numeric feature/label correlations", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    pivot = correlations.pivot(index="feature", columns="label", values="pearson_correlation").fillna(0)
    height = max(3, min(12, 0.45 * len(pivot.index) + 1.5))
    width = max(5, min(14, 1.2 * len(pivot.columns) + 4))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(pivot.values, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("Feature / label Pearson correlation")
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


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = [str(column) for column in frame.columns]
    body = [[_markdown_cell(row[column]) for column in frame.columns] for _, row in frame.iterrows()]
    width_rows = [columns, *body]
    widths = [max(len(values[index]) for values in width_rows) for index in range(len(columns))]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    separator = ["-" * width for width in widths]
    return "\n".join([render_row(columns), render_row(separator), *[render_row(row) for row in body]])


def _markdown_cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _render_html(markdown: str, correlations: pd.DataFrame) -> str:
    escaped_markdown = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    table = correlations.head(25).to_html(index=False, classes="data-table") if not correlations.empty else ""
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
  <pre>{escaped_markdown}</pre>
  <h2>Top Correlations</h2>
  {table}
  <img src="feature_label_correlation_heatmap.png" alt="Feature label correlation heatmap">
</body>
</html>
"""


def _round(value: float) -> float | None:
    if np.isnan(value) or np.isinf(value):
        return None
    return round(value, 6)
