from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_INPUT_SUFFIXES = (".csv", ".parquet")


MODEL_RESULT_COLUMNS = [
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


def _is_supported_input_path(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES


def _supported_input_formats_message() -> str:
    return " or ".join(SUPPORTED_INPUT_SUFFIXES)


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path, low_memory=False)
    elif suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input file type for {path}. Expected {_supported_input_formats_message()}.")

    if "Date" in frame.columns:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame


def _read_table_with_date_index(path: Path) -> pd.DataFrame:
    frame = _read_table(path)
    if "Date" in frame.columns:
        return frame.set_index("Date")
    return frame


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_csv(path: Path, frame: pd.DataFrame) -> Path:
    frame.to_csv(path, index=False)
    return path


def _numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float64")


def _round(value: float) -> float | None:
    if np.isnan(value) or np.isinf(value):
        return None
    return round(value, 6)


def _date_columns(columns: list[str]) -> list[str]:
    return [column for column in columns if str(column).lower() == "date"]


def _sample_rows(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=random_state).sort_index()


def _numeric_feature_columns(
    merged: pd.DataFrame,
    feature_columns: list[str],
    min_non_null: int = 10,
) -> list[str]:
    columns: list[str] = []
    for column in feature_columns:
        values = _numeric_series(merged[column])
        if values.notna().sum() >= min_non_null and values.nunique(dropna=True) >= 2:
            columns.append(column)
    return columns


def _model_results_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=MODEL_RESULT_COLUMNS)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["score_primary", "samples"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


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
