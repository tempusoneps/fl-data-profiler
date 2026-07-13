from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _date_columns,
    _html_markdown_details,
    _markdown_table,
    _merge_inputs,
    _numeric_series,
    _read_table_with_date_index,
    _round,
    _sample_rows,
    _select_targets,
    _write_csv,
    _write_json,
)


MAX_ROWS = 50_000
MAX_LABEL_CLASSES = 20
MAX_MISSING_RATIO = 0.5
MIN_NON_NULL = 10
MIN_DISTINCT_VALUES = 2
N_BINS = 10
MAX_CANDIDATE_FEATURES = 24
TOP_1D_FEATURES = 16
RANDOM_STATE = 42
MIN_CELL_SUPPORT = 5
MIN_MODEL_SAMPLES = 40
TEST_SIZE = 0.25


PAIR_SCORE_COLUMNS = [
    "feature_x",
    "feature_y",
    "label",
    "samples",
    "separability",
    "linearity",
    "region_purity",
    "overlap",
    "recommendation",
]


@dataclass(frozen=True)
class VisualRegionsRunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    model_rows: int
    numeric_features: list[str]
    categorical_labels: list[str]
    candidate_features: list[str]
    feature_pairs: int
    thresholds: dict[str, object]


def _categorical_label_columns(
    merged: pd.DataFrame,
    label_columns: list[str],
    max_classes: int = MAX_LABEL_CLASSES,
) -> list[str]:
    selected: list[str] = []
    for column in label_columns:
        values = merged[column].dropna()
        unique_count = int(values.nunique(dropna=True))
        if 2 <= unique_count <= max_classes:
            selected.append(column)
    return selected


def _prepare_numeric_feature_frame(
    merged: pd.DataFrame,
    feature_columns: list[str],
    max_missing_ratio: float = MAX_MISSING_RATIO,
    min_non_null: int = MIN_NON_NULL,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    date_columns = set(_date_columns(feature_columns))
    prepared: dict[str, pd.Series] = {}
    exclusions: list[dict[str, object]] = []
    row_count = len(merged)
    for column in feature_columns:
        if column in date_columns:
            exclusions.append({"column": column, "reason": "date_column"})
            continue
        values = _numeric_series(merged[column])
        non_null = int(values.notna().sum())
        if non_null == 0:
            exclusions.append({"column": column, "reason": "non_numeric"})
            continue
        missing_ratio = 1.0 if row_count == 0 else 1.0 - (non_null / row_count)
        if missing_ratio > max_missing_ratio or non_null < min_non_null:
            exclusions.append({"column": column, "reason": "too_many_missing"})
            continue
        distinct = int(values.nunique(dropna=True))
        if distinct < MIN_DISTINCT_VALUES:
            exclusions.append({"column": column, "reason": "constant_or_too_few_values"})
            continue
        prepared[column] = values
    return pd.DataFrame(prepared, index=merged.index), exclusions


def _quantile_bin_features(features: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    binned: dict[str, pd.Series] = {}
    for column in features.columns:
        values = features[column]
        valid = values.dropna()
        if valid.nunique(dropna=True) < 2:
            continue
        ranks = values.rank(method="first", na_option="keep")
        bins = pd.qcut(ranks, q=min(n_bins, int(valid.nunique())), labels=False, duplicates="drop")
        binned[column] = bins.astype("float64")
    result = pd.DataFrame(binned, index=features.index)
    if result.empty:
        return result
    return result.fillna(255).astype("uint8")


def _score_1d_candidates(bin_frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in bin_frame.columns:
        feature_bins = bin_frame[feature]
        valid_feature = feature_bins != 255
        for label in labels.columns:
            frame = pd.DataFrame({"bin": feature_bins, "label": labels[label]}).loc[valid_feature].dropna()
            if frame.empty or frame["label"].nunique(dropna=True) < 2:
                continue
            prior = frame["label"].value_counts(normalize=True)
            base_purity = float(prior.max())
            weighted_purity = 0.0
            for _, group in frame.groupby("bin", observed=True):
                purity = float(group["label"].value_counts(normalize=True).max())
                weighted_purity += purity * (len(group) / len(frame))
            score = max(0.0, weighted_purity - base_purity)
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "samples": len(frame),
                    "score": _round(score),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["feature", "label", "samples", "score"])
    return pd.DataFrame(rows).sort_values(["score", "samples"], ascending=[False, False]).reset_index(drop=True)


def _select_candidate_features(
    candidate_scores: pd.DataFrame,
    valid_features: list[str],
    max_features: int = MAX_CANDIDATE_FEATURES,
    random_state: int = RANDOM_STATE,
) -> list[str]:
    selected: list[str] = []
    if not candidate_scores.empty:
        sort_columns = ["score"]
        if "samples" in candidate_scores.columns:
            sort_columns.append("samples")
        ranked = candidate_scores.sort_values(sort_columns, ascending=[False] * len(sort_columns))
        for feature in ranked["feature"]:
            if feature not in selected:
                selected.append(str(feature))
            if len(selected) >= min(TOP_1D_FEATURES, max_features):
                break
    remaining = [feature for feature in valid_features if feature not in selected]
    if remaining and len(selected) < max_features:
        rng = np.random.default_rng(random_state)
        sample_size = min(len(remaining), max_features - len(selected))
        sampled = list(rng.choice(np.array(remaining, dtype=object), size=sample_size, replace=False))
        selected.extend(str(feature) for feature in sampled)
    return selected[:max_features]
