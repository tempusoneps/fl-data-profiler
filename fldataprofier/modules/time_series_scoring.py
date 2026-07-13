from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

from fldataprofier.modules.base import ModuleResult
from fldataprofier.utils import (
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


SCORE_COLUMNS = [
    "feature",
    "label",
    "score_name",
    "mean_score",
    "mean_abs_score",
    "score_std",
    "valid_folds",
    "positive_fold_ratio",
    "samples",
]


@dataclass(frozen=True)
class PreparedData:
    merged: pd.DataFrame
    feature_columns: list[str]
    target_columns: list[str]
    join_strategy: str


def walk_forward_splits(
    n_rows: int,
    min_train_size: int = 100,
    test_size: int = 50,
    step_size: int = 50,
    max_folds: int = 20,
) -> list[tuple[int, int, int, int]]:
    for name, value in {
        "n_rows": n_rows,
        "min_train_size": min_train_size,
        "test_size": test_size,
        "step_size": step_size,
        "max_folds": max_folds,
    }.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    splits: list[tuple[int, int, int, int]] = []
    train_end = min_train_size
    while train_end + test_size <= n_rows and len(splits) < max_folds:
        splits.append((0, train_end, train_end, train_end + test_size))
        train_end += step_size
    return splits


def load_prepared_data(
    feature_csv: Path,
    label_csv: Path,
    join_key: str | None,
    targets: list[str] | None,
) -> PreparedData:
    features = _read_table_with_date_index(feature_csv)
    labels = _read_table_with_date_index(label_csv)
    merged, feature_columns, label_columns, join_strategy = _merge_inputs(
        features,
        labels,
        join_key,
    )
    if isinstance(merged.index, pd.DatetimeIndex):
        merged = merged.sort_index()
    return PreparedData(
        merged=merged,
        feature_columns=feature_columns,
        target_columns=_select_targets(label_columns, targets),
        join_strategy=join_strategy,
    )


def prepare_numeric_matrix(
    df: pd.DataFrame,
    exclude: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    excluded = exclude or set()
    columns: dict[str, pd.Series] = {}
    for column in df.columns:
        if column in excluded:
            continue
        values = _numeric_series(df[column])
        if values.notna().sum() == 0 or values.nunique(dropna=True) < 2:
            continue
        columns[column] = values

    frame = pd.DataFrame(columns, index=df.index)
    return frame, list(frame.columns)


def impute_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    medians = frame.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)


def aggregate_scores(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    grouped = frame.groupby(["feature", "label", "score_name"], dropna=False)
    result = grouped.agg(
        mean_score=("score", "mean"),
        mean_abs_score=("score", lambda values: float(np.abs(values).mean())),
        score_std=("score", "std"),
        valid_folds=("score", "count"),
        positive_fold_ratio=("score", lambda values: float((values > 0).mean())),
        samples=("samples", "sum"),
    ).reset_index()

    for column in ["mean_score", "mean_abs_score", "score_std", "positive_fold_ratio"]:
        result[column] = result[column].map(lambda value: _round(float(value)))
    result["samples"] = result["samples"].fillna(0).astype(int)
    result["valid_folds"] = result["valid_folds"].fillna(0).astype(int)
    return result.sort_values(
        ["mean_abs_score", "valid_folds", "feature"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def write_score_artifacts(
    report_dir: Path,
    raw_scores: pd.DataFrame,
    summary: pd.DataFrame,
    raw_name: str = "fold_scores.csv",
) -> list[Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        _write_csv(report_dir / raw_name, raw_scores),
        _write_csv(report_dir / "feature_scores.csv", summary),
        _write_csv(report_dir / "top_features.csv", summary.head(50)),
    ]
    return artifacts


def write_standard_report(
    report_dir: Path,
    module_name: str,
    feature_csv: Path,
    label_csv: Path,
    join_strategy: str,
    targets: list[str],
    feature_scores: pd.DataFrame,
    extra_summary: dict[str, object] | None = None,
) -> list[Path]:
    payload = {
        "module": module_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feature_csv": str(feature_csv),
        "label_csv": str(label_csv),
        "join_strategy": join_strategy,
        "targets": targets,
        "score_rows": int(len(feature_scores)),
    }
    if extra_summary:
        payload.update(extra_summary)

    markdown = _render_report_markdown(module_name, payload, feature_scores.head(25))
    html = _render_report_html(markdown, feature_scores.head(25))
    return [
        _write_json(report_dir / "summary.json", payload),
        _write_text(report_dir / "report.md", markdown),
        _write_text(report_dir / "report.html", html),
    ]


def build_result(
    report_dir: Path,
    module_name: str,
    feature_csv: Path,
    label_csv: Path,
    prepared: PreparedData,
    feature_scores: pd.DataFrame,
    artifacts: list[Path],
    extra_summary: dict[str, object] | None = None,
) -> ModuleResult:
    artifacts.extend(
        write_standard_report(
            report_dir,
            module_name,
            feature_csv,
            label_csv,
            prepared.join_strategy,
            prepared.target_columns,
            feature_scores,
            extra_summary,
        )
    )
    return ModuleResult(report_dir=report_dir, artifacts=artifacts)


def information_coefficient_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    min_train_size: int = 100,
    test_size: int = 50,
    step_size: int = 50,
    max_folds: int = 20,
    min_samples: int = 20,
) -> list[dict[str, object]]:
    features, numeric_features = prepare_numeric_matrix(frame[feature_columns])
    splits = walk_forward_splits(len(frame), min_train_size, test_size, step_size, max_folds)
    if not splits and len(frame) >= min_samples:
        splits = [(0, 0, 0, len(frame))]

    rows: list[dict[str, object]] = []
    for fold, (_, _, test_start, test_end) in enumerate(splits):
        for label in target_columns:
            y = _numeric_series(frame[label]).iloc[test_start:test_end]
            if y.notna().sum() < min_samples or y.nunique(dropna=True) < 2:
                continue
            for feature in numeric_features:
                x = features[feature].iloc[test_start:test_end]
                pair = pd.concat([x, y], axis=1).dropna()
                if len(pair) < min_samples or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                    continue
                for score_name, method in (("pearson_ic", "pearson"), ("rank_ic", "spearman")):
                    score = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
                    if pd.notna(score):
                        rows.append(
                            {
                                "fold": fold,
                                "feature": feature,
                                "label": label,
                                "score_name": score_name,
                                "score": float(score),
                                "samples": int(len(pair)),
                            }
                        )
    return rows


def permutation_importance_rows(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    n_estimators: int = 100,
    random_state: int = 42,
    min_train_size: int = 100,
    test_size: int = 50,
    step_size: int = 50,
    max_folds: int = 20,
) -> list[dict[str, object]]:
    feature_frame, numeric_features = prepare_numeric_matrix(frame[feature_columns])
    if not numeric_features:
        return []
    x_all = impute_numeric_frame(feature_frame)
    splits = walk_forward_splits(len(frame), min_train_size, test_size, step_size, max_folds)
    rows: list[dict[str, object]] = []

    for fold, (_, train_end, test_start, test_end) in enumerate(splits):
        x_train = x_all.iloc[:train_end]
        x_test = x_all.iloc[test_start:test_end]
        for label in target_columns:
            y_raw = frame[label]
            train_target = y_raw.iloc[:train_end]
            test_target = y_raw.iloc[test_start:test_end]
            train_mask = train_target.notna()
            test_mask = test_target.notna()
            if train_mask.sum() < 20 or test_mask.sum() < 10:
                continue

            classification = is_classification_target(train_target)
            if classification:
                encoder = LabelEncoder()
                y_train = encoder.fit_transform(train_target[train_mask].astype(str))
                known = test_target[test_mask].astype(str).isin(encoder.classes_)
                if len(np.unique(y_train)) < 2 or known.sum() < 10:
                    continue
                x_fit = x_train.loc[train_mask]
                x_eval = x_test.loc[test_mask].loc[known.index[known]]
                y_eval = encoder.transform(test_target[test_mask].loc[known.index[known]].astype(str))
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=random_state + fold,
                )
                model.fit(x_fit, y_train)
                baseline = accuracy_score(y_eval, model.predict(x_eval))
                metric_name = "accuracy_drop"
            else:
                y_train_series = _numeric_series(train_target[train_mask])
                y_eval_series = _numeric_series(test_target[test_mask])
                valid_train = y_train_series.notna()
                valid_eval = y_eval_series.notna()
                if valid_train.sum() < 20 or valid_eval.sum() < 10 or y_train_series[valid_train].nunique() < 2:
                    continue
                x_fit = x_train.loc[y_train_series[valid_train].index]
                x_eval = x_test.loc[y_eval_series[valid_eval].index]
                y_train = y_train_series[valid_train].to_numpy()
                y_eval = y_eval_series[valid_eval].to_numpy()
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features="sqrt",
                    n_jobs=1,
                    random_state=random_state + fold,
                )
                model.fit(x_fit, y_train)
                baseline = r2_score(y_eval, model.predict(x_eval))
                metric_name = "r2_drop"

            rng = np.random.default_rng(random_state + fold)
            for feature in numeric_features:
                x_permuted = x_eval.copy()
                x_permuted[feature] = rng.permutation(x_permuted[feature].to_numpy())
                if classification:
                    permuted = accuracy_score(y_eval, model.predict(x_permuted))
                    support = 0.0
                else:
                    permuted = r2_score(y_eval, model.predict(x_permuted))
                    support = _correlation_support(x_eval[feature], y_eval)
                drop = float(baseline - permuted)
                rows.append(
                    {
                        "fold": fold,
                        "feature": feature,
                        "label": label,
                        "score_name": "permutation_importance",
                        "metric": metric_name,
                        "baseline": _round(float(baseline)),
                        "permutation_drop": drop,
                        "correlation_support": support,
                        "score": max(drop, 0.0) + support,
                        "samples": int(len(x_eval)),
                    }
                )
    return rows


def mutual_information_scores(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    random_state: int = 42,
) -> pd.DataFrame:
    feature_frame, numeric_features = prepare_numeric_matrix(frame[feature_columns])
    if not numeric_features:
        return pd.DataFrame(columns=["feature", "label", "score_name", "score", "samples"])
    x_all = impute_numeric_frame(feature_frame)
    rows: list[dict[str, object]] = []
    for label in target_columns:
        y_raw = frame[label]
        mask = y_raw.notna()
        if mask.sum() < 20:
            continue
        x = x_all.loc[mask]
        if is_classification_target(y_raw[mask]):
            y = LabelEncoder().fit_transform(y_raw[mask].astype(str))
            if len(np.unique(y)) < 2:
                continue
            values = mutual_info_classif(x, y, discrete_features=False, random_state=random_state)
            score_name = "mutual_info_classif"
        else:
            y_series = _numeric_series(y_raw[mask])
            valid = y_series.notna()
            if valid.sum() < 20 or y_series[valid].nunique() < 2:
                continue
            x = x.loc[y_series[valid].index]
            values = mutual_info_regression(x, y_series[valid], random_state=random_state)
            score_name = "mutual_info_regression"
        for feature, score in zip(numeric_features, values, strict=False):
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "score_name": score_name,
                    "score": _round(float(score)),
                    "samples": int(len(x)),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["feature", "label", "score_name", "score", "samples"])
    return result.sort_values(["score", "samples"], ascending=[False, False]).reset_index(drop=True)


def is_classification_target(series: pd.Series) -> bool:
    numeric = _numeric_series(series)
    return numeric.notna().sum() == 0 or series.nunique(dropna=True) <= 10


def _correlation_support(feature: pd.Series, target: np.ndarray) -> float:
    pair = pd.concat(
        [
            pd.Series(feature.to_numpy(), dtype="float64"),
            pd.Series(target, dtype="float64"),
        ],
        axis=1,
    ).dropna()
    if len(pair) < 2 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return 0.0
    corr = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    return abs(float(corr)) if pd.notna(corr) else 0.0


def _render_report_markdown(
    module_name: str,
    payload: dict[str, object],
    top_scores: pd.DataFrame,
) -> str:
    return "\n\n".join(
        [
            f"# {module_name} report",
            f"- Created at: `{payload['created_at']}`",
            f"- Join strategy: {payload['join_strategy']}",
            f"- Targets: {', '.join(str(target) for target in payload['targets'])}",
            "## Top features",
            _markdown_table(top_scores),
            "",
        ]
    )


def _render_report_html(markdown: str, top_scores: pd.DataFrame) -> str:
    table = top_scores.to_html(index=False, escape=True) if not top_scores.empty else "<p>No rows.</p>"
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Feature scoring report</title></head>
<body>
{_html_markdown_details(markdown)}
{table}
</body>
</html>
"""


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path
