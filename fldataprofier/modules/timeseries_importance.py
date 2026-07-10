from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from tqdm.auto import tqdm

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.time_series_scoring import (
    build_result,
    information_coefficient_rows,
    load_prepared_data,
    permutation_importance_rows,
)
from fldataprofier.utils import _write_csv


class TimeSeriesImportanceModule:
    name = "timeseries_importance"

    def __init__(
        self,
        n_estimators: int = 100,
        random_state: int = 42,
        progress: bool | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.progress = progress

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        progress_enabled = sys.stderr.isatty() if self.progress is None else self.progress
        with tqdm(
            total=5,
            desc=self.name,
            unit="step",
            disable=not progress_enabled,
        ) as progress_bar:
            progress_bar.set_postfix_str("load")
            prepared = load_prepared_data(feature_csv, label_csv, join_key, targets)
            report_dir = output_dir / self.name
            report_dir.mkdir(parents=True, exist_ok=True)
            progress_bar.update(1)

            progress_bar.set_postfix_str("rank_ic")
            rank_ic_rows = [
                row
                for row in information_coefficient_rows(
                    prepared.merged,
                    prepared.feature_columns,
                    prepared.target_columns,
                )
                if row.get("score_name") == "rank_ic"
            ]
            progress_bar.update(1)

            progress_bar.set_postfix_str("permutation")
            permutation_rows = permutation_importance_rows(
                prepared.merged,
                prepared.feature_columns,
                prepared.target_columns,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            progress_bar.update(1)

            progress_bar.set_postfix_str("combine")
            component_scores = pd.DataFrame([*rank_ic_rows, *permutation_rows])
            feature_scores = _combined_scores(component_scores)
            progress_bar.update(1)

            progress_bar.set_postfix_str("write")
            artifacts = [
                _write_csv(report_dir / "component_scores.csv", component_scores),
                _write_csv(report_dir / "feature_scores.csv", feature_scores),
                _write_csv(report_dir / "top_features.csv", feature_scores.head(50)),
            ]
            result = build_result(
                report_dir,
                self.name,
                feature_csv,
                label_csv,
                prepared,
                feature_scores,
                artifacts,
                {
                    "n_estimators": self.n_estimators,
                    "random_state": self.random_state,
                    "progress_enabled": progress_enabled,
                },
            )
            progress_bar.update(1)
            return result


def _combined_scores(component_scores: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature",
        "label",
        "combined_score",
        "mean_abs_score",
        "component_count",
        "valid_folds",
        "samples",
    ]
    if component_scores.empty:
        return pd.DataFrame(columns=columns)

    frame = component_scores.copy()
    frame["abs_score"] = frame["score"].abs()
    max_by_component = frame.groupby(["label", "score_name"])["abs_score"].transform("max").replace(0, 1)
    frame["normalized_score"] = frame["abs_score"] / max_by_component
    result = frame.groupby(["feature", "label"], dropna=False).agg(
        combined_score=("normalized_score", "mean"),
        mean_abs_score=("abs_score", "mean"),
        component_count=("score_name", "nunique"),
        valid_folds=("score", "count"),
        samples=("samples", "sum"),
    ).reset_index()
    return result.sort_values(
        ["combined_score", "component_count", "valid_folds"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
