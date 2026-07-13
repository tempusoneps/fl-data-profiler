from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.time_series_scoring import (
    build_result,
    impute_numeric_frame,
    load_prepared_data,
    mutual_information_scores,
    prepare_numeric_matrix,
)
from fldataprofier.utils import _write_csv


class MRMRModule:
    name = "mrmr"

    def __init__(
        self,
        max_features: int = 50,
        random_state: int = 42,
        progress: bool | None = None,
    ) -> None:
        self.max_features = max_features
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
        with ModuleProgress(self.name, total=4, enabled=self.progress) as progress_bar:
            prepared = load_prepared_data(feature_csv, label_csv, join_key, targets)
            report_dir = output_dir / self.name
            report_dir.mkdir(parents=True, exist_ok=True)
            progress_bar.step("load")
            scores = _mrmr_scores(prepared, self.max_features, self.random_state)
            progress_bar.step("score")
            artifacts = [
                _write_csv(report_dir / "feature_scores.csv", scores),
                _write_csv(report_dir / "top_features.csv", scores.head(50)),
            ]
            progress_bar.step("artifacts")
            result = build_result(
                report_dir,
                self.name,
                feature_csv,
                label_csv,
                prepared,
                scores,
                artifacts,
                {"progress_enabled": progress_bar.enabled},
            )
            progress_bar.step("write")
            return result


def _mrmr_scores(prepared, max_features: int, random_state: int) -> pd.DataFrame:
    relevance = mutual_information_scores(
        prepared.merged,
        prepared.feature_columns,
        prepared.target_columns,
        random_state=random_state,
    )
    feature_frame, _ = prepare_numeric_matrix(prepared.merged[prepared.feature_columns])
    feature_frame = impute_numeric_frame(feature_frame)
    rows: list[dict[str, object]] = []
    for label, group in relevance.groupby("label", dropna=False):
        selected: list[str] = []
        candidates = group.sort_values("score", ascending=False).to_dict(orient="records")
        for _ in range(min(max_features, len(candidates))):
            best: dict[str, object] | None = None
            for candidate in candidates:
                feature = str(candidate["feature"])
                if feature in selected:
                    continue
                redundancy = _mean_abs_corr(feature_frame, feature, selected)
                mrmr_score = float(candidate["score"] or 0.0) - redundancy
                entry = {
                    "feature": feature,
                    "label": label,
                    "score_name": "mrmr",
                    "score": mrmr_score,
                    "relevance": candidate["score"],
                    "redundancy": redundancy,
                    "mrmr_score": mrmr_score,
                    "samples": candidate["samples"],
                }
                if best is None or mrmr_score > float(best["mrmr_score"]):
                    best = entry
            if best is None:
                break
            selected.append(str(best["feature"]))
            rows.append(best)
    return pd.DataFrame(rows).sort_values(["mrmr_score", "relevance"], ascending=[False, False]).reset_index(drop=True)


def _mean_abs_corr(frame: pd.DataFrame, feature: str, selected: list[str]) -> float:
    if not selected:
        return 0.0
    values = []
    for other in selected:
        corr = frame[feature].corr(frame[other], method="spearman")
        if pd.notna(corr):
            values.append(abs(float(corr)))
    return float(np.mean(values)) if values else 0.0
