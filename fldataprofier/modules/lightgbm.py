from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.time_series_scoring import (
    build_result,
    impute_numeric_frame,
    is_classification_target,
    load_prepared_data,
    prepare_numeric_matrix,
)
from fldataprofier.utils import _numeric_series, _write_csv


class LightGBMModule:
    name = "lightgbm"

    def __init__(self, n_estimators: int = 100, random_state: int = 42) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        prepared = load_prepared_data(feature_csv, label_csv, join_key, targets)
        report_dir = output_dir / self.name
        report_dir.mkdir(parents=True, exist_ok=True)
        scores, backend = _gbm_scores(prepared, self.n_estimators, self.random_state)
        artifacts = [
            _write_csv(report_dir / "feature_scores.csv", scores),
            _write_csv(report_dir / "top_features.csv", scores.head(50)),
        ]
        return build_result(
            report_dir,
            self.name,
            feature_csv,
            label_csv,
            prepared,
            scores,
            artifacts,
            {"backend": backend, "n_estimators": self.n_estimators},
        )


def _gbm_scores(prepared, n_estimators: int, random_state: int) -> tuple[pd.DataFrame, str]:
    lightgbm = _load_lightgbm()
    feature_frame, features = prepare_numeric_matrix(prepared.merged[prepared.feature_columns])
    x_all = impute_numeric_frame(feature_frame)
    rows: list[dict[str, object]] = []
    backend = "lightgbm" if lightgbm is not None else "sklearn_random_forest_fallback"
    for label in prepared.target_columns:
        y_raw = prepared.merged[label]
        mask = y_raw.notna()
        if mask.sum() < 30:
            continue
        x = x_all.loc[mask]
        if is_classification_target(y_raw[mask]):
            y = LabelEncoder().fit_transform(y_raw[mask].astype(str))
            if len(np.unique(y)) < 2:
                continue
            if lightgbm is not None:
                model = lightgbm.LGBMClassifier(n_estimators=n_estimators, random_state=random_state, verbose=-1)
            else:
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        else:
            y_series = _numeric_series(y_raw[mask])
            valid = y_series.notna()
            if valid.sum() < 30 or y_series[valid].nunique() < 2:
                continue
            x = x.loc[y_series[valid].index]
            y = y_series[valid].to_numpy()
            if lightgbm is not None:
                model = lightgbm.LGBMRegressor(n_estimators=n_estimators, random_state=random_state, verbose=-1)
            else:
                model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        model.fit(x, y)
        importances = getattr(model, "feature_importances_", np.zeros(len(features)))
        for feature, score in zip(features, importances, strict=False):
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "score_name": "gbm_importance",
                    "score": float(score),
                    "importance_type": "split",
                    "backend": backend,
                    "samples": int(len(x)),
                }
            )
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores, backend
    return scores.sort_values("score", ascending=False).reset_index(drop=True), backend


def _load_lightgbm():
    try:
        import lightgbm
    except Exception:
        return None
    return lightgbm
