from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.time_series_scoring import (
    build_result,
    impute_numeric_frame,
    is_classification_target,
    load_prepared_data,
    prepare_numeric_matrix,
)
from fldataprofier.utils import _numeric_series, _write_csv


class RegularizedLinearModule:
    name = "regularized_linear"

    def __init__(self, random_state: int = 42, progress: bool | None = None) -> None:
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
            scores = _linear_scores(prepared, self.random_state)
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


def _linear_scores(prepared, random_state: int) -> pd.DataFrame:
    feature_frame, features = prepare_numeric_matrix(prepared.merged[prepared.feature_columns])
    x_all = impute_numeric_frame(feature_frame)
    rows: list[dict[str, object]] = []
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
            model = make_pipeline(
                StandardScaler(),
                LogisticRegressionCV(
                    Cs=5,
                    cv=3,
                    penalty="l1",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=random_state,
                ),
            )
            model_type = "logistic_l1_cv"
        else:
            y_series = _numeric_series(y_raw[mask])
            valid = y_series.notna()
            if valid.sum() < 30 or y_series[valid].nunique() < 2:
                continue
            x = x.loc[y_series[valid].index]
            y = y_series[valid].to_numpy()
            model = make_pipeline(StandardScaler(), ElasticNetCV(cv=3, random_state=random_state))
            model_type = "elastic_net_cv"
        model.fit(x, y)
        estimator = model.steps[-1][1]
        coefs = np.asarray(estimator.coef_)
        if coefs.ndim > 1:
            coefs = coefs.mean(axis=0)
        for feature, coefficient in zip(features, coefs, strict=False):
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "score_name": "abs_coefficient",
                    "score": abs(float(coefficient)),
                    "coefficient": float(coefficient),
                    "abs_coefficient": abs(float(coefficient)),
                    "model_type": model_type,
                    "samples": int(len(x)),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
