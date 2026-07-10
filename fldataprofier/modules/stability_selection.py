from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.time_series_scoring import (
    build_result,
    impute_numeric_frame,
    is_classification_target,
    load_prepared_data,
    prepare_numeric_matrix,
)
from fldataprofier.utils import _numeric_series, _write_csv


class StabilitySelectionModule:
    name = "stability_selection"

    def __init__(self, n_resamples: int = 30, random_state: int = 42) -> None:
        self.n_resamples = n_resamples
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
        scores = _stability_scores(prepared, self.n_resamples, self.random_state)
        artifacts = [
            _write_csv(report_dir / "feature_scores.csv", scores),
            _write_csv(report_dir / "top_features.csv", scores.head(50)),
        ]
        return build_result(report_dir, self.name, feature_csv, label_csv, prepared, scores, artifacts)


def _stability_scores(prepared, n_resamples: int, random_state: int) -> pd.DataFrame:
    feature_frame, features = prepare_numeric_matrix(prepared.merged[prepared.feature_columns])
    x_all = impute_numeric_frame(feature_frame)
    rng = np.random.default_rng(random_state)
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
            model_factory = lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty="l1", solver="liblinear", C=0.2, max_iter=1000),
            )
        else:
            y_series = _numeric_series(y_raw[mask])
            valid = y_series.notna()
            if valid.sum() < 30 or y_series[valid].nunique() < 2:
                continue
            x = x.loc[y_series[valid].index]
            y = y_series[valid].to_numpy()
            model_factory = lambda: make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=5000))
        counts = dict.fromkeys(features, 0)
        valid_resamples = 0
        sample_size = max(20, int(len(x) * 0.7))
        for _ in range(n_resamples):
            sample_index = rng.choice(len(x), size=sample_size, replace=False)
            model = model_factory()
            model.fit(x.iloc[sample_index], np.asarray(y)[sample_index])
            estimator = model.steps[-1][1]
            coefs = getattr(estimator, "coef_", np.array([]))
            coefs = np.asarray(coefs)
            if coefs.ndim > 1:
                coefs = np.abs(coefs).max(axis=0)
            selected = np.abs(coefs) > 1e-9
            for feature, is_selected in zip(features, selected, strict=False):
                counts[feature] += int(bool(is_selected))
            valid_resamples += 1
        for feature in features:
            frequency = counts[feature] / valid_resamples if valid_resamples else 0.0
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "score_name": "selection_frequency",
                    "score": frequency,
                    "selection_frequency": frequency,
                    "samples": int(len(x)),
                    "valid_resamples": valid_resamples,
                }
            )
    return pd.DataFrame(rows).sort_values("selection_frequency", ascending=False).reset_index(drop=True)
