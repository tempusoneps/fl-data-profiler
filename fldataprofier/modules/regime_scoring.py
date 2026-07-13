from __future__ import annotations

from pathlib import Path

import pandas as pd

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.time_series_scoring import aggregate_scores, build_result, load_prepared_data
from fldataprofier.modules.time_series_scoring import information_coefficient_rows
from fldataprofier.utils import _numeric_series, _write_csv


class RegimeScoringModule:
    name = "regime_scoring"

    def __init__(self, n_regimes: int = 3, progress: bool | None = None) -> None:
        self.n_regimes = n_regimes
        self.progress = progress

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        with ModuleProgress(self.name, total=5, enabled=self.progress) as progress_bar:
            prepared = load_prepared_data(feature_csv, label_csv, join_key, targets)
            report_dir = output_dir / self.name
            report_dir.mkdir(parents=True, exist_ok=True)
            progress_bar.step("load")
            regimes = _assign_regimes(prepared.merged, prepared.feature_columns, prepared.target_columns, self.n_regimes)
            progress_bar.step("regimes")
            rows: list[dict[str, object]] = []
            for regime, index in regimes.groupby(regimes).groups.items():
                regime_frame = prepared.merged.loc[index]
                regime_rows = information_coefficient_rows(
                    regime_frame,
                    prepared.feature_columns,
                    prepared.target_columns,
                    min_train_size=50,
                    test_size=25,
                    step_size=25,
                    min_samples=10,
                )
                for row in regime_rows:
                    row["regime"] = regime
                rows.extend(regime_rows)
            raw = pd.DataFrame(rows)
            summary = aggregate_scores(raw.to_dict(orient="records"))
            if not raw.empty:
                regimes_by_feature = raw.groupby(["feature", "label", "score_name"])["regime"].nunique().reset_index()
                regimes_by_feature = regimes_by_feature.rename(columns={"regime": "regime_count"})
                summary = summary.merge(regimes_by_feature, on=["feature", "label", "score_name"], how="left")
                summary["regime"] = "all"
            else:
                summary["regime"] = []
                summary["regime_count"] = []
            progress_bar.step("score")
            artifacts = [
                _write_csv(report_dir / "fold_scores.csv", raw),
                _write_csv(report_dir / "feature_scores.csv", summary),
                _write_csv(report_dir / "top_features.csv", summary.head(50)),
            ]
            progress_bar.step("artifacts")
            result = build_result(
                report_dir,
                self.name,
                feature_csv,
                label_csv,
                prepared,
                summary,
                artifacts,
                {
                    "n_regimes": self.n_regimes,
                    "regime_count": int(regimes.nunique(dropna=True)),
                    "progress_enabled": progress_bar.enabled,
                },
            )
            progress_bar.step("write")
            return result


def _assign_regimes(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    n_regimes: int,
) -> pd.Series:
    regime_source = None
    for column in feature_columns:
        lowered = str(column).lower()
        if any(token in lowered for token in ("volatility", "atr", "range")):
            candidate = _numeric_series(frame[column])
            if candidate.notna().sum() >= n_regimes:
                regime_source = candidate
                break
    if regime_source is None:
        for label in target_columns:
            candidate = _numeric_series(frame[label])
            if candidate.notna().sum() >= n_regimes:
                regime_source = candidate.diff().abs().rolling(20, min_periods=1).mean()
                break
    if regime_source is None:
        regime_source = pd.Series(range(len(frame)), index=frame.index)

    labels = ["low", "mid", "high"][:n_regimes]
    if len(labels) < n_regimes:
        labels = [f"regime_{index + 1}" for index in range(n_regimes)]
    return pd.qcut(regime_source.rank(method="first"), q=n_regimes, labels=labels, duplicates="drop")
