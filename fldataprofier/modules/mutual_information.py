from __future__ import annotations

from pathlib import Path

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.time_series_scoring import build_result, load_prepared_data, mutual_information_scores
from fldataprofier.utils import _write_csv


class MutualInformationModule:
    name = "mutual_information"

    def __init__(self, random_state: int = 42) -> None:
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
        scores = mutual_information_scores(
            prepared.merged,
            prepared.feature_columns,
            prepared.target_columns,
            random_state=self.random_state,
        )
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
            {"random_state": self.random_state},
        )
