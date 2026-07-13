from __future__ import annotations

from pathlib import Path

import pandas as pd

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.time_series_scoring import (
    aggregate_scores,
    build_result,
    information_coefficient_rows,
    load_prepared_data,
    write_score_artifacts,
)


class InformationCoefficientModule:
    name = "information_coefficient"

    def __init__(self, progress: bool | None = None) -> None:
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
            progress_bar.step("load")
            raw = pd.DataFrame(
                information_coefficient_rows(
                    prepared.merged,
                    prepared.feature_columns,
                    prepared.target_columns,
                )
            )
            progress_bar.step("score")
            summary = aggregate_scores(raw.to_dict(orient="records"))
            progress_bar.step("aggregate")
            artifacts = write_score_artifacts(report_dir, raw, summary)
            result = build_result(
                report_dir,
                self.name,
                feature_csv,
                label_csv,
                prepared,
                summary,
                artifacts,
                {"progress_enabled": progress_bar.enabled},
            )
            progress_bar.step("write")
            return result
