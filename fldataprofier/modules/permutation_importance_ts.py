from __future__ import annotations

from pathlib import Path

import pandas as pd

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.time_series_scoring import (
    aggregate_scores,
    build_result,
    load_prepared_data,
    permutation_importance_rows,
    write_score_artifacts,
)


class PermutationImportanceTSModule:
    name = "permutation_importance_ts"

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
        raw = pd.DataFrame(
            permutation_importance_rows(
                prepared.merged,
                prepared.feature_columns,
                prepared.target_columns,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
        )
        summary = aggregate_scores(raw.to_dict(orient="records"))
        artifacts = write_score_artifacts(report_dir, raw, summary)
        return build_result(
            report_dir,
            self.name,
            feature_csv,
            label_csv,
            prepared,
            summary,
            artifacts,
            {"n_estimators": self.n_estimators, "random_state": self.random_state},
        )
