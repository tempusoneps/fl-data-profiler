from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.time_series_scoring import (
    build_result,
    impute_numeric_frame,
    load_prepared_data,
    mutual_information_scores,
    prepare_numeric_matrix,
)
from fldataprofier.utils import _write_csv


class FeatureInteractionsModule:
    name = "feature_interactions"

    def __init__(self, max_base_features: int = 12, max_pairs: int = 80) -> None:
        self.max_base_features = max_base_features
        self.max_pairs = max_pairs

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
        generated = _generate_interactions(prepared, self.max_base_features, self.max_pairs)
        interaction_frame = pd.concat([prepared.merged, generated.drop(columns=["__meta__"], errors="ignore")], axis=1)
        interaction_columns = [column for column in generated.columns if column != "__meta__"]
        scores = mutual_information_scores(
            interaction_frame,
            interaction_columns,
            prepared.target_columns,
        )
        metadata = pd.DataFrame(generated.attrs.get("metadata", []))
        scores = scores.merge(metadata, left_on="feature", right_on="interaction", how="left")
        artifacts = [
            _write_csv(report_dir / "generated_interactions.csv", metadata),
            _write_csv(report_dir / "feature_scores.csv", scores),
            _write_csv(report_dir / "top_features.csv", scores.head(50)),
        ]
        return build_result(report_dir, self.name, feature_csv, label_csv, prepared, scores, artifacts)


def _generate_interactions(prepared, max_base_features: int, max_pairs: int) -> pd.DataFrame:
    base_scores = mutual_information_scores(prepared.merged, prepared.feature_columns, prepared.target_columns)
    if base_scores.empty:
        return pd.DataFrame()
    base_features = list(dict.fromkeys(base_scores["feature"].head(max_base_features).tolist()))
    feature_frame, _ = prepare_numeric_matrix(prepared.merged[base_features])
    feature_frame = impute_numeric_frame(feature_frame)
    generated: dict[str, pd.Series] = {}
    metadata: list[dict[str, object]] = []
    pair_count = 0
    for left, right in combinations(base_features, 2):
        if pair_count >= max_pairs:
            break
        operations = {
            "product": feature_frame[left] * feature_frame[right],
            "difference": feature_frame[left] - feature_frame[right],
            "ratio": feature_frame[left] / feature_frame[right].where(feature_frame[right].abs() > 1e-12),
        }
        for operation, values in operations.items():
            name = f"{left}__{operation}__{right}"
            generated[name] = values
            metadata.append(
                {
                    "interaction": name,
                    "left_feature": left,
                    "right_feature": right,
                    "operation": operation,
                }
            )
        pair_count += 1
    result = pd.DataFrame(generated, index=prepared.merged.index)
    result.attrs["metadata"] = metadata
    return result
