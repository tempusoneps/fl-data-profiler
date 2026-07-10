from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.kmean import (
    MAX_LABEL_CLASSES,
    MIN_SAMPLES,
    RANDOM_STATE,
    TEST_SIZE,
    _categorical_label_columns,
    _cluster_distribution,
    _cluster_distribution_frame,
    _label_profile_frame,
    _ranked_successful,
    _render_html,
    _render_markdown,
    _results_frame,
    _skipped_result,
)
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _date_columns,
    _merge_inputs,
    _numeric_feature_columns,
    _numeric_series,
    _read_table_with_date_index,
    _round,
    _sample_rows,
    _select_targets,
    _write_csv,
    _write_json,
)


MAX_ROWS = 50_000
MAX_SILHOUETTE_SAMPLES = 5_000


@dataclass(frozen=True)
class KMeansGpuRunMetadata:
    module: str
    backend: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    model_rows: int
    numeric_features: list[str]
    categorical_labels: list[str]
    ignored_columns: list[str]
    feature_pairs: int
    combinations: int
    max_silhouette_samples: int


class KMeansGpuRelationshipsModule:
    name = "kmeans_gpu"

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        gpu_kmeans = _load_gpu_kmeans()
        features = _read_table_with_date_index(feature_csv)
        labels = _read_table_with_date_index(label_csv)
        merged, feature_columns, label_columns, join_strategy = _merge_inputs(
            features, labels, join_key
        )

        ignored_columns = _date_columns([*feature_columns, *label_columns])
        feature_columns = [column for column in feature_columns if column not in ignored_columns]
        label_columns = [column for column in label_columns if column not in ignored_columns]
        selected_targets = _select_targets(label_columns, targets)

        numeric_features = _numeric_feature_columns(merged, feature_columns)
        categorical_labels = _categorical_label_columns(merged, selected_targets)
        model_frame = _sample_rows(
            merged[[*numeric_features, *categorical_labels]],
            MAX_ROWS,
            RANDOM_STATE,
        )

        feature_pairs = list(combinations(numeric_features, 2))
        results, cluster_distribution = _fit_kmeans_gpu_reports(
            model_frame,
            numeric_features,
            feature_pairs,
            categorical_labels,
            gpu_kmeans,
        )

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = KMeansGpuRunMetadata(
            module=self.name,
            backend="cuml",
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=DatasetShape(*features.shape),
            label_shape=DatasetShape(*labels.shape),
            merged_shape=DatasetShape(*merged.shape),
            model_rows=len(model_frame),
            numeric_features=numeric_features,
            categorical_labels=categorical_labels,
            ignored_columns=ignored_columns,
            feature_pairs=len(feature_pairs),
            combinations=len(feature_pairs) * len(categorical_labels),
            max_silhouette_samples=MAX_SILHOUETTE_SAMPLES,
        )

        numeric_features_frame = pd.DataFrame({"feature": numeric_features})
        categorical_labels_frame = _label_profile_frame(merged, categorical_labels)

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "numeric_features": numeric_features,
                    "categorical_labels": categorical_labels,
                    "top_results": _ranked_successful(results).head(100).to_dict(orient="records"),
                    "cluster_distribution": cluster_distribution.head(200).to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "numeric_features.csv", numeric_features_frame),
            _write_csv(run_dir / "categorical_labels.csv", categorical_labels_frame),
            _write_csv(run_dir / "kmean_results.csv", results),
            _write_csv(run_dir / "cluster_label_distribution.csv", cluster_distribution),
        ]

        markdown = _render_markdown(metadata, results, categorical_labels_frame)
        md_path = run_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")
        artifacts.append(md_path)

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, results), encoding="utf-8")
        artifacts.append(html_path)

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _load_gpu_kmeans() -> type[Any]:
    try:
        from cuml.cluster import KMeans as CumlKMeans
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "kmeans_gpu requires RAPIDS cuML. Install a cuML build that matches your "
            "NVIDIA driver/CUDA environment, then rerun with --module kmeans_gpu."
        ) from exc
    return CumlKMeans


def _fit_kmeans_gpu_reports(
    merged: pd.DataFrame,
    numeric_features: list[str],
    feature_pairs: list[tuple[str, str]],
    label_columns: list[str],
    gpu_kmeans: type[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    numeric_frame = merged[numeric_features].apply(_numeric_series)
    numeric_values = numeric_frame.to_numpy(dtype=np.float32, copy=True)
    feature_positions = {feature: index for index, feature in enumerate(numeric_features)}
    label_cache = {
        label: _PreparedLabel(merged[label].astype(str), merged[label].notna().to_numpy())
        for label in label_columns
    }

    for feature_1, feature_2 in feature_pairs:
        positions = [feature_positions[feature_1], feature_positions[feature_2]]
        pair_values = numeric_values[:, positions]
        pair_valid = ~np.isnan(pair_values).any(axis=1)
        for label in label_columns:
            result, distribution = _fit_single_gpu_kmeans(
                feature_1,
                feature_2,
                label,
                pair_values,
                pair_valid,
                label_cache[label],
                gpu_kmeans,
            )
            result_rows.append(result)
            distribution_rows.extend(distribution)

    return _results_frame(result_rows), _cluster_distribution_frame(distribution_rows)


@dataclass(frozen=True)
class _PreparedLabel:
    values: pd.Series
    valid_mask: np.ndarray


def _fit_single_gpu_kmeans(
    feature_1: str,
    feature_2: str,
    label: str,
    pair_values: np.ndarray,
    pair_valid: np.ndarray,
    prepared_label: _PreparedLabel,
    gpu_kmeans: type[Any],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    valid = pair_valid & prepared_label.valid_mask
    samples = int(valid.sum())
    if samples < MIN_SAMPLES:
        return _skipped_result(feature_1, feature_2, label, samples, "not_enough_samples"), []

    x = pair_values[valid]
    label_values = prepared_label.values[valid]
    class_count = int(label_values.nunique(dropna=True))
    if class_count < 2:
        return _skipped_result(feature_1, feature_2, label, samples, "label_has_less_than_2_classes"), []
    if class_count > MAX_LABEL_CLASSES:
        return _skipped_result(feature_1, feature_2, label, samples, "label_has_too_many_classes"), []

    test_count = int(np.ceil(samples * TEST_SIZE))
    train_count = samples - test_count
    if train_count < class_count:
        return _skipped_result(feature_1, feature_2, label, samples, "not_enough_train_samples_for_clusters"), []

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(label_values)
    class_sizes = np.bincount(encoded)
    stratify = (
        encoded
        if np.min(class_sizes) >= 2 and test_count >= class_count and train_count >= class_count
        else None
    )
    train_x_raw, test_x_raw, train_labels_raw, test_labels_raw = train_test_split(
        x,
        label_values.to_numpy(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x_raw).astype(np.float32, copy=False)
    test_x = scaler.transform(test_x_raw).astype(np.float32, copy=False)

    model = gpu_kmeans(n_clusters=class_count, n_init=10, random_state=RANDOM_STATE)
    model.fit(train_x)
    train_clusters = _to_numpy(model.predict(train_x)).astype(int, copy=False)
    test_clusters = _to_numpy(model.predict(test_x)).astype(int, copy=False)

    train_labels = encoder.transform(train_labels_raw.astype(str))
    test_labels = encoder.transform(test_labels_raw.astype(str))
    distribution = _cluster_distribution(
        feature_1,
        feature_2,
        label,
        test_clusters,
        pd.Series(test_labels_raw),
    )

    return (
        {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "label": label,
            "status": "ok",
            "samples": samples,
            "train_samples": int(len(train_x)),
            "test_samples": int(len(test_x)),
            "label_classes": int(class_count),
            "clusters": int(class_count),
            "train_adjusted_rand": _round(float(adjusted_rand_score(train_labels, train_clusters))),
            "test_adjusted_rand": _round(float(adjusted_rand_score(test_labels, test_clusters))),
            "train_normalized_mutual_info": _round(float(normalized_mutual_info_score(train_labels, train_clusters))),
            "test_normalized_mutual_info": _round(float(normalized_mutual_info_score(test_labels, test_clusters))),
            "test_homogeneity": _round(float(homogeneity_score(test_labels, test_clusters))),
            "test_completeness": _round(float(completeness_score(test_labels, test_clusters))),
            "test_v_measure": _round(float(v_measure_score(test_labels, test_clusters))),
            "test_silhouette": _safe_sampled_silhouette(test_x, test_clusters),
            "train_inertia": _round(float(_to_numpy(getattr(model, "inertia_", np.nan)))),
            "note": "gpu=cuml",
        },
        distribution,
    )


def _safe_sampled_silhouette(features: np.ndarray, clusters: np.ndarray) -> float | None:
    unique_clusters = np.unique(clusters)
    if len(unique_clusters) < 2 or len(unique_clusters) >= len(features):
        return None
    sample_features, sample_clusters = _silhouette_sample(
        features,
        clusters,
        MAX_SILHOUETTE_SAMPLES,
        RANDOM_STATE,
    )
    return _round(float(silhouette_score(sample_features, sample_clusters)))


def _silhouette_sample(
    features: np.ndarray,
    clusters: np.ndarray,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(features) <= max_samples:
        return features, clusters
    rng = np.random.default_rng(random_state)
    indices = np.sort(rng.choice(len(features), size=max_samples, replace=False))
    return features[indices], clusters[indices]


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)
