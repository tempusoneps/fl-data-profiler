from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    completeness_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _html_markdown_details,
    _read_table_with_date_index,
    _date_columns,
    _markdown_table,
    _merge_inputs,
    _numeric_feature_columns,
    _numeric_series,
    _round,
    _select_targets,
    _write_csv,
    _write_json,
)


MAX_ROWS = 50_000
MAX_LABEL_CLASSES = 50
MIN_SAMPLES = 20
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_TOP_FEATURES_PER_LABEL = 30
MAX_PAIR_CORRELATION = 0.90


def _select_top_features_for_label(
    df: pd.DataFrame,
    numeric_features: list[str],
    label: str,
    top_k: int = MAX_TOP_FEATURES_PER_LABEL,
) -> list[str]:
    if len(numeric_features) <= top_k:
        return numeric_features
    frame = df[[*numeric_features, label]].dropna()
    if len(frame) < MIN_SAMPLES:
        return numeric_features[:top_k]

    label_values = frame[label].astype(str)
    if label_values.nunique(dropna=True) < 2:
        return numeric_features[:top_k]

    encoder = LabelEncoder()
    y = encoder.fit_transform(label_values)
    x = frame[numeric_features].apply(_numeric_series).fillna(0.0).to_numpy()

    try:
        scores, _ = f_classif(x, y)
        scores = np.nan_to_num(scores, nan=0.0)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [numeric_features[i] for i in top_indices]
    except Exception:
        return numeric_features[:top_k]


def _select_sequential_rows(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.iloc[:max_rows]


def _map_clusters_to_labels(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    if len(y_true) == 0:
        return y_pred
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(c, c) for c in y_pred])


def _clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    cm = contingency_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    matched_count = cm[row_ind, col_ind].sum()
    return float(matched_count / len(y_true))


def _clustering_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    mapped_pred = _map_clusters_to_labels(y_true, y_pred)
    return float(balanced_accuracy_score(y_true, mapped_pred))


def _clustering_f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    mapped_pred = _map_clusters_to_labels(y_true, y_pred)
    return float(f1_score(y_true, mapped_pred, average="weighted", zero_division=0))



@dataclass(frozen=True)
class KMeanRunMetadata:
    module: str
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


class KMeanRelationshipsModule:
    name = "kmean"

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
        model_frame = _select_sequential_rows(
            merged[[*numeric_features, *categorical_labels]],
            MAX_ROWS,
        )

        label_pairs_map: dict[str, list[tuple[str, str]]] = {}
        total_combinations = 0
        for label in categorical_labels:
            top_feats = _select_top_features_for_label(
                model_frame, numeric_features, label, MAX_TOP_FEATURES_PER_LABEL
            )
            pairs = list(combinations(top_feats, 2))
            label_pairs_map[label] = pairs
            total_combinations += len(pairs)

        with ModuleProgress(self.name, total=total_combinations, enabled=self.progress) as progress_bar:
            results, cluster_distribution = _fit_kmeans_reports(
                model_frame,
                label_pairs_map,
                categorical_labels,
                progress_bar,
            )

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = KMeanRunMetadata(
            module=self.name,
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
            feature_pairs=sum(len(p) for p in label_pairs_map.values()),
            combinations=total_combinations,
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


def _categorical_label_columns(merged: pd.DataFrame, label_columns: list[str]) -> list[str]:
    columns: list[str] = []
    for column in label_columns:
        values = merged[column].dropna()
        unique_count = int(values.nunique(dropna=True))
        if 2 <= unique_count <= MAX_LABEL_CLASSES:
            columns.append(column)
    return columns


def _label_profile_frame(merged: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    rows = []
    for label in labels:
        values = merged[label]
        rows.append(
            {
                "label": label,
                "dtype": str(values.dtype),
                "non_null": int(values.notna().sum()),
                "unique": int(values.nunique(dropna=True)),
                "top_values": "; ".join(
                    f"{value}={count}" for value, count in values.value_counts(dropna=False).head(10).items()
                ),
            }
        )
    return pd.DataFrame(rows, columns=["label", "dtype", "non_null", "unique", "top_values"])


def _fit_kmeans_reports(
    merged: pd.DataFrame,
    label_pairs_map: dict[str, list[tuple[str, str]]] | list[tuple[str, str]],
    label_columns: list[str],
    progress_bar: ModuleProgress | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []

    for label in label_columns:
        pairs = label_pairs_map.get(label, []) if isinstance(label_pairs_map, dict) else label_pairs_map
        if not pairs:
            continue
        unique_features = list({feat for pair in pairs for feat in pair})
        sub_df = merged[unique_features].apply(_numeric_series)
        corr_matrix = sub_df.corr().abs()

        for feature_1, feature_2 in pairs:
            corr_val = corr_matrix.loc[feature_1, feature_2] if (feature_1 in corr_matrix and feature_2 in corr_matrix) else 0.0
            if not np.isnan(corr_val) and corr_val > MAX_PAIR_CORRELATION:
                result_rows.append(
                    _skipped_result(feature_1, feature_2, label, len(merged), "high_feature_correlation")
                )
            else:
                x = merged[[feature_1, feature_2]].apply(_numeric_series)
                result, distribution = _fit_single_kmeans(feature_1, feature_2, label, x, merged[label])
                result_rows.append(result)
                distribution_rows.extend(distribution)

            if progress_bar is not None:
                progress_bar.step(f"{feature_1}+{feature_2}->{label}")

    return _results_frame(result_rows), _cluster_distribution_frame(distribution_rows)


def _fit_single_kmeans(
    feature_1: str,
    feature_2: str,
    label: str,
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    frame = pd.concat([features, labels.rename(label)], axis=1).dropna()
    if len(frame) < MIN_SAMPLES:
        return _skipped_result(feature_1, feature_2, label, len(frame), "not_enough_samples"), []

    label_values = frame[label].astype(str)
    class_count = int(label_values.nunique(dropna=True))
    if class_count < 2:
        return _skipped_result(feature_1, feature_2, label, len(frame), "label_has_less_than_2_classes"), []
    if class_count > MAX_LABEL_CLASSES:
        return _skipped_result(feature_1, feature_2, label, len(frame), "label_has_too_many_classes"), []
    test_count = int(np.ceil(len(frame) * TEST_SIZE))
    train_count = len(frame) - test_count
    if train_count < class_count:
        return _skipped_result(feature_1, feature_2, label, len(frame), "not_enough_train_samples_for_clusters"), []

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(label_values)
    train_frame, test_frame = train_test_split(
        frame,
        test_size=TEST_SIZE,
        shuffle=False,
    )

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_frame[[feature_1, feature_2]])
    test_x = scaler.transform(test_frame[[feature_1, feature_2]])
    if _distinct_row_count(train_x) < class_count:
        return _skipped_result(
            feature_1,
            feature_2,
            label,
            len(frame),
            "not_enough_distinct_points_for_clusters",
        ), []

    model = KMeans(n_clusters=class_count, n_init=1, random_state=RANDOM_STATE)
    model.fit(train_x)
    train_clusters = model.predict(train_x)
    test_clusters = model.predict(test_x)

    train_labels = encoder.transform(train_frame[label].astype(str))
    test_labels = encoder.transform(test_frame[label].astype(str))
    distribution = _cluster_distribution(
        feature_1,
        feature_2,
        label,
        test_clusters,
        test_frame[label].astype(str),
    )

    return (
        {
            "feature_1": feature_1,
            "feature_2": feature_2,
            "label": label,
            "status": "ok",
            "samples": int(len(frame)),
            "train_samples": int(len(train_frame)),
            "test_samples": int(len(test_frame)),
            "label_classes": int(class_count),
            "clusters": int(class_count),
            "train_accuracy": _round(float(_clustering_accuracy(train_labels, train_clusters) * 100)),
            "test_accuracy": _round(float(_clustering_accuracy(test_labels, test_clusters) * 100)),
            "train_balanced_accuracy": _round(float(_clustering_balanced_accuracy(train_labels, train_clusters) * 100)),
            "test_balanced_accuracy": _round(float(_clustering_balanced_accuracy(test_labels, test_clusters) * 100)),
            "train_f1_weighted": _round(float(_clustering_f1_weighted(train_labels, train_clusters) * 100)),
            "test_f1_weighted": _round(float(_clustering_f1_weighted(test_labels, test_clusters) * 100)),
            "train_adjusted_rand": _round(float(adjusted_rand_score(train_labels, train_clusters))),
            "test_adjusted_rand": _round(float(adjusted_rand_score(test_labels, test_clusters))),
            "train_normalized_mutual_info": _round(float(normalized_mutual_info_score(train_labels, train_clusters))),
            "test_normalized_mutual_info": _round(float(normalized_mutual_info_score(test_labels, test_clusters))),
            "test_homogeneity": _round(float(homogeneity_score(test_labels, test_clusters))),
            "test_completeness": _round(float(completeness_score(test_labels, test_clusters))),
            "test_v_measure": _round(float(v_measure_score(test_labels, test_clusters))),
            "test_silhouette": _safe_silhouette(test_x, test_clusters),
            "train_inertia": _round(float(model.inertia_)),
            "note": "",
        },
        distribution,
    )


def _skipped_result(
    feature_1: str,
    feature_2: str,
    label: str,
    samples: int,
    note: str,
) -> dict[str, object]:
    return {
        "feature_1": feature_1,
        "feature_2": feature_2,
        "label": label,
        "status": "skipped",
        "samples": int(samples),
        "train_samples": 0,
        "test_samples": 0,
        "label_classes": None,
        "clusters": None,
        "train_accuracy": None,
        "test_accuracy": None,
        "train_balanced_accuracy": None,
        "test_balanced_accuracy": None,
        "train_f1_weighted": None,
        "test_f1_weighted": None,
        "train_adjusted_rand": None,
        "test_adjusted_rand": None,
        "train_normalized_mutual_info": None,
        "test_normalized_mutual_info": None,
        "test_homogeneity": None,
        "test_completeness": None,
        "test_v_measure": None,
        "test_silhouette": None,
        "train_inertia": None,
        "note": note,
    }


MAX_SILHOUETTE_SAMPLES = 1_000


def _safe_silhouette(features: np.ndarray, clusters: np.ndarray) -> float | None:
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
    rng = np.random.defaultrng(random_state)
    indices = np.sort(rng.choice(len(features), size=max_samples, replace=False))
    return features[indices], clusters[indices]


def _distinct_row_count(values: np.ndarray) -> int:
    return int(np.unique(values, axis=0).shape[0])


def _cluster_distribution(
    feature_1: str,
    feature_2: str,
    label: str,
    clusters: np.ndarray,
    labels: pd.Series,
) -> list[dict[str, object]]:
    frame = pd.DataFrame({"cluster": clusters, "label_value": labels.to_numpy()})
    cluster_sizes = frame["cluster"].value_counts().to_dict()
    rows: list[dict[str, object]] = []
    for (cluster, label_value), count in frame.value_counts(["cluster", "label_value"]).items():
        cluster_total = int(cluster_sizes[cluster])
        rows.append(
            {
                "feature_1": feature_1,
                "feature_2": feature_2,
                "label": label,
                "cluster": int(cluster),
                "label_value": str(label_value),
                "count": int(count),
                "cluster_pct": _round(float(count / cluster_total * 100)) if cluster_total else None,
            }
        )
    return rows


def _results_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "feature_1",
        "feature_2",
        "label",
        "status",
        "samples",
        "train_samples",
        "test_samples",
        "label_classes",
        "clusters",
        "train_accuracy",
        "test_accuracy",
        "train_balanced_accuracy",
        "test_balanced_accuracy",
        "train_f1_weighted",
        "test_f1_weighted",
        "train_adjusted_rand",
        "test_adjusted_rand",
        "train_normalized_mutual_info",
        "test_normalized_mutual_info",
        "test_homogeneity",
        "test_completeness",
        "test_v_measure",
        "test_silhouette",
        "train_inertia",
        "note",
    ]
    return pd.DataFrame(rows, columns=columns)


def _cluster_distribution_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    columns = [
        "feature_1",
        "feature_2",
        "label",
        "cluster",
        "label_value",
        "count",
        "cluster_pct",
    ]
    return pd.DataFrame(rows, columns=columns)


def _ranked_successful(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    successful = results[results["status"] == "ok"].copy()
    if successful.empty:
        return successful
    return successful.sort_values(
        ["test_balanced_accuracy", "test_f1_weighted", "test_accuracy", "test_adjusted_rand", "samples"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)



def _render_markdown(
    metadata: KMeanRunMetadata,
    results: pd.DataFrame,
    label_profile: pd.DataFrame,
) -> str:
    ranked = _ranked_successful(results)
    top_results = (
        _markdown_table(ranked.head(20))
        if not ranked.empty
        else "No successful KMeans clustering runs were available."
    )
    skipped = results[results["status"] == "skipped"] if not results.empty else results
    skipped_summary = (
        _markdown_table(skipped["note"].value_counts().rename_axis("reason").reset_index(name="count"))
        if not skipped.empty
        else "No skipped combinations."
    )
    ignored = ", ".join(metadata.ignored_columns) if metadata.ignored_columns else "none"
    return f"""# KMeans Feature-Pair / Label Clustering Report

## Run

- Module: `{metadata.module}`
- Created at: `{metadata.created_at}`
- Feature CSV: `{metadata.feature_csv}`
- Label CSV: `{metadata.label_csv}`
- Join strategy: {metadata.join_strategy}
- Feature shape: {metadata.feature_shape.rows} rows x {metadata.feature_shape.columns} columns
- Label shape: {metadata.label_shape.rows} rows x {metadata.label_shape.columns} columns
- Merged shape: {metadata.merged_shape.rows} rows x {metadata.merged_shape.columns} columns
- Model rows: {metadata.model_rows}
- Ignored columns: {ignored}
- Numeric features: {len(metadata.numeric_features)}
- Categorical/text labels: {len(metadata.categorical_labels)}
- Feature pairs: {metadata.feature_pairs}
- Feature-pair/label combinations evaluated: {metadata.combinations}

## Step Outputs

### Step 1: Numeric Features

{", ".join(metadata.numeric_features) if metadata.numeric_features else "No numeric features were available."}

### Step 1: Categorical/Text Labels

{_markdown_table(label_profile) if not label_profile.empty else "No categorical/text labels were available."}

### Steps 2-4: KMeans Evaluation

Every numeric feature pair is combined with every categorical/text label. Each row in `kmean_results.csv` is one feature-pair/label combination, including skipped combinations.

{top_results}

## Skipped Combination Summary

{skipped_summary}

## Artifacts

- `summary.json`
- `numeric_features.csv`
- `categorical_labels.csv`
- `kmean_results.csv`
- `cluster_label_distribution.csv`
"""


def _render_html(markdown: str, results: pd.DataFrame) -> str:
    ranked = _ranked_successful(results)
    table = ranked.head(50).to_html(index=False, classes="data-table") if not ranked.empty else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KMeans Feature-Pair / Label Clustering Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    pre {{ white-space: pre-wrap; background: #f5f7fa; padding: 16px; border-radius: 6px; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin-top: 24px; }}
    .data-table th, .data-table td {{ border: 1px solid #d9e2ec; padding: 8px; text-align: left; }}
    .data-table th {{ background: #edf2f7; }}
  </style>
</head>
<body>
  {_html_markdown_details(markdown)}
  <h2>Top KMeans Results</h2>
  {table}
</body>
</html>
"""
