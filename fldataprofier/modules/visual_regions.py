from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.progress import ModuleProgress
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _date_columns,
    _html_markdown_details,
    _markdown_table,
    _merge_inputs,
    _numeric_series,
    _read_table_with_date_index,
    _round,
    _sample_rows,
    _select_targets,
    _write_csv,
    _write_json,
)


MAX_ROWS = 50_000
MAX_LABEL_CLASSES = 20
MAX_MISSING_RATIO = 0.5
MIN_NON_NULL = 10
MIN_DISTINCT_VALUES = 2
N_BINS = 10
MAX_CANDIDATE_FEATURES = 24
TOP_1D_FEATURES = 16
RANDOM_STATE = 42
MIN_CELL_SUPPORT = 5
MIN_MODEL_SAMPLES = 40
TEST_SIZE = 0.25


PAIR_SCORE_COLUMNS = [
    "feature_x",
    "feature_y",
    "label",
    "samples",
    "separability",
    "linearity",
    "region_purity",
    "overlap",
    "recommendation",
]


@dataclass(frozen=True)
class VisualRegionsRunMetadata:
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
    candidate_features: list[str]
    feature_pairs: int
    thresholds: dict[str, object]


def _categorical_label_columns(
    merged: pd.DataFrame,
    label_columns: list[str],
    max_classes: int = MAX_LABEL_CLASSES,
) -> list[str]:
    selected: list[str] = []
    for column in label_columns:
        values = merged[column].dropna()
        unique_count = int(values.nunique(dropna=True))
        if 2 <= unique_count <= max_classes:
            selected.append(column)
    return selected


def _prepare_numeric_feature_frame(
    merged: pd.DataFrame,
    feature_columns: list[str],
    max_missing_ratio: float = MAX_MISSING_RATIO,
    min_non_null: int = MIN_NON_NULL,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    date_columns = set(_date_columns(feature_columns))
    prepared: dict[str, pd.Series] = {}
    exclusions: list[dict[str, object]] = []
    row_count = len(merged)
    for column in feature_columns:
        if column in date_columns:
            exclusions.append({"column": column, "reason": "date_column"})
            continue
        values = _numeric_series(merged[column])
        non_null = int(values.notna().sum())
        if non_null == 0:
            exclusions.append({"column": column, "reason": "non_numeric"})
            continue
        missing_ratio = 1.0 if row_count == 0 else 1.0 - (non_null / row_count)
        if missing_ratio > max_missing_ratio or non_null < min_non_null:
            exclusions.append({"column": column, "reason": "too_many_missing"})
            continue
        distinct = int(values.nunique(dropna=True))
        if distinct < MIN_DISTINCT_VALUES:
            exclusions.append({"column": column, "reason": "constant_or_too_few_values"})
            continue
        prepared[column] = values
    return pd.DataFrame(prepared, index=merged.index), exclusions


def _quantile_bin_features(features: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    binned: dict[str, pd.Series] = {}
    for column in features.columns:
        values = features[column]
        valid = values.dropna()
        if valid.nunique(dropna=True) < 2:
            continue
        ranks = values.rank(method="first", na_option="keep")
        bins = pd.qcut(ranks, q=min(n_bins, int(valid.nunique())), labels=False, duplicates="drop")
        binned[column] = bins.astype("float64")
    result = pd.DataFrame(binned, index=features.index)
    if result.empty:
        return result
    return result.fillna(255).astype("uint8")


def _score_1d_candidates(bin_frame: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in bin_frame.columns:
        feature_bins = bin_frame[feature]
        valid_feature = feature_bins != 255
        for label in labels.columns:
            frame = pd.DataFrame({"bin": feature_bins, "label": labels[label]}).loc[valid_feature].dropna()
            if frame.empty or frame["label"].nunique(dropna=True) < 2:
                continue
            prior = frame["label"].value_counts(normalize=True)
            base_purity = float(prior.max())
            weighted_purity = 0.0
            for _, group in frame.groupby("bin", observed=True):
                purity = float(group["label"].value_counts(normalize=True).max())
                weighted_purity += purity * (len(group) / len(frame))
            score = max(0.0, weighted_purity - base_purity)
            rows.append(
                {
                    "feature": feature,
                    "label": label,
                    "samples": len(frame),
                    "score": _round(score),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["feature", "label", "samples", "score"])
    return pd.DataFrame(rows).sort_values(["score", "samples"], ascending=[False, False]).reset_index(drop=True)


def _select_candidate_features(
    candidate_scores: pd.DataFrame,
    valid_features: list[str],
    max_features: int = MAX_CANDIDATE_FEATURES,
    random_state: int = RANDOM_STATE,
) -> list[str]:
    selected: list[str] = []
    if not candidate_scores.empty:
        sort_columns = ["score"]
        if "samples" in candidate_scores.columns:
            sort_columns.append("samples")
        ranked = candidate_scores.sort_values(sort_columns, ascending=[False] * len(sort_columns))
        for feature in ranked["feature"]:
            if feature not in selected:
                selected.append(str(feature))
            if len(selected) >= min(TOP_1D_FEATURES, max_features):
                break
    remaining = [feature for feature in valid_features if feature not in selected]
    if remaining and len(selected) < max_features:
        rng = np.random.default_rng(random_state)
        sample_size = min(len(remaining), max_features - len(selected))
        sampled = list(rng.choice(np.array(remaining, dtype=object), size=sample_size, replace=False))
        selected.extend(str(feature) for feature in sampled)
    return selected[:max_features]

def _evaluate_2d_grid_purity(df: pd.DataFrame, x_bin_col: str, y_bin_col: str, x_val_col: str, y_val_col: str, label_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    global_counts = df[label_col].value_counts(normalize=True)
    cells = []
    for (x_bin, y_bin), group in df.groupby([x_bin_col, y_bin_col]):
        sample_count = len(group)
        label_counts = group[label_col].value_counts()
        majority_label = label_counts.idxmax()
        purity = label_counts.max() / sample_count
        global_prior = global_counts.get(majority_label, 1e-5)
        lift = purity / global_prior
        cells.append({
            "x_bin": x_bin,
            "y_bin": y_bin,
            "sample_count": sample_count,
            "majority_label": majority_label,
            "purity": purity,
            "lift": lift,
            "x_min": group[x_val_col].min(),
            "x_max": group[x_val_col].max(),
            "y_min": group[y_val_col].min(),
            "y_max": group[y_val_col].max(),
        })
    return pd.DataFrame(cells)

def _merge_contiguous_regions(grid_cells: pd.DataFrame, raw_features: pd.DataFrame, feature_x: str, feature_y: str, label_col: str, min_purity: float, min_samples: int) -> pd.DataFrame:
    if grid_cells is None or grid_cells.empty:
        return pd.DataFrame()
    filtered = grid_cells[(grid_cells["purity"] >= min_purity) & (grid_cells["sample_count"] >= min_samples)].copy()
    if filtered.empty:
        return pd.DataFrame()
    
    merged = []
    for label, group in filtered.groupby("majority_label"):
        x_min = group["x_min"].min()
        x_max = group["x_max"].max()
        y_min = group["y_min"].min()
        y_max = group["y_max"].max()
        
        total_samples = group["sample_count"].sum()
        avg_purity = (group["purity"] * group["sample_count"]).sum() / total_samples
        avg_lift = (group["lift"] * group["sample_count"]).sum() / total_samples
        
        merged.append({
            "majority_label": label,
            "purity": avg_purity,
            "sample_count": total_samples,
            "lift": avg_lift,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        })
    return pd.DataFrame(merged)

def _extract_2d_rules(merged_df: pd.DataFrame, feature_columns: list[str], label_columns: list[str], n_bins: int=8, min_samples: int=15, min_purity: float=0.70) -> pd.DataFrame:
    rules = []
    if len(feature_columns) < 2 or not label_columns:
        return pd.DataFrame(columns=["rule_text"])
    label_col = label_columns[0]
    
    bin_frame = _quantile_bin_features(merged_df[feature_columns], n_bins=n_bins)
    candidate_scores = _score_1d_candidates(bin_frame, merged_df[label_columns])
    candidates = _select_candidate_features(candidate_scores, feature_columns)
    if len(candidates) < 2:
        candidates = feature_columns[:2]
        
    for fx, fy in combinations(candidates, 2):
        df_pair = merged_df[[fx, fy, label_col]].dropna()
        if df_pair.empty:
            continue
        try:
            bin_x = pd.qcut(df_pair[fx], q=n_bins, labels=False, duplicates="drop")
            bin_y = pd.qcut(df_pair[fy], q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        df_pair["x_bin"] = bin_x
        df_pair["y_bin"] = bin_y
        
        cells = _evaluate_2d_grid_purity(df_pair, "x_bin", "y_bin", fx, fy, label_col)
        merged_regions = _merge_contiguous_regions(cells, None, fx, fy, label_col, min_purity, min_samples)
        
        for _, region in merged_regions.iterrows():
            sample_count = region["sample_count"]
            purity = region["purity"]
            lift = region["lift"]
            score = purity * np.log2(max(2, sample_count)) * lift
            rule_text = f"IF {fx} ∈ [{region['x_min']:.2f}, {region['x_max']:.2f}] AND {fy} ∈ [{region['y_min']:.2f}, {region['y_max']:.2f}] THEN Label = {region['majority_label']}"
            
            rules.append({
                "rank": 0,
                "feature_x": fx,
                "feature_y": fy,
                "range_x_min": region["x_min"],
                "range_x_max": region["x_max"],
                "range_y_min": region["y_min"],
                "range_y_max": region["y_max"],
                "target_label": region["majority_label"],
                "purity_pct": purity,
                "sample_count": sample_count,
                "coverage_pct": sample_count / len(merged_df),
                "lift_ratio": lift,
                "rule_score": score,
                "rule_text": rule_text
            })
            
    res = pd.DataFrame(rules)
    if res.empty:
        return pd.DataFrame(columns=["rule_text"])
    res = res.sort_values("rule_score", ascending=False).reset_index(drop=True)
    res["rank"] = res.index + 1
    return res

def _render_rules_markdown(metadata: VisualRegionsRunMetadata, rules_df: pd.DataFrame, candidate_scores: pd.DataFrame) -> str:
    lines = [
        f"# Visual Regions (2D Rules)",
        "",
        f"**Module:** `{metadata.module}`",
        f"**Generated:** `{metadata.created_at}`",
        f"**Feature Set:** `{metadata.feature_shape.rows} rows, {metadata.feature_shape.columns} cols`",
        f"**Label Set:** `{metadata.label_shape.rows} rows, {metadata.label_shape.columns} cols`",
        f"**Merged Set:** `{metadata.merged_shape.rows} rows, {metadata.merged_shape.columns} cols`",
        ""
    ]
    
    if rules_df.empty:
        lines.append("No 2D rules extracted.")
    else:
        lines.append("## Top Rules")
        lines.append("")
        for _, rule in rules_df.head(10).iterrows():
            lines.append(f"- **Rank {rule['rank']}**: {rule['rule_text']} (Purity: {rule['purity_pct']:.2%}, Samples: {rule['sample_count']})")
        lines.append("")
        lines.append("## Rule Details")
        lines.append("")
        lines.append(_markdown_table(rules_df))
    return "\n".join(lines)

def _render_rules_html(markdown: str, rules_df: pd.DataFrame, candidate_scores: pd.DataFrame) -> str:
    return _html_markdown_details(markdown)


class VisualRegionsModule:
    name = "visual_regions"

    def __init__(self, n_bins: int=8, min_samples_per_region: int=15, min_purity: float=0.70, progress=None):
        self.n_bins = n_bins
        self.min_samples_per_region = min_samples_per_region
        self.min_purity = min_purity
        self.progress = progress or ModuleProgress(module_name=self.name, total=5)

    def run(self, feature_csv: str | Path, label_csv: str | Path, output_dir: str | Path, join_key: str | None = None, targets: list[str] | None = None) -> ModuleResult:
        feature_path = Path(feature_csv)
        label_path = Path(label_csv)
        out_path = Path(output_dir) / self.name
        out_path.mkdir(parents=True, exist_ok=True)
        
        self.progress.step("load")
        features = _read_table_with_date_index(feature_path)
        labels = _read_table_with_date_index(label_path)
        
        self.progress.step("prepare")
        merged, feature_cols, label_cols, _ = _merge_inputs(features, labels, join_key=join_key)
        label_cols = _select_targets(label_cols, targets)
        
        self.progress.step("extract_rules")
        rules_df = _extract_2d_rules(merged, feature_cols, label_cols, n_bins=self.n_bins, min_samples=self.min_samples_per_region, min_purity=self.min_purity)
        
        self.progress.step("artifacts")
        metadata = VisualRegionsRunMetadata(
            module=self.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_csv=feature_path.name,
            label_csv=label_path.name,
            join_strategy="inner",
            feature_shape=DatasetShape(rows=len(features), columns=len(features.columns)),
            label_shape=DatasetShape(rows=len(labels), columns=len(labels.columns)),
            merged_shape=DatasetShape(rows=len(merged), columns=len(merged.columns)),
            model_rows=len(merged),
            numeric_features=feature_cols,
            categorical_labels=label_cols,
            candidate_features=[],
            feature_pairs=0,
            thresholds={"n_bins": self.n_bins, "min_purity": self.min_purity}
        )
        
        _write_json(out_path / "summary.json", asdict(metadata))
        _write_csv(out_path / "rules_2d.csv", rules_df)
        
        self.progress.step("report")
        candidate_scores = pd.DataFrame()
        md_content = _render_rules_markdown(metadata, rules_df, candidate_scores)
        (out_path / "report.md").write_text(md_content, encoding="utf-8")
        html_content = _render_rules_html(md_content, rules_df, candidate_scores)
        (out_path / "report.html").write_text(html_content, encoding="utf-8")
        
        return ModuleResult(
            report_dir=out_path,
            artifacts=[
                out_path / "summary.json",
                out_path / "rules_2d.csv",
                out_path / "report.md",
                out_path / "report.html"
            ]
        )
