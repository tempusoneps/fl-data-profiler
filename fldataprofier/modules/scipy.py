from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, stats

from fldataprofier.modules.base import ModuleResult
from fldataprofier.modules.statistics import DatasetShape
from fldataprofier.utils import (
    _html_markdown_details,
    _read_table_with_date_index,
    _date_columns,
    _markdown_table,
    _merge_inputs,
    _numeric_series,
    _round,
    _select_targets,
    _write_csv,
    _write_json,
)


@dataclass(frozen=True)
class ScipyRunMetadata:
    module: str
    created_at: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    features: list[str]
    targets: list[str]
    ignored_columns: list[str]


class ScipyRelationshipsModule:
    name = "scipy"

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

        run_dir = output_dir / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        pairwise = _pairwise_tests(merged, feature_columns, selected_targets)
        combined = _combined_two_feature_tests(merged, feature_columns, selected_targets)

        metadata = ScipyRunMetadata(
            module=self.name,
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=DatasetShape(*features.shape),
            label_shape=DatasetShape(*labels.shape),
            merged_shape=DatasetShape(*merged.shape),
            features=feature_columns,
            targets=selected_targets,
            ignored_columns=ignored_columns,
        )

        artifacts = [
            _write_json(
                run_dir / "summary.json",
                {
                    "metadata": asdict(metadata),
                    "top_pairwise_relationships": pairwise.head(50).to_dict(orient="records"),
                    "combined_two_feature_tests": combined.to_dict(orient="records"),
                },
            ),
            _write_csv(run_dir / "pairwise.csv", pairwise),
            _write_csv(run_dir / "two_feature.csv", combined),
        ]

        markdown = _render_markdown(metadata, pairwise, combined)
        md_path = run_dir / "report.md"
        md_path.write_text(markdown, encoding="utf-8")
        artifacts.append(md_path)

        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(markdown, pairwise, combined), encoding="utf-8")
        artifacts.append(html_path)

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)


def _pairwise_tests(
    merged: pd.DataFrame, feature_columns: list[str], label_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in feature_columns:
        for label in label_columns:
            row = _feature_label_test(merged[feature], merged[label])
            if row is None:
                continue
            rows.append({"feature": feature, "label": label, **row})

    columns = [
        "feature",
        "label",
        "test",
        "statistic",
        "p_value",
        "effect_size",
        "effect_name",
        "samples",
        "feature_type",
        "label_type",
        "note",
    ]
    result = pd.DataFrame(rows, columns=columns)
    if result.empty:
        return result
    return result.sort_values(
        ["p_value", "effect_size", "samples"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)


def _feature_label_test(feature: pd.Series, label: pd.Series) -> dict[str, object] | None:
    frame = pd.concat([feature, label], axis=1).dropna()
    if len(frame) < 3:
        return None

    x = frame.iloc[:, 0]
    y = frame.iloc[:, 1]
    x_numeric = _numeric_series(x)
    y_numeric = _numeric_series(y)
    x_is_numeric = x_numeric.notna().sum() == len(frame)
    y_is_numeric = y_numeric.notna().sum() == len(frame)
    y_unique = y.nunique(dropna=True)

    if x_is_numeric and y_is_numeric and x_numeric.nunique() >= 2 and y_numeric.nunique() >= 2:
        numeric_pair = pd.concat([x_numeric, y_numeric], axis=1).dropna()
        if len(numeric_pair) < 3:
            return None
        pearson = stats.pearsonr(numeric_pair.iloc[:, 0].to_numpy(), numeric_pair.iloc[:, 1].to_numpy())
        spearman = stats.spearmanr(numeric_pair.iloc[:, 0].to_numpy(), numeric_pair.iloc[:, 1].to_numpy())
        return {
            "test": "pearsonr",
            "statistic": _round(float(pearson.statistic)),
            "p_value": _round(float(pearson.pvalue)),
            "effect_size": _round(abs(float(pearson.statistic))),
            "effect_name": "abs_pearson_r",
            "samples": int(len(numeric_pair)),
            "feature_type": "numeric",
            "label_type": "numeric",
            "note": f"spearman_r={_round(float(spearman.statistic))}, spearman_p={_round(float(spearman.pvalue))}",
        }

    if x_is_numeric and y_unique == 2:
        groups = [x_numeric[y == value].dropna().to_numpy() for value in y.dropna().unique()]
        if len(groups[0]) < 2 or len(groups[1]) < 2:
            return None
        ttest = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        mannwhitney = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        effect = _cohens_d(groups[0], groups[1])
        return {
            "test": "welch_ttest",
            "statistic": _round(float(ttest.statistic)),
            "p_value": _round(float(ttest.pvalue)),
            "effect_size": _round(abs(effect)),
            "effect_name": "abs_cohens_d",
            "samples": int(len(frame)),
            "feature_type": "numeric",
            "label_type": "binary",
            "note": f"mannwhitney_u={_round(float(mannwhitney.statistic))}, mannwhitney_p={_round(float(mannwhitney.pvalue))}",
        }

    if x_is_numeric and 2 < y_unique <= 50:
        groups = [x_numeric[y == value].dropna().to_numpy() for value in y.dropna().unique()]
        groups = [group for group in groups if len(group) >= 2]
        if len(groups) < 2:
            return None
        anova = stats.f_oneway(*groups)
        kruskal = stats.kruskal(*groups)
        return {
            "test": "one_way_anova",
            "statistic": _round(float(anova.statistic)),
            "p_value": _round(float(anova.pvalue)),
            "effect_size": None,
            "effect_name": "not_computed",
            "samples": int(len(frame)),
            "feature_type": "numeric",
            "label_type": "categorical",
            "note": f"kruskal_h={_round(float(kruskal.statistic))}, kruskal_p={_round(float(kruskal.pvalue))}",
        }

    if not x_is_numeric and not y_is_numeric:
        table = pd.crosstab(x, y)
        if table.shape[0] < 2 or table.shape[1] < 2:
            return None
        chi2 = stats.chi2_contingency(table)
        return {
            "test": "chi2_contingency",
            "statistic": _round(float(chi2.statistic)),
            "p_value": _round(float(chi2.pvalue)),
            "effect_size": _round(_cramers_v(chi2.statistic, int(table.to_numpy().sum()), table.shape)),
            "effect_name": "cramers_v",
            "samples": int(len(frame)),
            "feature_type": "categorical",
            "label_type": "categorical",
            "note": f"dof={chi2.dof}",
        }

    return None


def _combined_two_feature_tests(
    merged: pd.DataFrame, feature_columns: list[str], label_columns: list[str]
) -> pd.DataFrame:
    numeric_features = [
        column
        for column in feature_columns
        if _numeric_series(merged[column]).notna().sum() == merged[column].notna().sum()
        and _numeric_series(merged[column]).nunique(dropna=True) >= 2
    ]
    if len(numeric_features) != 2:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "label",
                "test",
                "r_squared",
                "adjusted_r_squared",
                "f_statistic",
                "p_value",
                "samples",
                "note",
            ]
        )

    rows: list[dict[str, object]] = []
    for feature_1, feature_2 in combinations(numeric_features, 2):
        for label in label_columns:
            row = _two_feature_linear_test(
                merged[[feature_1, feature_2]],
                _numeric_series(merged[label]),
            )
            if row is None:
                continue
            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "label": label,
                    **row,
                }
            )

    result = pd.DataFrame(
        rows,
        columns=[
            "feature_1",
            "feature_2",
            "label",
            "test",
            "r_squared",
            "adjusted_r_squared",
            "f_statistic",
            "p_value",
            "samples",
            "note",
        ],
    )
    if result.empty:
        return result
    return result.sort_values(["p_value", "r_squared"], ascending=[True, False], na_position="last").reset_index(drop=True)


def _two_feature_linear_test(features: pd.DataFrame, label: pd.Series) -> dict[str, object] | None:
    frame = pd.concat([features.apply(_numeric_series), label.rename("label")], axis=1).dropna()
    if len(frame) < 5 or frame["label"].nunique() < 2:
        return None

    x = frame.iloc[:, :2].to_numpy(dtype=float)
    y = frame["label"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), x])
    coefficients, *_ = linalg.lstsq(design, y)
    fitted = design @ coefficients
    residual = y - fitted

    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return None

    n = len(y)
    predictors = 2
    r_squared = 1 - (ss_res / ss_tot)
    adjusted = 1 - (1 - r_squared) * (n - 1) / (n - predictors - 1)
    f_stat = (r_squared / predictors) / ((1 - r_squared) / (n - predictors - 1))
    p_value = stats.f.sf(f_stat, predictors, n - predictors - 1)

    return {
        "test": "scipy_lstsq_ols_f_test",
        "r_squared": _round(float(r_squared)),
        "adjusted_r_squared": _round(float(adjusted)),
        "f_statistic": _round(float(f_stat)),
        "p_value": _round(float(p_value)),
        "samples": int(n),
        "note": "Linear numeric-label test for exactly two numeric features; use statsmodels/sklearn later for richer modeling.",
    }


def _cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    pooled = np.sqrt(((group_a.var(ddof=1) + group_b.var(ddof=1)) / 2))
    if pooled == 0:
        return 0.0
    return float((group_a.mean() - group_b.mean()) / pooled)


def _cramers_v(chi2_statistic: float, sample_count: int, shape: tuple[int, int]) -> float:
    denominator = sample_count * (min(shape) - 1)
    if denominator <= 0:
        return 0.0
    return float(np.sqrt(chi2_statistic / denominator))


def _render_markdown(
    metadata: ScipyRunMetadata, pairwise: pd.DataFrame, combined: pd.DataFrame
) -> str:
    pairwise_table = (
        _markdown_table(pairwise.head(20))
        if not pairwise.empty
        else "No valid SciPy pairwise tests were available."
    )
    combined_table = (
        _markdown_table(combined)
        if not combined.empty
        else "No two-feature numeric label test was available; review pairwise tests instead."
    )
    ignored = ", ".join(metadata.ignored_columns) if metadata.ignored_columns else "none"
    return f"""# SciPy Feature/Label Relationship Report

## Run

- Module: `{metadata.module}`
- Created at: `{metadata.created_at}`
- Feature CSV: `{metadata.feature_csv}`
- Label CSV: `{metadata.label_csv}`
- Join strategy: {metadata.join_strategy}
- Feature shape: {metadata.feature_shape.rows} rows x {metadata.feature_shape.columns} columns
- Label shape: {metadata.label_shape.rows} rows x {metadata.label_shape.columns} columns
- Merged shape: {metadata.merged_shape.rows} rows x {metadata.merged_shape.columns} columns
- Ignored columns: {ignored}
- Features: {", ".join(metadata.features)}
- Targets: {", ".join(metadata.targets)}

## Two Feature Test

{combined_table}

## Pairwise Tests

{pairwise_table}

## Artifacts

- `summary.json`
- `pairwise.csv`
- `two_feature.csv`
"""


def _render_html(markdown: str, pairwise: pd.DataFrame, combined: pd.DataFrame) -> str:
    combined_table = combined.to_html(index=False, classes="data-table") if not combined.empty else ""
    pairwise_table = pairwise.head(50).to_html(index=False, classes="data-table") if not pairwise.empty else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SciPy Feature/Label Relationship Report</title>
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
  <h2>Two Feature Test</h2>
  {combined_table}
  <h2>Pairwise Tests</h2>
  {pairwise_table}
</body>
</html>
"""
