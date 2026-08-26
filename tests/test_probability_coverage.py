from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_coverage import (
    ProbabilityCoverageModule,
    _compute_quantile_bins,
    _evaluate_feature_crosstab_coverage,
)
from fldataprofiler.registry import get_module


def make_test_datasets(
    base_dir: Path,
    rows: int = 400,
    seed: int = 42,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    # Feature 1: Monotonic relationship with target
    x1 = np.linspace(-2.0, 2.0, rows)
    # Feature 2: High value concentrated target
    x2 = rng.uniform(-1.0, 1.0, rows)
    noise = rng.standard_normal(rows)

    # Target: 1 when x1 > 0.0 else 0
    target = (x1 > 0.0).astype(int)

    feature_path = base_dir / "feature.csv"
    label_path = base_dir / "label.csv"

    pd.DataFrame({
        "Date": dates,
        "x1": x1,
        "x2": x2,
        "noise": noise,
    }).to_csv(feature_path, index=False)

    pd.DataFrame({
        "Date": dates,
        "target": target,
    }).to_csv(label_path, index=False)

    return feature_path, label_path


class ProbabilityCoverageModuleTests(unittest.TestCase):
    def test_registry_lookup(self) -> None:
        mod1 = get_module("probability_coverage")
        mod2 = get_module("coverage")
        mod3 = get_module("probabilitycoverage")
        self.assertIsInstance(mod1, ProbabilityCoverageModule)
        self.assertIsInstance(mod2, ProbabilityCoverageModule)
        self.assertIsInstance(mod3, ProbabilityCoverageModule)

    def test_quantile_binning(self) -> None:
        s = pd.Series(np.linspace(1, 100, 100))
        bins = _compute_quantile_bins(s, n_bins=20)
        self.assertEqual(len(bins), 100)
        self.assertEqual(bins.min(), 1)
        self.assertEqual(bins.max(), 20)
        self.assertEqual((bins == 1).sum(), 5)

    def test_crosstab_coverage_evaluation(self) -> None:
        rows = 400
        x = pd.Series(np.linspace(-2.0, 2.0, rows))
        target = pd.Series((x > 0.0).astype(int))

        mat_sum, summaries, cell_details = _evaluate_feature_crosstab_coverage(
            feature_series=x,
            target_series=target,
            feature_name="x1",
            target_name="target",
            n_quantiles=20,
            min_probability=0.55,
            min_support=10,
            min_lift=1.0,
        )

        self.assertIsNotNone(mat_sum)
        self.assertTrue(len(summaries) > 0)
        self.assertTrue(len(cell_details) > 0)
        self.assertEqual(mat_sum["qualified_cells"], 20)  # 10 bins for class 0 (100%) and 10 bins for class 1 (100%)
        self.assertEqual(mat_sum["qualified_bins"], 20)

        # For class 1 (x > 0.0), top 10 quantiles should be 100% -> 10 qualified bins
        sum_cls1 = [s for s in summaries if s["target_class"] == 1][0]
        self.assertEqual(sum_cls1["qualified_bins"], 10)
        self.assertEqual(sum_cls1["total_bins"], 20)
        self.assertEqual(sum_cls1["bin_coverage_pct"], 50.0)
        self.assertEqual(sum_cls1["sample_coverage_pct"], 50.0)
        self.assertEqual(sum_cls1["weighted_qualified_prob"], 1.0)

    def test_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            f_path, l_path = make_test_datasets(base, rows=300)
            out_dir = base / "reports"

            module = ProbabilityCoverageModule()
            result = module.run(
                feature_csv=f_path,
                label_csv=l_path,
                output_dir=out_dir,
            )

            report_dir = out_dir / "probability_coverage"
            self.assertEqual(result.report_dir, report_dir)

            report_md = report_dir / "report.md"
            report_html = report_dir / "report.html"
            summary_json = report_dir / "summary.json"
            feature_scores_csv = report_dir / "feature_coverage_scores.csv"
            matrix_rankings_csv = report_dir / "matrix_coverage_rankings.csv"
            target_matrix_csv = report_dir / "crosstab_matrices_target.csv"
            cell_details_csv = report_dir / "quantile_crosstab_probabilities.csv"
            dist_png = report_dir / "probability_coverage_distribution.png"

            self.assertTrue(report_md.exists())
            self.assertTrue(report_html.exists())
            self.assertTrue(summary_json.exists())
            self.assertTrue(feature_scores_csv.exists())
            self.assertTrue(matrix_rankings_csv.exists())
            self.assertTrue(target_matrix_csv.exists())
            self.assertTrue(cell_details_csv.exists())
            self.assertTrue(dist_png.exists())

            # Verify summary.json content
            with summary_json.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["metadata"]["module"], "probability_coverage")
            self.assertIn("top_coverage_matrices", data)

            # Verify sorting: qualified_bins should be non-increasing
            df_scores = pd.read_csv(feature_scores_csv)
            if len(df_scores) > 1:
                q_bins = df_scores["qualified_bins"].values
                self.assertTrue(all(q_bins[i] >= q_bins[i+1] for i in range(len(q_bins)-1)))

    def test_max_label_classes_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            f_path = base / "feature.csv"
            l_path = base / "label.csv"

            # Create a discrete target (2 classes) and a continuous target (100 unique values)
            pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=200, freq="5min"),
                "f1": np.linspace(-1, 1, 200),
            }).to_csv(f_path, index=False)

            pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=200, freq="5min"),
                "discrete_tgt": np.random.choice([0, 1], size=200),
                "continuous_tgt": np.linspace(0, 100, 200),  # 200 unique values > max_label_classes
            }).to_csv(l_path, index=False)

            module = ProbabilityCoverageModule()
            result = module.run(
                feature_csv=f_path,
                label_csv=l_path,
                output_dir=base / "reports",
            )

            with open(result.report_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)

            # Continuous target must be skipped, only discrete target evaluated
            self.assertEqual(summary["metadata"]["targets"], ["discrete_tgt"])

    def test_categorical_feature_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            f_path = base / "feature.csv"
            l_path = base / "label.csv"

            # Create:
            # - continuous numeric feature (200 unique float values)
            # - low-cardinality discrete feature (only 2 unique values [0, 1])
            # - categorical string feature (['A', 'B'])
            pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=200, freq="5min"),
                "continuous_feat": np.linspace(-1, 1, 200),
                "binary_indicator": np.random.choice([0, 1], size=200),
                "string_cat": np.random.choice(["A", "B"], size=200),
            }).to_csv(f_path, index=False)

            pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=200, freq="5min"),
                "target": np.random.choice([0, 1], size=200),
            }).to_csv(l_path, index=False)

            module = ProbabilityCoverageModule()
            result = module.run(
                feature_csv=f_path,
                label_csv=l_path,
                output_dir=base / "reports",
            )

            with open(result.report_dir / "summary.json", "r", encoding="utf-8") as f:
                summary = json.load(f)

            # Only the continuous numeric feature should be evaluated
            self.assertEqual(summary["metadata"]["features_evaluated"], 1)
            top_m = summary["top_coverage_matrices"]
            self.assertEqual(len(top_m), 1)
            self.assertEqual(top_m[0]["feature"], "continuous_feat")


if __name__ == "__main__":
    unittest.main()
