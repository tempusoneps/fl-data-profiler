from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability import (
    ProbabilityModule,
    _compute_feature_target_probabilities,
    _compute_quantile_bins,
)


def make_test_datasets(
    base_dir: Path,
    rows: int = 400,
    seed: int = 42,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    # Feature 1: Monotonic signal strongly correlated with binary target
    signal = np.linspace(-3.0, 3.0, rows)
    # Feature 2: Random noise
    noise = rng.standard_normal(rows)
    # Feature 3: Duplicated values
    duplicates = np.tile([1.0, 2.0, 3.0, 4.0], int(np.ceil(rows / 4)))[:rows]

    # Binary label: 1 if signal > 0 else 0
    binary_target = (signal + rng.standard_normal(rows) * 0.2 > 0).astype(int)

    # Multiclass label: 0 (down), 1 (flat), 2 (up)
    multi_target = np.where(signal < -1.0, "down", np.where(signal > 1.0, "up", "flat"))

    feature_path = base_dir / "feature.csv"
    label_path = base_dir / "label.csv"

    pd.DataFrame(
        {
            "Date": dates,
            "signal": signal,
            "noise": noise,
            "duplicates": duplicates,
        }
    ).to_csv(feature_path, index=False)

    pd.DataFrame(
        {
            "Date": dates,
            "binary_target": binary_target,
            "multi_target": multi_target,
        }
    ).to_csv(label_path, index=False)

    return feature_path, label_path


class ProbabilityModuleTests(unittest.TestCase):
    def test_quantile_binning_equal_frequency_and_duplicates(self) -> None:
        # 100 samples, 20 bins -> 5 samples per bin
        series = pd.Series(np.repeat([10.0, 20.0, 30.0, 40.0, 50.0], 20))
        bins = _compute_quantile_bins(series, n_bins=20)
        self.assertEqual(len(bins), 100)
        self.assertEqual(bins.min(), 1)
        self.assertEqual(bins.max(), 20)
        # Each bin should have exactly 5 samples
        counts = bins.value_counts()
        self.assertEqual(len(counts), 20)
        self.assertTrue((counts == 5).all())

    def test_probability_spread_and_monotonicity(self) -> None:
        signal = pd.Series(np.linspace(0, 100, 200))
        # Perfect step target
        target = pd.Series(np.where(signal > 50, 1, 0))

        score_rows, quantile_rows = _compute_feature_target_probabilities(
            signal, target, "signal", "target", n_bins=20
        )
        self.assertTrue(len(score_rows) > 0)
        self.assertTrue(len(quantile_rows) > 0)

        # Look at class 1 score row
        class_1_score = next(r for r in score_rows if r["target_class"] == 1)
        self.assertEqual(class_1_score["prob_spread"], 1.0)
        self.assertGreater(float(class_1_score["monotonicity"]), 0.85)
        self.assertGreater(float(class_1_score["information_value"]), 1.0)

    def test_multiclass_categorical_targets(self) -> None:
        signal = pd.Series(np.linspace(-3.0, 3.0, 300))
        multi_target = pd.Series(
            np.where(signal < -1.0, "down", np.where(signal > 1.0, "up", "flat"))
        )

        score_rows, quantile_rows = _compute_feature_target_probabilities(
            signal, multi_target, "signal", "multi_target", n_bins=20
        )
        classes = {r["target_class"] for r in score_rows}
        self.assertEqual(classes, {"down", "flat", "up"})

        # Verify sum of probabilities across classes in each bin is 1.0
        qdf = pd.DataFrame(quantile_rows)
        bin_prob_sums = qdf.groupby("bin_index")["conditional_prob"].sum()
        for _, prob_sum in bin_prob_sums.items():
            self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_entropy_and_woe_calculation(self) -> None:
        # Clean separable signal
        signal = pd.Series(np.linspace(-10, 10, 100))
        target = pd.Series(np.where(signal > 0, 1, 0))

        _, quantile_rows = _compute_feature_target_probabilities(
            signal, target, "signal", "target", n_bins=10
        )
        # Bins at the extreme ends should have zero entropy (pure bins)
        first_bin = next(r for r in quantile_rows if r["bin_index"] == 1 and r["target_class"] == 1)
        last_bin = next(r for r in quantile_rows if r["bin_index"] == 10 and r["target_class"] == 1)
        self.assertEqual(first_bin["conditional_prob"], 0.0)
        self.assertEqual(first_bin["entropy"], 0.0)
        self.assertEqual(last_bin["conditional_prob"], 1.0)
        self.assertEqual(last_bin["entropy"], 0.0)

    def test_probability_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=200)
            output_dir = tmp_path / "reports"

            module = ProbabilityModule(n_bins=20)
            result = module.run(feature_csv, label_csv, output_dir)

            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "probability")

            # Check required artifacts exist
            expected_files = [
                "feature_probability_scores.csv",
                "quantile_conditional_probabilities.csv",
                "probability_distribution.png",
                "summary.json",
                "report.md",
                "report.html",
            ]
            for filename in expected_files:
                file_path = result.report_dir / filename
                self.assertTrue(file_path.exists(), f"Missing artifact {filename}")
                self.assertGreater(file_path.stat().st_size, 0)

            # Check scores CSV contents
            scores = pd.read_csv(result.report_dir / "feature_probability_scores.csv")
            self.assertIn("feature", scores.columns)
            self.assertIn("target", scores.columns)
            self.assertIn("target_class", scores.columns)
            self.assertIn("information_value", scores.columns)
            self.assertIn("prob_spread", scores.columns)
            self.assertIn("monotonicity", scores.columns)

            # Check quantiles CSV contents
            quantiles = pd.read_csv(result.report_dir / "quantile_conditional_probabilities.csv")
            self.assertIn("bin_index", quantiles.columns)
            self.assertIn("conditional_prob", quantiles.columns)
            self.assertIn("woe", quantiles.columns)
            self.assertIn("entropy", quantiles.columns)

            # Check summary.json contents
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability")
            self.assertIn("feature_shape", summary)
            self.assertIn("top_features", summary)


    def test_missing_values_and_constant_features(self) -> None:
        # Series with NaNs and a constant series
        feature_with_nans = pd.Series([np.nan, 1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        target_with_nans = pd.Series([0, np.nan, 1, 0, 1, 0, 1, 0, 1, 0])
        scores, quantiles = _compute_feature_target_probabilities(
            feature_with_nans, target_with_nans, "nan_feat", "nan_target", n_bins=5
        )
        self.assertTrue(len(scores) > 0)
        self.assertTrue(len(quantiles) > 0)

        # Constant feature should return empty scores
        constant_feature = pd.Series([1.0] * 50)
        target = pd.Series([0, 1] * 25)
        c_scores, c_quantiles = _compute_feature_target_probabilities(
            constant_feature, target, "constant", "target", n_bins=5
        )
        self.assertEqual(len(c_scores), 0)
        self.assertEqual(len(c_quantiles), 0)

    def test_custom_target_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=200)
            output_dir = tmp_path / "reports"

            module = ProbabilityModule(n_bins=10)
            result = module.run(
                feature_csv,
                label_csv,
                output_dir,
                targets=["binary_target"],
            )
            scores = pd.read_csv(result.report_dir / "feature_probability_scores.csv")
            self.assertTrue((scores["target"] == "binary_target").all())

    def test_cli_probability_run(self) -> None:
        from fldataprofiler.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=150)
            output_dir = tmp_path / "reports"

            exit_code = main([
                "fit",
                str(feature_csv),
                str(label_csv),
                "--module", "probability",
                "--output-dir", str(output_dir),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "probability" / "report.md").exists())
            self.assertTrue((output_dir / "probability" / "report.html").exists())
            self.assertTrue((output_dir / "probability" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()

