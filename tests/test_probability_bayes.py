from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_bayes import (
    ProbabilityBayesModule,
    _compute_feature_target_bayes_probabilities,
    _compute_log_bayes_factor,
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


class ProbabilityBayesModuleTests(unittest.TestCase):
    def test_quantile_binning_equal_frequency_and_duplicates(self) -> None:
        series = pd.Series(np.repeat([10.0, 20.0, 30.0, 40.0, 50.0], 20))
        bins = _compute_quantile_bins(series, n_bins=20)
        self.assertEqual(len(bins), 100)
        self.assertEqual(bins.min(), 1)
        self.assertEqual(bins.max(), 20)
        counts = bins.value_counts()
        self.assertEqual(len(counts), 20)
        self.assertTrue((counts == 5).all())

    def test_log_bayes_factor_calculation(self) -> None:
        # Extreme evidence against H0
        lbf_strong = _compute_log_bayes_factor(
            k_events=50, n_k=50, alpha_0=5.0, beta_0=5.0, p_0=0.5
        )
        self.assertGreater(lbf_strong, 2.3)  # Strong evidence

        # Null-like evidence (k_events = n_k * p_0)
        lbf_null = _compute_log_bayes_factor(
            k_events=25, n_k=50, alpha_0=5.0, beta_0=5.0, p_0=0.5
        )
        self.assertLessEqual(lbf_null, 0.5)

    def test_bayesian_shrinkage_behavior(self) -> None:
        # 50 zeros in bin 1, 50 ones in bin 2
        feature = pd.Series(np.linspace(1.0, 100.0, 100))
        target = pd.Series([0] * 50 + [1] * 50)

        score_rows, quantile_rows = _compute_feature_target_bayes_probabilities(
            feature, target, "feature", "target", n_bins=2, prior_strength=10.0
        )
        self.assertTrue(len(quantile_rows) > 0)
        bin1_class1 = next(
            r for r in quantile_rows if r["bin_index"] == 1 and r["target_class"] == 1
        )
        self.assertEqual(bin1_class1["raw_prob"], 0.0)
        self.assertGreater(bin1_class1["bayes_prob"], 0.0)  # Shrunk towards base rate (0.5)

        # Credible intervals should contain posterior mean
        for r in quantile_rows:
            self.assertLessEqual(r["ci_lower_95"], r["bayes_prob"] + 1e-6)
            self.assertGreaterEqual(r["ci_upper_95"], r["bayes_prob"] - 1e-6)
            self.assertGreater(r["ci_width"], 0.0)
            self.assertTrue(np.isfinite(r["bayes_woe"]))
            self.assertTrue(np.isfinite(r["bayes_iv_contribution"]))

    def test_bayes_probability_spread_and_monotonicity(self) -> None:
        # 1000 samples -> 50 samples per bin (m=10 has reasonable shrinkage)
        signal = pd.Series(np.linspace(0, 100, 1000))
        target = pd.Series(np.where(signal > 50, 1, 0))

        score_rows, quantile_rows = _compute_feature_target_bayes_probabilities(
            signal, target, "signal", "target", n_bins=20, prior_strength=10.0
        )
        self.assertTrue(len(score_rows) > 0)
        class_1_score = next(r for r in score_rows if r["target_class"] == 1)
        self.assertGreater(float(class_1_score["bayes_prob_spread"]), 0.7)
        self.assertGreater(float(class_1_score["bayes_monotonicity"]), 0.85)
        self.assertGreater(float(class_1_score["bayes_information_value"]), 0.5)

    def test_multiclass_categorical_bayes_targets(self) -> None:
        signal = pd.Series(np.linspace(-3.0, 3.0, 300))
        multi_target = pd.Series(
            np.where(signal < -1.0, "down", np.where(signal > 1.0, "up", "flat"))
        )

        score_rows, quantile_rows = _compute_feature_target_bayes_probabilities(
            signal, multi_target, "signal", "multi_target", n_bins=20, prior_strength=10.0
        )
        classes = {r["target_class"] for r in score_rows}
        self.assertEqual(classes, {"down", "flat", "up"})

        # Verify sum of Bayesian posterior probabilities across classes in each bin is 1.0
        qdf = pd.DataFrame(quantile_rows)
        bin_prob_sums = qdf.groupby("bin_index")["bayes_prob"].sum()
        for _, prob_sum in bin_prob_sums.items():
            self.assertAlmostEqual(prob_sum, 1.0, places=4)

    def test_probability_bayes_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=200)
            output_dir = tmp_path / "reports"

            module = ProbabilityBayesModule(n_bins=20, prior_strength=10.0)
            result = module.run(feature_csv, label_csv, output_dir)

            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "probability_bayes")

            # Check required artifacts exist
            expected_files = [
                "bayes_probability_scores.csv",
                "quantile_bayes_probabilities.csv",
                "bayes_probability_distribution.png",
                "summary.json",
                "report.md",
                "report.html",
            ]
            for filename in expected_files:
                file_path = result.report_dir / filename
                self.assertTrue(file_path.exists(), f"Missing artifact {filename}")
                self.assertGreater(file_path.stat().st_size, 0)

            # Check scores CSV contents
            scores = pd.read_csv(result.report_dir / "bayes_probability_scores.csv")
            self.assertIn("feature", scores.columns)
            self.assertIn("target", scores.columns)
            self.assertIn("target_class", scores.columns)
            self.assertIn("bayes_information_value", scores.columns)
            self.assertIn("bayes_prob_spread", scores.columns)
            self.assertIn("bayes_monotonicity", scores.columns)
            self.assertIn("mean_log_bayes_factor", scores.columns)
            self.assertIn("mean_ci_width", scores.columns)

            # Check quantiles CSV contents
            quantiles = pd.read_csv(result.report_dir / "quantile_bayes_probabilities.csv")
            self.assertIn("bin_index", quantiles.columns)
            self.assertIn("raw_prob", quantiles.columns)
            self.assertIn("bayes_prob", quantiles.columns)
            self.assertIn("ci_lower_95", quantiles.columns)
            self.assertIn("ci_upper_95", quantiles.columns)
            self.assertIn("log_bayes_factor", quantiles.columns)
            self.assertIn("bayes_woe", quantiles.columns)
            self.assertIn("bayes_entropy", quantiles.columns)

            # Check summary.json contents
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability_bayes")
            self.assertIn("feature_shape", summary)
            self.assertIn("top_features", summary)

    def test_cli_probability_bayes_run(self) -> None:
        from fldataprofiler.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=150)
            output_dir = tmp_path / "reports"

            exit_code = main(
                [
                    "fit",
                    str(feature_csv),
                    str(label_csv),
                    "--module",
                    "probability_bayes",
                    "--output-dir",
                    str(output_dir),
                    "--target",
                    "binary_target",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "probability_bayes" / "report.md").exists())
            self.assertTrue((output_dir / "probability_bayes" / "report.html").exists())
            self.assertTrue((output_dir / "probability_bayes" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
