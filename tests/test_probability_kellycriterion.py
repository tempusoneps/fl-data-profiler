from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_kellycriterion import (
    ProbabilityKellyCriterionModule,
    _compute_feature_target_kelly_profiles,
    _compute_kelly_metrics,
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


class ProbabilityKellyCriterionModuleTests(unittest.TestCase):
    def test_kelly_metrics_calculation(self) -> None:
        # Win rate 60%, Payoff 1.5:1
        # EV = 0.6 * 1.5 - 0.4 * 1.0 = 0.9 - 0.4 = +0.5R
        # f* = 0.5 / 1.5 = 1/3 ~ 0.3333
        raw_kelly, half_kelly, quarter_kelly, ev, growth, rec = _compute_kelly_metrics(
            p_win=0.60, payoff_b=1.5
        )
        self.assertAlmostEqual(ev, 0.50, places=4)
        self.assertAlmostEqual(raw_kelly, 0.333333, places=4)
        self.assertAlmostEqual(half_kelly, 0.166667, places=4)
        self.assertAlmostEqual(quarter_kelly, 0.083333, places=4)
        self.assertGreater(growth, 0.0)
        self.assertEqual(rec, "STRONG_BET")

        # Win rate 30%, Payoff 1.5:1
        # Breakeven win rate = 1 / (1 + 1.5) = 40%
        # EV = 0.3 * 1.5 - 0.7 * 1.0 = 0.45 - 0.7 = -0.25R
        # f* = -0.25 / 1.5 < 0
        raw_kelly_neg, half_k_neg, quarter_k_neg, ev_neg, growth_neg, rec_neg = (
            _compute_kelly_metrics(p_win=0.30, payoff_b=1.5)
        )
        self.assertLess(raw_kelly_neg, 0.0)
        self.assertEqual(half_k_neg, 0.0)
        self.assertEqual(quarter_k_neg, 0.0)
        self.assertEqual(growth_neg, 0.0)
        self.assertEqual(rec_neg, "AVOID_NO_BET")

    def test_feature_target_kelly_profiles(self) -> None:
        signal = pd.Series(np.linspace(0, 100, 1000))
        target = pd.Series(np.where(signal > 50, 1, 0))

        score_rows, quantile_rows = _compute_feature_target_kelly_profiles(
            signal, target, "signal", "target", n_bins=20, payoff_ratio_b=1.5
        )
        self.assertTrue(len(score_rows) > 0)
        self.assertTrue(len(quantile_rows) > 0)

        class_1_score = next(r for r in score_rows if r["target_class"] == 1)
        self.assertGreater(float(class_1_score["max_kelly_fraction"]), 0.8)
        self.assertGreater(float(class_1_score["max_expected_value"]), 1.0)
        self.assertGreater(float(class_1_score["kelly_monotonicity"]), 0.85)
        self.assertGreater(int(class_1_score["favorable_bins_count"]), 0)

    def test_multiclass_categorical_kelly_targets(self) -> None:
        signal = pd.Series(np.linspace(-3.0, 3.0, 300))
        multi_target = pd.Series(
            np.where(signal < -1.0, "down", np.where(signal > 1.0, "up", "flat"))
        )

        score_rows, quantile_rows = _compute_feature_target_kelly_profiles(
            signal, multi_target, "signal", "multi_target", n_bins=20, payoff_ratio_b=2.0
        )
        classes = {r["target_class"] for r in score_rows}
        self.assertEqual(classes, {"down", "flat", "up"})
        self.assertEqual(len(quantile_rows), 60)

    def test_probability_kellycriterion_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=200)
            output_dir = tmp_path / "reports"

            module = ProbabilityKellyCriterionModule(n_bins=20, payoff_ratio_b=1.5)
            result = module.run(feature_csv, label_csv, output_dir)

            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "probability_kellycriterion")

            # Check required artifacts exist
            expected_files = [
                "kelly_probability_scores.csv",
                "quantile_kelly_probabilities.csv",
                "kelly_distribution.png",
                "summary.json",
                "report.md",
                "report.html",
            ]
            for filename in expected_files:
                file_path = result.report_dir / filename
                self.assertTrue(file_path.exists(), f"Missing artifact {filename}")
                self.assertGreater(file_path.stat().st_size, 0)

            # Check scores CSV contents
            scores = pd.read_csv(result.report_dir / "kelly_probability_scores.csv")
            self.assertIn("feature", scores.columns)
            self.assertIn("target", scores.columns)
            self.assertIn("target_class", scores.columns)
            self.assertIn("max_kelly_fraction", scores.columns)
            self.assertIn("max_half_kelly", scores.columns)
            self.assertIn("max_expected_value", scores.columns)
            self.assertIn("favorable_bins_count", scores.columns)

            # Check quantiles CSV contents
            quantiles = pd.read_csv(result.report_dir / "quantile_kelly_probabilities.csv")
            self.assertIn("bin_index", quantiles.columns)
            self.assertIn("win_prob", quantiles.columns)
            self.assertIn("kelly_fraction_f", quantiles.columns)
            self.assertIn("half_kelly", quantiles.columns)
            self.assertIn("expected_value_ev", quantiles.columns)
            self.assertIn("action_recommendation", quantiles.columns)

            # Check summary.json contents
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability_kellycriterion")
            self.assertIn("feature_shape", summary)
            self.assertIn("top_features", summary)

    def test_cli_probability_kellycriterion_run(self) -> None:
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
                    "probability_kellycriterion",
                    "--output-dir",
                    str(output_dir),
                    "--target",
                    "binary_target",
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "probability_kellycriterion" / "report.md").exists())
            self.assertTrue((output_dir / "probability_kellycriterion" / "report.html").exists())
            self.assertTrue((output_dir / "probability_kellycriterion" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
