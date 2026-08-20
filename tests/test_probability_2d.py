from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_2d import (
    Probability2DModule,
    _compute_1d_iv_and_spread,
    _compute_pair_target_probabilities,
    _compute_quantile_bins,
    _prescreen_candidate_features,
)


def make_test_datasets(
    base_dir: Path,
    rows: int = 500,
    seed: int = 42,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    # Feature 1 & 2: Synergistic interaction (XOR-like region)
    x1 = rng.uniform(-2.0, 2.0, rows)
    x2 = rng.uniform(-2.0, 2.0, rows)
    # Feature 3: Linear monotonic signal
    signal = np.linspace(-3.0, 3.0, rows)
    # Feature 4: Noise
    noise = rng.standard_normal(rows)

    # Binary label: 1 if (x1 > 0 and x2 > 0) or (x1 < 0 and x2 < 0) else 0 (XOR structure)
    xor_signal = ((x1 > 0) & (x2 > 0)) | ((x1 < 0) & (x2 < 0))
    binary_target = (xor_signal | (rng.uniform(0, 1, rows) < 0.05)).astype(int)

    # Multiclass label: "bull" if x1 > 0.5 and x2 > 0.5, "bear" if x1 < -0.5 and x2 < -0.5, else "neutral"
    multi_target = np.where(
        (x1 > 0.5) & (x2 > 0.5),
        "bull",
        np.where((x1 < -0.5) & (x2 < -0.5), "bear", "neutral"),
    )

    feature_path = base_dir / "feature.csv"
    label_path = base_dir / "label.csv"

    pd.DataFrame(
        {
            "Date": dates,
            "x1": x1,
            "x2": x2,
            "signal": signal,
            "noise": noise,
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


class Probability2DModuleTests(unittest.TestCase):
    def test_2d_joint_quantile_binning(self) -> None:
        s1 = pd.Series(np.linspace(0, 100, 100))
        s2 = pd.Series(np.linspace(0, 50, 100))

        bins1 = _compute_quantile_bins(s1, n_bins=10)
        bins2 = _compute_quantile_bins(s2, n_bins=10)

        self.assertEqual(len(bins1), 100)
        self.assertEqual(len(bins2), 100)
        self.assertEqual(bins1.min(), 1)
        self.assertEqual(bins1.max(), 10)
        self.assertEqual(bins2.min(), 1)
        self.assertEqual(bins2.max(), 10)

    def test_2d_conditional_probability_and_multiclass_sums(self) -> None:
        rows = 300
        rng = np.random.default_rng(42)
        x1 = pd.Series(rng.uniform(-2, 2, rows))
        x2 = pd.Series(rng.uniform(-2, 2, rows))
        multi_target = pd.Series(
            np.where(
                (x1 > 0.5) & (x2 > 0.5),
                "bull",
                np.where((x1 < -0.5) & (x2 < -0.5), "bear", "neutral"),
            )
        )

        f1_stats = _compute_1d_iv_and_spread(x1, multi_target, n_bins=10)
        f2_stats = _compute_1d_iv_and_spread(x2, multi_target, n_bins=10)

        pair_scores, cell_rows = _compute_pair_target_probabilities(
            x1,
            x2,
            multi_target,
            "x1",
            "x2",
            "multi_target",
            n_bins=10,
            f1_1d_stats=f1_stats,
            f2_1d_stats=f2_stats,
        )

        self.assertTrue(len(pair_scores) > 0)
        self.assertTrue(len(cell_rows) > 0)

        # Verify all 3 classes are present
        classes = {r["target_class"] for r in pair_scores}
        self.assertEqual(classes, {"bull", "bear", "neutral"})

        # Check cell probabilities sum to 1.0 for each (bin_x, bin_y) cell
        cell_df = pd.DataFrame(cell_rows)
        cell_prob_sums = cell_df.groupby(["bin_x", "bin_y"])["conditional_prob"].sum()
        for _, p_sum in cell_prob_sums.items():
            self.assertAlmostEqual(p_sum, 1.0, places=4)

    def test_synergy_gain_on_xor_interaction(self) -> None:
        # Generate clean XOR dataset where 1D features have near 0 spread, but 2D interaction is strong
        rows = 600
        rng = np.random.default_rng(123)
        x1 = rng.uniform(-1, 1, rows)
        x2 = rng.uniform(-1, 1, rows)
        # XOR label
        y = ((x1 > 0) == (x2 > 0)).astype(int)

        s1 = pd.Series(x1)
        s2 = pd.Series(x2)
        sy = pd.Series(y)

        f1_stats = _compute_1d_iv_and_spread(s1, sy, n_bins=10)
        f2_stats = _compute_1d_iv_and_spread(s2, sy, n_bins=10)

        pair_scores, _ = _compute_pair_target_probabilities(
            s1,
            s2,
            sy,
            "x1",
            "x2",
            "y",
            n_bins=10,
            f1_1d_stats=f1_stats,
            f2_1d_stats=f2_stats,
        )

        class_1_score = next(r for r in pair_scores if r["target_class"] == 1)
        # 1D spreads should be relatively low for pure XOR
        self.assertLess(class_1_score["prob_spread_f1"], 0.35)
        self.assertLess(class_1_score["prob_spread_f2"], 0.35)
        # 2D spread should be very high (~0.9 - 1.0)
        self.assertGreater(class_1_score["prob_spread_2d"], 0.70)
        # Synergy gain must be strictly positive
        self.assertGreater(class_1_score["synergy_gain"], 0.35)
        self.assertGreater(class_1_score["iv_2d"], class_1_score["iv_f1"])

    def test_sweet_spot_rule_extraction_with_min_support(self) -> None:
        rows = 400
        rng = np.random.default_rng(99)
        x1 = rng.uniform(0, 10, rows)
        x2 = rng.uniform(0, 10, rows)
        # Target concentrated in high x1 and high x2
        y = ((x1 > 7.0) & (x2 > 7.0)).astype(int)

        s1 = pd.Series(x1)
        s2 = pd.Series(x2)
        sy = pd.Series(y)

        f1_stats = _compute_1d_iv_and_spread(s1, sy, n_bins=10)
        f2_stats = _compute_1d_iv_and_spread(s2, sy, n_bins=10)

        pair_scores, _ = _compute_pair_target_probabilities(
            s1,
            s2,
            sy,
            "x1",
            "x2",
            "target",
            n_bins=10,
            f1_1d_stats=f1_stats,
            f2_1d_stats=f2_stats,
            min_support=10,
        )

        class_1_score = next(r for r in pair_scores if r["target_class"] == 1)
        self.assertGreater(class_1_score["sweet_spot_prob"], 0.8)
        self.assertGreater(class_1_score["sweet_spot_lift"], 2.0)
        self.assertGreaterEqual(class_1_score["sweet_spot_samples"], 5)
        self.assertIn("<=", class_1_score["sweet_spot_rule"])
        self.assertIn("AND", class_1_score["sweet_spot_rule"])
        self.assertGreaterEqual(class_1_score["sweet_spot_bin_x"], 8)
        self.assertGreaterEqual(class_1_score["sweet_spot_bin_y"], 8)

    def test_prescreen_candidate_features(self) -> None:
        rows = 200
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "strong": np.linspace(0, 10, rows),
                "weak": rng.standard_normal(rows),
                "target": (np.linspace(0, 10, rows) > 5).astype(int),
            }
        )

        candidates, stats = _prescreen_candidate_features(
            df,
            numeric_features=["strong", "weak"],
            valid_targets=["target"],
            max_candidates=1,
            n_bins=10,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0], "strong")
        self.assertIn("strong", stats)

    def test_missing_values_and_constant_features(self) -> None:
        # Missing values handling
        s1 = pd.Series([np.nan, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        s2 = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0])
        target = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

        pair_scores, cell_rows = _compute_pair_target_probabilities(
            s1, s2, target, "s1", "s2", "target", n_bins=5, f1_1d_stats={}, f2_1d_stats={}
        )
        self.assertTrue(len(pair_scores) > 0)
        self.assertTrue(len(cell_rows) > 0)

        # Constant feature should return empty
        constant_s = pd.Series([1.0] * 50)
        normal_s = pd.Series(np.linspace(0, 10, 50))
        t = pd.Series([0, 1] * 25)
        c_scores, c_cells = _compute_pair_target_probabilities(
            constant_s, normal_s, t, "c", "n", "t", n_bins=5, f1_1d_stats={}, f2_1d_stats={}
        )
        self.assertEqual(len(c_scores), 0)
        self.assertEqual(len(c_cells), 0)

    def test_probability_2d_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=250)
            output_dir = tmp_path / "reports"

            module = Probability2DModule(n_bins=10, max_candidates=4)
            result = module.run(feature_csv, label_csv, output_dir)

            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "probability_2d")

            expected_files = [
                "pair_probability_scores.csv",
                "cell_conditional_probabilities.csv",
                "probability_2d_heatmaps.png",
                "summary.json",
                "report.md",
                "report.html",
            ]
            for filename in expected_files:
                file_path = result.report_dir / filename
                self.assertTrue(file_path.exists(), f"Missing artifact {filename}")
                self.assertGreater(file_path.stat().st_size, 0)

            # Check pair scores CSV contents
            pair_scores = pd.read_csv(result.report_dir / "pair_probability_scores.csv")
            self.assertIn("feature_x", pair_scores.columns)
            self.assertIn("feature_y", pair_scores.columns)
            self.assertIn("target", pair_scores.columns)
            self.assertIn("target_class", pair_scores.columns)
            self.assertIn("iv_2d", pair_scores.columns)
            self.assertIn("synergy_gain", pair_scores.columns)
            self.assertIn("sweet_spot_rule", pair_scores.columns)
            self.assertIn("sweet_spot_prob", pair_scores.columns)
            self.assertIn("sweet_spot_lift", pair_scores.columns)

            # Check cell probabilities CSV contents
            cell_probs = pd.read_csv(result.report_dir / "cell_conditional_probabilities.csv")
            self.assertIn("bin_x", cell_probs.columns)
            self.assertIn("bin_y", cell_probs.columns)
            self.assertIn("conditional_prob", cell_probs.columns)
            self.assertIn("lift", cell_probs.columns)
            self.assertIn("woe", cell_probs.columns)
            self.assertIn("iv_contribution", cell_probs.columns)
            self.assertIn("entropy", cell_probs.columns)

            # Check summary.json contents
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability_2d")
            self.assertIn("feature_shape", summary)
            self.assertIn("top_pairs", summary)
            self.assertIn("top_sweet_spots", summary)
            self.assertIn("summary_metrics", summary)

    def test_custom_target_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=200)
            output_dir = tmp_path / "reports"

            module = Probability2DModule(n_bins=10, max_candidates=4)
            result = module.run(
                feature_csv,
                label_csv,
                output_dir,
                targets=["binary_target"],
            )
            pair_scores = pd.read_csv(result.report_dir / "pair_probability_scores.csv")
            self.assertTrue((pair_scores["target"] == "binary_target").all())

    def test_cli_probability_2d_run(self) -> None:
        from fldataprofiler.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_datasets(tmp_path, rows=150)
            output_dir = tmp_path / "reports"

            # Test primary name
            exit_code = main([
                "fit",
                str(feature_csv),
                str(label_csv),
                "--module", "probability_2d",
                "--output-dir", str(output_dir),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "probability_2d" / "report.md").exists())
            self.assertTrue((output_dir / "probability_2d" / "report.html").exists())
            self.assertTrue((output_dir / "probability_2d" / "summary.json").exists())

            # Test probability2d alias
            output_dir_alias = tmp_path / "reports_alias"
            exit_code_alias = main([
                "fit",
                str(feature_csv),
                str(label_csv),
                "--module", "probability2d",
                "--output-dir", str(output_dir_alias),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code_alias, 0)
            self.assertTrue((output_dir_alias / "probability_2d" / "report.md").exists())


if __name__ == "__main__":
    unittest.main()
