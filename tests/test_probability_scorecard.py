from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_scorecard import (
    ProbabilityScorecardModule,
    _build_scorecard,
    _compute_woe_and_iv,
)


def make_scorecard_test_datasets(
    base_dir: Path,
    rows: int = 500,
    seed: int = 42,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    # Features:
    # f1 and f2 are strongly predictive of positive probability
    f1 = rng.standard_normal(rows)
    f2 = rng.standard_normal(rows)
    f3 = rng.uniform(-3, 3, rows)  # Noise feature

    logits = 1.2 * f1 - 0.9 * f2 + rng.standard_normal(rows) * 0.4
    probs = 1.0 / (1.0 + np.exp(-logits))
    binary_target = (rng.uniform(0, 1, rows) < probs).astype(int)

    feature_path = base_dir / "feature.parquet"
    label_path = base_dir / "label.csv"

    pd.DataFrame(
        {
            "Date": dates,
            "f1_alpha": f1,
            "f2_alpha": f2,
            "f3_noise": f3,
        }
    ).to_parquet(feature_path, index=False)

    pd.DataFrame(
        {
            "Date": dates,
            "binary_target": binary_target,
        }
    ).to_csv(label_path, index=False)

    return feature_path, label_path


class ProbabilityScorecardModuleTests(unittest.TestCase):
    def test_compute_woe_and_iv(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        x = pd.Series(rng.standard_normal(n))
        y = pd.Series((x > 0).astype(int))

        woe_map, iv, bin_details, woe_series = _compute_woe_and_iv(x, y, n_bins=5)
        self.assertGreater(len(woe_map), 0)
        self.assertGreater(iv, 0.5)
        self.assertEqual(len(woe_series), n)

    def test_build_scorecard(self) -> None:
        rng = np.random.default_rng(42)
        n = 400
        f1 = rng.standard_normal(n)
        f2 = rng.standard_normal(n)
        logits = 1.5 * f1 - 1.0 * f2
        probs = 1 / (1 + np.exp(-logits))
        y = (rng.uniform(0, 1, n) < probs).astype(int)

        df = pd.DataFrame({"f1": f1, "f2": f2})
        points_df, calibration_df, metrics, score_series, prob_series = _build_scorecard(
            model_frame=df,
            feature_columns=["f1", "f2"],
            target_series=pd.Series(y),
            base_score=600,
            pdo=20,
            n_bins=5,
        )

        self.assertGreater(metrics["auc"], 0.70)
        self.assertGreater(metrics["ks_statistic"], 0.30)
        self.assertIn("f1", points_df["feature"].values)
        self.assertIn("f2", points_df["feature"].values)
        self.assertEqual(len(score_series), n)

    def test_probability_scorecard_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_path, label_path = make_scorecard_test_datasets(tmp_path, rows=300)
            output_dir = tmp_path / "reports"

            module = ProbabilityScorecardModule(n_bins=5, max_features=3)
            result = module.run(feature_path, label_path, output_dir)

            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "probability_scorecard")

            expected_files = [
                "scorecard_points.csv",
                "score_to_probability.csv",
                "score_distribution_plot.png",
                "summary.json",
                "report.md",
                "report.html",
            ]
            for filename in expected_files:
                file_path = result.report_dir / filename
                self.assertTrue(file_path.exists(), f"Missing artifact {filename}")
                self.assertGreater(file_path.stat().st_size, 0)

            # Check points CSV
            pts_df = pd.read_csv(result.report_dir / "scorecard_points.csv")
            self.assertIn("feature", pts_df.columns)
            self.assertIn("bin", pts_df.columns)
            self.assertIn("points", pts_df.columns)
            self.assertIn("woe", pts_df.columns)

            # Check summary.json
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability_scorecard")
            self.assertIn("summary_metrics", summary)
            self.assertGreater(summary["summary_metrics"]["roc_auc"], 0.60)

    def test_cli_probability_scorecard_run(self) -> None:
        from fldataprofiler.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_path, label_path = make_scorecard_test_datasets(tmp_path, rows=150)
            output_dir = tmp_path / "reports"

            exit_code = main([
                "fit",
                str(feature_path),
                str(label_path),
                "--module", "probability_scorecard",
                "--output-dir", str(output_dir),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "probability_scorecard" / "report.md").exists())
            self.assertTrue((output_dir / "probability_scorecard" / "scorecard_points.csv").exists())

            # Test alias
            output_dir_alias = tmp_path / "reports_alias"
            exit_code_alias = main([
                "fit",
                str(feature_path),
                str(label_path),
                "--module", "scorecard",
                "--output-dir", str(output_dir_alias),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code_alias, 0)
            self.assertTrue((output_dir_alias / "probability_scorecard" / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
