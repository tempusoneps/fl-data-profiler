from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_drift import (
    ProbabilityDriftModule,
    _compute_psi,
    _evaluate_feature_drift,
)


def make_test_drift_datasets(
    base_dir: Path,
    rows: int = 500,
    seed: int = 42,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    # Feature 1: Stable signal across all folds
    stable_signal = np.linspace(-3.0, 3.0, rows)

    # Feature 2: Drifting feature (distribution shifts from left to right over time)
    drifting_feat = np.concatenate([
        rng.normal(loc=-2.0, scale=0.5, size=rows // 2),
        rng.normal(loc=2.0, scale=0.5, size=rows - rows // 2),
    ])

    # Feature 3: Regime-flipping feature (correlation flips from + to - midway)
    flipping_feat = np.linspace(-3.0, 3.0, rows)

    # Binary label: correlates with stable_signal throughout
    # But for flipping_feat, early half is correlated (+), late half is anti-correlated (-)
    target_early = (flipping_feat[:rows // 2] > 0).astype(int)
    target_late = (flipping_feat[rows // 2:] < 0).astype(int)
    binary_target = np.concatenate([target_early, target_late])

    # Multiclass label: 3 classes
    multi_target = np.where(stable_signal < -1.0, "down", np.where(stable_signal > 1.0, "up", "flat"))

    feature_path = base_dir / "feature.csv"
    label_path = base_dir / "label.csv"

    pd.DataFrame(
        {
            "Date": dates,
            "stable_signal": stable_signal,
            "drifting_feat": drifting_feat,
            "flipping_feat": flipping_feat,
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


class ProbabilityDriftModuleTests(unittest.TestCase):
    def test_psi_calculation(self) -> None:
        # Identical distributions -> PSI = 0
        dist1 = np.array([0.1, 0.2, 0.3, 0.4])
        dist2 = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(_compute_psi(dist1, dist2), 0.0, places=5)

        # Shifted distribution -> PSI > 0
        shifted_dist = np.array([0.4, 0.3, 0.2, 0.1])
        psi_val = _compute_psi(shifted_dist, dist1)
        self.assertGreater(psi_val, 0.2)

    def test_evaluate_feature_drift_stability_and_flips(self) -> None:
        rows = 400
        t = np.linspace(0, 8 * np.pi, rows)
        stationary_signal = pd.Series(np.sin(t) + np.random.default_rng(42).normal(0, 0.1, rows))
        target = pd.Series((stationary_signal > 0).astype(int))

        summary, fold_metrics, quantiles = _evaluate_feature_drift(
            stationary_signal, target, "stationary_signal", "target", n_bins=10, n_folds=4
        )
        self.assertTrue(len(summary) > 0)
        self.assertTrue(len(fold_metrics) > 0)
        self.assertTrue(len(quantiles) > 0)

        stable_item = next(s for s in summary if s["target_class"] == 1)
        self.assertEqual(stable_item["monotonicity_flips"], 0)
        self.assertGreater(float(stable_item["stability_ratio"]), 1.0)
        self.assertIn(stable_item["drift_status"], ["STABLE", "MODERATE_DRIFT"])

    def test_probability_drift_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_drift_datasets(tmp_path, rows=300)
            output_dir = tmp_path / "reports"

            module = ProbabilityDriftModule(n_bins=10, n_folds=3)
            result = module.run(feature_csv, label_csv, output_dir)

            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "probability_drift")

            expected_files = [
                "feature_drift_scores.csv",
                "fold_probability_metrics.csv",
                "quantile_drift_probabilities.csv",
                "probability_drift_charts.png",
                "summary.json",
                "report.md",
                "report.html",
            ]
            for filename in expected_files:
                file_path = result.report_dir / filename
                self.assertTrue(file_path.exists(), f"Missing artifact {filename}")
                self.assertGreater(file_path.stat().st_size, 0)

            # Check scores CSV contents
            scores = pd.read_csv(result.report_dir / "feature_drift_scores.csv")
            self.assertIn("feature", scores.columns)
            self.assertIn("target", scores.columns)
            self.assertIn("target_class", scores.columns)
            self.assertIn("drift_status", scores.columns)
            self.assertIn("mean_iv", scores.columns)
            self.assertIn("stability_ratio", scores.columns)
            self.assertIn("max_psi", scores.columns)
            self.assertIn("monotonicity_flips", scores.columns)

            # Check fold metrics CSV contents
            fold_metrics = pd.read_csv(result.report_dir / "fold_probability_metrics.csv")
            self.assertIn("fold", fold_metrics.columns)
            self.assertIn("psi", fold_metrics.columns)
            self.assertIn("iv", fold_metrics.columns)
            self.assertIn("monotonicity", fold_metrics.columns)

            # Check summary.json
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability_drift")
            self.assertIn("top_stable_features", summary)
            self.assertIn("summary_metrics", summary)

    def test_cli_probability_drift_run(self) -> None:
        from fldataprofiler.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_test_drift_datasets(tmp_path, rows=200)
            output_dir = tmp_path / "reports"

            exit_code = main([
                "fit",
                str(feature_csv),
                str(label_csv),
                "--module", "probability_drift",
                "--output-dir", str(output_dir),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "probability_drift" / "report.md").exists())
            self.assertTrue((output_dir / "probability_drift" / "report.html").exists())
            self.assertTrue((output_dir / "probability_drift" / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
