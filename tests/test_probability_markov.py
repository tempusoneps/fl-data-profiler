from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_markov import (
    ProbabilityMarkovModule,
    _compute_markov_transitions_for_feature,
    _compute_quantile_bins,
)


def make_markov_test_datasets(
    base_dir: Path,
    rows: int = 500,
    seed: int = 42,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    # Features:
    # f1 exhibits high alpha when transitioning from low state (Q1) to high state (Q4/Q5)
    f1 = rng.standard_normal(rows)
    f2 = rng.uniform(-3, 3, rows)  # Noise feature

    # Calculate states for f1
    ranks = pd.Series(f1).rank(method="first")
    states = pd.qcut(ranks, q=5, labels=False) + 1

    y = np.zeros(rows, dtype=int)
    for i in range(1, rows):
        prev_s = states.iloc[i - 1]
        curr_s = states.iloc[i]
        # Strong trigger: Transition from Q1 to Q3/Q4/Q5 produces 85% win rate
        if prev_s == 1 and curr_s >= 3:
            y[i] = int(rng.uniform(0, 1) < 0.85)
        else:
            y[i] = int(rng.uniform(0, 1) < 0.20)

    # Multiclass target
    multi_target = np.where(y == 1, "long_signal", "flat")

    feature_path = base_dir / "feature.parquet"
    label_path = base_dir / "label.csv"

    pd.DataFrame(
        {
            "Date": dates,
            "f1_momentum": f1,
            "f2_noise": f2,
        }
    ).to_parquet(feature_path, index=False)

    pd.DataFrame(
        {
            "Date": dates,
            "binary_target": y,
            "multi_target": multi_target,
        }
    ).to_csv(label_path, index=False)

    return feature_path, label_path


class ProbabilityMarkovModuleTests(unittest.TestCase):
    def test_quantile_bins(self) -> None:
        s = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        bins = _compute_quantile_bins(s, n_bins=5)
        self.assertEqual(len(bins), 10)
        self.assertEqual(bins.min(), 1)
        self.assertEqual(bins.max(), 5)

    def test_compute_markov_transitions(self) -> None:
        rng = np.random.default_rng(42)
        n = 300
        f = pd.Series(rng.standard_normal(n))
        ranks = f.rank(method="first")
        states = pd.qcut(ranks, q=5, labels=False) + 1

        y = pd.Series(np.zeros(n, dtype=int))
        for i in range(1, n):
            if states.iloc[i - 1] == 1 and states.iloc[i] >= 3:
                y.iloc[i] = 1

        transitions, meta_f = _compute_markov_transitions_for_feature(
            feature_series=f,
            target_series=y,
            feature_name="f_test",
            target_name="y_test",
            n_bins=5,
            min_samples=5,
        )

        self.assertGreater(len(transitions), 0)
        self.assertIn("mean_transition_entropy", meta_f)

        # Check structure of transitions
        df = pd.DataFrame(transitions)
        self.assertIn("transition_label", df.columns)
        self.assertIn("excess_probability", df.columns)
        self.assertIn("win_rate", df.columns)
        self.assertIn("lift", df.columns)
        self.assertIn("p_value_fisher", df.columns)

    def test_probability_markov_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_path, label_path = make_markov_test_datasets(tmp_path, rows=300)
            output_dir = tmp_path / "reports"

            module = ProbabilityMarkovModule(n_bins=5, min_pattern_samples=10)
            result = module.run(feature_path, label_path, output_dir)

            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "probability_markov")

            expected_files = [
                "markov_transitions.csv",
                "top_sequential_patterns.csv",
                "markov_heatmap.png",
                "summary.json",
                "report.md",
                "report.html",
            ]
            for filename in expected_files:
                file_path = result.report_dir / filename
                self.assertTrue(file_path.exists(), f"Missing artifact {filename}")
                self.assertGreater(file_path.stat().st_size, 0)

            # Check transitions CSV
            trans_df = pd.read_csv(result.report_dir / "markov_transitions.csv")
            self.assertIn("feature", trans_df.columns)
            self.assertIn("transition_label", trans_df.columns)
            self.assertIn("win_rate", trans_df.columns)
            self.assertIn("excess_probability", trans_df.columns)

            # Check summary.json
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability_markov")
            self.assertIn("summary_metrics", summary)
            self.assertGreater(summary["summary_metrics"]["total_transitions_evaluated"], 0)

    def test_cli_probability_markov_run(self) -> None:
        from fldataprofiler.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_path, label_path = make_markov_test_datasets(tmp_path, rows=150)
            output_dir = tmp_path / "reports"

            exit_code = main([
                "fit",
                str(feature_path),
                str(label_path),
                "--module", "probability_markov",
                "--output-dir", str(output_dir),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "probability_markov" / "report.md").exists())
            self.assertTrue((output_dir / "probability_markov" / "markov_transitions.csv").exists())

            # Test alias
            output_dir_alias = tmp_path / "reports_alias"
            exit_code_alias = main([
                "fit",
                str(feature_path),
                str(label_path),
                "--module", "markov",
                "--output-dir", str(output_dir_alias),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code_alias, 0)
            self.assertTrue((output_dir_alias / "probability_markov" / "report.html").exists())


if __name__ == "__main__":
    unittest.main()
