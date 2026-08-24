from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_prim import (
    ProbabilityPrimModule,
    _compute_box_metrics,
    _extract_prim_rules_for_target,
    _generate_python_rule_code,
    _patient_peel_box,
)


def make_prim_test_datasets(
    base_dir: Path,
    rows: int = 500,
    seed: int = 42,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    # Features:
    # f1 and f2 contain a strong sweet spot bump when f1 in [1.0, 3.0] and f2 in [1.0, 3.0]
    f1 = rng.uniform(-4.0, 4.0, rows)
    f2 = rng.uniform(-4.0, 4.0, rows)
    f3 = rng.uniform(-2.0, 2.0, rows)  # Noise feature
    f4 = rng.standard_normal(rows)  # Noise feature

    # Target: bump in f1 in [1.0, 3.0] and f2 in [1.0, 3.0]
    in_bump = (f1 >= 1.0) & (f1 <= 3.0) & (f2 >= 1.0) & (f2 <= 3.0)
    # Win rate inside bump is ~85%, outside is ~15%
    prob = np.where(in_bump, 0.85, 0.15)
    binary_target = (rng.uniform(0, 1, rows) < prob).astype(int)

    # Multiclass target: "bull" if in bump, else "bear" or "neutral"
    multi_target = np.where(
        in_bump & (rng.uniform(0, 1, rows) < 0.8),
        "bull",
        np.where(f1 < -1.0, "bear", "neutral"),
    )

    feature_path = base_dir / "feature.csv"
    label_path = base_dir / "label.csv"

    pd.DataFrame(
        {
            "Date": dates,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f4": f4,
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


class ProbabilityPrimModuleTests(unittest.TestCase):
    def test_patient_peeling_finds_bump(self) -> None:
        rng = np.random.default_rng(123)
        rows = 400
        f1 = rng.uniform(-5.0, 5.0, rows)
        f2 = rng.uniform(-5.0, 5.0, rows)
        # Clear bump in [1.0, 3.0] x [1.0, 3.0]
        in_bump = (f1 >= 1.0) & (f1 <= 3.0) & (f2 >= 1.0) & (f2 <= 3.0)
        y = np.where(in_bump, 1, 0)

        df = pd.DataFrame({"f1": f1, "f2": f2})
        target = pd.Series(y)

        best_box = _patient_peel_box(
            df=df,
            features=["f1", "f2"],
            target_binary=target,
            alpha=0.05,
            min_box_samples=20,
        )

        self.assertIsNotNone(best_box)
        self.assertGreater(best_box["win_rate"], 0.70)
        self.assertGreaterEqual(best_box["sample_count"], 15)
        # Bounds should capture region around [1.0, 3.0]
        f1_min, f1_max = best_box["bounds"]["f1"]
        f2_min, f2_max = best_box["bounds"]["f2"]
        self.assertGreater(f1_max, f1_min)
        self.assertGreater(f2_max, f2_min)

    def test_compute_box_metrics(self) -> None:
        metrics = _compute_box_metrics(
            box_pos=40,
            box_total=50,
            total_pos=100,
            total_samples=500,
        )
        self.assertEqual(metrics["sample_count"], 50)
        self.assertAlmostEqual(metrics["support"], 0.10, places=4)
        self.assertEqual(metrics["target_positive_count"], 40)
        self.assertAlmostEqual(metrics["win_rate"], 0.80, places=4)
        self.assertAlmostEqual(metrics["baseline_rate"], 0.20, places=4)
        self.assertAlmostEqual(metrics["lift"], 4.0, places=4)
        self.assertLess(metrics["p_value_fisher"], 1e-5)
        self.assertGreater(metrics["credible_interval_low_95"], 0.60)
        self.assertLess(metrics["credible_interval_high_95"], 0.95)

    def test_generate_python_rule_code(self) -> None:
        rules = [
            {
                "rule_id": "rule_1",
                "dimension": "2D",
                "features": "f1, f2",
                "bounds_condition": "0.9500 <= f1 <= 3.1000 and 1.0500 <= f2 <= 2.9500",
                "python_condition": "0.9500 <= row.get('f1', float('nan')) <= 3.1000 and 1.0500 <= row.get('f2', float('nan')) <= 2.9500",
                "win_rate": 0.85,
                "lift": 4.25,
                "support": 0.08,
                "sample_count": 40,
                "target": "target",
                "target_class": 1,
            }
        ]
        code_str = _generate_python_rule_code(rules)
        # Verify valid Python AST
        ast.parse(code_str)

        # Execute the code in isolated namespace
        namespace: dict[str, object] = {}
        exec(code_str, namespace)
        self.assertIn("predict_prim_rules", namespace)
        self.assertIn("evaluate_prim_rules", namespace)

        predict_fn = namespace["predict_prim_rules"]
        # In-box sample should return 1
        in_sample = {"f1": 2.0, "f2": 2.0, "f3": 0.0}
        self.assertEqual(predict_fn(in_sample), 1)

        # Out-of-box sample should return 0
        out_sample = {"f1": -3.0, "f2": -3.0, "f3": 0.0}
        self.assertEqual(predict_fn(out_sample), 0)

        # Evaluate DataFrame helper
        eval_fn = namespace["evaluate_prim_rules"]
        test_df = pd.DataFrame([in_sample, out_sample])
        pred_series = eval_fn(test_df)
        self.assertEqual(list(pred_series), [1, 0])

    def test_probability_prim_module_run_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_prim_test_datasets(tmp_path, rows=300)
            output_dir = tmp_path / "reports"

            module = ProbabilityPrimModule(max_candidates=4, min_box_samples=15)
            result = module.run(feature_csv, label_csv, output_dir)

            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "probability_prim")

            expected_files = [
                "prim_rules.csv",
                "rule_code_python.py",
                "prim_rules_plot.png",
                "summary.json",
                "report.md",
                "report.html",
            ]
            for filename in expected_files:
                file_path = result.report_dir / filename
                self.assertTrue(file_path.exists(), f"Missing artifact {filename}")
                self.assertGreater(file_path.stat().st_size, 0)

            # Check prim_rules.csv
            rules_df = pd.read_csv(result.report_dir / "prim_rules.csv")
            self.assertIn("rule_id", rules_df.columns)
            self.assertIn("dimension", rules_df.columns)
            self.assertIn("features", rules_df.columns)
            self.assertIn("bounds_condition", rules_df.columns)
            self.assertIn("sample_count", rules_df.columns)
            self.assertIn("support", rules_df.columns)
            self.assertIn("win_rate", rules_df.columns)
            self.assertIn("baseline_rate", rules_df.columns)
            self.assertIn("lift", rules_df.columns)
            self.assertIn("p_value_fisher", rules_df.columns)
            self.assertIn("credible_interval_low_95", rules_df.columns)
            self.assertIn("credible_interval_high_95", rules_df.columns)

            # Check summary.json
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability_prim")
            self.assertIn("top_rules", summary)
            self.assertIn("summary_metrics", summary)
            self.assertGreater(summary["summary_metrics"]["rules_discovered"], 0)

    def test_custom_target_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_prim_test_datasets(tmp_path, rows=200)
            output_dir = tmp_path / "reports"

            module = ProbabilityPrimModule(max_candidates=4, min_box_samples=15)
            result = module.run(
                feature_csv,
                label_csv,
                output_dir,
                targets=["binary_target"],
            )
            rules_df = pd.read_csv(result.report_dir / "prim_rules.csv")
            self.assertTrue((rules_df["target"] == "binary_target").all())

    def test_cli_probability_prim_run(self) -> None:
        from fldataprofiler.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_prim_test_datasets(tmp_path, rows=150)
            output_dir = tmp_path / "reports"

            # Test primary name
            exit_code = main([
                "fit",
                str(feature_csv),
                str(label_csv),
                "--module", "probability_prim",
                "--output-dir", str(output_dir),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "probability_prim" / "report.md").exists())
            self.assertTrue((output_dir / "probability_prim" / "rule_code_python.py").exists())
            self.assertTrue((output_dir / "probability_prim" / "summary.json").exists())

            # Test prim alias
            output_dir_alias = tmp_path / "reports_alias"
            exit_code_alias = main([
                "fit",
                str(feature_csv),
                str(label_csv),
                "--module", "prim",
                "--output-dir", str(output_dir_alias),
                "--target", "binary_target",
            ])
            self.assertEqual(exit_code_alias, 0)
            self.assertTrue((output_dir_alias / "probability_prim" / "report.md").exists())

    def test_probability_prim_config_and_env_overrides(self) -> None:
        from fldataprofiler.modules.probability_prim import ProbabilityPrimConfig

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            feature_csv, label_csv = make_prim_test_datasets(tmp_path, rows=200)
            output_dir = tmp_path / "reports"

            cfg = ProbabilityPrimConfig(
                min_box_samples=20,
                min_support=0.01,
                objective="support_weighted",
                alpha=0.05,
                max_candidates=4,
            )
            module = ProbabilityPrimModule(config=cfg)
            result = module.run(
                feature_csv,
                label_csv,
                output_dir,
                targets=["binary_target"],
            )
            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["objective"], "support_weighted")
            self.assertEqual(summary["min_support"], 0.01)


if __name__ == "__main__":
    unittest.main()
