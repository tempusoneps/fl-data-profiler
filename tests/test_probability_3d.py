from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofiler.modules.probability_3d import (
    Probability3DModule,
    _compute_1d_iv_and_spread,
    _compute_quantile_bins,
    _compute_triplet_target_probabilities,
    _prescreen_candidate_features,
)


def make_test_datasets(
    base_dir: Path,
    rows: int = 500,
    seed: int = 42,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    x1 = rng.uniform(-2.0, 2.0, rows)
    x2 = rng.uniform(-2.0, 2.0, rows)
    x3 = rng.uniform(-2.0, 2.0, rows)
    noise = rng.standard_normal(rows)

    # 3D synergistic target: 1 if all 3 positive or all 3 negative
    synergy_mask = ((x1 > 0) & (x2 > 0) & (x3 > 0)) | ((x1 < 0) & (x2 < 0) & (x3 < 0))
    binary_target = (synergy_mask | (rng.uniform(0, 1, rows) < 0.05)).astype(int)

    multi_target = np.where(
        (x1 > 0.5) & (x2 > 0.5) & (x3 > 0.5),
        "bull",
        np.where((x1 < -0.5) & (x2 < -0.5) & (x3 < -0.5), "bear", "neutral"),
    )

    feature_path = base_dir / "feature.csv"
    label_path = base_dir / "label.csv"

    pd.DataFrame(
        {
            "Date": dates,
            "x1": x1,
            "x2": x2,
            "x3": x3,
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


class Probability3DModuleTests(unittest.TestCase):
    def test_3d_quantile_binning(self) -> None:
        s = pd.Series(np.linspace(0, 100, 100))
        bins = _compute_quantile_bins(s, n_bins=5)
        self.assertEqual(len(bins), 100)
        self.assertEqual(bins.min(), 1)
        self.assertEqual(bins.max(), 5)

    def test_3d_triplet_probabilities_and_synergy(self) -> None:
        rows = 400
        rng = np.random.default_rng(42)
        x1 = pd.Series(rng.uniform(-2, 2, rows))
        x2 = pd.Series(rng.uniform(-2, 2, rows))
        x3 = pd.Series(rng.uniform(-2, 2, rows))
        target = pd.Series(
            np.where(
                (x1 > 0.5) & (x2 > 0.5) & (x3 > 0.5),
                "bull",
                np.where((x1 < -0.5) & (x2 < -0.5) & (x3 < -0.5), "bear", "neutral"),
            )
        )

        f1_stats = _compute_1d_iv_and_spread(x1, target, n_bins=5)
        f2_stats = _compute_1d_iv_and_spread(x2, target, n_bins=5)
        f3_stats = _compute_1d_iv_and_spread(x3, target, n_bins=5)

        triplet_scores, voxel_probs = _compute_triplet_target_probabilities(
            f1_series=x1,
            f2_series=x2,
            f3_series=x3,
            target_series=target,
            f1_name="x1",
            f2_name="x2",
            f3_name="x3",
            target_name="target",
            n_bins=5,
            f1_1d_stats=f1_stats,
            f2_1d_stats=f2_stats,
            f3_1d_stats=f3_stats,
            min_support=5,
        )

        self.assertGreater(len(triplet_scores), 0)
        self.assertGreater(len(voxel_probs), 0)

        bull_score = [s for s in triplet_scores if s["target_class"] == "bull"][0]
        self.assertGreater(bull_score["sweet_spot_prob"], bull_score["base_rate"])
        self.assertGreater(bull_score["sweet_spot_lift"], 1.0)
        self.assertIn("x1", bull_score["sweet_spot_rule"])
        self.assertIn("x2", bull_score["sweet_spot_rule"])
        self.assertIn("x3", bull_score["sweet_spot_rule"])

    def test_e2e_probability_3d_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            feat_csv, lbl_csv = make_test_datasets(base_dir, rows=300)
            out_dir = base_dir / "reports"

            module = Probability3DModule(n_bins=5, max_candidates=4, min_support=5)
            result = module.run(feat_csv, lbl_csv, out_dir)

            self.assertTrue((result.report_dir / "summary.json").exists())
            self.assertTrue((result.report_dir / "triplet_probability_scores.csv").exists())
            self.assertTrue((result.report_dir / "voxel_conditional_probabilities.csv").exists())
            self.assertTrue((result.report_dir / "probability_3d_heatmaps.png").exists())
            self.assertTrue((result.report_dir / "report.md").exists())
            self.assertTrue((result.report_dir / "report.html").exists())

            summary = json.loads((result.report_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["module"], "probability_3d")
            self.assertIn("summary_metrics", summary)
            self.assertIn("top_triplets", summary)

    def test_probability_3d_module_loads_config(self) -> None:
        from fldataprofiler.modules.probability_3d import Probability3DConfig

        mod = Probability3DModule()
        self.assertIsInstance(mod.config, Probability3DConfig)
        self.assertEqual(mod.n_bins, 5)
        self.assertEqual(mod.max_candidates, 10)
        self.assertEqual(mod.min_support, 20)

        custom_cfg = Probability3DConfig(n_bins=4, max_candidates=8, min_support=30)
        mod_custom = Probability3DModule(config=custom_cfg)
        self.assertEqual(mod_custom.n_bins, 4)
        self.assertEqual(mod_custom.max_candidates, 8)
        self.assertEqual(mod_custom.min_support, 30)


if __name__ == "__main__":
    unittest.main()

