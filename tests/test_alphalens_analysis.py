from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fldataprofier.modules.alphalens_analysis import AlphalensAnalysisModule
from fldataprofier.registry import get_module


def make_alphalens_synthetic_dataset(base_dir: Path, rows: int = 300) -> tuple[Path, Path]:
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")

    # Generate price random walk
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.005, rows)
    close_price = 1000.0 * np.exp(np.cumsum(returns))

    # Generate a predictive signal: positively correlated with forward price change (t to t+5)
    fwd_5 = pd.Series(close_price).pct_change(5).shift(-5).fillna(0).values
    signal = fwd_5 * 10.0 + np.random.normal(0, 0.01, rows)

    # Generate an inverted signal: negatively correlated with forward return
    inverted_signal = -fwd_5 * 8.0 + np.random.normal(0, 0.01, rows)

    # Pure noise feature
    noise = np.random.normal(0, 1.0, rows)

    feature_path = base_dir / "feature.csv"
    label_path = base_dir / "label.csv"

    pd.DataFrame(
        {
            "Date": dates,
            "Close": close_price,
            "signal_alpha": signal,
            "inverted_alpha": inverted_signal,
            "noise_feat": noise,
            "constant_feat": 100.0,
        }
    ).to_csv(feature_path, index=False)

    pd.DataFrame(
        {
            "Date": dates,
            "target_return": fwd_5,
            "allow_entry": np.where(fwd_5 > 0, "yes", "no"),
        }
    ).to_csv(label_path, index=False)

    return feature_path, label_path


class AlphalensAnalysisModuleTests(unittest.TestCase):
    def test_registry_lookup(self) -> None:
        mod1 = get_module("alphalens_analysis")
        mod2 = get_module("alphalens")
        self.assertIsInstance(mod1, AlphalensAnalysisModule)
        self.assertIsInstance(mod2, AlphalensAnalysisModule)

    def test_alphalens_analysis_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_csv, label_csv = make_alphalens_synthetic_dataset(tmp_path)
            output_dir = tmp_path / "reports"

            module = AlphalensAnalysisModule(progress=False)
            result = module.run(
                feature_csv=feature_csv,
                label_csv=label_csv,
                output_dir=output_dir,
            )

            # 1. Verify report directory and artifacts
            self.assertTrue(result.report_dir.exists())
            self.assertEqual(result.report_dir.name, "alphalens")

            report_md = result.report_dir / "report.md"
            summary_json = result.report_dir / "summary.json"
            metrics_csv = result.report_dir / "factor_metrics.csv"
            quantile_csv = result.report_dir / "quantile_returns.csv"

            self.assertTrue(report_md.exists())
            self.assertTrue(summary_json.exists())
            self.assertTrue(metrics_csv.exists())
            self.assertTrue(quantile_csv.exists())

            # 2. Verify metrics content
            metrics_df = pd.read_csv(metrics_csv)
            self.assertIn("feature", metrics_df.columns)
            self.assertIn("ir", metrics_df.columns)
            self.assertIn("long_short_spread", metrics_df.columns)
            self.assertIn("monotonicity_score", metrics_df.columns)

            # Signal feature should rank top by absolute IR
            top_feature = metrics_df.iloc[0]["feature"]
            self.assertIn(top_feature, {"signal_alpha", "inverted_alpha"})

            # Signal alpha should have positive spread
            signal_row = metrics_df[metrics_df["feature"] == "signal_alpha"].iloc[0]
            self.assertGreater(signal_row["rank_ic"], 0.1)
            self.assertGreater(signal_row["long_short_spread"], 0.0)

            # Inverted alpha should have negative rank IC
            inv_row = metrics_df[metrics_df["feature"] == "inverted_alpha"].iloc[0]
            self.assertLess(inv_row["rank_ic"], -0.1)

            # 3. Verify charts created
            self.assertTrue((result.report_dir / "quantile_returns.png").exists())
            self.assertTrue((result.report_dir / "ic_decay.png").exists())
            self.assertTrue((result.report_dir / "cumulative_spread.png").exists())


if __name__ == "__main__":
    unittest.main()
