from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd


class SignalAnalysisTests(unittest.TestCase):
    def test_signal_analysis_runs_successfully_numeric(self) -> None:
        from fldataprofiler.modules.signal_analysis import SignalAnalysisModule

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = 60
            features = pd.DataFrame(
                {
                    "pin_bar_signal": [1.0 if i % 4 == 0 else 0.0 for i in range(rows)],
                    "rsi_divergence_signal": [-1.0 if i % 5 == 0 else 0.0 for i in range(rows)],
                    "atr_breakout_signal": [1.0 if i % 3 == 0 else 0.0 for i in range(rows)],
                    "non_signal_feat": [float(i) for i in range(rows)],
                }
            )
            labels = pd.DataFrame(
                {
                    "allow_entry": [
                        "Yes - Buy"
                        if i % 3 == 0
                        else ("Yes - Sell" if i % 4 == 0 else "No - Sideway")
                        for i in range(rows)
                    ]
                }
            )

            feat_csv = tmp_path / "features.csv"
            lbl_csv = tmp_path / "labels.csv"
            features.to_csv(feat_csv, index=False)
            labels.to_csv(lbl_csv, index=False)

            result = SignalAnalysisModule(progress=False).run(
                feat_csv,
                lbl_csv,
                tmp_path / "out",
            )

            self.assertTrue(result.report_dir.exists())
            report_dir = tmp_path / "out" / "signal_analysis"
            self.assertTrue((report_dir / "report.md").exists())
            self.assertTrue((report_dir / "report.html").exists())
            self.assertTrue((report_dir / "signal_probability_matrix.csv").exists())
            self.assertTrue((report_dir / "signal_trap_diagnosis.csv").exists())
            self.assertTrue((report_dir / "top_clean_signals.csv").exists())
            self.assertTrue((report_dir / "summary.json").exists())

            prob_df = pd.read_csv(report_dir / "signal_probability_matrix.csv")
            self.assertFalse(prob_df.empty)
            self.assertIn("conditional_prob", prob_df.columns)
            self.assertIn("lift", prob_df.columns)
            self.assertIn("ci_lower_95", prob_df.columns)

            trap_df = pd.read_csv(report_dir / "signal_trap_diagnosis.csv")
            self.assertFalse(trap_df.empty)
            self.assertIn("clean_edge", trap_df.columns)
            self.assertIn("adverse_risk_ratio", trap_df.columns)

    def test_signal_analysis_with_string_signals(self) -> None:
        from fldataprofiler.modules.signal_analysis import SignalAnalysisModule

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = 60
            features = pd.DataFrame(
                {
                    "macd_signal": ["buy" if i % 4 == 0 else "hold" for i in range(rows)],
                    "rsi_signal": ["sell" if i % 5 == 0 else "none" for i in range(rows)],
                    "bb_signal": ["buy" if i % 3 == 0 else "hold" for i in range(rows)],
                }
            )
            labels = pd.DataFrame(
                {
                    "allow_entry": [
                        "Yes - Buy"
                        if i % 3 == 0
                        else ("Yes - Sell" if i % 4 == 0 else "No - Sideway")
                        for i in range(rows)
                    ]
                }
            )

            feat_csv = tmp_path / "features.csv"
            lbl_csv = tmp_path / "labels.csv"
            features.to_csv(feat_csv, index=False)
            labels.to_csv(lbl_csv, index=False)

            result = SignalAnalysisModule(progress=False).run(
                feat_csv,
                lbl_csv,
                tmp_path / "out",
            )

            self.assertTrue(result.report_dir.exists())
            report_dir = tmp_path / "out" / "signal_analysis"
            prob_df = pd.read_csv(report_dir / "signal_probability_matrix.csv")
            self.assertFalse(prob_df.empty)

            trap_df = pd.read_csv(report_dir / "signal_trap_diagnosis.csv")
            self.assertFalse(trap_df.empty)
            self.assertTrue((trap_df["true_alpha_pct"] > 0).any())


if __name__ == "__main__":
    unittest.main()
