from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd


class SignalAnalysisTests(unittest.TestCase):
    def test_signal_analysis_runs_successfully(self) -> None:
        from fldataprofier.modules.signal_analysis import SignalAnalysisModule

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
                        "Yes - Buy" if i % 3 == 0 else ("Yes - Sell" if i % 4 == 0 else "No - Sideway")
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

            report_dir = tmp_path / "out" / "signal_analysis"
            self.assertTrue((report_dir / "report.md").exists())
            self.assertTrue((report_dir / "single_signal_scores.csv").exists())
            self.assertTrue((report_dir / "combined_signal_importance.csv").exists())
            self.assertTrue((report_dir / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
