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

    def test_yearly_stability_calculation(self) -> None:
        from fldataprofiler.modules.signal_analysis import _compute_yearly_stability, _compute_trap_diagnosis

        dates = pd.date_range("2021-01-01", periods=180, freq="D")
        features = pd.DataFrame(
            {"sig1_signal": [1.0 if i % 2 == 0 else 0.0 for i in range(180)]},
            index=dates,
        )
        labels = pd.DataFrame(
            {
                "allow_entry": [
                    "Yes - Buy" if d.year == 2021 else ("Yes - Sell" if d.year == 2022 else "Yes - Buy")
                    for d in dates
                ]
            },
            index=dates,
        )
        merged = features.join(labels)
        trap_df, _ = _compute_trap_diagnosis(merged, ["sig1_signal"], "allow_entry")
        yearly_df, ranking_df = _compute_yearly_stability(
            merged, ["sig1_signal"], "allow_entry", trap_df, min_triggers=20
        )
        self.assertFalse(yearly_df.empty)
        self.assertIn("clean_edge", yearly_df.columns)
        self.assertIn("status", yearly_df.columns)
        self.assertFalse(ranking_df.empty)
        self.assertIn("consistency_pct", ranking_df.columns)
        self.assertIn("stability_grade", ranking_df.columns)

    def test_yearly_stability_multi_year_and_fallback(self) -> None:
        from fldataprofiler.modules.signal_analysis import _compute_yearly_stability, _compute_trap_diagnosis

        # 3 years: 2021 (win), 2022 (loss), 2023 (insufficient triggers < 20)
        dates_2021 = pd.date_range("2021-01-01", periods=100, freq="D")
        dates_2022 = pd.date_range("2022-01-01", periods=100, freq="D")
        dates_2023 = pd.date_range("2023-01-01", periods=30, freq="D")
        all_dates = dates_2021.append(dates_2022).append(dates_2023)

        sig_values = []
        lbl_values = []
        for d in all_dates:
            if d.year == 2021:
                sig_values.append(1.0)
                lbl_values.append("Yes - Buy")
            elif d.year == 2022:
                sig_values.append(1.0)
                lbl_values.append("Yes - Sell")  # counter-trend reversal trap
            else:
                # 2023: only 5 triggers (< 20)
                if len([x for x in sig_values[-len(dates_2023):] if x == 1.0]) < 5:
                    sig_values.append(1.0)
                else:
                    sig_values.append(0.0)
                lbl_values.append("Yes - Buy")

        df = pd.DataFrame({"alpha_sig": sig_values, "allow_entry": lbl_values}, index=all_dates)
        trap_df, _ = _compute_trap_diagnosis(df, ["alpha_sig"], "allow_entry")
        yearly_df, ranking_df = _compute_yearly_stability(
            df, ["alpha_sig"], "allow_entry", trap_df, min_triggers=20
        )

        self.assertEqual(len(yearly_df), 3)
        row_2021 = yearly_df[yearly_df["year"] == 2021].iloc[0]
        self.assertEqual(row_2021["status"], "valid")
        self.assertGreater(row_2021["clean_edge"], 0)

        row_2022 = yearly_df[yearly_df["year"] == 2022].iloc[0]
        self.assertEqual(row_2022["status"], "valid")
        self.assertLess(row_2022["clean_edge"], 0)

        row_2023 = yearly_df[yearly_df["year"] == 2023].iloc[0]
        self.assertEqual(row_2023["status"], "insufficient_data")
        self.assertTrue(pd.isna(row_2023["clean_edge"]))

        self.assertEqual(len(ranking_df), 1)
        rk = ranking_df.iloc[0]
        self.assertEqual(rk["years_evaluated"], 2)
        self.assertEqual(rk["positive_years"], 1)
        self.assertEqual(rk["consistency_pct"], 50.0)

        # Fallback test with non-datetime index
        df_no_date = df.reset_index(drop=True)
        yearly_fb, ranking_fb = _compute_yearly_stability(
            df_no_date, ["alpha_sig"], "allow_entry", trap_df, min_triggers=20
        )
        self.assertEqual(len(yearly_fb), 1)
        self.assertEqual(yearly_fb.iloc[0]["year"], "All")

    def test_yearly_heatmap_chart_generation(self) -> None:
        from fldataprofiler.modules.signal_analysis import _write_yearly_stability_heatmap

        with tempfile.TemporaryDirectory() as tmp:
            chart_path = Path(tmp) / "signal_yearly_stability.png"
            yearly_df = pd.DataFrame(
                [
                    {"signal_name": "sig1_signal", "signal_state": "buy", "year": 2021, "clean_edge": 15.0, "status": "valid"},
                    {"signal_name": "sig1_signal", "signal_state": "buy", "year": 2022, "clean_edge": -5.0, "status": "valid"},
                    {"signal_name": "sig2_signal", "signal_state": "buy", "year": 2021, "clean_edge": None, "status": "insufficient_data"},
                ]
            )
            top_clean = pd.DataFrame([{"signal_name": "sig1_signal", "signal_state": "buy"}])
            res = _write_yearly_stability_heatmap(chart_path, yearly_df, top_clean)
            self.assertIsNotNone(res)
            self.assertTrue(chart_path.exists())
            self.assertGreater(chart_path.stat().st_size, 0)

            # Test empty yearly_df returns None
            empty_path = Path(tmp) / "empty.png"
            res_empty = _write_yearly_stability_heatmap(empty_path, pd.DataFrame(), top_clean)
            self.assertIsNone(res_empty)
            self.assertFalse(empty_path.exists())

            # Test fallback when top_clean is empty
            fallback_path = Path(tmp) / "fallback.png"
            res_fb = _write_yearly_stability_heatmap(fallback_path, yearly_df, pd.DataFrame())
            self.assertIsNotNone(res_fb)
            self.assertTrue(fallback_path.exists())

    def test_signal_analysis_yearly_artifacts_and_reports(self) -> None:
        import json
        from fldataprofiler.modules.signal_analysis import SignalAnalysisModule

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dates = pd.date_range("2021-01-01", periods=120, freq="D")
            features = pd.DataFrame(
                {
                    "Date": dates,
                    "trend_signal": [1.0 if i % 2 == 0 else 0.0 for i in range(120)],
                }
            )
            labels = pd.DataFrame(
                {
                    "Date": dates,
                    "allow_entry": ["Yes - Buy" if i % 3 == 0 else "No - Sideway" for i in range(120)],
                }
            )
            feat_csv = tmp_path / "features.csv"
            lbl_csv = tmp_path / "labels.csv"
            features.to_csv(feat_csv, index=False)
            labels.to_csv(lbl_csv, index=False)

            result = SignalAnalysisModule(progress=False).run(feat_csv, lbl_csv, tmp_path / "out")
            report_dir = result.report_dir

            self.assertTrue((report_dir / "signal_yearly_stability.csv").exists())
            self.assertTrue((report_dir / "signal_stability_ranking.csv").exists())
            self.assertTrue((report_dir / "signal_yearly_stability.png").exists())

            # Check top_clean_signals.csv enrichment
            top_clean = pd.read_csv(report_dir / "top_clean_signals.csv")
            self.assertIn("consistency_pct", top_clean.columns)
            self.assertIn("worst_year_edge", top_clean.columns)

            # Check summary.json
            with open(report_dir / "summary.json", encoding="utf-8") as f:
                summary_data = json.load(f)
            self.assertIn("years_analyzed", summary_data)
            self.assertIn("top_stable_signals", summary_data)
            self.assertIn("yearly_stability", summary_data)

            # Check Markdown and HTML content
            md_text = (report_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("Multi-Year Stability & Consistency Analysis", md_text)
            self.assertIn("signal_yearly_stability.png", md_text)

            html_text = (report_dir / "report.html").read_text(encoding="utf-8")
            self.assertIn("Multi-Year Stability", html_text)


if __name__ == "__main__":
    unittest.main()
