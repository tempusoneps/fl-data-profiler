from __future__ import annotations

import unittest
from pathlib import Path


class HtmlReportTests(unittest.TestCase):
    def test_markdown_source_is_collapsed_and_escaped(self) -> None:
        from fldataprofiler.utils import _html_markdown_details

        html = _html_markdown_details("# Title\n<script>alert(1)</script>")

        self.assertIn('<details class="markdown-source">', html)
        self.assertIn("<summary>Markdown source</summary>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)

    def test_report_renderers_do_not_show_full_markdown_as_visible_pre(self) -> None:
        module_dir = Path("src/fldataprofiler/modules")
        offenders: list[str] = []
        for path in module_dir.glob("*.py"):
            text = path.read_text()
            if "<pre>{escaped_markdown}</pre>" in text or "<pre>{markdown}</pre>" in text:
                offenders.append(str(path))

        self.assertEqual([], offenders)

    def test_markdown_reports_contain_execution_time(self) -> None:
        import json
        import tempfile

        import pandas as pd

        from fldataprofiler.registry import get_module

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feat_csv = tmp_path / "feat.csv"
            label_csv = tmp_path / "label.csv"
            out_dir = tmp_path / "out"

            df_feat = pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=50, freq="1h"),
                "feat_a": range(50),
                "feat_b": range(50, 100),
            })
            df_label = pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=50, freq="1h"),
                "target": [0, 1] * 25,
            })
            df_feat.to_csv(feat_csv, index=False)
            df_label.to_csv(label_csv, index=False)

            for module_name in ["statistics", "scipy", "information_coefficient"]:
                mod = get_module(module_name)
                res = mod.run(feature_csv=feat_csv, label_csv=label_csv, output_dir=out_dir)
                md_path = res.report_dir / "report.md"
                self.assertTrue(md_path.exists())
                md_text = md_path.read_text(encoding="utf-8")
                self.assertIn("- Execution time: `", md_text)
                summary_path = res.report_dir / "summary.json"
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    metadata = summary.get("metadata", summary)
                    self.assertIn("execution_time", metadata)

    def test_all_core_modules_support_progress(self) -> None:
        import tempfile
        from unittest.mock import patch

        import pandas as pd

        from fldataprofiler.registry import get_module

        class FakeTqdm:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs
                self.updates: list[int] = []

            def __enter__(self) -> FakeTqdm:
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                pass

            def set_postfix_str(self, label: str) -> None:
                pass

            def update(self, n: int = 1) -> None:
                self.updates.append(n)

        progress_instances: list[FakeTqdm] = []

        def fake_tqdm(*args, **kwargs):
            instance = FakeTqdm(*args, **kwargs)
            progress_instances.append(instance)
            return instance

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            feat_csv = tmp_path / "feat.csv"
            label_csv = tmp_path / "label.csv"
            out_dir = tmp_path / "out"

            df_feat = pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=60, freq="1h"),
                "feat_a": range(60),
                "feat_b": range(60, 120),
            })
            df_label = pd.DataFrame({
                "Date": pd.date_range("2024-01-01", periods=60, freq="1h"),
                "target": [0, 1] * 30,
            })
            df_feat.to_csv(feat_csv, index=False)
            df_label.to_csv(label_csv, index=False)

            with patch("fldataprofiler.modules.progress.tqdm", fake_tqdm):
                for module_name in ["statistics", "scipy", "statsmodels", "sklearn", "boruta", "shap"]:
                    mod = get_module(module_name)
                    mod.progress = True
                    mod.run(feature_csv=feat_csv, label_csv=label_csv, output_dir=out_dir)

        self.assertEqual(6, len(progress_instances))
        for progress in progress_instances:
            self.assertGreaterEqual(progress.kwargs["total"], 3)
            self.assertFalse(progress.kwargs["disable"])
            self.assertEqual(progress.kwargs["total"], sum(progress.updates))

    def test_statistics_heatmap_top_features_selection(self) -> None:
        import pandas as pd

        from fldataprofiler.modules.statistics import _select_top_heatmap_features

        # Create sample correlations with 50 features and 2 labels
        rows = []
        for i in range(50):
            rows.append({
                "feature": f"f_{i}",
                "label": "label_1",
                "pearson_correlation": 0.01 * (i + 1),
                "abs_correlation": 0.01 * (i + 1),
            })
            rows.append({
                "feature": f"f_{i}",
                "label": "label_2",
                "pearson_correlation": -0.005 * (50 - i),
                "abs_correlation": 0.005 * (50 - i),
            })
        df = pd.DataFrame(rows)

        # Select top-k per label (top 5 per label, max 8)
        selected = _select_top_heatmap_features(df, top_k_per_label=5, max_total=8)
        self.assertLessEqual(len(selected), 8)
        # Should include top features for label_1 (e.g. f_49, f_48) and label_2 (f_0, f_1)
        self.assertIn("f_49", selected)
        self.assertIn("f_0", selected)


if __name__ == "__main__":
    unittest.main()

