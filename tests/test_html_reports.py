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



if __name__ == "__main__":
    unittest.main()
