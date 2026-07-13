from __future__ import annotations

import unittest
from pathlib import Path


class HtmlReportTests(unittest.TestCase):
    def test_markdown_source_is_collapsed_and_escaped(self) -> None:
        from fldataprofier.utils import _html_markdown_details

        html = _html_markdown_details("# Title\n<script>alert(1)</script>")

        self.assertIn('<details class="markdown-source">', html)
        self.assertIn('<summary>Markdown source</summary>', html)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', html)
        self.assertNotIn('<script>alert(1)</script>', html)

    def test_report_renderers_do_not_show_full_markdown_as_visible_pre(self) -> None:
        module_dir = Path("fldataprofier/modules")
        offenders: list[str] = []
        for path in module_dir.glob("*.py"):
            text = path.read_text()
            if "<pre>{escaped_markdown}</pre>" in text or "<pre>{markdown}</pre>" in text:
                offenders.append(str(path))

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
