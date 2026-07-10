from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fldataprofier import cli
from fldataprofier.utils import _read_table, _read_table_with_date_index


class InputFormatTests(unittest.TestCase):
    def test_cli_rejects_unsupported_feature_extension_before_loading_module(self) -> None:
        stderr = io.StringIO()

        with patch("fldataprofier.cli.get_module") as get_module, contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                cli.main([
                    "fit",
                    "feature.json",
                    "label.csv",
                    "--module",
                    "statistics",
                ])

        self.assertEqual(2, raised.exception.code)
        get_module.assert_not_called()
        self.assertIn("feature_csv must be a .csv or .parquet file", stderr.getvalue())

    def test_read_table_with_date_index_loads_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.parquet"
            source = pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=3, freq="h"),
                    "feature": [1.0, 2.0, 3.0],
                }
            )
            source.to_parquet(path, index=False)

            frame = _read_table_with_date_index(path)

        self.assertEqual("Date", frame.index.name)
        self.assertEqual(["feature"], list(frame.columns))
        self.assertEqual([1.0, 2.0, 3.0], frame["feature"].tolist())

    def test_read_table_loads_parquet_without_indexing_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.parquet"
            source = pd.DataFrame(
                {
                    "Date": pd.date_range("2024-01-01", periods=2, freq="h"),
                    "target": ["up", "down"],
                }
            )
            source.to_parquet(path, index=False)

            frame = _read_table(path)

        self.assertEqual(["Date", "target"], list(frame.columns))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame["Date"]))


if __name__ == "__main__":
    unittest.main()
