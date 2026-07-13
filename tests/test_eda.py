from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class EdaTests(unittest.TestCase):
    def test_progress_tracks_eda_phases(self) -> None:
        from fldataprofier.modules.eda import EdaModule

        class FakeProgress:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs
                self.updates: list[int] = []
                self.postfixes: list[str] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> bool:
                return False

            def set_postfix_str(self, value: str) -> None:
                self.postfixes.append(value)

            def update(self, value: int) -> None:
                self.updates.append(value)

        progress_instances: list[FakeProgress] = []

        def fake_tqdm(*args, **kwargs):
            progress = FakeProgress(*args, **kwargs)
            progress_instances.append(progress)
            return progress

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_csv = tmp_path / "features.csv"
            label_csv = tmp_path / "labels.csv"
            pd.DataFrame(
                {
                    "feature_1": [1.0, 2.0, 3.0, 4.0],
                    "feature_2": [4.0, 3.0, 2.0, 1.0],
                    "category": ["a", "b", "a", "b"],
                }
            ).to_csv(feature_csv, index=False)
            pd.DataFrame(
                {
                    "target_1": [0, 1, 0, 1],
                    "target_2": [1.0, 1.5, 2.0, 2.5],
                }
            ).to_csv(label_csv, index=False)

            with patch("fldataprofier.modules.progress.tqdm", fake_tqdm):
                EdaModule(progress=True).run(
                    feature_csv,
                    label_csv,
                    tmp_path / "out",
                    targets=["target_1"],
                )

        self.assertEqual(1, len(progress_instances))
        self.assertEqual(7, progress_instances[0].kwargs["total"])
        self.assertFalse(progress_instances[0].kwargs["disable"])
        self.assertEqual([1, 1, 1, 1, 1, 1, 1], progress_instances[0].updates)
        self.assertEqual(
            ["load", "select", "profile", "summaries", "write_tables", "heatmaps", "report"],
            progress_instances[0].postfixes,
        )


if __name__ == "__main__":
    unittest.main()
