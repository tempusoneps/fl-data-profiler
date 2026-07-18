from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import pandas as pd

from fldataprofier.registry import get_module, list_modules
from fldataprofier.modules.flaml_module import FLAMLRelationshipsModule
from fldataprofier.modules.autogluon_module import AutoGluonRelationshipsModule
from fldataprofier.modules.pycaret_module import PyCaretRelationshipsModule


class AutoMLTests(unittest.TestCase):
    def test_registry_contains_automl_modules(self) -> None:
        modules = list_modules()
        self.assertIn("flaml", modules)
        self.assertIn("autogluon", modules)
        self.assertIn("pycaret", modules)

    def test_get_module_returns_correct_types(self) -> None:
        self.assertIsInstance(get_module("flaml"), FLAMLRelationshipsModule)
        self.assertIsInstance(get_module("autogluon"), AutoGluonRelationshipsModule)
        self.assertIsInstance(get_module("pycaret"), PyCaretRelationshipsModule)

    def test_missing_dependency_raises_importerror(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            features = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0]})
            labels = pd.DataFrame({"label_a": [0, 1, 0]})
            feature_csv = tmp_path / "features.csv"
            label_csv = tmp_path / "labels.csv"
            features.to_csv(feature_csv, index=False)
            labels.to_csv(label_csv, index=False)

            with patch.dict("sys.modules", {"flaml": None}):
                with self.assertRaises(ImportError) as ctx:
                    get_module("flaml").run(feature_csv, label_csv, tmp_path / "out")
                self.assertIn("flaml", str(ctx.exception).lower())

            with patch.dict("sys.modules", {"autogluon": None, "autogluon.tabular": None}):
                with self.assertRaises(ImportError) as ctx:
                    get_module("autogluon").run(feature_csv, label_csv, tmp_path / "out")
                self.assertIn("autogluon", str(ctx.exception).lower())

            with patch.dict("sys.modules", {"pycaret": None}):
                with self.assertRaises(ImportError) as ctx:
                    get_module("pycaret").run(feature_csv, label_csv, tmp_path / "out")
                self.assertIn("pycaret", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
