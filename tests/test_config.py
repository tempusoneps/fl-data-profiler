from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from fldataprofiler.config import (
    get_default_config_path,
    get_global_config,
    get_module_config,
    get_prune_config,
    load_config,
)


class ConfigTests(unittest.TestCase):
    def test_default_config_file_exists_and_is_valid_json(self) -> None:
        path = get_default_config_path()
        self.assertTrue(path.exists())
        cfg = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("global", cfg)
        self.assertIn("prune", cfg)
        self.assertIn("modules", cfg)
        self.assertEqual(cfg["global"]["max_rows"], 50000)
        self.assertEqual(cfg["prune"]["max_corr"], 0.85)
        self.assertEqual(cfg["modules"]["probability_prim"]["min_box_samples"], 250)

    def test_load_config_and_helpers(self) -> None:
        cfg = load_config()
        self.assertIsInstance(cfg, dict)

        global_cfg = get_global_config(cfg)
        self.assertIn("output_dir", global_cfg)
        self.assertEqual(global_cfg["output_dir"], "reports")

        prune_cfg = get_prune_config(cfg)
        self.assertIn("max_corr", prune_cfg)
        self.assertEqual(prune_cfg["max_corr"], 0.85)

        prim_cfg = get_module_config("probability_prim", cfg)
        self.assertEqual(prim_cfg["min_box_samples"], 250)
        self.assertEqual(prim_cfg["objective"], "support_weighted")

    def test_custom_config_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            custom_cfg_file = tmp_path / "custom_config.json"
            custom_cfg_file.write_text(
                json.dumps(
                    {
                        "global": {"output_dir": "custom_reports"},
                        "prune": {"max_corr": 0.70},
                        "modules": {
                            "probability_prim": {
                                "min_box_samples": 400,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cfg = load_config(custom_cfg_file)
            self.assertEqual(cfg["global"]["output_dir"], "custom_reports")
            # Unoverridden field in global is preserved
            self.assertEqual(cfg["global"]["max_rows"], 50000)
            # Overridden field in prune
            self.assertEqual(cfg["prune"]["max_corr"], 0.70)
            # Overridden field in module
            self.assertEqual(cfg["modules"]["probability_prim"]["min_box_samples"], 400)
            # Unoverridden field in module is preserved
            self.assertEqual(cfg["modules"]["probability_prim"]["objective"], "support_weighted")

    def test_nonexistent_custom_config_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_config("non_existent_config_12345.json")


if __name__ == "__main__":
    unittest.main()
