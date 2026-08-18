from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class VisualRegionsHelperTests(unittest.TestCase):
    def test_preparation_excludes_invalid_features_and_converts_numeric_strings(self) -> None:
        from fldataprofier.modules.visual_regions import _prepare_numeric_feature_frame

        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=12, freq="D"),
                "numeric_string": [str(value) for value in range(12)],
                "constant": [1.0] * 12,
                "text": [f"row-{value}" for value in range(12)],
                "mostly_missing": [np.nan] * 10 + [1.0, 2.0],
                "usable": np.linspace(-1.0, 1.0, 12),
            }
        )

        prepared, exclusions = _prepare_numeric_feature_frame(
            frame,
            ["Date", "numeric_string", "constant", "text", "mostly_missing", "usable"],
            max_missing_ratio=0.5,
        )

        self.assertEqual(["numeric_string", "usable"], list(prepared.columns))
        self.assertTrue(np.issubdtype(prepared["numeric_string"].dtype, np.floating))
        self.assertEqual(
            {
                "Date": "date_column",
                "constant": "constant_or_too_few_values",
                "text": "non_numeric",
                "mostly_missing": "too_many_missing",
            },
            {row["column"]: row["reason"] for row in exclusions},
        )

    def test_categorical_label_detection_uses_cardinality_bounds(self) -> None:
        from fldataprofier.modules.visual_regions import _categorical_label_columns

        frame = pd.DataFrame(
            {
                "one": ["x"] * 8,
                "two": ["a", "b"] * 4,
                "many": [f"class-{index}" for index in range(8)],
            }
        )

        self.assertEqual(
            ["two"], _categorical_label_columns(frame, ["one", "two", "many"], max_classes=4)
        )

    def test_quantile_bins_are_uint8_and_reuse_feature_names(self) -> None:
        from fldataprofier.modules.visual_regions import _quantile_bin_features

        features = pd.DataFrame(
            {
                "left": np.linspace(0.0, 1.0, 20),
                "right": np.linspace(1.0, 2.0, 20),
            }
        )

        binned = _quantile_bin_features(features, n_bins=5)

        self.assertEqual(["left", "right"], list(binned.columns))
        self.assertEqual(np.dtype("uint8"), binned["left"].dtype)
        self.assertLessEqual(int(binned.max().max()), 4)

    def test_candidate_selection_keeps_top_and_deterministic_sample(self) -> None:
        from fldataprofier.modules.visual_regions import _select_candidate_features

        scores = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "label": ["side", "side", "side"],
                "score": [0.9, 0.8, 0.1],
            }
        )

        selected = _select_candidate_features(
            scores,
            ["f1", "f2", "f3", "f4", "f5"],
            max_features=4,
            random_state=7,
        )

        self.assertEqual(4, len(selected))
        self.assertEqual(["f1", "f2"], selected[:2])
        self.assertEqual(
            selected, _select_candidate_features(scores, ["f1", "f2", "f3", "f4", "f5"], 4, 7)
        )

    def test_evaluate_2d_grid_purity(self) -> None:
        from fldataprofier.modules.visual_regions import _evaluate_2d_grid_purity

        df = pd.DataFrame(
            {
                "x_bin": [0, 0, 1, 1, 1],
                "y_bin": [0, 0, 1, 1, 1],
                "x_val": [0.1, 0.2, 0.8, 0.9, 1.0],
                "y_val": [10, 12, 80, 90, 100],
                "label": ["A", "B", "B", "B", "B"],
            }
        )
        result = _evaluate_2d_grid_purity(df, "x_bin", "y_bin", "x_val", "y_val", "label")
        self.assertTrue("purity" in result.columns)
        self.assertTrue("lift" in result.columns)
        self.assertTrue("sample_count" in result.columns)
        self.assertTrue("majority_label" in result.columns)
        self.assertEqual(len(result), 2)

    def test_merge_contiguous_regions(self) -> None:
        from fldataprofier.modules.visual_regions import _merge_contiguous_regions

        grid_cells = pd.DataFrame(
            {
                "x_bin": [0, 1, 2],
                "y_bin": [0, 0, 0],
                "majority_label": ["A", "A", "B"],
                "purity": [1.0, 1.0, 1.0],
                "sample_count": [10, 10, 10],
                "lift": [2.0, 2.0, 2.0],
                "x_min": [0.0, 1.0, 2.0],
                "x_max": [1.0, 2.0, 3.0],
                "y_min": [0.0, 0.0, 0.0],
                "y_max": [1.0, 1.0, 1.0],
            }
        )

        merged = _merge_contiguous_regions(grid_cells, None, "fx", "fy", "label", 0.5, 5)
        self.assertGreater(len(merged), 0)

    def test_extract_2d_rules(self) -> None:
        from fldataprofier.modules.visual_regions import _extract_2d_rules

        df = pd.DataFrame(
            {
                "f1": np.random.rand(100),
                "f2": np.random.rand(100),
                "label": np.random.choice(["A", "B"], 100),
            }
        )

        rules = _extract_2d_rules(
            df, ["f1", "f2"], ["label"], n_bins=3, min_samples=2, min_purity=0.5
        )
        self.assertTrue("rule_text" in rules.columns)


class VisualRegionsModuleTests(unittest.TestCase):
    def test_run_creates_expected_artifacts(self) -> None:
        from fldataprofier.modules.visual_regions import VisualRegionsModule

        feature_df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "f1": [0.1, 0.2, 0.8, 0.9, 1.0],
                "f2": [10, 20, 80, 90, 100],
            }
        )
        label_df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "label": ["A", "A", "B", "B", "B"],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            feature_csv = temp_path / "features.csv"
            label_csv = temp_path / "labels.csv"
            feature_df.to_csv(feature_csv, index=False)
            label_df.to_csv(label_csv, index=False)

            module = VisualRegionsModule(n_bins=3, min_samples_per_region=2, min_purity=0.5)
            result = module.run(feature_csv, label_csv, temp_path, join_key="id")

            self.assertEqual("visual_regions", result.report_dir.name)
            self.assertEqual(4, len(result.artifacts))

            artifact_names = [a.name for a in result.artifacts]
            self.assertIn("summary.json", artifact_names)
            self.assertIn("rules_2d.csv", artifact_names)
            self.assertIn("report.md", artifact_names)
            self.assertIn("report.html", artifact_names)

    def test_registry_integration(self) -> None:
        from fldataprofier.modules.visual_regions import VisualRegionsModule
        from fldataprofier.registry import get_module, list_modules

        self.assertIn("visual_regions", list_modules())
        module = get_module("visual_regions")
        self.assertIsInstance(module, VisualRegionsModule)
