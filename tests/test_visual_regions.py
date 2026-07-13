from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

        self.assertEqual(["two"], _categorical_label_columns(frame, ["one", "two", "many"], max_classes=4))

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
        self.assertEqual(selected, _select_candidate_features(scores, ["f1", "f2", "f3", "f4", "f5"], 4, 7))
