from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from sklearn.exceptions import ConvergenceWarning


class KMeanTests(unittest.TestCase):
    def test_skips_feature_pair_when_distinct_points_are_fewer_than_label_classes(self) -> None:
        from fldataprofier.modules.kmean import _fit_single_kmeans

        points = [(float(index), float(index % 2)) for index in range(6)]
        rows = points * 25
        features = pd.DataFrame(rows, columns=["feature_1", "feature_2"])
        labels = pd.Series([f"class_{index % 15}" for index in range(len(features))], name="label")

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result, distribution = _fit_single_kmeans(
                "feature_1",
                "feature_2",
                "label",
                features,
                labels,
            )

        convergence_warnings = [
            warning for warning in captured if issubclass(warning.category, ConvergenceWarning)
        ]
        self.assertEqual([], convergence_warnings)
        self.assertEqual("skipped", result["status"])
        self.assertEqual("not_enough_distinct_points_for_clusters", result["note"])
        self.assertEqual([], distribution)

    def test_progress_tracks_every_feature_pair_label_combination(self) -> None:
        from fldataprofier.modules.kmean import KMeanRelationshipsModule

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
            rows = 40
            features = pd.DataFrame(
                {
                    "feature_1": [float(index) for index in range(rows)],
                    "feature_2": [float(index % 7) for index in range(rows)],
                    "feature_3": [float((index * 3) % 11) for index in range(rows)],
                }
            )
            labels = pd.DataFrame(
                {
                    "label_a": ["up" if index % 2 else "down" for index in range(rows)],
                    "label_b": ["x" if index % 3 else "y" for index in range(rows)],
                }
            )
            feature_csv = tmp_path / "features.csv"
            label_csv = tmp_path / "labels.csv"
            features.to_csv(feature_csv, index=False)
            labels.to_csv(label_csv, index=False)

            with patch("fldataprofier.modules.progress.tqdm", fake_tqdm):
                KMeanRelationshipsModule(progress=True).run(
                    feature_csv,
                    label_csv,
                    tmp_path / "out",
                )

        self.assertEqual(1, len(progress_instances))
        self.assertEqual(6, progress_instances[0].kwargs["total"])
        self.assertFalse(progress_instances[0].kwargs["disable"])
        self.assertEqual([1, 1, 1, 1, 1, 1], progress_instances[0].updates)
        self.assertEqual(6, len(progress_instances[0].postfixes))

    def test_clustering_accuracy_computes_optimal_matching_accuracy(self) -> None:
        import numpy as np
        from fldataprofier.modules.kmean import _clustering_accuracy

        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred_perfect = np.array([1, 1, 1, 0, 0, 0])
        self.assertAlmostEqual(1.0, _clustering_accuracy(y_true, y_pred_perfect))

        y_pred_imperfect = np.array([1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(5 / 6, _clustering_accuracy(y_true, y_pred_imperfect))

    def test_select_sequential_rows_preserves_contiguous_order(self) -> None:
        from fldataprofier.modules.kmean import _select_sequential_rows

        df = pd.DataFrame({"a": range(100)})
        selected = _select_sequential_rows(df, 10)
        self.assertEqual(list(range(10)), selected["a"].tolist())


if __name__ == "__main__":
    unittest.main()
