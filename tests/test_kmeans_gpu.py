from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fldataprofier.registry import get_module, list_modules


class KMeansGpuTests(unittest.TestCase):
    def test_registry_exposes_kmeans_gpu_module(self) -> None:
        self.assertIn("kmeans_gpu", list_modules())
        self.assertEqual("kmeans_gpu", get_module("kmeans_gpu").name)

    def test_missing_cuml_raises_actionable_error(self) -> None:
        from fldataprofier.modules.kmeans_gpu import _load_gpu_kmeans

        with patch.dict("sys.modules", {"cuml": None}):
            with self.assertRaisesRegex(RuntimeError, "kmeans_gpu requires RAPIDS cuML"):
                _load_gpu_kmeans()

    def test_silhouette_sample_limits_expensive_metric(self) -> None:
        from fldataprofier.modules.kmeans_gpu import _silhouette_sample

        features = np.arange(200, dtype=float).reshape(100, 2)
        clusters = np.arange(100) % 3

        sample_features, sample_clusters = _silhouette_sample(
            features,
            clusters,
            max_samples=10,
            random_state=42,
        )

        self.assertEqual((10, 2), sample_features.shape)
        self.assertEqual((10,), sample_clusters.shape)
        self.assertTrue(set(sample_clusters).issubset({0, 1, 2}))

    def test_fit_reports_works_with_kmeans_compatible_backend(self) -> None:
        from fldataprofier.modules.kmeans_gpu import _fit_kmeans_gpu_reports

        frame = pd.DataFrame(
            {
                "feature_1": [0.0, 0.1, 1.0, 1.1] * 10,
                "feature_2": [0.0, 0.2, 1.0, 1.2] * 10,
                "label": ["low", "low", "high", "high"] * 10,
            }
        )

        results, distribution = _fit_kmeans_gpu_reports(
            frame,
            ["feature_1", "feature_2"],
            [("feature_1", "feature_2")],
            ["label"],
            _FakeKMeans,
        )

        self.assertEqual(["ok"], results["status"].tolist())
        self.assertEqual(["gpu=cuml"], results["note"].tolist())
        self.assertFalse(distribution.empty)


class _FakeKMeans:
    def __init__(self, n_clusters: int, n_init: int, random_state: int) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.inertia_ = 0.0

    def fit(self, values: np.ndarray) -> "_FakeKMeans":
        self.threshold_ = float(values[:, 0].mean())
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        return (values[:, 0] > self.threshold_).astype(int) % self.n_clusters


if __name__ == "__main__":
    unittest.main()
