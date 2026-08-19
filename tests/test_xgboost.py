from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class XGBoostTests(unittest.TestCase):
    def test_progress_tracks_every_target_model(self) -> None:
        from fldataprofiler.modules.xgboost import XGBoostRelationshipsModule

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

        def fake_classification(label, features, target):
            return (
                {
                    "label": label,
                    "task": "classification",
                    "model": "FakeXGBClassifier",
                    "samples": len(target),
                    "features": len(features.columns),
                    "score_train": 1.0,
                    "score_primary": 1.0,
                    "overfit_gap": 0.0,
                    "score_primary_name": "balanced_accuracy",
                    "mae": None,
                    "rmse": None,
                    "accuracy": 1.0,
                    "balanced_accuracy": 1.0,
                    "f1_weighted": 1.0,
                    "note": "fake",
                },
                [],
                [],
                ([], None),
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = 40
            features = pd.DataFrame(
                {
                    "feature_1": [float(index) for index in range(rows)],
                    "feature_2": [float(index % 5) for index in range(rows)],
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

            with (
                patch("fldataprofiler.modules.progress.tqdm", fake_tqdm),
                patch(
                    "fldataprofiler.modules.xgboost._fit_classification",
                    fake_classification,
                ),
            ):
                XGBoostRelationshipsModule(progress=True).run(
                    feature_csv,
                    label_csv,
                    tmp_path / "out",
                )

        self.assertEqual(1, len(progress_instances))
        self.assertEqual(2, progress_instances[0].kwargs["total"])
        self.assertFalse(progress_instances[0].kwargs["disable"])
        self.assertEqual([1, 1], progress_instances[0].updates)
        self.assertEqual(["label_a", "label_b"], progress_instances[0].postfixes)

    def test_categorical_and_numeric_features_passed(self) -> None:
        from fldataprofiler.modules.xgboost import XGBoostRelationshipsModule

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rows = 50
            features = pd.DataFrame(
                {
                    "num_feat": [float(i) for i in range(rows)],
                    "cat_feat": ["cat_A" if i % 2 == 0 else "cat_B" for i in range(rows)],
                }
            )
            labels = pd.DataFrame(
                {
                    "target_num": [float(i % 10) for i in range(rows)],
                }
            )
            feat_path = tmp_path / "features.csv"
            lbl_path = tmp_path / "labels.csv"
            features.to_csv(feat_path, index=False)
            labels.to_csv(lbl_path, index=False)

            result = XGBoostRelationshipsModule(progress=False).run(
                feat_path,
                lbl_path,
                tmp_path / "out",
            )
            self.assertTrue(result.report_dir.exists())
            self.assertTrue((tmp_path / "out" / "xgboost" / "report.md").exists())
            self.assertTrue((tmp_path / "out" / "xgboost" / "scores.csv").exists())


if __name__ == "__main__":
    unittest.main()
