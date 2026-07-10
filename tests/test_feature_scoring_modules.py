from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


def make_signal_dataset(base_dir: Path, rows: int = 260) -> tuple[Path, Path]:
    dates = pd.date_range("2024-01-01", periods=rows, freq="5min")
    signal = np.linspace(-2.0, 2.0, rows)
    seasonal = np.sin(np.arange(rows) / 9.0)
    noise = np.cos(np.arange(rows) / 5.0) * 0.03
    target = signal * 2.5 + seasonal * 0.2 + noise
    label_class = np.where(target > np.median(target), "up", "down")

    feature_path = base_dir / "feature.csv"
    label_path = base_dir / "label.csv"
    pd.DataFrame(
        {
            "Date": dates,
            "signal": signal,
            "seasonal": seasonal,
            "noise": np.cos(np.arange(rows)),
            "constant": 1.0,
        }
    ).to_csv(feature_path, index=False)
    pd.DataFrame(
        {
            "Date": dates,
            "target": target,
            "label_class": label_class,
        }
    ).to_csv(label_path, index=False)
    return feature_path, label_path


class FeatureScoringModuleTests(unittest.TestCase):
    def test_information_coefficient_ranks_signal_feature(self) -> None:
        from fldataprofier.modules.information_coefficient import InformationCoefficientModule

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_csv, label_csv = make_signal_dataset(tmp_path)
            result = InformationCoefficientModule().run(feature_csv, label_csv, tmp_path / "out")

            scores = pd.read_csv(result.report_dir / "feature_scores.csv")

        top = scores.sort_values("mean_abs_score", ascending=False).iloc[0]
        self.assertEqual("signal", top["feature"])
        self.assertIn(top["score_name"], {"pearson_ic", "rank_ic"})

    def test_permutation_importance_ts_ranks_predictive_feature(self) -> None:
        from fldataprofier.modules.permutation_importance_ts import PermutationImportanceTSModule

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_csv, label_csv = make_signal_dataset(tmp_path)
            result = PermutationImportanceTSModule(n_estimators=25, random_state=7).run(
                feature_csv,
                label_csv,
                tmp_path / "out",
            )

            scores = pd.read_csv(result.report_dir / "feature_scores.csv")

        self.assertEqual("signal", scores.iloc[0]["feature"])
        self.assertGreater(float(scores.iloc[0]["mean_score"]), 0)

    def test_timeseries_importance_combines_component_scores(self) -> None:
        from fldataprofier.modules.timeseries_importance import TimeSeriesImportanceModule

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_csv, label_csv = make_signal_dataset(tmp_path)
            result = TimeSeriesImportanceModule(n_estimators=25, random_state=11).run(
                feature_csv,
                label_csv,
                tmp_path / "out",
            )

            scores = pd.read_csv(result.report_dir / "feature_scores.csv")
            components = pd.read_csv(result.report_dir / "component_scores.csv")

        self.assertEqual("signal", scores.iloc[0]["feature"])
        self.assertTrue({"rank_ic", "permutation_importance"}.issubset(set(components["score_name"])))

    def test_timeseries_importance_uses_tqdm_when_progress_enabled(self) -> None:
        from fldataprofier.modules.timeseries_importance import TimeSeriesImportanceModule

        class FakeProgress:
            def __init__(self, *args, **kwargs) -> None:
                self.args = args
                self.kwargs = kwargs
                self.updates: list[int] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> bool:
                return False

            def set_postfix_str(self, value: str) -> None:
                self.postfix = value

            def update(self, value: int) -> None:
                self.updates.append(value)

        progress_instances: list[FakeProgress] = []

        def fake_tqdm(*args, **kwargs):
            progress = FakeProgress(*args, **kwargs)
            progress_instances.append(progress)
            return progress

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_csv, label_csv = make_signal_dataset(tmp_path)
            with patch("fldataprofier.modules.timeseries_importance.tqdm", fake_tqdm):
                TimeSeriesImportanceModule(
                    n_estimators=5,
                    random_state=11,
                    progress=True,
                ).run(feature_csv, label_csv, tmp_path / "out")

        self.assertEqual(1, len(progress_instances))
        self.assertEqual(5, progress_instances[0].kwargs["total"])
        self.assertFalse(progress_instances[0].kwargs["disable"])
        self.assertEqual([1, 1, 1, 1, 1], progress_instances[0].updates)

    def test_remaining_feature_selection_modules_write_scores(self) -> None:
        module_cases = [
            (
                "fldataprofier.modules.mutual_information",
                "MutualInformationModule",
                {},
                {"feature", "label", "score_name", "score"},
            ),
            (
                "fldataprofier.modules.mrmr",
                "MRMRModule",
                {"max_features": 20, "random_state": 3},
                {"relevance", "redundancy", "mrmr_score"},
            ),
            (
                "fldataprofier.modules.stability_selection",
                "StabilitySelectionModule",
                {"n_resamples": 8, "random_state": 13},
                {"selection_frequency"},
            ),
            (
                "fldataprofier.modules.regularized_linear",
                "RegularizedLinearModule",
                {"random_state": 17},
                {"coefficient", "abs_coefficient", "model_type"},
            ),
            (
                "fldataprofier.modules.lightgbm",
                "LightGBMModule",
                {"n_estimators": 10, "random_state": 19},
                {"importance_type", "score"},
            ),
            (
                "fldataprofier.modules.feature_interactions",
                "FeatureInteractionsModule",
                {"max_base_features": 4, "max_pairs": 12},
                {"interaction", "left_feature", "right_feature", "operation"},
            ),
            (
                "fldataprofier.modules.regime_scoring",
                "RegimeScoringModule",
                {"n_regimes": 3},
                {"regime", "score_name", "mean_abs_score"},
            ),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            feature_csv, label_csv = make_signal_dataset(tmp_path)
            for module_path, class_name, kwargs, expected_columns in module_cases:
                with self.subTest(module=module_path):
                    module = __import__(module_path, fromlist=[class_name])
                    module_class = getattr(module, class_name)
                    result = module_class(**kwargs).run(feature_csv, label_csv, tmp_path / "out")
                    scores = pd.read_csv(result.report_dir / "feature_scores.csv")

                    self.assertFalse(scores.empty)
                    self.assertTrue(expected_columns.issubset(set(scores.columns)))
                    self.assertTrue((result.report_dir / "summary.json").exists())

    def test_feature_scoring_modules_are_registered(self) -> None:
        from fldataprofier.registry import list_modules

        expected = {
            "timeseries_importance",
            "permutation_importance_ts",
            "information_coefficient",
            "mutual_information",
            "mrmr",
            "stability_selection",
            "regularized_linear",
            "lightgbm",
            "feature_interactions",
            "regime_scoring",
        }

        self.assertTrue(expected.issubset(set(list_modules())))


if __name__ == "__main__":
    unittest.main()
