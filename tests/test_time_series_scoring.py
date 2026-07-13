from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd


class TimeSeriesScoringUtilityTests(unittest.TestCase):
    def test_walk_forward_splits_are_expanding(self) -> None:
        from fldataprofier.modules.time_series_scoring import walk_forward_splits

        splits = walk_forward_splits(
            260,
            min_train_size=100,
            test_size=50,
            step_size=50,
        )

        self.assertEqual(
            [(0, 100, 100, 150), (0, 150, 150, 200), (0, 200, 200, 250)],
            splits[:3],
        )

    def test_aggregate_scores_sorts_by_abs_mean(self) -> None:
        from fldataprofier.modules.time_series_scoring import aggregate_scores

        result = aggregate_scores(
            [
                {
                    "feature": "weak",
                    "label": "target",
                    "score_name": "rank_ic",
                    "score": 0.1,
                    "samples": 50,
                },
                {
                    "feature": "strong",
                    "label": "target",
                    "score_name": "rank_ic",
                    "score": -0.8,
                    "samples": 50,
                },
                {
                    "feature": "strong",
                    "label": "target",
                    "score_name": "rank_ic",
                    "score": -0.6,
                    "samples": 50,
                },
            ]
        )

        self.assertEqual("strong", result.iloc[0]["feature"])
        self.assertEqual(-0.7, round(float(result.iloc[0]["mean_score"]), 3))
        self.assertEqual(0.7, round(float(result.iloc[0]["mean_abs_score"]), 3))


    def test_permutation_importance_uses_serial_random_forest_jobs(self) -> None:
        from fldataprofier.modules.time_series_scoring import permutation_importance_rows

        captured_n_jobs: list[int | None] = []

        class FakeRegressor:
            def __init__(self, *args, **kwargs) -> None:
                captured_n_jobs.append(kwargs.get("n_jobs"))

            def fit(self, x, y):
                self.mean_ = float(np.mean(y))
                return self

            def predict(self, x):
                return np.full(len(x), self.mean_)

        rows = 160
        frame = pd.DataFrame(
            {
                "signal": np.linspace(0.0, 1.0, rows),
                "noise": np.sin(np.arange(rows)),
                "target": np.linspace(0.0, 1.0, rows),
            }
        )

        with patch(
            "fldataprofier.modules.time_series_scoring.RandomForestRegressor",
            FakeRegressor,
        ):
            permutation_importance_rows(
                frame,
                ["signal", "noise"],
                ["target"],
                n_estimators=5,
                min_train_size=100,
                test_size=50,
                step_size=50,
                max_folds=1,
            )

        self.assertEqual([1], captured_n_jobs)


if __name__ == "__main__":
    unittest.main()
