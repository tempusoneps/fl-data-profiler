from __future__ import annotations

import unittest


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


if __name__ == "__main__":
    unittest.main()
