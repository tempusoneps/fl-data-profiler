from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fldataprofiler.feature_pruner import PruneConfig, load_scores, prune_features


def test_drop_high_null_features() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=10),
            "feat_good": np.random.randn(10),
            "feat_null": [np.nan] * 8 + [1.0, 2.0],  # 80% null
        }
    )
    config = PruneConfig(max_null=0.20)
    result = prune_features(df, config)

    assert "feat_good" in result.retained_features
    assert "feat_null" not in result.retained_features
    assert "feat_null" in result.dropped_by_reason["high_null"]
    assert "Date" in result.df_selected.columns


def test_drop_zero_and_low_variance_features() -> None:
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=10),
            "feat_const": [5.0] * 10,
            "feat_var": np.linspace(0, 10, 10),
        }
    )
    config = PruneConfig(min_variance=0.0)
    result = prune_features(df, config)

    assert "feat_var" in result.retained_features
    assert "feat_const" not in result.retained_features
    assert "feat_const" in result.dropped_by_reason["low_variance"]


def test_drop_collinear_features_unsupervised() -> None:
    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, 100)
    x2 = x1 * 0.99 + rng.normal(0, 0.01, 100)  # corr > 0.99
    x3 = rng.normal(0, 1, 100)  # independent

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=100),
            "feat_1": x1,
            "feat_2": x2,
            "feat_3": x3,
        }
    )
    config = PruneConfig(max_corr=0.85)
    result = prune_features(df, config)

    assert "feat_3" in result.retained_features
    assert len(result.retained_features) == 2
    assert "feat_2" in result.dropped_by_reason["collinear"]
    assert result.dropped_by_reason["collinear"]["feat_2"]["dropped_for"] == "feat_1"


def test_score_guided_tie_breaking(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    # feat_low has variance ~ 1, feat_high has variance ~ 100
    base = rng.normal(0, 1, 100)
    feat_low = base + rng.normal(0, 0.01, 100)
    feat_high = base * 10.0 + rng.normal(0, 0.01, 100)

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=100),
            "feat_low_var": feat_low,
            "feat_high_var": feat_high,
        }
    )

    # Without scores, higher variance feature (feat_high_var) is kept
    res_unsupervised = prune_features(df, PruneConfig(max_corr=0.85))
    assert "feat_high_var" in res_unsupervised.retained_features
    assert "feat_low_var" in res_unsupervised.dropped_by_reason["collinear"]

    # With scores, feat_low_var has higher score, so it should be retained instead
    scores_file = tmp_path / "scores.csv"
    scores_file.write_text("feature,importance\nfeat_low_var,0.95\nfeat_high_var,0.10\n")
    scores = load_scores(scores_file)

    res_supervised = prune_features(df, PruneConfig(max_corr=0.85), scores=scores)
    assert "feat_low_var" in res_supervised.retained_features
    assert "feat_high_var" not in res_supervised.retained_features
    assert "feat_high_var" in res_supervised.dropped_by_reason["collinear"]
    assert (
        res_supervised.dropped_by_reason["collinear"]["feat_high_var"]["dropped_for"]
        == "feat_low_var"
    )


def test_top_k_selection() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=50),
            "f1": rng.normal(0, 1, 50),
            "f2": rng.normal(0, 1, 50),
            "f3": rng.normal(0, 1, 50),
            "f4": rng.normal(0, 1, 50),
            "f5": rng.normal(0, 1, 50),
        }
    )
    scores = {"f1": 10.0, "f2": 8.0, "f3": 6.0, "f4": 4.0, "f5": 2.0}
    config = PruneConfig(top_k=3, max_corr=0.99)
    result = prune_features(df, config=config, scores=scores)

    assert result.retained_features == ["f1", "f2", "f3"]
    assert len(result.retained_features) == 3
    assert set(result.dropped_by_reason["top_k_cutoff"]) == {"f4", "f5"}
    assert result.summary["dropped_breakdown"]["top_k_cutoff_count"] == 2
    assert "Date" in result.df_selected.columns
    assert list(result.df_selected.columns) == ["Date", "f1", "f2", "f3"]


def test_load_scores_from_csv(tmp_path: Path) -> None:
    # 1. Standard feature, importance
    p1 = tmp_path / "scores_standard.csv"
    p1.write_text("feature,importance\nalpha,0.85\nbeta,0.12\n")
    s1 = load_scores(p1)
    assert s1 == {"alpha": 0.85, "beta": 0.12}

    # 2. Custom score column
    p2 = tmp_path / "scores_custom.csv"
    p2.write_text("name,metric_val,unrelated\nx,1.5,foo\ny,3.5,bar\n")
    s2 = load_scores(p2, score_col="metric_val")
    assert s2 == {"x": 1.5, "y": 3.5}

    # 3. Alternative candidate column name (e.g. abs_ic_mean, column)
    p3 = tmp_path / "scores_ic.csv"
    p3.write_text("column,abs_ic_mean\nf_one,0.25\nf_two,0.05\n")
    s3 = load_scores(p3)
    assert s3 == {"f_one": 0.25, "f_two": 0.05}

    # 4. Error on invalid score_col
    with pytest.raises(ValueError, match="Specified score column 'missing' not found"):
        load_scores(p1, score_col="missing")

    # 5. Error on non-existent file
    with pytest.raises(FileNotFoundError):
        load_scores(tmp_path / "non_existent.csv")

    # 6. Error when no numeric score column is present
    p4 = tmp_path / "scores_no_numeric.csv"
    p4.write_text("feature,comment\nf_one,bad\nf_two,worse\n")
    with pytest.raises(ValueError, match="No valid score column found"):
        load_scores(p4)
