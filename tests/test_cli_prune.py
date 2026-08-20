from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fldataprofiler.cli import main


def test_cli_prune_parquet(tmp_path: Path) -> None:
    feat_file = tmp_path / "feature.parquet"
    out_file = tmp_path / "selected_feature.parquet"
    summary_file = tmp_path / "prune_summary.json"

    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=50),
            "f1": rng.normal(0, 1, 50),
            "f2_const": [1.0] * 50,
            "f3_null": [np.nan] * 40 + list(rng.normal(0, 1, 10)),
        }
    )
    df.to_parquet(feat_file, index=False)

    exit_code = main(
        [
            "prune",
            str(feat_file),
            "--output",
            str(out_file),
            "--summary-json",
            str(summary_file),
            "--max-corr",
            "0.85",
            "--max-null",
            "0.20",
        ]
    )
    assert exit_code == 0
    assert out_file.exists()
    assert summary_file.exists()

    df_out = pd.read_parquet(out_file)
    assert "Date" in df_out.columns
    assert "f1" in df_out.columns
    assert "f2_const" not in df_out.columns
    assert "f3_null" not in df_out.columns

    summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary_data["input_path"] == str(feat_file)
    assert summary_data["output_path"] == str(out_file)
    assert summary_data["retained_features_count"] == 1
    assert "f2_const" in summary_data["dropped_by_reason"]["low_variance"]
    assert "f3_null" in summary_data["dropped_by_reason"]["high_null"]


def test_cli_prune_default_output_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    feat_file = tmp_path / "feature.csv"
    df = pd.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-02"],
            "f1": [1.0, 2.0],
        }
    )
    df.to_csv(feat_file, index=False)

    monkeypatch.chdir(tmp_path)
    exit_code = main(["prune", str(feat_file)])
    assert exit_code == 0
    assert (tmp_path / "datasets" / "selected_feature.csv").exists()
    assert (tmp_path / "reports" / "prune_summary.json").exists()

    df_out = pd.read_csv(tmp_path / "datasets" / "selected_feature.csv")
    assert list(df_out.columns) == ["Date", "f1"]


def test_cli_prune_with_scores_and_top_k(tmp_path: Path) -> None:
    feat_file = tmp_path / "feature.parquet"
    scores_file = tmp_path / "scores.csv"
    out_file = tmp_path / "custom_selected.parquet"

    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 1, 100)
    x2 = x1 * 0.99 + rng.normal(0, 0.01, 100)  # highly collinear with x1
    x3 = rng.normal(0, 1, 100)
    x4 = rng.normal(0, 1, 100)

    df = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=100),
            "feat_1": x1,
            "feat_2": x2,
            "feat_3": x3,
            "feat_4": x4,
        }
    )
    df.to_parquet(feat_file, index=False)

    # feat_2 has higher score than feat_1, so feat_2 should survive collinearity pruning
    # Also top-k=2 should keep the top 2 overall (feat_2 and feat_3 or feat_4)
    scores_df = pd.DataFrame(
        {
            "feature": ["feat_1", "feat_2", "feat_3", "feat_4"],
            "importance": [0.1, 0.9, 0.8, 0.7],
        }
    )
    scores_df.to_csv(scores_file, index=False)

    exit_code = main(
        [
            "prune",
            str(feat_file),
            "--scores-file",
            str(scores_file),
            "--score-col",
            "importance",
            "--max-corr",
            "0.85",
            "--top-k",
            "2",
            "--output",
            str(out_file),
            "--keep-col",
            "feat_4",
        ]
    )
    assert exit_code == 0
    assert out_file.exists()

    df_out = pd.read_parquet(out_file)
    assert "Date" in df_out.columns
    assert "feat_2" in df_out.columns
    assert "feat_4" in df_out.columns
    assert "feat_1" not in df_out.columns


def test_cli_prune_invalid_extension(tmp_path: Path) -> None:
    invalid_file = tmp_path / "data.txt"
    invalid_file.write_text("hello")

    with pytest.raises(SystemExit) as excinfo:
        main(["prune", str(invalid_file)])
    assert excinfo.value.code == 2
