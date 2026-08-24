from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PruneConfig:
    max_corr: float = 0.85
    corr_method: str = "pearson"
    max_null: float = 0.20
    min_variance: float = 0.0
    top_k: int | None = None
    keep_cols: list[str] | None = None


@dataclass
class PruneResult:
    df_selected: pd.DataFrame
    retained_features: list[str]
    dropped_by_reason: dict[str, Any]
    summary: dict[str, Any]


def load_scores(scores_path: Path | str, score_col: str | None = None) -> dict[str, float]:
    """Load feature importance or metric scores from a CSV file.

    Parameters
    ----------
    scores_path : Path | str
        Path to the scores CSV file.
    score_col : str | None
        Name of the score column. If None, auto-detected from common names.

    Returns
    -------
    dict[str, float]
        Dictionary mapping feature name to numerical score.
    """
    path = Path(scores_path)
    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")

    df_scores = pd.read_csv(path)
    if df_scores.empty and len(df_scores.columns) == 0:
        raise ValueError(f"Empty CSV file with no columns: {path}")

    # Identify feature name column
    feature_col: str | None = None
    candidate_feature_cols = ["feature", "Feature", "feature_name", "column", "name"]
    for cand in candidate_feature_cols:
        if cand in df_scores.columns:
            feature_col = cand
            break

    if feature_col is None:
        for col in df_scores.columns:
            if str(col).lower() in ("feature", "feature_name", "column", "name"):
                feature_col = str(col)
                break

    if feature_col is None and len(df_scores.columns) > 0:
        feature_col = str(df_scores.columns[0])

    if feature_col is None:
        raise ValueError(f"Could not identify a feature column in {path}")

    # Identify score column
    target_score_col: str | None = None
    if score_col is not None:
        if score_col not in df_scores.columns:
            raise ValueError(f"Specified score column '{score_col}' not found in {path}")
        target_score_col = score_col
    else:
        candidate_score_cols = [
            "importance",
            "abs_ic_mean",
            "ic_mean",
            "score",
            "weight",
            "f_statistic",
            "mutual_info",
        ]
        for cand in candidate_score_cols:
            for col in df_scores.columns:
                if col != feature_col and str(col).lower() == cand.lower():
                    target_score_col = str(col)
                    break
            if target_score_col is not None:
                break

        if target_score_col is None:
            # Fallback to first numeric column != feature_col
            for col in df_scores.columns:
                if col != feature_col:
                    if np.issubdtype(df_scores[col].dtype, np.number):
                        target_score_col = str(col)
                        break
                    # Attempt numeric coercion check
                    converted = pd.to_numeric(df_scores[col], errors="coerce")
                    if converted.notna().any():
                        target_score_col = str(col)
                        break

    if target_score_col is None:
        raise ValueError(f"No valid score column found in {path}")

    scores: dict[str, float] = {}
    for _, row in df_scores.iterrows():
        feat = row[feature_col]
        val = row[target_score_col]
        if pd.isna(feat) or pd.isna(val):
            continue
        try:
            val_float = float(val)
            if not np.isnan(val_float):
                scores[str(feat)] = val_float
        except (ValueError, TypeError):
            continue

    return scores


def _is_raw_level_feature(col: str, series: pd.Series) -> bool:
    """Detect non-stationary raw price levels, raw unnormalized volumes, and raw calendar indices."""
    name = col.lower().strip()

    if not np.issubdtype(series.dtype, np.number) or pd.api.types.is_bool_dtype(series):
        return False

    # Safe relative / normalized / stationary keywords and suffixes
    safe_keywords = (
        "_pct",
        "_percent",
        "_ratio",
        "_return",
        "_ret",
        "_norm",
        "_normalized",
        "_zscore",
        "_score",
        "_signal",
        "_diff_pct",
        "_change_pct",
        "_rel",
        "_spread_pct",
        "_dist_pct",
        "_percentile",
        "_rank",
        "_prop",
        "_std",
        "_rsi",
        "_mfi",
        "_slope",
        "_roc",
        "_bb_b",
        "_keltner",
        "_cross",
        "_diff_ratio",
    )
    if any(k in name for k in safe_keywords):
        return False

    # Exact raw price/volume names (retaining cyclical calendar features like month, day_of_month, day_of_week)
    exact_raw = {"open", "high", "low", "close", "volume", "year"}
    if name in exact_raw:
        return True

    # Raw multi-timeframe level prefixes
    raw_patterns = (
        "prev_day_open",
        "prev_day_high",
        "prev_day_low",
        "prev_day_close",
        "prev_day_volume",
        "prev_day_pivot",
        "prev_day_r1",
        "prev_day_r2",
        "prev_day_s1",
        "prev_day_s2",
        "prev_15m_open",
        "prev_15m_high",
        "prev_15m_low",
        "prev_15m_close",
        "prev_15m_volume",
        "prev_15m_pivot",
        "prev_15m_r1",
        "prev_15m_s1",
        "prev_30m_open",
        "prev_30m_high",
        "prev_30m_low",
        "prev_30m_close",
        "prev_30m_volume",
        "prev_1h_open",
        "prev_1h_high",
        "prev_1h_low",
        "prev_1h_close",
        "prev_1h_volume",
        "high_macro",
        "low_macro",
        "open_macro",
        "close_macro",
        "high_micro",
        "low_micro",
        "open_micro",
        "close_micro",
        "high_short",
        "low_short",
        "open_short",
        "close_short",
        "high_medium",
        "low_medium",
        "open_medium",
        "close_medium",
        "close_min_micro",
        "close_min_short",
        "close_max_micro",
        "close_max_short",
        "open_min_micro",
        "open_max_micro",
    )
    if any(name == p or name.startswith(p) for p in raw_patterns):
        return True

    clean = series.dropna()
    if len(clean) > 0:
        # Large raw integer volume check
        if "volume" in name and clean.median() > 1000:
            return True
        # Unscaled absolute price coordinate check (>200 level without safe keyword)
        if any(k in name for k in ("high", "low", "close", "open", "pivot", "vwap", "sma", "ema", "wma")):
            if float(clean.median()) > 200:
                return True

    return False


def prune_features(
    df: pd.DataFrame,
    config: PruneConfig | None = None,
    scores: dict[str, float] | None = None,
) -> PruneResult:
    if config is None:
        config = PruneConfig()

    dropped_by_reason: dict[str, Any] = {
        "non_stationary_levels": [],
        "high_null": [],
        "low_variance": [],
        "collinear": {},
        "top_k_cutoff": [],
    }

    keep_cols = list(config.keep_cols) if config.keep_cols else []

    # Auto-detect date/time/object/category columns to keep
    non_numeric_cols = [
        col
        for col in df.columns
        if not np.issubdtype(df[col].dtype, np.number)
        or pd.api.types.is_bool_dtype(df[col])
        or col in keep_cols
        or col.lower() in ("date", "timestamp", "datetime", "time", "id", "symbol")
    ]
    meta_cols = list(dict.fromkeys(non_numeric_cols + keep_cols))

    # Candidate numeric features
    candidate_cols = [
        c
        for c in df.columns
        if c not in meta_cols
        and np.issubdtype(df[c].dtype, np.number)
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    total_candidates = len(candidate_cols)

    # 1. Automatic Non-Stationary / Raw Level Filter (drops absolute price levels & unnormalized volumes)
    survived_stationary: list[str] = []
    for col in candidate_cols:
        if col not in keep_cols and _is_raw_level_feature(col, df[col]):
            dropped_by_reason["non_stationary_levels"].append(col)
        else:
            survived_stationary.append(col)

    # 2. Null filter (treating inf and -inf as invalid/null)
    survived_null: list[str] = []
    for col in survived_stationary:
        null_ratio = float((df[col].isna() | np.isinf(df[col])).mean())
        if null_ratio > config.max_null:
            dropped_by_reason["high_null"].append(col)
        else:
            survived_null.append(col)

    # 3. Variance filter
    survived_var: list[str] = []
    col_variances: dict[str, float] = {}
    for col in survived_null:
        clean_series = df[col].replace([np.inf, -np.inf], np.nan)
        var = float(clean_series.var(skipna=True))
        if np.isnan(var) or var <= config.min_variance:
            dropped_by_reason["low_variance"].append(col)
        else:
            survived_var.append(col)
            col_variances[col] = var

    # Sort survived candidates: by score if provided, else by variance descending
    if scores:
        ordered_candidates = sorted(
            survived_var,
            key=lambda c: (
                scores.get(c, float("-inf")),
                col_variances.get(c, 0.0),
            ),
            reverse=True,
        )
    else:
        ordered_candidates = sorted(
            survived_var,
            key=lambda c: col_variances.get(c, 0.0),
            reverse=True,
        )

    # 3. Multicollinearity filter
    if len(ordered_candidates) <= 1 or config.max_corr >= 1.0:
        survived_corr = list(ordered_candidates)
    else:
        df_corr_input = df[ordered_candidates].replace([np.inf, -np.inf], np.nan)
        corr_matrix = df_corr_input.corr(method=config.corr_method).abs()

        kept: list[str] = []
        dropped_collinear: set[str] = set()

        for col in ordered_candidates:
            if col in dropped_collinear:
                continue
            kept.append(col)
            for other in ordered_candidates:
                if other != col and other not in dropped_collinear and other not in kept:
                    val = corr_matrix.loc[col, other]
                    if not np.isnan(val) and val > config.max_corr:
                        dropped_collinear.add(other)
                        dropped_by_reason["collinear"][other] = {
                            "dropped_for": col,
                            "correlation": float(val),
                        }
        survived_corr = kept

    # 4. Top-K cutoff
    if config.top_k is not None and config.top_k > 0 and len(survived_corr) > config.top_k:
        retained = survived_corr[: config.top_k]
        for col in survived_corr[config.top_k :]:
            dropped_by_reason["top_k_cutoff"].append(col)
    else:
        retained = survived_corr

    # Construct final dataframe: meta columns + retained features
    final_cols = [c for c in df.columns if c in meta_cols or c in retained]
    df_selected = df[final_cols].copy()

    total_dropped = total_candidates - len(retained)
    summary = {
        "total_rows": len(df),
        "initial_features_count": total_candidates,
        "retained_features_count": len(retained),
        "dropped_features_count": total_dropped,
        "dropped_breakdown": {
            "non_stationary_levels_count": len(dropped_by_reason["non_stationary_levels"]),
            "high_null_count": len(dropped_by_reason["high_null"]),
            "low_variance_count": len(dropped_by_reason["low_variance"]),
            "collinear_count": len(dropped_by_reason["collinear"]),
            "top_k_cutoff_count": len(dropped_by_reason["top_k_cutoff"]),
        },
    }

    return PruneResult(
        df_selected=df_selected,
        retained_features=retained,
        dropped_by_reason=dropped_by_reason,
        summary=summary,
    )
