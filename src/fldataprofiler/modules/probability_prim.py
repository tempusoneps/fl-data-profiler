from __future__ import annotations

import itertools
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from fldataprofiler.config import get_module_config
from fldataprofiler.modules.base import ModuleResult
from fldataprofiler.modules.progress import ModuleProgress, StatusTimer
from fldataprofiler.modules.statistics import DatasetShape
from fldataprofiler.utils import (
    _date_columns,
    _format_duration,
    _html_markdown_details,
    _markdown_table,
    _merge_inputs,
    _numeric_feature_columns,
    _numeric_series,
    _read_table_with_date_index,
    _round,
    _sample_rows,
    _select_targets,
    _write_csv,
    _write_json,
)

MAX_ROWS = 50_000
MAX_LABEL_CLASSES = 50
MIN_NON_NULL = 10
RANDOM_STATE = 42
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_BOX_SAMPLES = 250
DEFAULT_MIN_SUPPORT = 0.005
DEFAULT_MAX_CANDIDATES = 16
DEFAULT_EXPAND_DELTA = 0.01
DEFAULT_OBJECTIVE = "support_weighted"
EPSILON = 1e-9


@dataclass
class ProbabilityPrimConfig:
    min_box_samples: int = DEFAULT_MIN_BOX_SAMPLES
    min_support: float = DEFAULT_MIN_SUPPORT
    alpha: float = DEFAULT_ALPHA
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    expand_delta: float = DEFAULT_EXPAND_DELTA
    objective: str = DEFAULT_OBJECTIVE


@dataclass(frozen=True)
class ProbabilityPrimRunMetadata:
    module: str
    created_at: str
    execution_time: str
    feature_csv: str
    label_csv: str
    join_strategy: str
    feature_shape: DatasetShape
    label_shape: DatasetShape
    merged_shape: DatasetShape
    alpha: float
    min_box_samples: int
    min_support: float
    objective: str
    max_candidates: int
    features_count: int
    candidate_features: list[str]
    targets: list[str]
    model_rows: int
    rules_count: int


def _compute_quantile_bins(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """Assign rank-based equal-frequency quantile bin indices (1 to n_bins)."""
    if len(series) == 0:
        return pd.Series(dtype=int, index=series.index)
    ranks = series.rank(method="first")
    actual_bins = min(n_bins, len(series))
    if actual_bins < 1:
        return pd.Series(1, index=series.index, dtype=int)
    bins = pd.qcut(ranks, q=actual_bins, labels=False) + 1
    return bins.astype(int)


def _compute_1d_iv(
    x: pd.Series,
    y_binary: pd.Series,
    n_bins: int = 10,
    precomputed_bins: pd.Series | None = None,
) -> float:
    """Calculate 1D Information Value for feature x against binary target y."""
    clean_x = _numeric_series(x)
    valid_mask = clean_x.notna() & y_binary.notna()
    x_val = clean_x[valid_mask]
    y_val = y_binary[valid_mask]

    if len(x_val) < 2 or y_val.nunique(dropna=True) < 2 or x_val.nunique(dropna=True) < 2:
        return 0.0

    if precomputed_bins is not None:
        bins = precomputed_bins[valid_mask]
    else:
        bins = _compute_quantile_bins(x_val, n_bins=n_bins)

    ct = pd.crosstab(bins, y_val)
    if ct.empty or 1 not in ct.columns or 0 not in ct.columns:
        return 0.0

    events = ct[1].values.astype(float)
    non_events = ct[0].values.astype(float)
    total_events = float(events.sum())
    total_non_events = float(non_events.sum())

    if total_events <= 0 or total_non_events <= 0:
        return 0.0

    dist_event = events / total_events
    dist_non_event = non_events / total_non_events

    p_e = np.maximum(dist_event, EPSILON)
    p_ne = np.maximum(dist_non_event, EPSILON)
    woe_k = np.log(p_e / p_ne)
    return float(np.sum((dist_event - dist_non_event) * woe_k))


def _prescreen_candidate_features(
    model_frame: pd.DataFrame,
    numeric_features: list[str],
    valid_targets: list[str],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    progress: bool = False,
    module_name: str = "probability_prim",
) -> list[str]:
    """Screen top candidate features based on max Information Value across targets."""
    feature_max_iv: dict[str, float] = {}

    show_progress = progress and len(numeric_features) > 20
    ps_bar = (
        ModuleProgress(
            f"{module_name} (prescreen)",
            total=len(numeric_features),
            unit="feat",
            enabled=True,
        )
        if show_progress
        else None
    )
    if ps_bar:
        ps_bar.__enter__()

    try:
        for f in numeric_features:
            max_iv_f = 0.0
            clean_x = _numeric_series(model_frame[f])
            bins = _compute_quantile_bins(clean_x, n_bins=10)

            for t in valid_targets:
                target_series = model_frame[t].dropna()
                unique_classes = target_series.unique()
                for c in unique_classes:
                    y_binary = (model_frame[t] == c).astype(int)
                    iv = _compute_1d_iv(clean_x, y_binary, n_bins=10, precomputed_bins=bins)
                    if iv > max_iv_f:
                        max_iv_f = iv
            feature_max_iv[f] = max_iv_f
            if ps_bar:
                ps_bar.step(f)
    finally:
        if ps_bar:
            ps_bar.__exit__(None, None, None)

    sorted_features = sorted(
        numeric_features, key=lambda f: feature_max_iv.get(f, 0.0), reverse=True
    )
    return sorted_features[:max_candidates]


def _compute_box_metrics(
    box_pos: int,
    box_total: int,
    total_pos: int,
    total_samples: int,
) -> dict[str, Any]:
    """Calculate statistical metrics for a PRIM box."""
    if box_total <= 0 or total_samples <= 0:
        return {
            "sample_count": 0,
            "support": 0.0,
            "target_positive_count": 0,
            "win_rate": 0.0,
            "baseline_rate": 0.0,
            "lift": 1.0,
            "p_value_fisher": 1.0,
            "credible_interval_low_95": 0.0,
            "credible_interval_high_95": 1.0,
        }

    box_neg = box_total - box_pos
    out_pos = max(0, total_pos - box_pos)
    out_neg = max(0, (total_samples - total_pos) - box_neg)

    support = box_total / total_samples
    win_rate = box_pos / box_total
    baseline_rate = total_pos / total_samples
    lift = win_rate / baseline_rate if baseline_rate > 0 else 1.0

    # Fisher exact test (greater alternative)
    contingency_table = [[box_pos, box_neg], [out_pos, out_neg]]
    try:
        _, p_value = stats.fisher_exact(contingency_table, alternative="greater")
    except Exception:
        p_value = 1.0

    # Bayesian 95% Credible Interval (Jeffreys Prior: Beta(0.5, 0.5))
    alpha_post = box_pos + 0.5
    beta_post = box_neg + 0.5
    ci_low = float(stats.beta.ppf(0.025, alpha_post, beta_post))
    ci_high = float(stats.beta.ppf(0.975, alpha_post, beta_post))

    return {
        "sample_count": int(box_total),
        "support": _round(support),
        "target_positive_count": int(box_pos),
        "win_rate": _round(win_rate),
        "baseline_rate": _round(baseline_rate),
        "lift": _round(lift),
        "p_value_fisher": _round(p_value) if p_value is not None else 1.0,
        "credible_interval_low_95": _round(ci_low),
        "credible_interval_high_95": _round(ci_high),
    }


def _score_peel_step(
    step: dict[str, Any], objective: str = DEFAULT_OBJECTIVE
) -> tuple[float, float, int]:
    """Score a peeling trajectory step according to the chosen objective."""
    m = step["metrics"]
    wr = float(m.get("win_rate") or 0.0)
    base = float(m.get("baseline_rate") or 0.0)
    lift = float(m.get("lift") or 1.0)
    n = int(step.get("sample_count") or 0)
    ci_low = float(m.get("credible_interval_low_95") or 0.0)

    if objective == "win_rate":
        return (wr, lift, n)
    elif objective == "edge_support":
        edge = max(0.0, wr - base) * n
        return (edge, wr, n)
    elif objective == "wilson_lower":
        return (ci_low, wr, n)
    else:  # "support_weighted"
        edge_sqrt = max(0.0, wr - base) * np.sqrt(max(1, n))
        return (float(edge_sqrt), wr, n)


def _patient_peel_box(
    df: pd.DataFrame,
    features: list[str],
    target_binary: pd.Series,
    alpha: float = DEFAULT_ALPHA,
    min_box_samples: int = DEFAULT_MIN_BOX_SAMPLES,
    expand_delta: float = DEFAULT_EXPAND_DELTA,
    objective: str = DEFAULT_OBJECTIVE,
) -> dict[str, Any] | None:
    """Run Patient Rule Induction Method (peeling and expansion) on feature subset."""
    valid_mask = target_binary.notna()
    for f in features:
        valid_mask &= df[f].notna()

    sub_df = df.loc[valid_mask, features].copy()
    sub_y = target_binary.loc[valid_mask].astype(int).to_numpy()

    n_samples = len(sub_df)
    if n_samples < min_box_samples or np.sum(sub_y) == 0:
        return None

    # Track current active sample indices in box
    active_indices = np.arange(n_samples)
    feature_arrays = {f: sub_df[f].to_numpy() for f in features}

    current_bounds: dict[str, tuple[float, float]] = {
        f: (float(feature_arrays[f].min()), float(feature_arrays[f].max())) for f in features
    }

    total_pos = int(np.sum(sub_y))

    trajectory: list[dict[str, Any]] = []

    # Record initial full dataset box
    init_metrics = _compute_box_metrics(total_pos, n_samples, total_pos, n_samples)
    trajectory.append(
        {
            "indices": active_indices.copy(),
            "bounds": {f: (b[0], b[1]) for f, b in current_bounds.items()},
            "win_rate": init_metrics["win_rate"],
            "sample_count": n_samples,
            "metrics": init_metrics,
        }
    )

    # Peeling Loop
    while len(active_indices) > min_box_samples:
        best_peel_candidate = None
        best_peel_mean = -1.0
        best_peel_support = 0

        current_n = len(active_indices)
        peel_count = max(1, int(np.ceil(alpha * current_n)))

        for f in features:
            f_vals = feature_arrays[f][active_indices]

            # Candidate 1: Peel lower boundary
            sorted_order = np.argsort(f_vals)
            cand_indices_low = active_indices[sorted_order[peel_count:]]

            if len(cand_indices_low) >= min_box_samples:
                mean_low = float(np.mean(sub_y[cand_indices_low]))
                if (mean_low > best_peel_mean) or (
                    np.isclose(mean_low, best_peel_mean)
                    and len(cand_indices_low) > best_peel_support
                ):
                    best_peel_mean = mean_low
                    best_peel_support = len(cand_indices_low)
                    new_lower = float(np.min(feature_arrays[f][cand_indices_low]))
                    cand_bounds = {f_k: (b[0], b[1]) for f_k, b in current_bounds.items()}
                    cand_bounds[f] = (new_lower, cand_bounds[f][1])
                    best_peel_candidate = {
                        "indices": cand_indices_low,
                        "bounds": cand_bounds,
                    }

            # Candidate 2: Peel upper boundary
            cand_indices_high = active_indices[sorted_order[:-peel_count]]
            if len(cand_indices_high) >= min_box_samples:
                mean_high = float(np.mean(sub_y[cand_indices_high]))
                if (mean_high > best_peel_mean) or (
                    np.isclose(mean_high, best_peel_mean)
                    and len(cand_indices_high) > best_peel_support
                ):
                    best_peel_mean = mean_high
                    best_peel_support = len(cand_indices_high)
                    new_upper = float(np.max(feature_arrays[f][cand_indices_high]))
                    cand_bounds = {f_k: (b[0], b[1]) for f_k, b in current_bounds.items()}
                    cand_bounds[f] = (cand_bounds[f][0], new_upper)
                    best_peel_candidate = {
                        "indices": cand_indices_high,
                        "bounds": cand_bounds,
                    }

        if best_peel_candidate is None or len(best_peel_candidate["indices"]) == len(
            active_indices
        ):
            break

        active_indices = best_peel_candidate["indices"]
        current_bounds = best_peel_candidate["bounds"]

        box_pos = int(np.sum(sub_y[active_indices]))
        box_n = len(active_indices)
        metrics = _compute_box_metrics(box_pos, box_n, total_pos, n_samples)

        trajectory.append(
            {
                "indices": active_indices.copy(),
                "bounds": {f: (b[0], b[1]) for f, b in current_bounds.items()},
                "win_rate": metrics["win_rate"],
                "sample_count": box_n,
                "metrics": metrics,
            }
        )

    # Select best box from trajectory (highest utility score with support >= min_box_samples)
    eligible = [step for step in trajectory if step["sample_count"] >= min_box_samples]
    if not eligible:
        eligible = trajectory

    eligible.sort(
        key=lambda s: _score_peel_step(s, objective=objective),
        reverse=True,
    )
    best_step = eligible[0]

    best_bounds = {f: (b[0], b[1]) for f, b in best_step["bounds"].items()}
    best_metrics = best_step["metrics"]

    # Box Expansion (Bottom-up Pasting to recover support while maintaining target win rate)
    target_threshold = (best_metrics["win_rate"] or 0.0) * (1.0 - expand_delta)

    for f in features:
        low, high = best_bounds[f]
        f_min = float(feature_arrays[f].min())
        f_max = float(feature_arrays[f].max())
        f_range = f_max - f_min if f_max > f_min else 1.0
        step_val = alpha * f_range

        # Try expanding lower
        cand_low = max(f_min, low - step_val)
        cand_mask = np.ones(n_samples, dtype=bool)
        for f_k in features:
            l_k = cand_low if f_k == f else best_bounds[f_k][0]
            u_k = best_bounds[f_k][1]
            cand_mask &= (feature_arrays[f_k] >= l_k) & (feature_arrays[f_k] <= u_k)

        cand_n = int(np.sum(cand_mask))
        if cand_n > 0:
            cand_pos = int(np.sum(sub_y[cand_mask]))
            cand_wr = cand_pos / cand_n
            if cand_wr >= target_threshold:
                best_bounds[f] = (cand_low, high)

        # Try expanding upper
        low, high = best_bounds[f]
        cand_high = min(f_max, high + step_val)
        cand_mask = np.ones(n_samples, dtype=bool)
        for f_k in features:
            l_k = best_bounds[f_k][0]
            u_k = cand_high if f_k == f else best_bounds[f_k][1]
            cand_mask &= (feature_arrays[f_k] >= l_k) & (feature_arrays[f_k] <= u_k)

        cand_n = int(np.sum(cand_mask))
        if cand_n > 0:
            cand_pos = int(np.sum(sub_y[cand_mask]))
            cand_wr = cand_pos / cand_n
            if cand_wr >= target_threshold:
                best_bounds[f] = (low, cand_high)

    # Re-evaluate final expanded box
    final_mask = np.ones(n_samples, dtype=bool)
    for f in features:
        final_mask &= (feature_arrays[f] >= best_bounds[f][0]) & (
            feature_arrays[f] <= best_bounds[f][1]
        )

    final_n = int(np.sum(final_mask))
    final_pos = int(np.sum(sub_y[final_mask]))
    final_metrics = _compute_box_metrics(final_pos, final_n, total_pos, n_samples)

    dim_label = f"{len(features)}D"
    feat_label = ", ".join(features)

    return {
        "dimension": dim_label,
        "features": feat_label,
        "features_list": features,
        "bounds": best_bounds,
        **final_metrics,
    }


def _format_bounds_condition(
    bounds: dict[str, tuple[float, float]], row_syntax: bool = False
) -> str:
    """Format bounds condition into human-readable or Python row-evaluable string."""
    clauses: list[str] = []
    for f, (low, high) in bounds.items():
        if row_syntax:
            if np.isclose(low, high):
                clauses.append(f"row.get('{f}', float('nan')) == {low:.4f}")
            else:
                clauses.append(f"{low:.4f} <= row.get('{f}', float('nan')) <= {high:.4f}")
        else:
            if np.isclose(low, high):
                clauses.append(f"{f} == {low:.4f}")
            else:
                clauses.append(f"{low:.4f} <= {f} <= {high:.4f}")
    return " and ".join(clauses)


def _extract_prim_rules_for_target(
    df: pd.DataFrame,
    candidate_features: list[str],
    target_series: pd.Series,
    target_name: str,
    alpha: float = DEFAULT_ALPHA,
    min_box_samples: int = DEFAULT_MIN_BOX_SAMPLES,
    expand_delta: float = DEFAULT_EXPAND_DELTA,
    objective: str = DEFAULT_OBJECTIVE,
) -> list[dict[str, Any]]:
    """Extract 1D, 2D, and 3D PRIM bump hunting rules for each class in target."""
    valid_classes = sorted(target_series.dropna().unique(), key=lambda v: str(v))
    discovered_rules: list[dict[str, Any]] = []

    # 1D combinations
    subsets_1d = [[f] for f in candidate_features]
    # 2D combinations
    subsets_2d = [list(p) for p in itertools.combinations(candidate_features, 2)]
    # 3D combinations (limit to top 20 to keep execution fast)
    subsets_3d = [list(t) for t in itertools.combinations(candidate_features[:8], 3)][:20]

    all_subsets = subsets_1d + subsets_2d + subsets_3d

    for c in valid_classes:
        target_binary = (target_series == c).astype(int)
        class_rules: list[dict[str, Any]] = []
        clean_c = (
            int(c)
            if isinstance(c, (int, np.integer))
            else (float(c) if isinstance(c, (float, np.floating)) else str(c))
        )

        for subset in all_subsets:
            rule_box = _patient_peel_box(
                df=df,
                features=subset,
                target_binary=target_binary,
                alpha=alpha,
                min_box_samples=min_box_samples,
                expand_delta=expand_delta,
                objective=objective,
            )
            if rule_box and (rule_box["lift"] or 0.0) > 1.05:
                rule_box["target"] = target_name
                rule_box["target_class"] = clean_c
                rule_box["bounds_condition"] = _format_bounds_condition(
                    rule_box["bounds"], row_syntax=False
                )
                rule_box["python_condition"] = _format_bounds_condition(
                    rule_box["bounds"], row_syntax=True
                )
                class_rules.append(rule_box)

        # Sort and deduplicate rules for this class
        class_rules.sort(
            key=lambda r: (
                r["lift"] or 0.0,
                r["win_rate"] or 0.0,
                r["support"] or 0.0,
            ),
            reverse=True,
        )

        # Keep top non-redundant rules (up to 15 per class)
        seen_conditions: set[str] = set()
        for r in class_rules:
            cond = r["bounds_condition"]
            if cond not in seen_conditions:
                seen_conditions.add(cond)
                discovered_rules.append(r)
            if len(seen_conditions) >= 15:
                break

    return discovered_rules


def _generate_python_rule_code(rules: list[dict[str, Any]]) -> str:
    """Generate standalone executable Python code implementing discovered PRIM decision rules."""
    lines = [
        '"""',
        "Auto-generated PRIM Decision Rules by fldataprofiler.probability_prim.",
        "Patient Rule Induction Method (Bump Hunting) for High-Probability Subspaces.",
        '"""',
        "from __future__ import annotations",
        "",
        "from typing import Any, Mapping",
        "import numpy as np",
        "import pandas as pd",
        "",
        "",
        "def predict_prim_rules(row: Mapping[str, Any]) -> int:",
        '    """',
        "    Evaluate PRIM bump-hunting rules sequentially on a dictionary-like data row.",
        "    Returns 1 if any high-probability rule condition is satisfied, else 0.",
        '    """',
    ]

    if not rules:
        lines.append("    # No rules discovered")
        lines.append("    return 0")
    else:
        for idx, rule in enumerate(rules, 1):
            rule_id = rule.get("rule_id", f"rule_{idx}")
            dim = rule.get("dimension", "")
            win_rate = rule.get("win_rate", 0.0)
            lift = rule.get("lift", 1.0)
            support = rule.get("support", 0.0)
            samples = rule.get("sample_count", 0)
            target = rule.get("target", "")
            target_class = rule.get("target_class", "")
            py_cond = rule.get("python_condition") or _format_bounds_condition(
                rule.get("bounds", {}), row_syntax=True
            )

            lines.append(
                f"    # {rule_id} ({dim}): Target={target}:{target_class} | WinRate={win_rate:.1%} | Lift={lift:.2f}x | Support={support:.1%} (N={samples})"
            )
            lines.append(f"    if {py_cond}:")
            lines.append("        return 1")
            lines.append("")
        lines.append("    return 0")

    lines.extend(
        [
            "",
            "",
            "def evaluate_prim_rules(df: pd.DataFrame) -> pd.Series:",
            '    """',
            "    Evaluate PRIM rules across an entire pandas DataFrame.",
            "    Returns a binary Series (1 if any rule matched, 0 otherwise).",
            '    """',
            "    if df.empty:",
            "        return pd.Series(dtype=int)",
            "    preds = df.apply(predict_prim_rules, axis=1)",
            "    return preds.astype(int)",
            "",
            "",
            "def get_prim_rules_metadata() -> list[dict[str, Any]]:",
            '    """Return metadata summary of all active PRIM rules."""',
            f"    return {repr([{'rule_id': r.get('rule_id', f'rule_{i+1}'), 'dimension': r.get('dimension'), 'features': r.get('features'), 'win_rate': r.get('win_rate'), 'lift': r.get('lift'), 'support': r.get('support'), 'sample_count': r.get('sample_count')} for i, r in enumerate(rules)])}",
            "",
        ]
    )

    return "\n".join(lines)


def _plot_prim_rules(
    model_frame: pd.DataFrame,
    rules_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot 2D scatter and sweet spot bounding box for top PRIM rules."""
    if model_frame.empty or rules_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No PRIM rules available for visualization", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Filter for 2D rules first, or fallback to any 2 features
    rules_2d = rules_df[rules_df["dimension"] == "2D"].head(4)
    if rules_2d.empty:
        rules_2d = rules_df.head(2)

    n_plots = len(rules_2d)
    if n_plots == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No 2D PRIM rules discovered", ha="center", va="center")
        ax.axis("off")
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path

    nrows = 1 if n_plots <= 2 else 2
    ncols = min(n_plots, 2)
    figsize = (8 * ncols, 6 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat_axes = axes.flatten()

    for i, (_, rule) in enumerate(rules_2d.iterrows()):
        ax = flat_axes[i]
        feat_list = [f.strip() for f in str(rule["features"]).split(",") if f.strip()]
        target_name = rule["target"]
        target_class = rule["target_class"]

        if len(feat_list) >= 2 and feat_list[0] in model_frame and feat_list[1] in model_frame:
            fx, fy = feat_list[0], feat_list[1]
            sub_df = model_frame[[fx, fy, target_name]].dropna()
            is_pos = sub_df[target_name] == target_class

            # Scatter background negative points
            ax.scatter(
                sub_df.loc[~is_pos, fx],
                sub_df.loc[~is_pos, fy],
                color="#94a3b8",
                alpha=0.4,
                s=18,
                label=f"Other ({target_name}!={target_class})",
            )
            # Scatter positive points
            ax.scatter(
                sub_df.loc[is_pos, fx],
                sub_df.loc[is_pos, fy],
                color="#2563eb",
                alpha=0.75,
                s=24,
                label=f"Positive ({target_name}={target_class})",
            )

            # Draw PRIM box if bounds exist
            bounds = rule.get("bounds")
            if isinstance(bounds, dict) and fx in bounds and fy in bounds:
                bx_min, bx_max = bounds[fx]
                by_min, by_max = bounds[fy]
                width = bx_max - bx_min
                height = by_max - by_min

                rect = patches.Rectangle(
                    (bx_min, by_min),
                    width,
                    height,
                    linewidth=2.5,
                    edgecolor="#dc2626",
                    facecolor="#fee2e2",
                    alpha=0.35,
                    label="PRIM Bump Box",
                )
                ax.add_patch(rect)
                # Border line
                rect_border = patches.Rectangle(
                    (bx_min, by_min),
                    width,
                    height,
                    linewidth=2.5,
                    edgecolor="#dc2626",
                    facecolor="none",
                )
                ax.add_patch(rect_border)

            win_rate = float(rule["win_rate"])
            lift = float(rule["lift"])
            samples = int(rule["sample_count"])
            p_val = float(rule["p_value_fisher"])

            ax.set_title(
                f"{rule['rule_id']}: {fx} × {fy}\nTarget: {target_name}={target_class} | Win Rate: {win_rate:.1%} ({lift:.2f}x Lift)\nSupport: {samples} samples | Fisher p={p_val:.2e}",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_xlabel(fx, fontsize=9)
            ax.set_ylabel(fy, fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, linestyle=":", alpha=0.6)
        else:
            ax.text(0.5, 0.5, f"Rule {rule['rule_id']} (1D/3D)", ha="center", va="center")
            ax.axis("off")

    for j in range(n_plots, len(flat_axes)):
        flat_axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _render_markdown(
    metadata: ProbabilityPrimRunMetadata,
    rules_df: pd.DataFrame,
) -> str:
    metadata_table = _markdown_table(
        pd.DataFrame(
            [
                {"Metric": "Module", "Value": metadata.module},
                {"Metric": "Created At", "Value": metadata.created_at},
                {"Metric": "Execution Time", "Value": metadata.execution_time},
                {"Metric": "Feature File", "Value": metadata.feature_csv},
                {"Metric": "Label File", "Value": metadata.label_csv},
                {"Metric": "Join Strategy", "Value": metadata.join_strategy},
                {"Metric": "Peeling Alpha (α)", "Value": f"{metadata.alpha:.2%}"},
                {"Metric": "Min Box Samples", "Value": metadata.min_box_samples},
                {"Metric": "Min Support", "Value": f"{metadata.min_support:.2%}"},
                {"Metric": "Optimization Objective", "Value": metadata.objective},
                {"Metric": "Candidate Features Screened", "Value": len(metadata.candidate_features)},
                {"Metric": "Targets Analyzed", "Value": ", ".join(metadata.targets)},
                {"Metric": "Evaluated Rows", "Value": metadata.model_rows},
                {"Metric": "PRIM Rules Discovered", "Value": metadata.rules_count},
            ]
        )
    )

    insights: list[str] = []
    if not rules_df.empty:
        best_lift = rules_df.sort_values("lift", ascending=False).iloc[0]
        insights.append(
            f"- **Top Bump Hunting Rule by Lift**: `{best_lift['rule_id']}` ({best_lift['dimension']}) achieved a **Win Rate of {best_lift['win_rate']:.1%}** ({best_lift['lift']:.2f}x Lift over baseline {best_lift['baseline_rate']:.1%}) with `{best_lift['sample_count']}` samples ({best_lift['support']:.1%} support) and Fisher exact p-value = `{best_lift['p_value_fisher']}`."
        )
        insights.append(
            f"- **Optimal Decision Condition**: `{best_lift['bounds_condition']}` (Target `{best_lift['target']}={best_lift['target_class']}`)."
        )
        insights.append(
            f"- **Bayesian Uncertainty**: Posterior 95% Credible Interval = `[{best_lift['credible_interval_low_95']:.1%}, {best_lift['credible_interval_high_95']:.1%}]`."
        )
    else:
        insights.append("- No high-probability PRIM bump boxes discovered with lift > 1.05.")

    insights_text = "\n".join(insights)

    display_cols = [
        "rule_id",
        "dimension",
        "features",
        "target",
        "target_class",
        "win_rate",
        "baseline_rate",
        "lift",
        "support",
        "sample_count",
        "credible_interval_low_95",
        "credible_interval_high_95",
        "p_value_fisher",
        "bounds_condition",
    ]
    existing_cols = [c for c in display_cols if c in rules_df.columns]
    rules_table = (
        _markdown_table(rules_df[existing_cols].head(25))
        if not rules_df.empty
        else "No rules discovered."
    )

    return f"""# Patient Rule Induction Method (PRIM & Bump Hunting) Report

## Executive Summary & Key Insights

{insights_text}

## Run Metadata

{metadata_table}

## Discovered High-Probability PRIM Decision Rules

{rules_table}

## Visual Sweet Spot Bump Hunting Plot

![PRIM Bump Hunting Plot](prim_rules_plot.png)

## Executable Python Code Integration

A ready-to-run Python rule evaluator module has been generated at `rule_code_python.py`.

```python
from rule_code_python import evaluate_prim_rules, predict_prim_rules

# Real-time scoring on a single record
signal = predict_prim_rules({{"feature_1": 1.25, "feature_2": -0.42}})

# Vectorized scoring on DataFrame
df["prim_signal"] = evaluate_prim_rules(df)
```

## Artifacts

- `summary.json`
- `prim_rules.csv`
- `rule_code_python.py`
- `prim_rules_plot.png`
- `report.html`
"""


def _render_html(
    metadata: ProbabilityPrimRunMetadata,
    markdown: str,
    rules_df: pd.DataFrame,
) -> str:
    display_cols = [
        "rule_id",
        "dimension",
        "features",
        "target",
        "target_class",
        "win_rate",
        "baseline_rate",
        "lift",
        "support",
        "sample_count",
        "credible_interval_low_95",
        "credible_interval_high_95",
        "p_value_fisher",
        "bounds_condition",
    ]
    existing_cols = [c for c in display_cols if c in rules_df.columns]
    rules_html = (
        rules_df[existing_cols].head(30).to_html(index=False, classes="data-table")
        if not rules_df.empty
        else "<p>No PRIM rules discovered.</p>"
    )

    max_win_rate = (
        f"{rules_df['win_rate'].max():.1%}"
        if not rules_df.empty and rules_df["win_rate"].max() is not None
        else "N/A"
    )
    max_lift = (
        f"{rules_df['lift'].max():.2f}x"
        if not rules_df.empty and rules_df["lift"].max() is not None
        else "N/A"
    )
    total_rules = str(len(rules_df))
    best_p_val = (
        f"{rules_df['p_value_fisher'].min():.2e}"
        if not rules_df.empty and rules_df["p_value_fisher"].min() is not None
        else "N/A"
    )

    details = _html_markdown_details(markdown)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>PRIM Bump Hunting & Patient Rule Induction Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            margin: 2rem auto;
            max-width: 1200px;
            padding: 0 1.5rem;
            color: #1e293b;
            background-color: #f8fafc;
        }}
        h1, h2, h3 {{ color: #0f172a; margin-top: 1.5rem; }}
        .card {{
            background: #ffffff;
            border-radius: 8px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            border: 1px solid #e2e8f0;
        }}
        .metrics-banner {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        .metric-card {{
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }}
        .metric-card.accent {{
            background: #f0fdf4;
            border-color: #bbf7d0;
        }}
        .metric-card.warning {{
            background: #fefce8;
            border-color: #fef08a;
        }}
        .metric-title {{ font-size: 0.8rem; color: #64748b; text-transform: uppercase; font-weight: 700; }}
        .metric-value {{ font-size: 1.6rem; color: #0f172a; font-weight: 700; margin-top: 0.25rem; }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }}
        .meta-item {{
            background: #f1f5f9;
            padding: 0.75rem 1rem;
            border-radius: 6px;
        }}
        .meta-label {{ font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: 600; }}
        .meta-val {{ font-size: 1rem; color: #0f172a; font-weight: 500; margin-top: 0.25rem; }}
        .table-container {{
            overflow-x: auto;
            margin: 1rem 0;
        }}
        table.data-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.88rem;
        }}
        table.data-table th, table.data-table td {{
            border: 1px solid #e2e8f0;
            padding: 8px 12px;
            text-align: left;
        }}
        table.data-table th {{
            background-color: #f8fafc;
            font-weight: 600;
            color: #334155;
        }}
        table.data-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .chart-img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            margin-top: 1rem;
        }}
        pre.code-box {{
            background: #0f172a;
            color: #f8fafc;
            padding: 1.2rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <h1>Patient Rule Induction Method (PRIM & Bump Hunting) Report</h1>

    <div class="metrics-banner">
        <div class="metric-card accent">
            <div class="metric-title">Peak Win Rate</div>
            <div class="metric-value">{max_win_rate}</div>
        </div>
        <div class="metric-card warning">
            <div class="metric-title">Max Lift</div>
            <div class="metric-value">{max_lift}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Rules Discovered</div>
            <div class="metric-value">{total_rules}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Best Fisher p-value</div>
            <div class="metric-value">{best_p_val}</div>
        </div>
    </div>

    <div class="card">
        <h2>Run Overview</h2>
        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-label">Module</div>
                <div class="meta-val">{metadata.module}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Execution Time</div>
                <div class="meta-val">{metadata.execution_time}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Peeling Rate (α)</div>
                <div class="meta-val">{metadata.alpha:.1%}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Min Box Samples</div>
                <div class="meta-val">{metadata.min_box_samples} samples ({metadata.min_support:.1%})</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Optimization Objective</div>
                <div class="meta-val">{metadata.objective}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Evaluated Rows</div>
                <div class="meta-val">{metadata.model_rows}</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Discovered High-Probability PRIM Decision Rules</h2>
        <div class="table-container">
            {rules_html}
        </div>
    </div>

    <div class="card">
        <h2>Visual Sweet Spot Bump Hunting Plot</h2>
        <img class="chart-img" src="prim_rules_plot.png" alt="PRIM Rules Visualization"/>
    </div>

    <div class="card">
        <h2>Executable Python Code Integration</h2>
        <p>A ready-to-run Python rule evaluator module has been generated at <code>rule_code_python.py</code>.</p>
        <pre class="code-box"><code>from rule_code_python import predict_prim_rules, evaluate_prim_rules

# Real-time single row prediction (returns 1 if rule triggered, 0 otherwise)
signal = predict_prim_rules({{'f1': 1.5, 'f2': 2.3}})

# Batch DataFrame evaluation
df['prim_rule_signal'] = evaluate_prim_rules(df)</code></pre>
    </div>

    {details}
</body>
</html>
"""


class ProbabilityPrimModule:
    name = "probability_prim"

    def __init__(
        self,
        config: ProbabilityPrimConfig | None = None,
        progress: bool | None = None,
        alpha: float | None = None,
        min_box_samples: int | None = None,
        min_support: float | None = None,
        max_candidates: int | None = None,
        expand_delta: float | None = None,
        objective: str | None = None,
    ) -> None:
        self.progress = progress
        if config is not None:
            base_cfg = config
        else:
            mod_cfg = get_module_config("probability_prim")
            base_cfg = ProbabilityPrimConfig(
                min_box_samples=int(mod_cfg.get("min_box_samples", DEFAULT_MIN_BOX_SAMPLES)),
                min_support=float(mod_cfg.get("min_support", DEFAULT_MIN_SUPPORT)),
                alpha=float(mod_cfg.get("alpha", DEFAULT_ALPHA)),
                max_candidates=int(mod_cfg.get("max_candidates", DEFAULT_MAX_CANDIDATES)),
                expand_delta=float(mod_cfg.get("expand_delta", DEFAULT_EXPAND_DELTA)),
                objective=str(mod_cfg.get("objective", DEFAULT_OBJECTIVE)),
            )

        # Check environment variable overrides for easy runtime tweaking
        env_min_samples = os.environ.get("PRIM_MIN_SAMPLES")
        env_min_support = os.environ.get("PRIM_MIN_SUPPORT")
        env_objective = os.environ.get("PRIM_OBJECTIVE")
        env_alpha = os.environ.get("PRIM_ALPHA")
        env_expand_delta = os.environ.get("PRIM_EXPAND_DELTA")
        env_max_candidates = os.environ.get("PRIM_MAX_CANDIDATES")

        self.alpha = (
            alpha
            if alpha is not None
            else (float(env_alpha) if env_alpha else base_cfg.alpha)
        )
        self.min_box_samples = (
            min_box_samples
            if min_box_samples is not None
            else (int(env_min_samples) if env_min_samples else base_cfg.min_box_samples)
        )
        self.min_support = (
            min_support
            if min_support is not None
            else (float(env_min_support) if env_min_support else base_cfg.min_support)
        )
        self.max_candidates = (
            max_candidates
            if max_candidates is not None
            else (
                int(env_max_candidates)
                if env_max_candidates
                else base_cfg.max_candidates
            )
        )
        self.expand_delta = (
            expand_delta
            if expand_delta is not None
            else (
                float(env_expand_delta)
                if env_expand_delta
                else base_cfg.expand_delta
            )
        )
        self.objective = (
            objective
            if objective is not None
            else (env_objective if env_objective else base_cfg.objective)
        )

    def run(
        self,
        feature_csv: str | Path,
        label_csv: str | Path,
        output_dir: str | Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        start_time = time.perf_counter()
        feature_path = Path(feature_csv)
        label_path = Path(label_csv)
        run_dir = Path(output_dir) / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        with StatusTimer(f"{self.name}: Loading & pre-screening", enabled=self.progress):
            features = _read_table_with_date_index(feature_path)
            labels = _read_table_with_date_index(label_path)
            merged, feature_columns, label_columns, join_strategy = _merge_inputs(
                features, labels, join_key
            )

            ignored_columns = set(_date_columns([*feature_columns, *label_columns]))
            feature_columns = [col for col in feature_columns if col not in ignored_columns]
            label_columns = [col for col in label_columns if col not in ignored_columns]

            selected_targets = _select_targets(label_columns, targets)
            numeric_features = _numeric_feature_columns(
                merged, feature_columns, min_non_null=MIN_NON_NULL
            )

            # Filter valid discrete targets
            valid_targets: list[str] = []
            for target_col in selected_targets:
                clean_target = merged[target_col].dropna()
                if 2 <= clean_target.nunique() <= MAX_LABEL_CLASSES:
                    valid_targets.append(target_col)

            model_frame = (
                _sample_rows(
                    merged[[*numeric_features, *valid_targets]],
                    MAX_ROWS,
                    RANDOM_STATE,
                )
                if valid_targets and numeric_features
                else merged
            )

            # Candidate Feature Screening
            candidate_features = _prescreen_candidate_features(
                model_frame,
                numeric_features,
                valid_targets,
                max_candidates=self.max_candidates,
            )

        all_rules: list[dict[str, Any]] = []

        # Adjust min_box_samples adaptively based on dataset size and configured min_support
        eff_min_samples = max(
            10,
            max(self.min_box_samples, int(self.min_support * len(model_frame))),
        )
        # Ensure feasibility on small synthetic test datasets
        eff_min_samples = min(eff_min_samples, max(10, int(len(model_frame) * 0.15)))

        total_evals = len(valid_targets)
        with ModuleProgress(
            self.name, total=max(1, total_evals), unit="target", enabled=self.progress
        ) as progress_bar:
            for target_col in valid_targets:
                rules_t = _extract_prim_rules_for_target(
                    df=model_frame,
                    candidate_features=candidate_features,
                    target_series=model_frame[target_col],
                    target_name=target_col,
                    alpha=self.alpha,
                    min_box_samples=eff_min_samples,
                    expand_delta=self.expand_delta,
                    objective=self.objective,
                )
                all_rules.extend(rules_t)
                progress_bar.step(f"PRIM->{target_col}")

            if total_evals == 0:
                progress_bar.step("no_valid_targets")

        # Rank and assign rule_id
        all_rules.sort(
            key=lambda r: (
                r["lift"] or 0.0,
                r["win_rate"] or 0.0,
                r["support"] or 0.0,
            ),
            reverse=True,
        )
        for idx, r in enumerate(all_rules, 1):
            r["rule_id"] = f"rule_{idx}"

        rules_df = pd.DataFrame(all_rules)

        # Generate Artifacts
        plot_path = run_dir / "prim_rules_plot.png"
        _plot_prim_rules(model_frame, rules_df, plot_path)

        python_code_str = _generate_python_rule_code(all_rules)
        python_code_path = run_dir / "rule_code_python.py"
        python_code_path.write_text(python_code_str, encoding="utf-8")

        rules_csv_path = _write_csv(
            run_dir / "prim_rules.csv",
            rules_df.drop(columns=["indices", "bounds", "python_condition"], errors="ignore"),
        )

        metadata = ProbabilityPrimRunMetadata(
            module=self.name,
            created_at=datetime.now(UTC).isoformat(),
            execution_time=_format_duration(time.perf_counter() - start_time),
            feature_csv=str(feature_csv),
            label_csv=str(label_csv),
            join_strategy=join_strategy,
            feature_shape=DatasetShape(*features.shape),
            label_shape=DatasetShape(*labels.shape),
            merged_shape=DatasetShape(*merged.shape),
            alpha=self.alpha,
            min_box_samples=eff_min_samples,
            min_support=self.min_support,
            objective=self.objective,
            max_candidates=self.max_candidates,
            features_count=len(numeric_features),
            candidate_features=candidate_features,
            targets=valid_targets,
            model_rows=len(model_frame),
            rules_count=len(all_rules),
        )

        def _clean_rule_for_json(r: dict[str, Any]) -> dict[str, Any]:
            return {
                "rule_id": str(r.get("rule_id", "")),
                "dimension": str(r.get("dimension", "")),
                "features": str(r.get("features", "")),
                "target": str(r.get("target", "")),
                "target_class": r.get("target_class"),
                "bounds_condition": str(r.get("bounds_condition", "")),
                "sample_count": int(r.get("sample_count", 0)),
                "support": float(r.get("support", 0.0)),
                "target_positive_count": int(r.get("target_positive_count", 0)),
                "win_rate": float(r.get("win_rate", 0.0)),
                "baseline_rate": float(r.get("baseline_rate", 0.0)),
                "lift": float(r.get("lift", 1.0)),
                "p_value_fisher": float(r.get("p_value_fisher", 1.0)),
                "credible_interval_low_95": float(r.get("credible_interval_low_95", 0.0)),
                "credible_interval_high_95": float(r.get("credible_interval_high_95", 1.0)),
            }

        summary_payload: dict[str, Any] = {
            **asdict(metadata),
            "top_rules": [_clean_rule_for_json(r) for r in all_rules[:10]]
            if all_rules
            else [],
            "summary_metrics": {
                "rules_discovered": len(all_rules),
                "max_win_rate": _round(float(rules_df["win_rate"].max()))
                if not rules_df.empty
                else 0.0,
                "max_lift": _round(float(rules_df["lift"].max()))
                if not rules_df.empty
                else 1.0,
                "best_rule_support": _round(float(rules_df["support"].iloc[0]))
                if not rules_df.empty
                else 0.0,
                "best_rule_p_value": _round(float(rules_df["p_value_fisher"].iloc[0]))
                if not rules_df.empty
                else 1.0,
            },
        }
        summary_json_path = _write_json(run_dir / "summary.json", summary_payload)

        markdown_report = _render_markdown(metadata, rules_df)
        report_md_path = run_dir / "report.md"
        report_md_path.write_text(markdown_report, encoding="utf-8")

        html_report = _render_html(metadata, markdown_report, rules_df)
        report_html_path = run_dir / "report.html"
        report_html_path.write_text(html_report, encoding="utf-8")

        artifacts = [
            summary_json_path,
            rules_csv_path,
            python_code_path,
            plot_path,
            report_md_path,
            report_html_path,
        ]

        return ModuleResult(report_dir=run_dir, artifacts=artifacts)
