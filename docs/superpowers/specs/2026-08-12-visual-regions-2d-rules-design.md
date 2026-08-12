# Visual Regions 2D Rules Mining Module Design

**Date**: 2026-08-12  
**Module Name**: `visual_regions`  
**Status**: Approved  

---

## 1. Overview & Goal

The `visual_regions` module identifies patterns and rules formed by combinations of two features with target labels. It partitions the 2D space of candidate feature pairs into quantile grid cells, measures label purity, merges contiguous high-purity cells into rectangular range rules ($[X_{min}, X_{max}] \times [Y_{min}, Y_{max}]$), and ranks the resulting rules.

The module outputs structured reports in CSV, JSON, Markdown, and HTML formats containing human-readable decision rules:
`IF Feature_X ∈ [a, b] AND Feature_Y ∈ [c, d] THEN Label = Z (Purity: P%, Support: N, Lift: L x)`

---

## 2. CLI Registration & Configuration

The module is registered in `fldataprofier/registry.py` under the name `"visual_regions"`.

### Configuration Parameters
- `n_bins` (default `8`): Number of quantile bins per numeric feature.
- `max_candidate_features` (default `16`): Top 1D features selected for 2D pair combinations.
- `max_pairs` (default `30`): Maximum feature pairs evaluated.
- `min_samples_per_region` (default `15`): Minimum samples required in a 2D region.
- `min_purity` (default `0.70`): Minimum purity threshold to qualify as a decision rule.
- `random_state` (default `42`): Seed for random sampling if needed.

---

## 3. Algorithm & Pipeline

1. **Preprocessing & 1D Candidate Scoring**:
   - Filter numeric columns, remove high-missing columns ($> 50\%$) or low-variance columns.
   - Bin features into quantile bins using `pd.qcut` with rank-based binning.
   - Compute 1D purity gain for each feature against labels and select the top `max_candidate_features`.

2. **2D Grid Purity Calculation**:
   - Build $N \times N$ contingency tables for each feature pair $(Feature_X, Feature_Y)$.
   - For each cell $(b_x, b_y)$, calculate:
     - `sample_count`: Total observations in cell.
     - `majority_label`: Label with highest frequency in cell.
     - `purity`: Fraction of majority label samples in cell.
     - `lift`: `purity / global_label_prior`.

3. **Contiguous Region Merging**:
   - Identify cells where `purity >= min_purity` and `sample_count >= min_samples_per_region`.
   - Merge adjacent cells sharing the same `majority_label` into rectangular bounds $[X_{min}, X_{max}] \times [Y_{min}, Y_{max}]$.
   - Re-aggregate total samples, overall purity, and lift for merged regions.

4. **Rule Ranking**:
   - Rank merged rules by composite score:
     $$\text{Score} = \text{Purity} \times \log_2(\text{Sample Count}) \times \text{Lift}$$

---

## 4. Output Artifacts & Report Structures

Generated files in `reports/<run_id>/visual_regions/`:

1. **`summary.json`**:
   - Metadata, shape, 1D candidate scores, and complete list of extracted 2D rules.

2. **`rules_2d.csv`**:
   - Table columns: `rank`, `feature_x`, `feature_y`, `range_x_min`, `range_x_max`, `range_y_min`, `range_y_max`, `target_label`, `purity_pct`, `sample_count`, `coverage_pct`, `lift_ratio`, `rule_text`.

3. **`report.md`**:
   - Executive summary of findings.
   - Formatted Markdown table of top 20 2D Region Rules (`IF ... THEN ...`).
   - Top predictive feature pairs table.

4. **`report.html`**:
   - Styled HTML report with embedded Markdown details and visual HTML/CSS rendering for top 2D rules and purity grids.

---

## 5. Verification Plan

- Run unit tests in `tests/test_visual_regions.py`.
- Verify module execution via `fldataprofier` CLI on synthetic/sample datasets.
- Ensure output files `summary.json`, `rules_2d.csv`, `report.md`, and `report.html` are created with correct content.
