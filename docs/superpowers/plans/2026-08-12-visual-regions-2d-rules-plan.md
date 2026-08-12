# Visual Regions 2D Rules Mining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `visual_regions` module in `fl-data-profiler` to extract, rank, and report 2D Region Purity decision rules (`IF Feature_X ∈ [a, b] AND Feature_Y ∈ [c, d] THEN Label = Z`) in CSV, JSON, Markdown, and HTML formats.

**Architecture:** Extend `fldataprofier/modules/visual_regions.py` with 2D quantile grid partitioning, contiguous cell region merging, rule scoring ($\text{Purity} \times \log_2(N) \times \text{Lift}$), and HTML/Markdown report exporters. Register the module in `fldataprofier/registry.py`.

**Tech Stack:** Python 3.10+, Pandas, NumPy, Scikit-learn (SimpleImputer), Pytest.

## Global Constraints

- Module CLI name: `visual_regions`
- Quantile Bins default: `8`
- Min samples per region default: `15`
- Min purity threshold default: `0.70`
- Target output directory: `output_dir / "visual_regions"`

---

### Task 1: Implement 2D Grid Binning, Region Merging, and Rule Ranking in `visual_regions.py`

**Files:**
- Modify: `fldataprofier/modules/visual_regions.py`
- Test: `tests/test_visual_regions.py`

**Interfaces:**
- Consumes: `fldataprofier.utils._merge_inputs`, `_read_table_with_date_index`, `_numeric_series`, `_round`, `_write_csv`, `_write_json`
- Produces: `_evaluate_2d_grid_purity`, `_merge_contiguous_regions`, `_extract_2d_rules`

- [ ] **Step 1: Write failing tests for 2D Grid Purity & Region Merging**

Create `tests/test_visual_regions.py`:
```python
import numpy as np
import pandas as pd
import pytest

from fldataprofier.modules.visual_regions import (
    _evaluate_2d_grid_purity,
    _merge_contiguous_regions,
    _extract_2d_rules,
)


def test_evaluate_2d_grid_purity():
    # Synthetic 2D data where X > 0.5 and Y > 0.5 is class 1
    df = pd.DataFrame(
        {
            "x_bin": [0, 0, 0, 1, 1, 1, 1, 1],
            "y_bin": [0, 0, 0, 1, 1, 1, 1, 1],
            "x_val": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9],
            "y_val": [0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.85, 0.9],
            "label": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )
    grid = _evaluate_2d_grid_purity(df, "x_bin", "y_bin", "x_val", "y_val", "label")
    assert not grid.empty
    assert "purity" in grid.columns


def test_extract_2d_rules():
    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 10, n)
    y = np.random.uniform(0, 10, n)
    # Target rule: x > 5 and y > 5 => label 1
    label = np.where((x > 5) & (y > 5), 1, 0)
    merged_df = pd.DataFrame({"feat_x": x, "feat_y": y, "target": label})

    rules_df = _extract_2d_rules(
        merged=merged_df,
        feature_columns=["feat_x", "feat_y"],
        label_columns=["target"],
        n_bins=4,
        min_samples=5,
        min_purity=0.65,
    )
    assert not rules_df.empty
    assert "rule_text" in rules_df.columns
```

- [ ] **Step 2: Run pytest to verify tests fail**

Run: `uv run pytest tests/test_visual_regions.py -v`
Expected: FAIL (`ImportError` or `NameError` for missing functions)

- [ ] **Step 3: Implement 2D Grid Binning, Cell Purity Evaluation, and Contiguous Region Merging in `visual_regions.py`**

In `fldataprofier/modules/visual_regions.py`:
Implement:
1. `_evaluate_2d_grid_purity(df, x_bin_col, y_bin_col, x_val_col, y_val_col, label_col)`:
   Computes grid cell counts, majority labels, cell purity, and lift over global class distribution.
2. `_merge_contiguous_regions(...)`:
   Groups adjacent high-purity cells (`purity >= min_purity`, `samples >= min_samples`) sharing the same target label into rectangular bounds $[X_{min}, X_{max}] \times [Y_{min}, Y_{max}]$.
3. `_extract_2d_rules(merged, feature_columns, label_columns, n_bins, min_samples, min_purity)`:
   Orchestrates 1D candidate selection, 2D pair evaluation, region merging, rule score computation ($\text{Purity} \times \log_2(N) \times \text{Lift}$), and string rule formatting (`IF ... THEN ...`).

- [ ] **Step 4: Run pytest to verify tests pass**

Run: `uv run pytest tests/test_visual_regions.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add fldataprofier/modules/visual_regions.py tests/test_visual_regions.py
git commit -m "feat(visual_regions): add 2D grid purity evaluation and region merging logic"
```

---

### Task 2: Implement `VisualRegionsModule` class and Artifact Exporters (`rules_2d.csv`, `report.md`, `report.html`)

**Files:**
- Modify: `fldataprofier/modules/visual_regions.py`
- Test: `tests/test_visual_regions.py`

**Interfaces:**
- Consumes: `_extract_2d_rules`, `ModuleProgress`, `ModuleResult`
- Produces: `VisualRegionsModule.run()`, `_render_rules_markdown()`, `_render_rules_html()`

- [ ] **Step 1: Write failing test for `VisualRegionsModule.run()`**

Add to `tests/test_visual_regions.py`:
```python
from pathlib import Path
from fldataprofier.modules.visual_regions import VisualRegionsModule


def test_visual_regions_module_run(tmp_path: Path):
    # Create sample CSVs
    np.random.seed(42)
    n = 60
    feat_df = pd.DataFrame({
        "id": range(n),
        "f1": np.random.uniform(0, 100, n),
        "f2": np.random.uniform(0, 10, n),
    })
    label_df = pd.DataFrame({
        "id": range(n),
        "target": np.where((feat_df["f1"] > 50) & (feat_df["f2"] > 5), 1, 0),
    })

    feat_path = tmp_path / "features.csv"
    label_path = tmp_path / "labels.csv"
    out_dir = tmp_path / "output"

    feat_df.to_csv(feat_path, index=False)
    label_df.to_csv(label_path, index=False)

    module = VisualRegionsModule(n_bins=4, min_samples_per_region=5, min_purity=0.60, progress=False)
    result = module.run(feat_path, label_path, out_dir, join_key="id")

    assert result.name == "visual_regions"
    assert (out_dir / "visual_regions" / "summary.json").exists()
    assert (out_dir / "visual_regions" / "rules_2d.csv").exists()
    assert (out_dir / "visual_regions" / "report.md").exists()
    assert (out_dir / "visual_regions" / "report.html").exists()
```

- [ ] **Step 2: Run pytest to verify test fails**

Run: `uv run pytest tests/test_visual_regions.py::test_visual_regions_module_run -v`
Expected: FAIL (`AttributeError` or missing `run` method / artifacts)

- [ ] **Step 3: Implement `VisualRegionsModule` class and exporters in `visual_regions.py`**

In `fldataprofier/modules/visual_regions.py`:
1. Define class `VisualRegionsModule`:
   ```python
   class VisualRegionsModule:
       name = "visual_regions"
       def __init__(self, n_bins: int = 8, min_samples_per_region: int = 15, min_purity: float = 0.70, progress: bool | None = None) -> None: ...
       def run(self, feature_csv: Path, label_csv: Path, output_dir: Path, join_key: str | None = None, targets: list[str] | None = None) -> ModuleResult: ...
   ```
2. Implement `_render_rules_markdown(...)` to output Executive Summary, Top 20 Rules Table, and Feature Pairs Table.
3. Implement `_render_rules_html(...)` wrapping Markdown & styled CSS tables.

- [ ] **Step 4: Run pytest to verify test passes**

Run: `uv run pytest tests/test_visual_regions.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add fldataprofier/modules/visual_regions.py tests/test_visual_regions.py
git commit -m "feat(visual_regions): add VisualRegionsModule class and report generators"
```

---

### Task 3: Register `visual_regions` in `registry.py` & End-to-End CLI Verification

**Files:**
- Modify: `fldataprofier/registry.py`
- Modify: `tests/test_visual_regions.py`

**Interfaces:**
- Consumes: `VisualRegionsModule`
- Produces: `_MODULES["visual_regions"]` in `fldataprofier.registry`

- [ ] **Step 1: Write test verifying registry inclusion**

Add to `tests/test_visual_regions.py`:
```python
from fldataprofier.registry import get_module, list_modules


def test_registry_contains_visual_regions():
    assert "visual_regions" in list_modules()
    module = get_module("visual_regions")
    assert module.name == "visual_regions"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_visual_regions.py::test_registry_contains_visual_regions -v`
Expected: FAIL (`Unknown module 'visual_regions'`)

- [ ] **Step 3: Register `visual_regions` in `fldataprofier/registry.py`**

In `fldataprofier/registry.py`:
Import `VisualRegionsModule` and add `"visual_regions": VisualRegionsModule()` to `_MODULES`.

- [ ] **Step 4: Run full test suite to verify everything passes**

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 5: Commit changes**

```bash
git add fldataprofier/registry.py tests/test_visual_regions.py
git commit -m "feat(registry): register visual_regions module in fldataprofier registry"
```
