# Profiling Module Contract & Architecture Reference

Every profiling module in `fl-data-profiling` must strictly adhere to the following architecture standards, contracts, and lifecycle patterns.

---

## 1. Module Protocol Interface

Modules are defined under `src/fldataprofiler/modules/<module_name>.py` and must implement `ProfilingModule` protocol:

```python
from __future__ import annotations
from pathlib import Path
from fldataprofiler.modules.base import ModuleResult

class ExampleModule:
    name: str = "example_module"
    description: str = "Brief 1-line description of what this module does"

    def run(
        self,
        feature_csv: Path,
        label_csv: Path,
        output_dir: Path,
        join_key: str | None = None,
        targets: list[str] | None = None,
    ) -> ModuleResult:
        ...
        return ModuleResult(report_dir=run_dir, artifacts=artifacts)
```

---

## 2. Standard 6-Artifact Output Contract

Every module execution must produce artifacts directly into `reports/<module_name>/`:

| Artifact | File Type | Purpose | Mandatory Content |
| :--- | :--- | :--- | :--- |
| `report.md` | Markdown | GitHub-flavored summary report | Executive summary, tables, decision rules, chart embed `![](chart.png)` |
| `report.html` | Interactive HTML | Standalone web report | Clean CSS, summary KPI metric cards, data tables, responsive images |
| `summary.json` | JSON | Metadata & Top Ranked Features/Rules | `metadata` (created_at, execution_time, shapes, parameters) + `top_results` |
| `*.csv` | CSV Tabular | Primary scoring table | E.g., `feature_scores.csv`, `pair_scores.csv` with standardized metrics |
| `*.csv` (secondary) | CSV Tabular | Detailed breakdown / bins / cells | E.g., `quantile_crosstab_probabilities.csv`, `cell_probabilities.csv` |
| `*.png` | Static Image (150+ DPI) | High-res empirical visualizations | Bar charts, heatmaps, ROC curves, or distribution charts |

---

## 3. Row Limit & Full Mode (`--full`)

1. Define `MAX_ROWS = 50_000` as default downsampling cap.
2. Ingest and merge features and labels via `_merge_inputs`.
3. Subsample data using `_sample_rows`:
   ```python
   sampled_df = _sample_rows(merged, max_rows=MAX_ROWS, random_state=RANDOM_STATE)
   ```
   `_sample_rows` automatically inspects `_FULL_ROW_MODE.get()`. When `--full` is specified in CLI, downsampling is bypassed.

---

## 4. Progress Reporting Protocol

Use `ModuleProgress` with 4 standard stages:
```python
with ModuleProgress(self.name, total=4) as progress_bar:
    progress_bar.step("Ingesting & pre-screening features")
    ...
    progress_bar.step("Computing scoring metrics")
    ...
    progress_bar.step("Rendering charts & visualizations")
    ...
    progress_bar.step("Writing reports and summaries")
```

---

## 5. Configuration Protocol

1. Register default parameters in `src/fldataprofiler/config.default.json`:
   ```json
   "example_module": {
     "n_quantiles": 20,
     "min_probability": 0.55,
     "min_support": 20
   }
   ```
2. In Python module, retrieve configs safely via `get_module_config`:
   ```python
   raw_cfg = get_module_config("example_module")
   cfg = ExampleConfig(
       n_quantiles=int(raw_cfg.get("n_quantiles", DEFAULT_N_QUANTILES)),
       min_probability=float(raw_cfg.get("min_probability", DEFAULT_MIN_PROBABILITY)),
   )
   ```

---

## 6. Registry & Aliases

In `src/fldataprofiler/registry.py`:
1. Import module:
   ```python
   from fldataprofiler.modules.example_module import ExampleModule
   ```
2. Add aliases in `_MODULES` dictionary:
   ```python
   "example_module": ExampleModule,
   "examplemodule": ExampleModule,
   "example": ExampleModule,
   ```
