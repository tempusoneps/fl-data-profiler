# Project Rules & Development Conventions

## 1. Keep Structure & Registry In Sync
Any change that adds, removes, or renames profiling modules in `src/fldataprofiler/modules/` must update:
- `src/fldataprofiler/registry.py` (Registry dictionary and module loader).
- `docs/.ai/STRUCTURE.md` and `docs/README.md`.
- Matching documentation file in `docs/<module>.md`.

## 2. Standardized Module Interface
All profiling modules must implement the `ProfilingModule` interface:
```python
def run(
    self,
    feature_csv: Path,
    label_csv: Path,
    output_dir: Path,
    join_key: str | None = None,
    targets: list[str] | None = None,
) -> ModuleResult:
    ...
```
And return `ModuleResult(report_dir=run_dir, artifacts=artifacts)`.

## 3. Standard Artifact Output Contract
Every module must generate the following standard artifacts in `reports/<module>/`:
- `report.md`: Markdown summary report.
- `report.html`: Interactive web report.
- `summary.json`: JSON metadata and top metrics.
- `*.csv`: Tabular scoring data (e.g. `feature_scores.csv`, `top_features.csv`, `kmean_results.csv`).

## 4. Full vs Subsampled Data Respect
Modules that utilize internal row limits (`MAX_ROWS`) must check `_FULL_ROW_MODE.get()` from `fldataprofiler.utils` so that `--full` correctly disables downsampling across all stages.

## 5. Run Targeted Tests First
When developing or debugging, run targeted tests (e.g. `uv run pytest tests/test_input_formats.py -q`). Do not run heavy end-to-end extraction tests repeatedly unless verifying full release readiness.

## 6. Do Not Commit Automatically
AI agents must **NOT** execute `git commit` commands automatically without explicit confirmation and approval from the user.\n