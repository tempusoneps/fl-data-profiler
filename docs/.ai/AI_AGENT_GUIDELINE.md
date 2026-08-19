## Repository Guidelines

#### Project Structure & Module Organization

`fl-data-profiling` is a Python 3.12 package for profiling relationships and predictive power between feature datasets (`feature.parquet` / `feature.csv`) and label datasets (`label.csv` / `label.parquet`). 

Source code lives in `src/fldataprofiler/`:
- `cli.py`: provides the CLI entry point (`fldataprofiler`).
- `registry.py`: manages registration, lookup, and aliasing of all profiling modules.
- `utils.py`: common data loading (`.csv`, `.parquet`), datetime indexing, merging, type casting, markdown/HTML rendering, and row limit/full mode context managers.
- `modules/`: contains 25 profiling modules implementing `ProfilingModule` interface (`run(feature_csv, label_csv, output_dir, join_key, targets) -> ModuleResult`).
- `modules/base.py`: defines `ProfilingModule` protocol and `ModuleResult` dataclass.
- `modules/progress.py`: provides progress reporting utilities across module stages.
- `modules/time_series_scoring.py`: common walk-forward splitting and scoring engine.

Tests are in `tests/`, human-facing documentation in `docs/`, datasets in `datasets/`, utility scripts in `scripts/`, and generated outputs in `reports/`.

#### Build, Test, and Development Commands

- `uv sync --dev`: install runtime and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run pytest`: run the full test suite.
- `uv run pytest tests/test_input_formats.py -q`: run only input format and CLI tests.
- `uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statistics`: run profiling via CLI.
- `uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module xgboost --full`: run ML profiling without subsampling.

#### Coding Style & Naming Conventions

Use standard Python style with 4-space indentation, explicit imports, and typed function signatures (`from __future__ import annotations`). Public modules, functions, and variables use `snake_case`; classes use `PascalCase`; tests use `test_*` names. 

Modules must implement `run(...) -> ModuleResult` and declare `name: str`. Generated artifacts must include `report.md`, `report.html`, `summary.json`, and module-specific `*.csv` / `*.png` files.

#### Testing Guidelines

The project uses `pytest`. Add or update tests in `tests/` corresponding to module behaviors:
- `tests/test_input_formats.py`: tests file parsing, parquet support, join keys, `--limit`, and `--full`.
- `tests/test_alphalens_analysis.py`: tests factor tearsheet analysis end-to-end.
- `tests/test_feature_scoring_modules.py`: tests feature selection and scoring modules.
- `tests/test_kmean.py`, `tests/test_visual_regions.py`, `tests/test_xgboost.py`, `tests/test_signal_analysis.py`, etc.: test specific module calculations.

Tests should use small synthetic DataFrames with `Date`, features, and targets to ensure fast execution.

#### Test Performance Guidelines

- Put fast, narrow unit tests around individual calculation functions before running end-to-end pipeline tests.
- For TDD and debugging, first run the smallest relevant test selection, such as `uv run pytest tests/test_input_formats.py::InputFormatTests::test_cli_limit_applies_to_feature_and_label_reads -q`.
- Do NOT run heavy end-to-end tests repeatedly during intermediate code editing.

#### Commit & Pull Request Guidelines

- Keep commit messages short, clear, and imperative (e.g. `add --full CLI flag`, `rename alphalens documentation`, `update module registry`).
- Never commit automatically without explicit user confirmation.

#### Agent-Specific Instructions

- Check `git status --short` before editing to avoid overwriting unrelated user changes.
- Ensure all file links use markdown clickable format (`[file.py](file:///path/to/file.py)`).\n