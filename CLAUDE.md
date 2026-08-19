# Agent Guide

This file is generated from project documentation. Do not edit it directly.


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
# Project Structure

```text
.
├── README.md                         # Main user guide and quick start.
├── AGENTS.md                         # Consolidated instructions for AI agents.
├── GEMINI.md                         # Consolidated instructions for Gemini CLI.
├── CLAUDE.md                         # Consolidated instructions for Claude Code.
├── pyproject.toml                    # Package metadata, dependencies, scripts.
├── uv.lock                           # Pinned dependencies for uv.
│
├── docs/                             # Comprehensive project & module documentation.
│   ├── README.md                     # Master index and module taxonomy.
│   │
│   ├── .ai/                          # Agent documentation and development guides.
│   │   ├── AI_AGENT_GUIDELINE.md     # Guidelines for AI agent development.
│   │   ├── CONFIGURATION.md          # CLI options and join/subsampling configuration.
│   │   ├── FEATURES_OVERVIEW.md      # High-level capabilities overview.
│   │   ├── INSTALLATION.md           # Setup instructions for uv and pip.
│   │   ├── MODULES.md                # Catalog of all 25 profiling modules.
│   │   ├── REF.md                    # Reference links and paper citations.
│   │   ├── RESOURCES.md              # Pointers to internal documentation and scripts.
│   │   ├── RULE.md                   # Project rules and conventions.
│   │   ├── STRUCTURE.md              # This repository architecture guide.
│   │   └── USAGE.md                  # CLI and Python API usage examples.
│   │
│   ├── alphalens.md                  # Factor tearsheet analysis documentation.
│   ├── autogluon.md                  # AutoGluon AutoML documentation.
│   ├── boruta.md                     # Boruta feature selection documentation.
│   ├── eda.md                        # Exploratory data analysis documentation.
│   ├── feature_interactions.md       # Pairwise feature interaction documentation.
│   ├── flaml.md                      # FLAML AutoML documentation.
│   ├── information_coefficient.md    # Walk-forward IC scoring documentation.
│   ├── kmean.md                      # 2D KMeans clustering documentation.
│   ├── lightgbm.md                   # LightGBM feature importance documentation.
│   ├── mrmr.md                       # mRMR feature selection documentation.
│   ├── mutual_information.md         # Mutual information documentation.
│   ├── permutation_importance_ts.md  # Time-series permutation importance docs.
│   ├── pycaret.md                    # PyCaret AutoML documentation.
│   ├── regime_scoring.md             # Regime-based feature scoring documentation.
│   ├── regularized_linear.md         # Regularized linear regression docs.
│   ├── scipy.md                      # SciPy statistical testing documentation.
│   ├── shap.md                       # SHAP values interpretability docs.
│   ├── signal_analysis.md            # Trading signal analysis documentation.
│   ├── sklearn.md                    # Scikit-Learn baseline documentation.
│   ├── stability_selection.md        # Stability selection documentation.
│   ├── statistics.md                 # Descriptive statistics documentation.
│   ├── statsmodels.md                # OLS/Logit econometric documentation.
│   ├── timeseries_importance.md      # Unified time-series importance docs.
│   ├── visual_regions.md             # 2D visual decision region rule docs.
│   └── xgboost.md                    # XGBoost gradient boosting documentation.
│
├── datasets/                         # Sample datasets and generated data.
│   ├── VN30F1M_5m.csv                # Sample raw OHLCV price bars.
│   ├── feature.parquet               # Extracted features dataset.
│   └── label.csv                     # Extracted classification/regression labels.
│
├── src/                             # Python package source root.
│   └── fldataprofiler/               # Main package module.
│       ├── __init__.py               # Package exports.
│       ├── cli.py                    # CLI argument parser and entry point.
│       ├── registry.py               # Module registry and factory.
│       ├── utils.py                  # Data ingestion, indexing, and rendering helpers.
│   │
│   └── modules/                      # 25 Profiling Module Implementations.
│       ├── __init__.py               # Module package marker.
│       ├── base.py                   # ProfilingModule protocol & ModuleResult.
│       ├── progress.py               # Shared progress reporting bar.
│       ├── time_series_scoring.py    # Walk-forward scoring engine.
│       ├── alphalens_analysis.py     # Alphalens factor analysis module.
│       ├── automl_autogluon.py       # AutoGluon tabular predictor module.
│       ├── automl_flaml.py           # FLAML lightweight AutoML module.
│       ├── automl_pycaret.py         # PyCaret low-code AutoML module.
│       ├── boruta.py                 # Boruta all-relevant selection module.
│       ├── eda.py                    # Exploratory data analysis module.
│       ├── feature_interactions.py   # Pairwise feature interaction module.
│       ├── information_coefficient.py# Walk-forward IC module.
│       ├── kmean.py                  # 2D KMeans clustering module.
│       ├── lightgbm.py               # LightGBM feature importance module.
│       ├── mrmr.py                   # mRMR selection module.
│       ├── mutual_information.py     # Mutual Information scoring module.
│       ├── permutation_importance_ts.py # Time-series permutation module.
│       ├── regime_scoring.py         # Market regime scoring module.
│       ├── regularized_linear.py     # Lasso/Ridge regularized linear module.
│       ├── scipy.py                  # SciPy hypothesis testing module.
│       ├── shap.py                   # TreeSHAP value attribution module.
│       ├── signal_analysis.py        # Trading signal analysis module.
│       ├── sklearn.py                # Scikit-Learn baseline module.
│       ├── stability_selection.py    # Stability selection module.
│       ├── statistics.py             # Descriptive statistics module.
│       ├── statsmodels.py            # Econometric OLS/Logit module.
│       ├── timeseries_importance.py  # Unified time-series importance module.
│       ├── visual_regions.py         # 2D decision region rule module.
│       └── xgboost.py                # XGBoost gradient boosting module.
│
├── reports/                          # Default directory for generated run artifacts.
├── scripts/                          # Utility and data preparation shell scripts.
│   ├── generate_agents_markdown.sh   # Rebuilds AGENTS.md, GEMINI.md, CLAUDE.md from docs/.ai/
│   ├── prepare_datasets.sh           # Downloads OHLCV data, generates features & labels.
│   └── run_modules.sh                # Sequentially executes profiling modules.
│
└── tests/                            # Automated test suite.
    ├── test_alphalens_analysis.py    # Tests for alphalens module.
    ├── test_automl.py                # Tests for AutoML modules.
    ├── test_eda.py                   # Tests for EDA module.
    ├── test_feature_scoring_modules.py # Tests for feature selection modules.
    ├── test_html_reports.py          # Tests for HTML report rendering.
    ├── test_input_formats.py         # Tests for CLI, parquet, limit, full flag.
    ├── test_kmean.py                 # Tests for KMeans module.
    ├── test_signal_analysis.py       # Tests for signal analysis module.
    ├── test_time_series_scoring.py   # Tests for walk-forward scoring.
    ├── test_visual_regions.py        # Tests for visual regions module.
    └── test_xgboost.py               # Tests for XGBoost module.
```\n