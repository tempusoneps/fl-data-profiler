# Configuration & CLI Parameters

`fl-data-profiling` is configured via command-line arguments passed to `fldataprofiler fit`:

## 1. CLI Options

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_csv` | `Path` (Positional) | Required | Path to input feature dataset (`.parquet` or `.csv`). |
| `label_csv` | `Path` (Positional) | Required | Path to input label dataset (`.parquet` or `.csv`). |
| `--config` | `Path` | `None` | Path to custom `config.json` file (overrides bundled `config.default.json`). |
| `--module` | `str` | `statistics` | Name of the profiling module to execute (choices from `list_modules()`). |
| `--output-dir` | `Path` | `reports` | Target directory for generated report artifacts. |
| `--join-key` | `str` | Auto | Column name used to join features and labels (defaults to common `Date` or index). |
| `--target` | `str` (Repeatable) | All | Specific label column(s) to focus analysis on. Can be passed multiple times. |
| `--limit` | `int` | `None` | Restrict initial data reading to the first $N$ rows (for rapid prototyping). |
| `--full` | `bool` (Flag) | `False` | Disable internal row subsampling (20k rows) in ML modules to analyze 100% of data. |

## 2. Project-Wide Default Configuration (`config.default.json`)

The package includes a built-in `config.default.json` bundled directly into the distribution:
- **Global Settings**: `output_dir`, `max_rows`, `random_state`, `min_non_null`, `max_label_classes`.
- **Prune Settings**: `max_corr`, `corr_method`, `max_null`, `min_variance`, `auto_drop_raw_levels`.
- **Module Settings**: `probability_prim`, `probability_markov`, `probability_scorecard`, `xgboost`, `lightgbm`, etc.

To override defaults:
1. Place a `config.json` in the current working directory.
2. Or pass `--config /path/to/custom_config.json` on the command line.

## 3. Input Data & Join Strategies

1. **Date Index Join (Default for Time-Series)**:
   If both datasets contain a `Date` column, inputs are automatically parsed as datetime and aligned by timestamp.
2. **Explicit Common Column (`--join-key`)**:
   If specified, datasets are merged via an inner join on the designated key column.
3. **Row Index Alignment**:
   If no timestamp or common join column is found, datasets are aligned row-by-row by index (row counts must match).

## 4. Subsampling Modes

- **Default Mode**: Heavy machine learning modules (`xgboost`, `sklearn`, `signal_analysis`, `statsmodels`, `alphalens`, `kmean`, `shap`, `boruta`, `autogluon`, `flaml`, `pycaret`) downsample to $10,000 - 50,000$ rows to protect CPU/RAM resources.
- **Full Mode (`--full`)**: Bypasses all downsampling and processes the entire dataset.\n