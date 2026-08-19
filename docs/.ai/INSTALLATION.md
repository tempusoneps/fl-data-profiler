# Installation

## Using `uv` (Recommended)

[`uv`](https://github.com/astral-sh/uv) is the recommended package manager for fast, deterministic environment setups.

```bash
# Clone the repository
git clone https://github.com/tempusoneps/fl-data-profiling.git
cd fl-data-profiling

# Sync dependencies
uv sync

# For development mode
uv sync --dev
```

## Using `pip`

```bash
# Recommended in a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

## Running Without Local Installation

You can run directly using `uvx` or `uv run`:

```bash
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statistics
```

## Requirements

- **Python**: `>= 3.12, < 3.14`
- **Core Dependencies**:
  - `numpy >= 2.2.6`
  - `pandas >= 2.3.0, < 3.0.0`
  - `pyarrow >= 18.0.0`
  - `scikit-learn >= 1.8.0`
  - `scipy >= 1.16.0`
  - `statsmodels >= 0.14.6`
  - `xgboost >= 3.0.0`
  - `shap >= 0.48.0`
  - `flaml >= 2.6.0`
  - `matplotlib >= 3.10.8`
  - `autofcholv` (Git dependency)
  - `labelohlcv` (Git dependency)\n