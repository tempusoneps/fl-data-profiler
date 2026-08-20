# Usage Guide

## 1. Using the Command Line Interface (CLI)

After installing, `fldataprofiler` is available in your shell:

### Basic Profiling Command
```bash
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module <module_name>
```

### Common Usage Examples

```bash
# 1. Quick Statistical Overview
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module statistics

# 2. Factor Tearsheet Analysis (Alphalens)
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module alphalens

# 3. XGBoost Modeling on a Specific Target Label
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module xgboost \
  --target allow_entry

# 4. Full Dataset Analysis (Disables 20k Subsampling)
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module xgboost \
  --full \
  --output-dir reports/xgboost_full

# 5. Fast Test Run with Row Limit
fldataprofiler fit datasets/feature.parquet datasets/label.csv \
  --module kmean \
  --limit 5000

# 6. Extract 2D Decision Bounding-Box Rules
fldataprofiler fit datasets/feature.parquet datasets/label.csv --module visual_regions
```

### Feature Pruning Command (`prune`)
```bash
# 1. Basic pruning (drops null > 20%, low-variance, and collinear |corr| > 0.85)
fldataprofiler prune datasets/feature.parquet

# 2. Score-guided pruning with custom output and top-K
fldataprofiler prune datasets/feature.parquet \
  -o datasets/selected_feature.parquet \
  --max-corr 0.80 \
  --scores-file reports/xgboost/feature_scores.csv \
  --top-k 30
```

## 2. Using the Python API

### Profiling Execution
```python
from pathlib import Path
from fldataprofiler.registry import get_module

# 1. Instantiate module
module = get_module("alphalens")

# 2. Execute profiling
result = module.run(
    feature_csv=Path("datasets/feature.parquet"),
    label_csv=Path("datasets/label.csv"),
    output_dir=Path("reports"),
    targets=["allow_entry"],
)

# 3. Inspect generated artifacts
print(f"Report written to: {result.report_dir}")
for artifact in result.artifacts:
    print(f"- {artifact}")
```

### Feature Pruning Programmatic API
```python
import pandas as pd
from fldataprofiler.feature_pruner import PruneConfig, prune_features, load_scores

df = pd.read_parquet("datasets/feature.parquet")
scores = load_scores(Path("reports/xgboost/feature_scores.csv"))

config = PruneConfig(max_corr=0.85, max_null=0.20, min_variance=0.0, top_k=30)
result = prune_features(df, config=config, scores=scores)

# Save cleaned dataframe
result.df_selected.to_parquet("datasets/selected_feature.parquet", index=False)
print(f"Retained {len(result.retained_features)} features")
```

## 3. Running Profiling Modules via Script

Use the automated shell script to run profiling modules sequentially:

```bash
# Run default 14 fast & recommended modules (~2-4 mins)
bash scripts/run_modules.sh

# Run with row limit (e.g. 1000 rows for fast testing)
bash scripts/run_modules.sh --limit 1000

# Run all 25 modules including slow/resource-intensive modules
bash scripts/run_modules.sh --all

# Run specific modules
bash scripts/run_modules.sh --modules statistics,eda,xgboost,lightgbm

# Skip specific modules
bash scripts/run_modules.sh --skip-modules kmean,visual_regions
```

\n