# Patient Rule Induction Method (PRIM & Bump Hunting)

## Overview

The **Patient Rule Induction Method (PRIM)**, originally formulated by Jerome H. Friedman and Nicholas I. Fisher (1999), is a non-parametric bump hunting technique designed to identify sub-boxes (hyper-rectangles) in multi-dimensional continuous feature space where the target event probability (win rate $\bar{Y}$) is substantially higher than the global baseline.

While standard decision trees split data abruptly along single axes, PRIM employs a "patient" iterative peeling strategy ($\alpha = 5\%$) along lower and upper feature boundaries, followed by a bottom-up pasting/expansion phase to recover sample support without compromising signal quality.

---

## Key Mathematical & Statistical Metrics

1. **Patient Peeling ($\alpha = 0.05$):**
   Iteratively trims $\alpha$ proportion of samples from the lower or upper boundary of each feature dimension, choosing the slice that yields the highest target mean $\bar{Y}_{\text{box}}$.
2. **Box Expansion (Pasting):**
   Expands bounding box limits to regain sample support $N_{\text{box}}$ while maintaining win rate above a tolerance threshold.
3. **Statistical Significance & Uncertainty:**
   - **Sample Support ($S = N_{\text{box}} / N_{\text{total}}$):** Proportion of the total dataset captured by the box.
   - **Win Rate ($\hat{p} = N_{Y=1} / N_{\text{box}}$):** Empirical probability of event occurrence inside the box.
   - **Lift ($\hat{p} / P(Y=1)$):** Multiplier over global base rate.
   - **Fisher's Exact Test $p$-value:** One-tailed significance test against null hypothesis of random distribution.
   - **Bayesian 95% Credible Interval:** Posterior Beta-Binomial interval $\text{Beta}(\alpha_0 + k, \beta_0 + n - k)$ under Jeffreys prior $\text{Beta}(0.5, 0.5)$.

---

## Generated Artifacts

| Artifact File | Description |
| :--- | :--- |
| `report.md` | Markdown report summarizing top discovered rules, lift, win rate, and sample support. |
| `report.html` | Interactive dashboard with KPI metrics cards and rule tables. |
| `summary.json` | Structured JSON metadata, summary metrics, and top rule definitions. |
| `prim_rules.csv` | Full tabular dataset of all discovered 1D, 2D, and 3D bump hunting rules. |
| `rule_code_python.py` | Standalone, ready-to-run Python code (`predict_prim_rules`, `evaluate_prim_rules`). |
| `prim_rules_plot.png` | 2D scatter visualization with highlighted bounding box sweet spots. |

---

## Configuration Options

PRIM Bump Hunting supports flexible configuration via `ProbabilityPrimConfig`, module constructor arguments, or environment variables:

| Parameter | Default | Env Variable | Description |
| :--- | :--- | :--- | :--- |
| `min_box_samples` | `250` | `PRIM_MIN_SAMPLES` | Absolute minimum number of sample bars inside a valid rule box. |
| `min_support` | `0.005` (0.5%) | `PRIM_MIN_SUPPORT` | Minimum fraction of dataset required ($N_{\text{box}} \ge \text{min\_support} \times N_{\text{total}}$). |
| `objective` | `"support_weighted"` | `PRIM_OBJECTIVE` | Trajectory optimization objective: `"support_weighted"`, `"win_rate"`, `"edge_support"`, `"wilson_lower"`. |
| `alpha` | `0.05` (5%) | `PRIM_ALPHA` | Fraction of samples peeled from feature boundaries at each iteration. |
| `max_candidates` | `16` | `PRIM_MAX_CANDIDATES` | Top candidate features selected by Information Value for 1D/2D/3D combinations. |
| `expand_delta` | `0.01` (1%) | `PRIM_EXPAND_DELTA` | Bottom-up pasting expansion tolerance threshold. |

---

## Usage Example

### Command Line Interface (CLI)

```bash
# Run with default robust statistical settings (N >= 250 samples)
uv run fldataprofiler fit datasets/selected_feature.parquet datasets/label.csv --module probability_prim

# Customize min support via environment variables on CLI:
PRIM_MIN_SAMPLES=500 PRIM_MIN_SUPPORT=0.01 uv run fldataprofiler fit datasets/selected_feature.parquet datasets/label.csv --module probability_prim
```

### Python API

```python
from fldataprofiler.modules.probability_prim import ProbabilityPrimConfig, ProbabilityPrimModule

# Configure via ProbabilityPrimConfig dataclass
config = ProbabilityPrimConfig(
    min_box_samples=300,
    min_support=0.01,           # At least 1% of total dataset (e.g. 500 samples)
    objective="support_weighted", # Balances high win rate with high statistical sample power
    alpha=0.05,
    max_candidates=16,
)

module = ProbabilityPrimModule(config=config)
result = module.run(
    feature_csv="datasets/selected_feature.parquet",
    label_csv="datasets/label.csv",
    output_dir="reports",
    join_key="Date",
    targets=["direction_filter", "allow_entry"],
)
```

