# Sequential State-Transition & Markov Probability

## Overview

The `probability_markov` module analyzes first-order sequential state-transition dynamics across continuous financial features. By discretizing features into quantile states ($Q_1 \dots Q_5$ quintiles) and tracking transitions ($S_{t-1} \to S_t$), the module uncovers high-alpha momentum triggers and regime-shift patterns that static univariate distributions fail to capture.

---

## Key Mathematical & Statistical Metrics

1. **State Discretization:**
   Features are partitioned into rank-based equal-frequency quantile states ($Q_1 \dots Q_M$, default $M=5$).
2. **Conditional Sequential Probability:**
   $$P(Y_{t+1} = c \mid S_t = Q_i \land S_{t-1} = Q_j) = \frac{\sum \mathbb{I}(Y_{t+1}=c, S_t=Q_i, S_{t-1}=Q_j)}{\sum \mathbb{I}(S_t=Q_i, S_{t-1}=Q_j)}$$
3. **Excess Alpha / Transition Momentum ($\Delta P$):**
   $$\Delta P = P(Y_{t+1} = c \mid S_t = Q_i \land S_{t-1} = Q_j) - P(Y_{t+1} = c \mid S_t = Q_i)$$
   Measures the incremental predictive power gained from historical transition trajectory versus stationary state alone.
4. **Statistical Significance & Uncertainty:**
   - **Fisher Exact / Chi-Square Contingency Test:** Evaluates whether transition pattern win rate is statistically different from baseline.
   - **Bayesian 95% Credible Interval:** Quantifies sample uncertainty under Beta-Binomial posterior.
   - **Transition Entropy:** Measures transition dispersion and chain predictability.

---

## Generated Artifacts

| Artifact File | Description |
| :--- | :--- |
| `report.md` | Markdown summary report highlighting top sequential alpha patterns. |
| `report.html` | Interactive web report with transition tables and metrics cards. |
| `summary.json` | JSON metadata and top pattern metrics. |
| `markov_transitions.csv` | Full $M \times M$ transition matrix records across all features and target classes. |
| `top_sequential_patterns.csv` | Filtered list of statistically verified, high-alpha sequential transition triggers. |
| `markov_heatmap.png` | 2D heatmaps illustrating transition win rates ($Q_{t-1} \times Q_t$). |

---

## Usage Example

### Command Line Interface (CLI)

```bash
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability_markov --target label_direction
```

### Python API

```python
from fldataprofiler.modules.probability_markov import ProbabilityMarkovModule

module = ProbabilityMarkovModule(n_bins=5, min_pattern_samples=15)
result = module.run(
    feature_csv="datasets/feature.parquet",
    label_csv="datasets/label.csv",
    output_dir="reports",
    join_key="Date",
    targets=["label_direction"],
)
```
