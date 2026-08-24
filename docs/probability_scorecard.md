# Multivariate WoE & Bayesian Log-Odds Scorecard

## Overview

The `probability_scorecard` module constructs an industry-standard additive scorecard model based on **Weight of Evidence (WoE)** binning and **Multivariate Logistic Regression**. It transforms continuous and complex feature distributions into intuitive, transparent integer point allocations, providing an interpretable point-based decision scorecard for quantitative trading and financial machine learning.

---

## Key Mathematical & Statistical Metrics

1. **Weight of Evidence ($\text{WoE}$) & Information Value ($\text{IV}$):**
   $$\text{WoE}_i = \ln\left(\frac{P(X \in \text{Bin}_i \mid Y=1)}{P(X \in \text{Bin}_i \mid Y=0)}\right), \quad \text{IV} = \sum_{i=1}^M (P(Y=1 \mid \text{Bin}_i) - P(Y=0 \mid \text{Bin}_i)) \cdot \text{WoE}_i$$
2. **Multivariate Logistic Weighting:**
   $$\text{Log-Odds} = \beta_0 + \sum_{k=1}^m \beta_k \cdot \text{WoE}(X_k)$$
3. **Scorecard Point Scaling:**
   $$\text{Factor} = \frac{\text{PDO}}{\ln(2)}, \quad \text{Offset} = \text{BaseScore} - \text{Factor} \cdot \ln(\text{BaseOdds})$$
   Feature points for bin $i$:
   $$\text{Points}_{k, i} = \text{round}\left(\text{Factor} \cdot \beta_k \cdot \text{WoE}_{k, i}\right)$$
   (Default parameters: $\text{BaseScore} = 600$, $\text{BaseOdds} = 1.0$, $\text{PDO} = 20$).
4. **Model Discrimination & Calibration:**
   - **Kolmogorov-Smirnov ($\text{KS-statistic}$):** Maximum separation between cumulative score distributions of events vs non-events ($KS = \max |F_{Y=1}(s) - F_{Y=0}(s)|$).
   - **ROC AUC:** Overall rank discrimination metric.
   - **Decile Calibration Table:** Empirical win rates mapped across score ranges.

---

## Generated Artifacts

| Artifact File | Description |
| :--- | :--- |
| `report.md` | Comprehensive Markdown report with scorecard rules, point tables, and KS statistics. |
| `report.html` | Interactive dashboard with metrics banners and scrollable point lookup tables. |
| `summary.json` | JSON metadata and performance metrics (AUC, KS, feature weights). |
| `scorecard_points.csv` | Full lookup table of point allocations per feature bin range. |
| `score_to_probability.csv` | Empirical and calibrated probability lookup across score deciles. |
| `score_distribution_plot.png` | Dual distribution plot comparing scores of $Y=1$ vs $Y=0$ with calibration curve. |

---

## Usage Example

### Command Line Interface (CLI)

```bash
uv run fldataprofiler fit datasets/feature.parquet datasets/label.csv --module probability_scorecard --target label_trend
```

### Python API

```python
from fldataprofiler.modules.probability_scorecard import ProbabilityScorecardModule

module = ProbabilityScorecardModule(base_score=600, pdo=20, n_bins=10, max_features=10)
result = module.run(
    feature_csv="datasets/feature.parquet",
    label_csv="datasets/label.csv",
    output_dir="reports",
    join_key="Date",
    targets=["label_trend"],
)
```
