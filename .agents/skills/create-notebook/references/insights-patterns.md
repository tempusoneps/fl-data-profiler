# Quantitative Insights & Factor Analytics Patterns

Use this structure when building interactive analytics and profiling notebooks in `fl-data-profiling`:

### 1. Macro & Environment Setup
- **Standard Imports**: `pandas`, `numpy`, `pandas_ta`, `matplotlib.pyplot`, `seaborn`, `scipy.stats.chi2_contingency`, and `from utils import load_analytics_dataset, show_3_distribution_charts, show_3_sns_charts`.
- **Data Ingestion**: Use `load_analytics_dataset()` to load aligned feature & label tables, followed by chronological date slicing (`data = dataset[...]`).

### 2. Comprehensive Factor Analysis Architecture
Every feature and label investigated must be presented through a complete empirical workflow:
- **Crosstab Distribution Tables**: Show both raw sample counts (`pd.crosstab(feat, label)`) and normalized percentage probabilities (`div(sum, axis=0) * 100`).
- **20-Quantile Discretization**: For continuous features, bin using `pd.qcut(data[col], q=20, duplicates='drop')` to isolate non-linear sweet spots and tail regimes.
- **Empirical Visualizations (At least 1 dedicated chart per feature/label)**:
  - **Stacked Bar Charts (Count)**: To visualize raw sample density across classes/time.
  - **100% Normalized Stacked Bar Charts (Probability)**: To display conditional win-rates and quantile probability spreads ($\Delta P$).
  - **KDE Density / Contour Plots (`sns.kdeplot`)**: For continuous interaction pairs and seasonal distributions.
  - **Tradable Opportunity Bar Charts**: Aggregate tradable actions (`Yes - Buy + Yes - Sell`) across quantiles.
  - **Statistical Testing**: Compute Chi-Square contingency stats (`chi2_contingency`) and p-values to validate independence vs significance.

### 3. Broad Quantitative Perspective & Self-Contained Runtime
- Do **not** merely summarize AI-generated text or truncated tables.
- **Never mention or read files from the `reports/` directory** (`reports/` does not exist in the analytics runtime environment). All counts, quantiles, win-rates, and figures must be computed purely from `data` in memory.
- Enable interactive data exploration with full data arrays, enabling traders and quantitative researchers to discover nuanced regime filters, time-of-day dynamics, and non-linear alpha anomalies.

### 4. Quality Checklist Verification
- [ ] Every feature & label discussed has a Crosstab table and exactly 1 dedicated chart.
- [ ] No mention or code reading from `reports/` directory.
- [ ] The notebook runs top-to-bottom without dangling variables.
- [ ] Titles, axes, and legends are formatted cleanly with appropriate colormaps.
