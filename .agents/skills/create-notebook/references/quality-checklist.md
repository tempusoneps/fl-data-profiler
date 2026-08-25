# Quality Checklist for Analytics & Insights Notebooks

Before delivering or executing a notebook:

- [ ] **Data Ingestion & Slicing**: Early cells load data via `load_analytics_dataset()` and apply appropriate chronological filtering (`dataset.index > ...`).
- [ ] **Empirical Chart Requirement**: Every feature/factor and interaction pair discussed in insights has **exactly 1 dedicated chart** (Stacked Bar, 100% Probability Bar, KDE density/contour, 2D Heatmap, or Scatter).
- [ ] **Broad Quantitative Overview**: Full Crosstab distribution tables (both Count and normalized %) and Chi-Square significance stats are present.
- [ ] **No `reports/` Dependency**: No code cells read from or reference files in the `reports/` directory (all data is dynamically computed from `load_analytics_dataset()`).
- [ ] **Clean Top-to-Bottom Execution**: The notebook executes cleanly from top-to-bottom without dangling variables or hidden state.
- [ ] **Tidy Aesthetics**: Figures have clean labels, titles, gridlines, and suitable colormaps (`viridis`, `coolwarm`, `Spectral`, `YlGnBu`).
- [ ] **Domain Summary**: Markdown cells concisely interpret quantitative sweet spots, win-rate lifts, and actionable rules.
