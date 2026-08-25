---
name: create-notebook
description: Use when creating, scaffolding, or editing Jupyter notebooks (`.ipynb`) for quantitative analytics, feature profiling explorations, or factor insights in fl-data-profiling; leverages bundled insights template, reference architecture, and new_notebook.py.
---

# Jupyter Notebook Skill (`create-notebook`)

Create clean, reproducible, and exploratory Jupyter notebooks for data profiling, factor analytics, and quantitative trading insights in `fl-data-profiling`.

---

## 4 Core Principles & Golden Rules

When creating or scaffolding analytics notebooks from insights:

1. **Empirical Chart for Every Factor (1 Chart is Sufficient)**:
   Every feature and label discussed in the insights **MUST** be visualized via a clear empirical chart (e.g., Stacked Bar Chart, 100% Normalized Probability Bar Chart, KDE Density/Contour Plot, 2D Heatmap, or Scatter Plot with Hue). Exactly 1 well-constructed chart per feature/interaction pair is sufficient.
2. **Broad Quantitative Overview (Not Narrow AI Text Rehash)**:
   Notebooks must provide a comprehensive, exploratory overview of the data across all classes/categories. Do **not** merely echo or summarize AI-generated text. Instead, deliver full Crosstab tables (raw counts + normalized percentages), 20-Quantile binning, Chi-Square significance testing, and Tradable Opportunity ratios so traders can interactively explore the underlying data.
3. **Template & Architecture Compliance**:
   All notebooks must strictly inherit the structure from `.agents/skills/create-notebook/assets/insights-template.ipynb`:
   - `## 1. Import Library` (`pandas`, `numpy`, `pandas_ta`, `matplotlib`, `seaborn`, `chi2_contingency`, `load_analytics_dataset`).
   - `## 2. Load Price Data` (`dataset = load_analytics_dataset()`, date slicing).
   - `## 3+. Analytics: <Feature / Pair> vs <Label>` (Crosstab $\to$ Chart $\to$ Statistical Check).
4. **No Mention or Reading of `reports/` Directory (Self-Contained Analytics)**:
   Do **NOT** mention, reference, or write code that reads files from the `reports/` directory (e.g., `../reports/probability/...`, `feature_scores.csv`, `pair_probability_scores.csv`, etc.). The `reports/` directory does not exist in the analytics runtime environment. All quantiles, probabilities, crosstabs, metrics, and charts must be computed directly and dynamically from `data` loaded via `load_analytics_dataset()`.

---

## Standard Factor Analysis Workflows

### Pattern A: Discrete / Categorical Feature vs Label (e.g., `time_int`, `month`, `signal`)
```python
# 1. Crosstab Distribution
ct = pd.crosstab(data['time_int'], data['allow_entry'])
display(ct)

# 2. Dedicated Empirical Chart (Stacked Bar Chart by Count)
ax = ct.plot(kind='bar', stacked=True, figsize=(16, 7), colormap='viridis', edgecolor='black', linewidth=0.5)
plt.title('Count Distribution of Allow Entry by Time Int', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Time Int', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
```

### Pattern B: Continuous Feature vs Label (e.g., `realized_volatility`, `bb_width_sma_medium`, `session_vwap_z`)
```python
# 1. Discretize into 20 Quantiles & Crosstab (Count & %)
data['feat_q20'] = pd.qcut(data['bb_width_sma_medium'], q=20, duplicates='drop')
ct_count = pd.crosstab(data['feat_q20'], data['allow_entry'])
ct_pct = ct_count.div(ct_count.sum(axis=1), axis=0) * 100
display(ct_pct.round(2))

# 2. Dedicated Empirical Chart (100% Stacked Probability Bar Chart)
ax = ct_pct.plot(kind='bar', stacked=True, figsize=(15, 7), colormap='viridis', edgecolor='black', linewidth=0.5)
plt.title('Probability Distribution by BB Width (20 Quantiles)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('BB Width Quantiles', fontsize=12)
plt.ylabel('Tỷ lệ (%)', fontsize=12)
plt.legend(title='Category', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 3. Statistical Testing (Chi-Square & Tradable Ratio)
chi2, p, dof, _ = chi2_contingency(ct_count)
print(f"Chi-square: {chi2:.4f}, p-value: {p:.4e}, DoF: {dof}")
```

### Pattern C: 2D Interaction Pair vs Label (e.g., `high_macro` × `pvi`, `month` × `keltner_reversal`)
```python
# 1. 2D 10x10 Quantile Pivot / Heatmap
data['macro_bin'] = pd.qcut(data['high_macro'], q=10, labels=[f'M{i+1}' for i in range(10)], duplicates='drop')
data['pvi_bin'] = pd.qcut(data['pvi'], q=10, labels=[f'P{i+1}' for i in range(10)], duplicates='drop')
prob_matrix = data.pivot_table(index='pvi_bin', columns='macro_bin', values='is_buy', aggfunc='mean', observed=False) * 100

# 2. Dedicated Empirical Chart (2D Heatmap / 2D KDE Contour)
plt.figure(figsize=(12, 6))
sns.heatmap(prob_matrix, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Win-Rate (%)'}, linewidths=0.5)
plt.title('2D Probability Heatmap: high_macro vs pvi (Buy Win-Rate %)', fontsize=13, fontweight='bold')
plt.xlabel('high_macro Quantiles')
plt.ylabel('pvi Quantiles')
plt.tight_layout()
plt.show()
```

---

## Scaffolding Workflow

Run the helper script directly via `uv run python`:

```bash
uv run python .agents/skills/create-notebook/scripts/new_notebook.py \
  --title "VN30 Factor Probability Analysis" \
  --out notebooks/factor_probability_analysis.ipynb
```

---

## Reference Map

- `assets/insights-template.ipynb`: Base Jupyter notebook template.
- `references/insights-patterns.md`: Deep dive on quantitative analytics patterns.
- `references/quality-checklist.md`: Final delivery verification checklist.
