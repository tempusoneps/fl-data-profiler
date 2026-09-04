# Module Catalog

`fl-data-profiling` provides 29 specialized profiling modules accessible via `--module <name>`:

## 1. Factor & Signal Analysis
- **`alphalens`**: Factor tearsheet analysis, forward return quantiles (Q1-Q5), IC decay ($t+1, t+5, t+15, t+60$), Information Ratio (IR), and long-short cumulative spread curve.
- **`probability`**: 20-bin Quantile Conditional Probability distributions, Information Value (IV), Weight of Evidence (WoE), Probability Spread, Monotonicity, and Shannon Entropy.
- **`probability_bayes`** (hoặc `probabilitybayes`): Bayesian Quantile Conditional Probability with Beta-Binomial / Dirichlet-Multinomial shrinkage, 95% Credible Intervals, Bayes Factor ($BF_{10}$), Bayes WoE/IV, and uncertainty bounds.
- **`probability_kellycriterion`** (hoặc `probability_kelly`, `kelly`): Kelly Criterion position sizing ($f^*$, Half-Kelly, Quarter-Kelly), Expected Value ($EV$), Expected Capital Growth Rate, and action recommendations across 20 quantile bins.
- **`probability_2d`** (hoặc `probability2d`): 2D Joint Probability Heatmaps on $10 \times 10$ quantile grids, 2D Information Value ($IV_{2D}$), Synergy Gain, and Sweet Spot Rule extraction.
- **`probability_coverage`** (hoặc `coverage`, `probabilitycoverage`): 2D Joint Probability Coverage on $10 \times 10$ grids, ranking matrices by count of high-probability cells ($P \ge \text{min\_x}$), sample coverage %, and decision rules.
- **`probability_3d`** (hoặc `probability3d`): 3D Joint Probability Hyper-Voxels on $5 \times 5 \times 5$ quantile grids, 3-Way Synergy Gain, and 3D Hyper-Voxel Sweet Spot Rule extraction.
- **`probability_drift`**: Time-series probability stability, Population Stability Index (PSI), IV alpha decay, and regime / monotonicity inversion checks across chronological time folds.
- **`probability_prim`** (hoặc `prim`, `bump_hunting`, `prob_prim`): Patient Rule Induction Method (PRIM) bump hunting, multi-dimensional boundary peeling ($\alpha = 0.05$), box expansion, Bayesian 95% Credible Intervals, Fisher exact test $p$-values, 2D sweet spot visualization, and executable Python rule generator (`rule_code_python.py`).
- **`probability_markov`** (hoặc `markov`, `sequential_probability`, `transition_probability`): Sequential state-transition conditional probability ($P(Y_{t+1} \mid S_t \cap S_{t-1})$), Markov transition entropy, excess alpha momentum triggers ($\Delta P$), and 2D state transition heatmaps.
- **`probability_scorecard`** (hoặc `scorecard`, `woe_scorecard`, `logodds`): Multivariate WoE additive scorecard model, log-odds point scaling (Base Score 600, PDO 20), Kolmogorov-Smirnov (KS) statistic separation, and decile probability calibration curves.
- **`information_coefficient`**: Walk-forward time-series Pearson IC and Spearman Rank IC across rolling time folds.
- **`signal_analysis`**: Discrete conditional probability, market trap diagnosis (True Alpha vs Whipsaw/Reversal Trap), and multi-year stability & consistency analysis.
- **`regime_scoring`**: Market regime-aware feature scoring across volatility/trend segments.

## 2. Exploratory Data Analysis & Statistics
- **`eda`**: Comprehensive exploratory data analysis: missingness profile, numeric & categorical distributions, correlation heatmaps.
- **`statistics`**: Descriptive statistics, Pearson linear correlations, and label quantile mean distributions.
- **`scipy`**: Mathematical hypothesis testing ($t$-test, ANOVA $F$, Kruskal-Wallis, Mann-Whitney U, Chi-square, Cohen's $d$, Cramer's $V$).
- **`statsmodels`**: Econometric OLS and Logit regression: $eta$ coefficients, standard errors, $t$-statistics, $p$-values, 95% confidence intervals, AIC/BIC.

## 3. Clustering & Decision Rules
- **`kmean`**: Unsupervised KMeans clustering on 2D feature pairs $(F_1, F_2)$ to measure target label separability and purity.
- **`visual_regions`**: Quantile grid discretization and 2D bounding-box decision rule generation (`IF F1 IN [...] AND F2 IN [...] THEN Class X`).

## 4. Feature Selection & Scoring
- **`mutual_information`**: Model-agnostic non-linear dependency estimation via Mutual Information.
- **`permutation_importance_ts`**: Time-series out-of-fold permutation importance using Random Forest models.
- **`timeseries_importance`**: Normalized unified multi-criteria score combining IC, Permutation drop, MI, and correlation support.
- **`mrmr`**: Minimum Redundancy Maximum Relevance feature selection.
- **`stability_selection`**: Subsampling-based stability selection with randomized regularized regression.
- **`feature_interactions`**: Automated discovery and evaluation of pairwise feature interaction terms ($F_1 	imes F_2$, $F_1 / F_2$, $F_1 - F_2$).
- **`boruta`**: All-relevant feature selection using Random Forest shadow feature contrast.

## 5. Machine Learning Models
- **`xgboost`**: Gradient boosted decision trees with Gain/Weight/Cover importance, confusion matrices, and regression curves.
- **`lightgbm`**: High-speed histogram-based GBDT feature importance (Split & Gain).
- **`shap`**: Model interpretability via TreeSHAP and Mean Absolute SHAP values.
- **`sklearn`**: Scikit-Learn linear baselines (SGDClassifier & Ridge Regression).
- **`regularized_linear`**: Lasso (L1) and Ridge (L2) feature shrinkage and coefficient importance.

## 6. Automated Machine Learning (AutoML)
- **`autogluon`**: Amazon AutoGluon multi-layer stacking ensemble and leaderboard.
- **`flaml`**: Microsoft FLAML fast and cost-frugal AutoML search.
- **`pycaret`**: Low-code AutoML comparing over 15 estimators.\n