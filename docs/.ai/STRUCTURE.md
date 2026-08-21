# Project Structure

```text
.
├── README.md                         # Main user guide and quick start.
├── AGENTS.md                         # Consolidated instructions for AI agents.
├── GEMINI.md                         # Consolidated instructions for Gemini CLI.
├── CLAUDE.md                         # Consolidated instructions for Claude Code.
├── pyproject.toml                    # Package metadata, dependencies, scripts.
├── uv.lock                           # Pinned dependencies for uv.
│
├── docs/                             # Comprehensive project & module documentation.
│   ├── README.md                     # Master index and module taxonomy.
│   │
│   ├── .ai/                          # Agent documentation and development guides.
│   │   ├── AI_AGENT_GUIDELINE.md     # Guidelines for AI agent development.
│   │   ├── CONFIGURATION.md          # CLI options and join/subsampling configuration.
│   │   ├── FEATURES_OVERVIEW.md      # High-level capabilities overview.
│   │   ├── INSTALLATION.md           # Setup instructions for uv and pip.
│   │   ├── MODULES.md                # Catalog of all 25 profiling modules.
│   │   ├── REF.md                    # Reference links and paper citations.
│   │   ├── RESOURCES.md              # Pointers to internal documentation and scripts.
│   │   ├── RULE.md                   # Project rules and conventions.
│   │   ├── STRUCTURE.md              # This repository architecture guide.
│   │   └── USAGE.md                  # CLI and Python API usage examples.
│   │
│   ├── alphalens.md                  # Factor tearsheet analysis documentation.
│   ├── autogluon.md                  # AutoGluon AutoML documentation.
│   ├── boruta.md                     # Boruta feature selection documentation.
│   ├── eda.md                        # Exploratory data analysis documentation.
│   ├── feature_interactions.md       # Pairwise feature interaction documentation.
│   ├── flaml.md                      # FLAML AutoML documentation.
│   ├── information_coefficient.md    # Walk-forward IC scoring documentation.
│   ├── kmean.md                      # 2D KMeans clustering documentation.
│   ├── lightgbm.md                   # LightGBM feature importance documentation.
│   ├── mrmr.md                       # mRMR feature selection documentation.
│   ├── mutual_information.md         # Mutual information documentation.
│   ├── permutation_importance_ts.md  # Time-series permutation importance docs.
│   ├── probability.md                # 20-bin Quantile Conditional Probability & WoE/IV docs.
│   ├── probability_2d.md             # 2D Joint Probability Heatmap & Sweet Spots docs.
│   ├── probability_3d.md             # 3D Joint Probability & Hyper Sweet Spots docs.
│   ├── probability_bayes.md          # Bayesian Probability & Credible Intervals docs.
│   ├── probability_drift.md          # Probability drift, PSI & Alpha stability docs.
│   ├── probability_kellycriterion.md # Kelly Criterion & Position Sizing docs.
│   ├── pycaret.md                    # PyCaret AutoML documentation.
│   ├── regime_scoring.md             # Regime-based feature scoring documentation.
│   ├── regularized_linear.md         # Regularized linear regression docs.
│   ├── scipy.md                      # SciPy statistical testing documentation.
│   ├── shap.md                       # SHAP values interpretability docs.
│   ├── signal_analysis.md            # Trading signal analysis documentation.
│   ├── sklearn.md                    # Scikit-Learn baseline documentation.
│   ├── stability_selection.md        # Stability selection documentation.
│   ├── statistics.md                 # Descriptive statistics documentation.
│   ├── statsmodels.md                # OLS/Logit econometric documentation.
│   ├── timeseries_importance.md      # Unified time-series importance docs.
│   ├── visual_regions.md             # 2D visual decision region rule docs.
│   └── xgboost.md                    # XGBoost gradient boosting documentation.
│
├── datasets/                         # Sample datasets and generated data.
│   ├── VN30F1M_5m.csv                # Sample raw OHLCV price bars.
│   ├── feature.parquet               # Extracted features dataset.
│   └── label.csv                     # Extracted classification/regression labels.
│
├── src/                             # Python package source root.
│   └── fldataprofiler/               # Main package module.
│       ├── __init__.py               # Package exports.
│       ├── cli.py                    # CLI argument parser and entry point (fit & prune).
│       ├── feature_pruner.py         # Feature pruning & multicollinearity engine.
│       ├── registry.py               # Module registry and factory.
│       ├── utils.py                  # Data ingestion, indexing, and rendering helpers.
│   │
│   └── modules/                      # 29 Profiling Module Implementations.
│       ├── __init__.py               # Module package marker.
│       ├── base.py                   # ProfilingModule protocol & ModuleResult.
│       ├── progress.py               # Shared progress reporting bar.
│       ├── time_series_scoring.py    # Walk-forward scoring engine.
│       ├── alphalens_analysis.py     # Alphalens factor analysis module.
│       ├── automl_autogluon.py       # AutoGluon tabular predictor module.
│       ├── automl_flaml.py           # FLAML lightweight AutoML module.
│       ├── automl_pycaret.py         # PyCaret low-code AutoML module.
│       ├── boruta.py                 # Boruta all-relevant selection module.
│       ├── eda.py                    # Exploratory data analysis module.
│       ├── feature_interactions.py   # Pairwise feature interaction module.
│       ├── information_coefficient.py# Walk-forward IC module.
│       ├── kmean.py                  # 2D KMeans clustering module.
│       ├── lightgbm.py               # LightGBM feature importance module.
│       ├── mrmr.py                   # mRMR selection module.
│       ├── mutual_information.py     # Mutual Information scoring module.
│       ├── permutation_importance_ts.py # Time-series permutation module.
│       ├── probability.py            # 20-bin Quantile Conditional Probability & WoE/IV module.
│       ├── probability_2d.py         # 2D Joint Probability Heatmap & Sweet Spots module.
│       ├── probability_3d.py         # 3D Joint Probability & Hyper Sweet Spots module.
│       ├── probability_bayes.py      # Bayesian Probability & Credible Intervals module.
│       ├── probability_drift.py      # Probability drift, PSI & Alpha stability module.
│       ├── probability_kellycriterion.py # Kelly Criterion & Position Sizing module.
│       ├── regime_scoring.py         # Market regime scoring module.
│       ├── regularized_linear.py     # Lasso/Ridge regularized linear module.
│       ├── scipy.py                  # SciPy hypothesis testing module.
│       ├── shap.py                   # TreeSHAP value attribution module.
│       ├── signal_analysis.py        # Trading signal analysis module.
│       ├── sklearn.py                # Scikit-Learn baseline module.
│       ├── stability_selection.py    # Stability selection module.
│       ├── statistics.py             # Descriptive statistics module.
│       ├── statsmodels.py            # Econometric OLS/Logit module.
│       ├── timeseries_importance.py  # Unified time-series importance module.
│       ├── visual_regions.py         # 2D decision region rule module.
│       └── xgboost.py                # XGBoost gradient boosting module.
│
├── reports/                          # Default directory for generated run artifacts.
├── scripts/                          # Utility and data preparation shell scripts.
│   ├── generate_agents_markdown.sh   # Rebuilds AGENTS.md, GEMINI.md, CLAUDE.md from docs/.ai/
│   ├── prepare_datasets.sh           # Downloads OHLCV data, generates features & labels.
│   └── run_modules.sh                # Sequentially executes profiling modules.
│
└── tests/                            # Automated test suite.
    ├── test_alphalens_analysis.py    # Tests for alphalens module.
    ├── test_automl.py                # Tests for AutoML modules.
    ├── test_cli_prune.py             # Tests for prune CLI command.
    ├── test_eda.py                   # Tests for EDA module.
    ├── test_feature_pruner.py        # Tests for feature pruning engine.
    ├── test_feature_scoring_modules.py # Tests for feature selection modules.
    ├── test_html_reports.py          # Tests for HTML report rendering.
    ├── test_input_formats.py         # Tests for CLI, parquet, limit, full flag.
    ├── test_kmean.py                 # Tests for KMeans module.
    ├── test_probability.py           # Tests for probability module.
    ├── test_probability_2d.py        # Tests for probability_2d module.
    ├── test_probability_3d.py        # Tests for probability_3d module.
    ├── test_probability_bayes.py     # Tests for probability_bayes module.
    ├── test_probability_drift.py     # Tests for probability_drift module.
    ├── test_probability_kellycriterion.py # Tests for probability_kellycriterion module.
    ├── test_signal_analysis.py       # Tests for signal analysis module.
    ├── test_time_series_scoring.py   # Tests for walk-forward scoring.
    ├── test_visual_regions.py        # Tests for visual regions module.
    └── test_xgboost.py               # Tests for XGBoost module.
```\n