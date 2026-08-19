# Key Features & System Capabilities

* **25 Specialized Profiling Modules**: Covers factor research (Alphalens), econometric regression (OLS/Logit), classical statistical hypothesis testing (SciPy), unsupervised clustering (KMeans), decision region extraction (Visual 2D Rules), information theory, machine learning (XGBoost, LightGBM, SHAP, Boruta), and automated ML (AutoGluon, FLAML, PyCaret).
* **Time-Series Aware Validation**: Implements non-shuffled, walk-forward cross-validation splits to prevent data leakage and look-ahead bias on financial time-series.
* **Dual Format Ingestion**: Natively reads both `.parquet` (fast, compressed, columnar) and `.csv` files for features and labels.
* **Multi-Layer Reporting Artifacts**: Generates human-readable GitHub Markdown (`report.md`), interactive styled HTML (`report.html`), structured machine-readable metadata (`summary.json`), and comprehensive numerical tables (`*.csv`).
* **High-Performance Vectorization & Full Data Support**: Supports lightning-fast statistical scans on millions of rows with optional `--full` execution for full-dataset training.
* **Dual CLI Entry Points**: Seamless command-line invocation via `fldataprofiler` or alias `fldataprofier`.\n