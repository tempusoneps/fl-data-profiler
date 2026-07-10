# Feature Scoring Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ten CLI modules that help identify valuable features for predicting labels from time-series feature and label files.

**Architecture:** Add shared time-series scoring utilities first, then implement each module as a small `ProfilingModule` registered in `fldataprofier/registry.py`. Each module reads `.csv` and `.parquet` through existing shared loaders, merges by `Date`/index or `join_key`, writes consistent artifacts, and keeps time order intact to reduce leakage.

**Tech Stack:** Python 3.12, pandas >=2.3.0,<3.0.0, numpy, scipy, scikit-learn, statsmodels, xgboost, optional lightgbm, optional catboost.

## Global Constraints

- All shell commands must be prefixed with `rtk`.
- Use existing CLI contract: `fldataprofiler fit FEATURES LABELS --module MODULE_NAME`.
- Supported input formats remain `.csv` and `.parquet`; reject other suffixes before report creation.
- Each module returns `ModuleResult` and writes artifacts under the selected `output_dir`.
- Default split style is expanding walk-forward validation ordered by time.
- Do not add GPU flags to existing modules; GPU-specific behavior belongs in separate modules such as `kmeans_gpu`.
- Preserve user changes in the worktree; never revert unrelated files.
- Keep implementation scoped to feature scoring modules, tests, registry, and docs.

---

## File Structure

- Create `fldataprofier/modules/time_series_scoring.py`: shared walk-forward split generator, numeric frame preparation, safe metric helpers, score aggregation, ranking artifact writers.
- Create `fldataprofier/modules/information_coefficient.py`: Pearson IC and Rank IC per walk-forward fold.
- Create `fldataprofier/modules/permutation_importance_ts.py`: time-series permutation importance with RandomForest models.
- Create `fldataprofier/modules/timeseries_importance.py`: combined walk-forward score that blends model importance, permutation importance, and Rank IC.
- Create `fldataprofier/modules/mutual_information.py`: mutual information scoring for regression and classification labels.
- Create `fldataprofier/modules/mrmr.py`: minimum-redundancy maximum-relevance ranking.
- Create `fldataprofier/modules/stability_selection.py`: repeated subsample feature stability scoring.
- Create `fldataprofier/modules/regularized_linear.py`: Lasso, ElasticNet, LogisticRegression L1/L2 coefficient ranking.
- Create `fldataprofier/modules/lightgbm.py`: optional LightGBM/CatBoost feature importance module with graceful missing dependency errors.
- Create `fldataprofier/modules/feature_interactions.py`: interaction feature search using pairwise products/differences/ratios and model-based ranking.
- Create `fldataprofier/modules/regime_scoring.py`: regime-aware scoring by volatility/return regimes.
- Modify `fldataprofier/registry.py`: register all new modules.
- Modify `README.md`: document module names, purpose, command examples, optional dependencies, and artifact names.
- Modify `pyproject.toml`: cap pandas below 3.0 and add optional dependency group for LightGBM/CatBoost if project style supports extras.
- Create `tests/test_time_series_scoring.py`: utility unit tests.
- Create `tests/test_information_coefficient.py`: IC module CLI and artifact tests.
- Create `tests/test_permutation_importance_ts.py`: permutation module smoke and ranking tests.
- Create `tests/test_timeseries_importance.py`: combined score smoke and registry tests.
- Create `tests/test_feature_selection_modules.py`: smoke tests for the remaining seven modules.

---

### Task 1: Shared Time-Series Scoring Utilities

**Files:**
- Create: `fldataprofier/modules/time_series_scoring.py`
- Test: `tests/test_time_series_scoring.py`

**Interfaces:**
- Produces: `walk_forward_splits(n_rows: int, min_train_size: int = 100, test_size: int = 50, step_size: int = 50, max_folds: int = 20) -> list[tuple[int, int, int, int]]`
- Produces: `prepare_numeric_matrix(df: pd.DataFrame, exclude: set[str]) -> tuple[pd.DataFrame, list[str]]`
- Produces: `aggregate_scores(rows: Iterable[dict[str, object]]) -> pd.DataFrame`
- Produces: `write_score_artifacts(report_dir: Path, raw_scores: pd.DataFrame, summary: pd.DataFrame) -> list[Path]`

- [ ] **Step 1: Write failing utility tests**

```python
def test_walk_forward_splits_are_expanding():
    from fldataprofier.modules.time_series_scoring import walk_forward_splits

    splits = walk_forward_splits(260, min_train_size=100, test_size=50, step_size=50)

    assert splits[:3] == [(0, 100, 100, 150), (0, 150, 150, 200), (0, 200, 200, 250)]


def test_aggregate_scores_sorts_by_abs_mean():
    from fldataprofier.modules.time_series_scoring import aggregate_scores

    result = aggregate_scores([
        {"feature": "weak", "label": "target", "score_name": "rank_ic", "score": 0.1, "samples": 50},
        {"feature": "strong", "label": "target", "score_name": "rank_ic", "score": -0.8, "samples": 50},
        {"feature": "strong", "label": "target", "score_name": "rank_ic", "score": -0.6, "samples": 50},
    ])

    assert result.iloc[0]["feature"] == "strong"
    assert round(float(result.iloc[0]["mean_score"]), 3) == -0.7
    assert round(float(result.iloc[0]["mean_abs_score"]), 3) == 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `rtk uv run python -m unittest tests.test_time_series_scoring -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'fldataprofier.modules.time_series_scoring'`.

- [ ] **Step 3: Implement shared utilities**

Implement `time_series_scoring.py` with:
- strict positive integer validation for split parameters;
- no split when `n_rows < min_train_size + test_size`;
- `pd.to_numeric(errors="coerce")` for features;
- removal of all-null and constant numeric columns;
- aggregation columns: `feature`, `label`, `score_name`, `mean_score`, `mean_abs_score`, `score_std`, `valid_folds`, `positive_fold_ratio`, `samples`;
- descending sort by `mean_abs_score`, then `valid_folds`, then `feature`.

- [ ] **Step 4: Run utility tests**

Run: `rtk uv run python -m unittest tests.test_time_series_scoring -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add fldataprofier/modules/time_series_scoring.py tests/test_time_series_scoring.py
rtk git commit -m "feat: add time series scoring utilities"
```

---

### Task 2: Information Coefficient Module

**Files:**
- Create: `fldataprofier/modules/information_coefficient.py`
- Modify: `fldataprofier/registry.py`
- Test: `tests/test_information_coefficient.py`

**Interfaces:**
- Consumes: `walk_forward_splits`, `prepare_numeric_matrix`, `aggregate_scores`, `write_score_artifacts`
- Produces module name: `information_coefficient`
- Produces artifacts: `fold_scores.csv`, `feature_scores.csv`, `top_features.csv`, `summary.json`, `report.md`, `report.html`

- [ ] **Step 1: Write failing module tests**

```python
def test_information_coefficient_ranks_signal_feature(tmp_path):
    from fldataprofier.modules.information_coefficient import InformationCoefficientModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=260)
    result = InformationCoefficientModule().run(feature_csv, label_csv, tmp_path / "out")

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    top = scores.sort_values("mean_abs_score", ascending=False).iloc[0]
    assert top["feature"] == "signal"
    assert top["score_name"] in {"pearson_ic", "rank_ic"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk uv run python -m unittest tests.test_information_coefficient -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement module**

Implement `InformationCoefficientModule.run(...)` by:
- loading inputs with `_read_table_with_date_index`;
- merging with `_merge_inputs`;
- selecting labels with `_select_targets`;
- sorting by datetime index when present;
- computing Pearson IC via `Series.corr(method="pearson")`;
- computing Rank IC via `Series.corr(method="spearman")`;
- requiring at least 20 non-null paired samples and at least two unique values on feature and label;
- writing artifacts using shared utility writers plus a short markdown/html report.

- [ ] **Step 4: Register module**

Add to `fldataprofier/registry.py`:

```python
from .modules.information_coefficient import InformationCoefficientModule

InformationCoefficientModule(),
```

- [ ] **Step 5: Run focused tests**

Run: `rtk uv run python -m unittest tests.test_information_coefficient -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add fldataprofier/modules/information_coefficient.py fldataprofier/registry.py tests/test_information_coefficient.py
rtk git commit -m "feat: add information coefficient module"
```

---

### Task 3: Time-Series Permutation Importance Module

**Files:**
- Create: `fldataprofier/modules/permutation_importance_ts.py`
- Modify: `fldataprofier/registry.py`
- Test: `tests/test_permutation_importance_ts.py`

**Interfaces:**
- Consumes: shared utility functions from Task 1
- Produces module name: `permutation_importance_ts`
- Produces artifacts: `fold_scores.csv`, `feature_scores.csv`, `top_features.csv`, `summary.json`, `report.md`, `report.html`

- [ ] **Step 1: Write failing tests**

```python
def test_permutation_importance_ts_ranks_predictive_feature(tmp_path):
    from fldataprofier.modules.permutation_importance_ts import PermutationImportanceTSModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=260)
    result = PermutationImportanceTSModule(n_estimators=25, random_state=7).run(
        feature_csv, label_csv, tmp_path / "out"
    )

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    assert scores.iloc[0]["feature"] == "signal"
    assert scores.iloc[0]["mean_score"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk uv run python -m unittest tests.test_permutation_importance_ts -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement module**

Implement walk-forward permutation scoring with:
- `RandomForestRegressor` and `r2_score` for numeric labels with more than 10 unique values;
- `RandomForestClassifier` and `accuracy_score` for categorical or low-cardinality labels;
- deterministic permutation via `np.random.default_rng(random_state + fold_index)`;
- score = baseline metric minus permuted metric;
- skip folds where the train or test target has fewer than two classes/unique values;
- defaults `n_estimators=100`, `max_features="sqrt"`, `n_jobs=-1`, `random_state=42`.

- [ ] **Step 4: Register module**

Add `PermutationImportanceTSModule()` to the registry under name `permutation_importance_ts`.

- [ ] **Step 5: Run focused tests**

Run: `rtk uv run python -m unittest tests.test_permutation_importance_ts -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add fldataprofier/modules/permutation_importance_ts.py fldataprofier/registry.py tests/test_permutation_importance_ts.py
rtk git commit -m "feat: add time series permutation importance"
```

---

### Task 4: Combined Time-Series Importance Module

**Files:**
- Create: `fldataprofier/modules/timeseries_importance.py`
- Modify: `fldataprofier/registry.py`
- Test: `tests/test_timeseries_importance.py`

**Interfaces:**
- Consumes: IC and permutation calculation helpers; if helpers are module-local in earlier tasks, extract them into `time_series_scoring.py` in this task.
- Produces module name: `timeseries_importance`
- Produces artifacts: `component_scores.csv`, `feature_scores.csv`, `top_features.csv`, `summary.json`, `report.md`, `report.html`

- [ ] **Step 1: Write failing test**

```python
def test_timeseries_importance_combines_component_scores(tmp_path):
    from fldataprofier.modules.timeseries_importance import TimeSeriesImportanceModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=260)
    result = TimeSeriesImportanceModule(n_estimators=25, random_state=11).run(
        feature_csv, label_csv, tmp_path / "out"
    )

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    assert scores.iloc[0]["feature"] == "signal"
    assert {"rank_ic", "permutation_importance"}.issubset(
        set(pd.read_csv(result.report_dir / "component_scores.csv")["score_name"])
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `rtk uv run python -m unittest tests.test_timeseries_importance -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement module**

Implement component blending:
- collect absolute Rank IC scores and permutation drops per feature/label/fold;
- normalize each component within `label` and `score_name` by max absolute value;
- combined score = average normalized component score;
- keep `component_count` and `valid_folds` in `feature_scores.csv`;
- write component scores separately for auditability.

- [ ] **Step 4: Register module and run tests**

Run: `rtk uv run python -m unittest tests.test_timeseries_importance -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add fldataprofier/modules/timeseries_importance.py fldataprofier/registry.py tests/test_timeseries_importance.py
rtk git commit -m "feat: add combined time series importance"
```

---

### Task 5: Mutual Information Module

**Files:**
- Create: `fldataprofier/modules/mutual_information.py`
- Modify: `fldataprofier/registry.py`
- Test: add case to `tests/test_feature_selection_modules.py`

**Interfaces:**
- Produces module name: `mutual_information`
- Uses `sklearn.feature_selection.mutual_info_regression` and `mutual_info_classif`

- [ ] **Step 1: Write failing smoke test**

```python
def test_mutual_information_writes_feature_scores(tmp_path):
    from fldataprofier.modules.mutual_information import MutualInformationModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=220)
    result = MutualInformationModule(random_state=5).run(feature_csv, label_csv, tmp_path / "out")

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    assert "signal" in set(scores["feature"])
    assert (result.report_dir / "summary.json").exists()
```

- [ ] **Step 2: Implement module**

Fit MI on complete numeric rows per label:
- regression path for numeric labels with more than 10 unique values;
- classification path for low-cardinality labels;
- fill missing feature values with train-column median;
- aggregate per label without walk-forward because MI is a univariate relevance screen.

- [ ] **Step 3: Run focused tests and commit**

Run: `rtk uv run python -m unittest tests.test_feature_selection_modules -v`

Expected: PASS for mutual information case.

Commit:

```bash
rtk git add fldataprofier/modules/mutual_information.py fldataprofier/registry.py tests/test_feature_selection_modules.py
rtk git commit -m "feat: add mutual information module"
```

---

### Task 6: mRMR Module

**Files:**
- Create: `fldataprofier/modules/mrmr.py`
- Modify: `fldataprofier/registry.py`
- Test: add case to `tests/test_feature_selection_modules.py`

**Interfaces:**
- Produces module name: `mrmr`
- Consumes MI relevance scores from Task 5 helper if extracted; otherwise compute MI directly.

- [ ] **Step 1: Write failing smoke test**

```python
def test_mrmr_outputs_redundancy_adjusted_scores(tmp_path):
    from fldataprofier.modules.mrmr import MRMRModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=220)
    result = MRMRModule(max_features=20, random_state=3).run(feature_csv, label_csv, tmp_path / "out")

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    assert {"relevance", "redundancy", "mrmr_score"}.issubset(scores.columns)
```

- [ ] **Step 2: Implement module**

Implement greedy mRMR:
- relevance = MI(feature, label);
- redundancy = mean absolute Spearman correlation with already selected features;
- mRMR score = relevance - redundancy;
- select at most `max_features` per label;
- skip constant features and rows without target.

- [ ] **Step 3: Run focused tests and commit**

Run: `rtk uv run python -m unittest tests.test_feature_selection_modules -v`

Expected: PASS including mRMR.

Commit:

```bash
rtk git add fldataprofier/modules/mrmr.py fldataprofier/registry.py tests/test_feature_selection_modules.py
rtk git commit -m "feat: add mrmr feature selection"
```

---

### Task 7: Stability Selection Module

**Files:**
- Create: `fldataprofier/modules/stability_selection.py`
- Modify: `fldataprofier/registry.py`
- Test: add case to `tests/test_feature_selection_modules.py`

**Interfaces:**
- Produces module name: `stability_selection`

- [ ] **Step 1: Write failing smoke test**

```python
def test_stability_selection_writes_selection_frequency(tmp_path):
    from fldataprofier.modules.stability_selection import StabilitySelectionModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=240)
    result = StabilitySelectionModule(n_resamples=8, random_state=13).run(
        feature_csv, label_csv, tmp_path / "out"
    )

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    assert "selection_frequency" in scores.columns
    assert scores["selection_frequency"].between(0, 1).all()
```

- [ ] **Step 2: Implement module**

Use repeated subsamples:
- sample 70 percent of rows without replacement per resample;
- StandardScaler + Lasso for regression labels;
- StandardScaler + LogisticRegression penalty `l1`, solver `liblinear` for classification labels;
- selected = non-zero absolute coefficient;
- score = selection count / valid resamples.

- [ ] **Step 3: Run focused tests and commit**

Run: `rtk uv run python -m unittest tests.test_feature_selection_modules -v`

Expected: PASS including stability selection.

Commit:

```bash
rtk git add fldataprofier/modules/stability_selection.py fldataprofier/registry.py tests/test_feature_selection_modules.py
rtk git commit -m "feat: add stability selection module"
```

---

### Task 8: Regularized Linear Module

**Files:**
- Create: `fldataprofier/modules/regularized_linear.py`
- Modify: `fldataprofier/registry.py`
- Test: add case to `tests/test_feature_selection_modules.py`

**Interfaces:**
- Produces module name: `regularized_linear`

- [ ] **Step 1: Write failing smoke test**

```python
def test_regularized_linear_writes_coefficients(tmp_path):
    from fldataprofier.modules.regularized_linear import RegularizedLinearModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=220)
    result = RegularizedLinearModule(random_state=17).run(feature_csv, label_csv, tmp_path / "out")

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    assert {"coefficient", "abs_coefficient", "model_type"}.issubset(scores.columns)
```

- [ ] **Step 2: Implement module**

Fit scaled regularized models:
- `ElasticNetCV` for regression labels;
- `LogisticRegressionCV` with `l1` and `liblinear` for classification labels;
- report signed and absolute coefficients;
- include model diagnostics in `summary.json`.

- [ ] **Step 3: Run focused tests and commit**

Run: `rtk uv run python -m unittest tests.test_feature_selection_modules -v`

Expected: PASS including regularized linear.

Commit:

```bash
rtk git add fldataprofier/modules/regularized_linear.py fldataprofier/registry.py tests/test_feature_selection_modules.py
rtk git commit -m "feat: add regularized linear feature scoring"
```

---

### Task 9: LightGBM/CatBoost Module

**Files:**
- Create: `fldataprofier/modules/lightgbm.py`
- Modify: `fldataprofier/registry.py`
- Modify: `pyproject.toml`
- Test: add case to `tests/test_feature_selection_modules.py`

**Interfaces:**
- Produces module name: `lightgbm`
- Optional imports: `lightgbm.LGBMRegressor`, `lightgbm.LGBMClassifier`; fallback to `catboost` only if LightGBM is unavailable and CatBoost is installed.

- [ ] **Step 1: Write missing-dependency and smoke tests**

```python
def test_lightgbm_module_has_actionable_missing_dependency_message(tmp_path, monkeypatch):
    from fldataprofier.modules.lightgbm import LightGBMModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=220)
    monkeypatch.setattr("fldataprofier.modules.lightgbm._load_lightgbm", lambda: None)

    with pytest.raises(RuntimeError, match="Install optional dependency"):
        LightGBMModule().run(feature_csv, label_csv, tmp_path / "out")
```

- [ ] **Step 2: Implement module**

Implement lazy dependency loading:
- do not import LightGBM at package import time;
- raise `RuntimeError("Install optional dependency with: uv add lightgbm")` when unavailable;
- compute gain/split importances per label;
- use walk-forward folds for evaluation metrics and full-fit model for final importances.

- [ ] **Step 3: Add optional dependency metadata**

If project uses dependency groups, add:

```toml
[project.optional-dependencies]
gbm = ["lightgbm>=4.5.0", "catboost>=1.2.7"]
```

If the project does not use extras, document install commands in README and leave runtime dependencies unchanged.

- [ ] **Step 4: Run focused tests and commit**

Run: `rtk uv run python -m unittest tests.test_feature_selection_modules -v`

Expected: PASS with missing-dependency test passing even when LightGBM is absent.

Commit:

```bash
rtk git add fldataprofier/modules/lightgbm.py fldataprofier/registry.py pyproject.toml tests/test_feature_selection_modules.py
rtk git commit -m "feat: add optional lightgbm feature scoring"
```

---

### Task 10: Feature Interaction Search Module

**Files:**
- Create: `fldataprofier/modules/feature_interactions.py`
- Modify: `fldataprofier/registry.py`
- Test: add case to `tests/test_feature_selection_modules.py`

**Interfaces:**
- Produces module name: `feature_interactions`

- [ ] **Step 1: Write failing smoke test**

```python
def test_feature_interactions_writes_interaction_scores(tmp_path):
    from fldataprofier.modules.feature_interactions import FeatureInteractionsModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=220)
    result = FeatureInteractionsModule(max_base_features=8, max_pairs=20).run(
        feature_csv, label_csv, tmp_path / "out"
    )

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    assert {"interaction", "left_feature", "right_feature", "operation"}.issubset(scores.columns)
```

- [ ] **Step 2: Implement module**

Search interactions:
- select base features from top MI scores;
- generate operations `product`, `difference`, `ratio`;
- guard ratio denominator with absolute value > `1e-12`;
- score generated interactions using MI against each label;
- write `generated_interactions.csv` with formula metadata.

- [ ] **Step 3: Run focused tests and commit**

Run: `rtk uv run python -m unittest tests.test_feature_selection_modules -v`

Expected: PASS including feature interactions.

Commit:

```bash
rtk git add fldataprofier/modules/feature_interactions.py fldataprofier/registry.py tests/test_feature_selection_modules.py
rtk git commit -m "feat: add feature interaction search"
```

---

### Task 11: Regime-Aware Scoring Module

**Files:**
- Create: `fldataprofier/modules/regime_scoring.py`
- Modify: `fldataprofier/registry.py`
- Test: add case to `tests/test_feature_selection_modules.py`

**Interfaces:**
- Produces module name: `regime_scoring`

- [ ] **Step 1: Write failing smoke test**

```python
def test_regime_scoring_writes_regime_column(tmp_path):
    from fldataprofier.modules.regime_scoring import RegimeScoringModule

    feature_csv, label_csv = make_signal_dataset(tmp_path, rows=260)
    result = RegimeScoringModule(n_regimes=3).run(feature_csv, label_csv, tmp_path / "out")

    scores = pd.read_csv(result.report_dir / "feature_scores.csv")
    assert "regime" in scores.columns
    assert "regime_count" in pd.read_json(result.report_dir / "summary.json", typ="series")
```

- [ ] **Step 2: Implement module**

Define regimes:
- if feature columns contain `volatility`, `atr`, or `range`, use the first matching numeric column;
- otherwise compute rolling absolute return proxy from the first numeric label with `window=20`;
- bin into quantile regimes named `low`, `mid`, `high` for `n_regimes=3`;
- compute IC and Rank IC per regime;
- report regime-specific and overall aggregate scores.

- [ ] **Step 3: Run focused tests and commit**

Run: `rtk uv run python -m unittest tests.test_feature_selection_modules -v`

Expected: PASS including regime scoring.

Commit:

```bash
rtk git add fldataprofier/modules/regime_scoring.py fldataprofier/registry.py tests/test_feature_selection_modules.py
rtk git commit -m "feat: add regime aware feature scoring"
```

---

### Task 12: CLI Registry, README, and Full Verification

**Files:**
- Modify: `fldataprofier/registry.py`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Test: all tests

**Interfaces:**
- Produces registered modules: `timeseries_importance`, `permutation_importance_ts`, `information_coefficient`, `mutual_information`, `mrmr`, `stability_selection`, `regularized_linear`, `lightgbm`, `feature_interactions`, `regime_scoring`

- [ ] **Step 1: Verify registry exposes every module**

Add or update a registry test:

```python
def test_feature_scoring_modules_are_registered():
    from fldataprofier.registry import available_modules

    expected = {
        "timeseries_importance",
        "permutation_importance_ts",
        "information_coefficient",
        "mutual_information",
        "mrmr",
        "stability_selection",
        "regularized_linear",
        "lightgbm",
        "feature_interactions",
        "regime_scoring",
    }
    assert expected.issubset(set(available_modules()))
```

- [ ] **Step 2: Document usage**

Add README commands:

```bash
fldataprofiler fit VN30F1M_5m_feature.csv VN30F1M_5m_label.csv --module information_coefficient
fldataprofiler fit VN30F1M_5m_feature.parquet VN30F1M_5m_label.parquet --module permutation_importance_ts
fldataprofiler fit VN30F1M_5m_feature.csv VN30F1M_5m_label.csv --module timeseries_importance
```

Document that `lightgbm` requires optional install and that all reports emit `feature_scores.csv`, `top_features.csv`, `summary.json`, `report.md`, and `report.html` unless a module has extra component artifacts.

- [ ] **Step 3: Verify pandas cap**

Ensure `pyproject.toml` contains:

```toml
"pandas>=2.3.0,<3.0.0"
```

This prevents the cuDF/cuML compatibility failure where pandas 3 removed `pandas.api.types.is_interval`.

- [ ] **Step 4: Run full suite**

Run:

```bash
rtk uv run python -m unittest discover -s tests -v
rtk uv run python -m compileall -q fldataprofier
```

Expected: all tests PASS and compileall exits 0.

- [ ] **Step 5: Commit final docs and integration**

```bash
rtk git add README.md pyproject.toml fldataprofier/registry.py tests
rtk git commit -m "docs: document feature scoring modules"
```

---

## Self-Review

- Spec coverage: The plan covers all ten requested methods: time-series importance, permutation importance, information coefficient, mutual information, mRMR, stability selection, regularized linear, LightGBM/CatBoost, feature interactions, and regime scoring.
- Artifact consistency: Every module writes score CSVs plus JSON and markdown/html reports; modules with components write extra audit CSVs.
- Leakage control: Walk-forward validation is used for predictive model scoring; full-sample univariate screens are explicitly labeled as relevance screens.
- Dependency control: LightGBM/CatBoost are optional and lazily imported; pandas is capped below 3.0 to avoid cuDF/cuML breakage.
- Execution path: Tasks are independently testable and ordered so shared utilities land before modules that consume them.
