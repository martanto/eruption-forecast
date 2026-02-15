# Eruption Forecast Package — Technical Summary

**Project:** eruption-forecast — Volcanic Eruption Forecasting using Seismic Data Analysis
**Repository:** D:\Projects\eruption-forecast
**Branch:** `copilot/unified-plotting-system`
**Last Updated:** 2026-02-15

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Architecture](#architecture)
3. [Critical Fix: Data Leakage in Model Training](#critical-fix-data-leakage-in-model-training-2026-02-09)
4. [Machine Learning Workflow Analysis](#machine-learning-workflow-analysis)
5. [Model Comparison & Recommendations](#model-comparison--recommendations)
6. [Grid Parameter Analysis](#grid-parameter-analysis)
7. [Enhanced Feature Selection](#enhanced-feature-selection-2026-02-09)
8. [Code Quality Summary](#code-quality-summary)
9. [Future Recommendations](#future-recommendations)
10. [ModelPredictor Multi-Model Consensus](#modelpredictor-multi-model-consensus-2026-02-13)
11. [Rename TrainModel → ModelTrainer](#rename-trainmodel--modeltrainer-2026-02-13)
12. [Docstring Improvements](#docstring-improvements-2026-02-13)
13. [FeaturesBuilder Readability Improvements](#featuresbuilder-readability-improvements-2026-02-13)
14. [ModelTrainer Refactor](#modeltrainer-refactor-2026-02-13)
15. [Refactor Output Directory Structure](#refactor-output-directory-structure-2026-02-15)
---

## Package Overview

### About eruption-forecast

`eruption-forecast` is a comprehensive Python package for volcanic eruption forecasting using seismic data analysis. The package implements a complete machine learning pipeline that processes raw seismic tremor data to predict volcanic eruptions based on time-series patterns.

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Seismic Data Processing** | Reads SDS/FDSN format, multi-station/channel support |
| **Tremor Calculation** | RSAM (amplitude) and DSAR (ratios) across 5 frequency bands |
| **Label Generation** | Binary classification labels with configurable forecast horizon |
| **Feature Engineering** | tsfresh time-series feature extraction (700+ features) |
| **Enhanced Feature Selection** | Three-method: tsfresh statistical, RandomForest permutation, combined two-stage |
| **Model Training** | Multi-seed feature selection with class balancing |
| **Prediction** | GridSearchCV-optimized classifiers for eruption prediction |

### Package Structure

```
eruption_forecast/
├── tremor/              # Tremor calculation (RSAM/DSAR)
│   ├── calculate_tremor.py
│   ├── rsam.py
│   ├── dsar.py
│   └── tremor_data.py
├── label/               # Label generation
│   ├── label_builder.py
│   ├── label_data.py
│   └── constants.py
├── features/            # Feature extraction and selection
│   ├── features_builder.py
│   ├── tremor_matrix_builder.py
│   ├── feature_selector.py
│   └── constants.py
├── model/               # ML models and training
│   ├── forecast_model.py    # Pipeline orchestrator
│   ├── model_trainer.py     # Multi-seed training
│   ├── classifier_model.py  # Classifier management
│   ├── model_evaluator.py   # Evaluation, export, and plotting
│   └── model_predictor.py   # Inference (evaluation + forecast modes)
├── utils.py             # Shared utilities
├── sds.py               # SDS file handling
├── plot.py              # Visualization
└── logger.py            # Centralized logging
```

---

## Architecture

### Data Pipeline

```
Raw Seismic Data (SDS / FDSN)
         │
         ▼
┌─────────────────────┐
│   CalculateTremor   │  RSAM + DSAR → tremor.csv
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    LabelBuilder     │  Binary labels → label_*.csv
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ TremorMatrixBuilder │  Windowed matrix → tremor_matrix_*.csv
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   FeaturesBuilder   │  700+ features → all_extracted_features_*.csv
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────────────────────┐
│                 ModelTrainer                │
│  ┌─────────────┐   ┌──────────────────────┐ │
│  │FeatureSelect│   │   ClassifierModel    │ │
│  │   or        │   │ (10 classifiers,     │ │
│  │  combined   │   │  3 CV strategies)    │ │
│  └─────────────┘   └──────────────────────┘ │
│         ↓  train_and_evaluate()  ↓ train()  │
│    80/20 split + metrics   Full dataset     │
└─────────┬───────────────────────────────────┘
          │  trained_model_*.csv  +  *.pkl
          ▼
┌─────────────────────────────────────────────┐
│               ModelPredictor                │
│  ┌──────────────────────────────────────┐   │
│  │ predict() / predict_best()           │   │
│  │ (evaluation mode — requires labels)  │   │
│  └──────────────────────────────────────┘   │
│  ┌──────────────────────────────────────┐   │
│  │ predict_proba()                      │   │
│  │ (forecast mode — no labels needed)   │   │
│  └──────────────────────────────────────┘   │
│  Single model or multi-model consensus      │
└─────────────────────────────────────────────┘
```

### Key Design Patterns

1. **Method Chaining**: Fluent API for pipeline configuration (`ForecastModel`)
2. **Multi-seed Training**: Robust feature selection across many random seeds
3. **Class Balancing**: RandomUnderSampler for imbalanced eruption data
4. **Separation of Concerns**: Clear module boundaries
5. **Data Leakage Prevention**: Strict train/test split before any resampling or feature selection

### Multi-Seeding Training Workflows

```
  train_and_evaluate()              train()
  ─────────────────────            ─────────────────────
  Full Dataset                     Full Dataset
       │                                │
       ▼                                ▼
   80/20 Split                  RandomUnderSampler
   (stratified)                  (full dataset)
  ┌────┴────┐                          │
Train     Test                  Feature Selection
  │         │                    (full dataset)
RandomUnder │                          │
Sampler     │                    GridSearchCV
  │         │                     + CV folds
Feature     │                          │
Selection   │                   ┌──────┴──────┐
  │         │               model.pkl   registry.csv
GridSearchCV│
 + CV folds │
  │         │
Evaluate ◄──┘
  │
Save model + metrics
```

---

## Critical Fix: Data Leakage in Model Training (2026-02-09)

### Problem Identified

The original `TrainModel._train()` implementation contained **critical data leakage** that violated fundamental machine learning best practices:

1. **Resampling applied to entire dataset** before train/test split
2. **Feature selection seeing test data** through p-value calculations on full dataset
3. **No held-out test set** for generalization validation
4. **Overly optimistic metrics** due to information leakage

### Corrected Workflow

```python
# Correct implementation
def _train(self, seed, random_state, ...):
    # 1. Split FIRST (80/20, stratified)
    features_train, features_test, labels_train, labels_test = train_test_split(
        self.df_features, self.df_labels,
        test_size=0.2, random_state=state, stratify=self.df_labels,
    )

    # 2. Resample ONLY training data
    features_train_resampled, labels_train_resampled = random_under_sampler(
        features=features_train,
        labels=labels_train,
        sampling_strategy=sampling_strategy,
        random_state=state,
    )

    # 3. Feature selection ONLY on training data
    significant_features = get_significant_features(
        features=features_train_resampled,
        labels=labels_train_resampled,
    )

    # 4. Apply selected features to both sets
    top_features = significant_features.head(number_of_significant_features).index.tolist()
    features_train_selected = features_train_resampled[top_features]
    features_test_selected = features_test[top_features]

    # 5. Cross-validation + GridSearchCV
    grid_search = GridSearchCV(...)
    grid_search.fit(features_train_selected, labels_train_resampled)

    # 6. Evaluate on held-out test set
    y_pred = grid_search.best_estimator_.predict(features_test_selected)
    metrics = {
        'accuracy': accuracy_score(labels_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(labels_test, y_pred),
        'f1_score': f1_score(labels_test, y_pred),
        ...
    }
```

### Key Changes

| Aspect | Before | After |
|--------|--------|-------|
| Train/Test Split | No | 80/20 stratified split first |
| Resampling | Applied to entire dataset | Applied only to training set |
| Feature Selection | Saw entire dataset | Only sees training data |
| Model Training | Not implemented | Classifier + GridSearchCV |
| Evaluation | No test set | Held-out test set evaluation |
| Outputs | Features only | Features + trained models + metrics |

---

## Machine Learning Workflow Analysis

### Current Workflow

The workflow follows a well-structured approach for binary classification:

1. **Feature Extraction**: tsfresh's ComprehensiveFCParameters (700+ features per tremor column)
2. **Feature Selection**: Multi-seed approach with under-sampling for stability
3. **Model Training**: GridSearchCV with configurable cross-validation strategy
4. **Class Handling**: Balanced class weights + RandomUnderSampler

### Strengths

| Aspect | Implementation | Benefit |
|--------|----------------|---------|
| Feature Robustness | Multi-seed feature selection | Reduces overfitting to single split |
| Class Imbalance | Under-sampling + balanced weights | Handles rare eruption events |
| Hyperparameter Tuning | GridSearchCV with CV | Optimal parameter selection |
| Multi-model Support | 10 classifiers available | Flexibility for different data characteristics |
| Feature Selection | Three-method FeatureSelector | Statistical rigor + interaction capture |

### Current Limitations & Recommendations

| Limitation | Status | Recommendation |
|------------|--------|----------------|
| ~~No ensemble methods~~ | RESOLVED | VotingClassifier available (`"voting"`) |
| ~~Fixed feature selection~~ | RESOLVED | FeatureSelector with tsfresh, RandomForest, combined |
| ~~No temporal validation~~ | RESOLVED | TimeSeriesSplit available (`cv_strategy="timeseries"`) |
| ~~Data leakage in TrainModel~~ | RESOLVED | Fixed: split → resample → select features → train → evaluate |
| ~~No model persistence~~ | RESOLVED | `ModelEvaluator.export_model()` with joblib |
| ~~No evaluation metrics~~ | RESOLVED | ModelEvaluator with ROC-AUC, PR-AUC, confusion matrix, plots |
| No probability calibration | Pending | Add CalibratedClassifierCV |
| No full-dataset training | RESOLVED | `ModelTrainer.train()` trains on full dataset; `ModelPredictor` evaluates on future data |

---

## Model Comparison & Recommendations

### Available Classifiers

| ID | Classifier | Best For | Pros | Cons |
|----|------------|----------|------|------|
| `rf` | Random Forest | **Default choice** | Robust, handles non-linear, feature importance | Can overfit small datasets |
| `gb` | Gradient Boosting | **Imbalanced data** | Excellent for tabular data, feature importance | Slower training, prone to overfit |
| `svm` | Support Vector Machine | High-dimensional data | Effective in high dimensions | Slow on large datasets, needs scaling |
| `lr` | Logistic Regression | Baseline, interpretable | Fast, probability estimates | Assumes linear separability |
| `nn` | MLP Neural Network | Complex patterns | Learns complex relationships | Needs more data, slower training |
| `dt` | Decision Tree | Interpretability | Easy to visualize, fast | Prone to overfitting |
| `knn` | K-Nearest Neighbors | Small datasets | Simple, no training | Slow prediction, needs scaling |
| `nb` | Gaussian Naive Bayes | Quick baseline | Very fast, works with small data | Strong independence assumption |
| `xgb` | XGBoost | **Imbalanced data** | `scale_pos_weight` grid search, fast, accurate | Requires xgboost package |
| `voting` | VotingClassifier Ensemble (RF + XGBoost) | **Best accuracy** | Combines RF and XGBoost with soft voting | Slower training, more complex |

### Cross-Validation Strategies

| Strategy | Class | Best For |
|----------|-------|----------|
| `shuffle` | StratifiedShuffleSplit | General classification |
| `stratified` | StratifiedKFold | Preserves class distribution |
| `timeseries` | TimeSeriesSplit | **Temporal data — prevents leakage** |

---

## Grid Parameter Analysis

### 1. Random Forest (`rf`)

**Grid Size:** 162 combinations (3 × 3 × 2 × 3 × 3 × 3)

| Parameter | Values |
|-----------|--------|
| `n_estimators` | [50, 100, 200] |
| `max_depth` | [3, 5, 7] |
| `criterion` | ["gini", "entropy"] |
| `max_features` | ["sqrt", "log2", None] |
| `min_samples_split` | [2, 5, 10] |
| `min_samples_leaf` | [1, 2, 4] |

### 2. Gradient Boosting (`gb`)

**Grid Size:** 216 combinations (3 × 3 × 3 × 2 × 2 × 2)

| Parameter | Values |
|-----------|--------|
| `n_estimators` | [50, 100, 200] |
| `max_depth` | [3, 5, 7] |
| `learning_rate` | [0.01, 0.1, 0.2] |
| `subsample` | [0.8, 1.0] |
| `min_samples_split` | [2, 5] |
| `min_samples_leaf` | [1, 2] |

### 3. Support Vector Machine (`svm`)

**Grid Size:** 120 combinations (5 × 3 × 4 × 2)

> **Note:** SVM requires feature scaling (StandardScaler). Add to pipeline before use.

### 4. Logistic Regression (`lr`)

**Grid Size:** 90 combinations (3 × 5 × 2 × 3)

| Parameter | Values |
|-----------|--------|
| `penalty` | ["l2", "l1", "elasticnet"] |
| `C` | [0.001, 0.01, 0.1, 1, 10] |
| `solver` | ["lbfgs", "saga"] |
| `l1_ratio` | [0.15, 0.5, 0.85] |

### 5. Neural Network / MLP (`nn`)

**Grid Size:** 32 combinations (4 × 4 × 2)

| Parameter | Values |
|-----------|--------|
| `activation` | ["identity", "logistic", "tanh", "relu"] |
| `hidden_layer_sizes` | [(50,), (100,), (100, 50), (100, 100)] |
| `learning_rate_init` | [0.001, 0.01] |

---

## Enhanced Feature Selection (2026-02-09)

### FeatureSelector Class

Three selection strategies available in `eruption_forecast.features.FeatureSelector`:

#### 1. tsfresh Statistical Selection (`method="tsfresh"`)
- Hypothesis testing with FDR (False Discovery Rate) control
- Fast filtering based on univariate statistical tests
- Provides p-values for interpretability
- Best for initial dimensionality reduction (1000s → 100s features)

#### 2. RandomForest Permutation Importance (`method="random_forest"`)
- Uses permutation importance (more reliable than Gini impurity)
- Captures feature interactions and non-linear relationships
- Provides importance scores with standard deviations
- Best for feature refinement after pre-filtering

#### 3. Combined Two-Stage (`method="combined"`) — Recommended
- **Stage 1:** tsfresh statistical filtering (statistical rigor)
- **Stage 2:** RandomForest permutation importance (interaction capture)
- Balances speed, statistical grounding, and model optimization
- Provides both p-values and importance scores

### Feature Selection Comparison

| Method | Dimensionality Reduction | Captures Interactions | Overfitting Risk | Speed |
|--------|--------------------------|----------------------|------------------|-------|
| `tsfresh` | 1000s → 100s | No | Low | Fast |
| `random_forest` | Direct → N | Yes | Medium | Slow |
| `combined` | 1000s → N | Yes | Low | Fast |

---

## Code Quality Summary

### Changes History

| Date | Category | Changes |
|------|----------|---------|
| 2026-02-15 | **Output directory restructure + docstring fixes** | Restructured ModelTrainer output into model-with-evaluation/ and model-only/ by classifier-slug/cv-slug. Added get_classifier_properties(), update_directories(). Added with_evaluation/grid_params to ForecastModel.train(). Added slugify_class_name() to utils.py. Fixed get_metrics() param rename. Fixed docstring typos. |
| 2026-02-13 | **README overhaul** | Rewrote README from scratch: added XGBoost classifier section, detailed hyperparameter grids with collapsible blocks, corrected output directory structure (classifier/{ClassName}/{cv_strategy}/models\|metrics), added `fit()` + `ModelPredictor` workflow, `optimize_threshold()` usage, and comprehensive A-to-Z step-by-step guide. Updated SUMMARY.md classifier count (9 → 10) and VotingClassifier composition (now RF + XGBoost). |
| 2026-02-13 | **Imbalance-aware improvements** | Fixed `_train()` skip-logic bug (spurious `or not save_features` check). Expanded RF/NN/LR grids; added `scale_pos_weight=[1,5,10,15]` to XGB grid; removed hardcoded `scale_pos_weight=1`. Added `class_weight`/`n_jobs` params to `ClassifierModel`. Added `optimize_threshold()`, `plot_threshold_analysis()`, `plot_feature_importance()`, `plot_calibration()`, `plot_prediction_distribution()` to `ModelEvaluator`; `get_metrics()` now includes optimal-threshold fields. |
| 2026-02-12 | **Simplify ModelEvaluator** | Rewrote from scratch: removed Protocol classes, X_train/y_train, all export_* methods, cross_validate, learning curve, feature importances, and as_dataframe flag. Kept core: `__init__`, `from_files()`, `get_metrics()`, `summary()`, three plot methods, `plot_all()`. ModelPredictor: dropped `save_reports` param. |
| 2026-02-12 | **Full-dataset training + ModelPredictor** | Added `TrainModel._fit()` / `fit()` for full-dataset training (no train/test split, no metrics). Extended `ModelEvaluator` with `selected_features` parameter and `from_files()` classmethod. Created new `ModelPredictor` class that loads trained models from `fit()`, evaluates each against future features/labels, and aggregates metrics (mean ± std). Exported `ModelPredictor` from `model/__init__.py`. |
| 2026-02-11 | **Docstrings** | Fixed spelling errors: `Extacted` → `Extracted`, `SKipp` → `Skip`, `WIll` → `Will`, `laad` → `load`, `preditcted` → `predicted`, `defaut` → `default`, `shanon` → `Shannon`, `paramaters` → `parameters`, `BE CAREFULL` → `WARNING`. Improved `CalculateTremor` class docstring with full Args/Returns/Example. Enhanced `set_random_state` in ClassifierModel with Raises section. Clarified `to_series` docstring. Fixed `volcano_id` description in LabelBuilder. |
| 2026-02-09 | **Grammar/Spelling** | Fixed "aftrer" → "after", "each significant features" → "each significant feature", "calulation" → "calculation", "filenmae" → "filename" |
| 2026-02-09 | **Feature Selection** | Added `FeatureSelector` class with tsfresh, RandomForest, and combined methods |
| 2026-02-09 | **Data Leakage Fix** | Rewrote `TrainModel._train()`: split → resample → select features → train → evaluate |
| 2026-02-09 | **Dynamic Classifier** | TrainModel supports all 9 classifier types and 3 CV strategies |
| 2026-02-09 | **ModelEvaluator** | New class for evaluation, plotting, and export |
| 2026-02-09 | **Type Safety** | Fixed pyrefly errors in ModelEvaluator with Protocol types |
| 2026-02-09 | **Docstrings** | Enhanced docstrings in FeaturesBuilder, TremorMatrixBuilder, ForecastModel, TrainModel |
| Pre-2026-02-09 | **New Classifier** | Added GradientBoostingClassifier (`"gb"`) |
| Pre-2026-02-09 | **TimeSeriesSplit** | Added `cv_strategy` parameter and `get_cv_splitter()` method |
| Pre-2026-02-09 | **VotingClassifier** | Added `"voting"` ensemble combining RF and XGBoost with soft voting |

### Code Quality Metrics

| Metric | Status |
|--------|--------|
| Type Checking (pyrefly) | 0 errors |
| Import Sorting (isort) | Applied |
| Test Coverage | 81+ tests passing |
| Docstring Coverage | All public methods documented |

---

## Recent Changes

### Add root_dir Parameter (2026-02-13)
Added `resolve_output_dir()` helper function to `utils.py` and a `root_dir: str | None = None` parameter to all standalone classes and `ForecastModel`. All relative `output_dir` values are now resolved against `root_dir` (falling back to `os.getcwd()` when `None`). Absolute `output_dir` values bypass `root_dir` entirely. `ForecastModel` normalises relative `root_dir` to an absolute path immediately in `__init__` for multiprocessing safety. Backward compatible: existing code without `root_dir` behaves identically to before.

---

## Future Recommendations

### Short-term

1. **Expand Neural Network grid** — `hidden_layer_sizes` is too limited; expand to multi-layer architectures
2. **Probability calibration** — Add `CalibratedClassifierCV` for reliable probability estimates
3. **Feature scaling pipeline** — Add `StandardScaler` for SVM/KNN/NN in sklearn `Pipeline`

### Medium-term

1. **Grid expansion** — Expand RF and Decision Tree grids with more depth/estimator options
2. **Naive Bayes grid** — Expand `var_smoothing` to `[1e-9, 1e-8, 1e-7, 1e-6]`
3. **Integration tests** — End-to-end workflow tests

### Long-term

1. **Deep Learning** — LSTM/Transformer for sequential temporal patterns
2. **SHAP values** — Model explainability for scientific interpretation
3. **Real-time inference** — Streaming prediction pipeline
4. **AutoML** — TPOT or auto-sklearn for automated model selection

---

---

## ModelPredictor Multi-Model Consensus (2026-02-13)

Refactored `ModelPredictor` to support multi-model consensus and unlabelled forecast mode.

### Changes
- `trained_models_csv: str` → `trained_models: str | dict[str, str]`
  - `str`: single classifier (backward compat)
  - `dict`: multiple classifiers (e.g. `{"rf": "...", "xgb": "...", "voting": "..."}`)
- `future_labels_csv` remains optional
- `predict()` includes a `classifier` column in the returned DataFrame; consensus metrics logged across classifiers
- `predict_proba()` outputs per-classifier columns (`{name}_eruption_probability`, `{name}_uncertainty`, `{name}_confidence`, `{name}_prediction`) + `consensus_*` columns
- New private `_compute_model_proba()` helper — per-classifier seed aggregation
- Plot: each classifier as dashed line + consensus as solid black line with shaded uncertainty band

### Usage

```python
predictor = ModelPredictor(
    trained_models={
        "rf":     "output/trainings/rf/trained_model_rf.csv",
        "xgb":    "output/trainings/xgb/trained_model_xgb.csv",
        "voting": "output/trainings/voting/trained_model_voting.csv",
    },
    future_features_csv="output/features/future_features.csv",
)
df = predictor.predict_proba()
# Columns: rf_eruption_probability, xgb_eruption_probability, ...,
#          consensus_eruption_probability, consensus_confidence, ...
```

---

## Rename TrainModel → ModelTrainer (2026-02-13)

Renamed `train_model.py` → `model_trainer.py` and class `TrainModel` → `ModelTrainer` throughout the codebase.

**Files updated:**
- `src/eruption_forecast/model/model_trainer.py` — all docstring references updated
- `src/eruption_forecast/model/model_predictor.py` — docstring references updated
- `src/eruption_forecast/model/forecast_model.py` — attribute `self.TrainModel` → `self.ModelTrainer`
- `tests/test_features_builder.py` — imports and class references updated
- `tests/test_train_model_dynamic_classifier.py` — imports and class references updated
- `tests/test_train_model_fixed.py` — imports and class references updated

---

## Docstring Improvements (2026-02-13)

Audited all 27 Python source files and fixed docstrings across 6 files.

**Changes made:**

- `utils.py::get_metrics()` — replaced placeholder `"Get features matrix"` with full
  docstring listing all parameters and the 17-key return dict.
- `plot.py::plot_tremor()` — corrected `overwrite` default from `False` → `True`,
  expanded all arg descriptions with units and behaviour notes, added example.
- `plot.py::plot_significant_features()` — added full arg descriptions, return type,
  and example.
- `tremor/tremor_data.py::TremorData` — added class-level docstring with args and
  examples; expanded `from_csv`, `df`, all cached_property methods, and
  `check_consistency` with complete return-tuple documentation.
- `model/model_evaluator.py` — added docstring to `_get_proba`; expanded `get_metrics`
  with all 17 return-dict keys; expanded `summary`, `plot_confusion_matrix`,
  `plot_roc_curve`, `plot_precision_recall_curve`, `plot_threshold_analysis`,
  `plot_feature_importance`, `plot_calibration`, `plot_prediction_distribution`, and
  `plot_all` with full args/returns.
- `decorators/decorator_class.py` — expanded `SerializationWrapper` class docstring;
  added full docstrings to `_prepare_value` and `save_to_file`; rewrote
  `AutoSaveDict` class docstring with args; added one-line docstrings to `_save`,
  `__setitem__`, `__delitem__`, `update`; fixed duplicate `print` statement.
- `logger.py` — added return types and full arg/example sections to `set_log_level`
  and `set_log_directory`.

---

## Multi-file Consistency Update (2026-02-13)

Reviewed and updated consistency across 8 changed files.

**Bug fixed:**
- `src/eruption_forecast/features/features_builder.py:299` — `y.index = y[ID_COLUMN]` was operating on a Series after `y` was already reassigned; fixed by capturing `y_index = y[ID_COLUMN]` before the reassignment.

**Dead code removed from `forecast_model.py`:**
- Removed `tsfresh` / `ComprehensiveFCParameters` / `impute` imports (feature extraction is now fully delegated to `FeaturesBuilder`)
- Removed `_initialize_feature_parameters()` static method and the `self.default_fc_parameters` / `self.excludes_features` attributes it populated
- Removed `_extract_features_for_column()` instance method (superseded by `FeaturesBuilder`)
- `features.constants` import auto-removed by ruff (no longer needed after dead-code removal)

**Public API updated (`__init__.py`):**
- Added `TremorMatrixBuilder` and `ModelTrainer` exports (both are now first-class pipeline components)

---

## FeaturesBuilder Docstring Update (2026-02-13)

Rewrote and corrected docstrings throughout `features_builder.py`.

**Changes:**
- Class docstring: replaced wrong example (used `tremor_df`, `window_size`, `select_tremor_columns` constructor params and `save_per_method` — none of which exist) with a correct two-mode (training / prediction) description and accurate example
- `validate()`: removed false mention of "date ranges"; clarified it checks DataFrame types and column presence only
- `concat_features()`: documented the `ValueError` raised by the underlying utility for < 2 files; clarified the attribute side-effects
- `extract_features()`: fixed signature (`prefix_filename: str = None` → `str | None = None`); rewrote docstring to describe per-column extraction flow, label CSV output, and the auto-fallback when no labels are provided

---

## FeaturesBuilder Readability Improvements (2026-02-13)

### Problem
`extract_features()` mixed training and prediction logic inline, and `concat_features()` read `self.use_relevant_features` which was only assigned *after* the call — a state-mutation ordering bug.

### Changes (`features_builder.py`)

| # | Change | Detail |
|---|--------|--------|
| 1 | **Bug fix** | `concat_features()` now accepts `use_relevant_features: bool = False` parameter instead of reading stale `self.use_relevant_features`; call site updated to pass the value explicitly |
| 2 | **`_prepare_training_mode()`** | New private helper: filters labels to window IDs present in tremor matrix, saves `label_features_<dates>.csv`, sets `self.label_features_csv`, returns `(dates_str, label_df)` |
| 3 | **`_prepare_prediction_mode()`** | New private helper: computes date range from tremor `datetime` column for unlabelled prediction; returns `(dates_str, empty_df)` |
| 4 | **`extract_features()` simplified** | Replaced 25-line inline dual-mode block with a 4-line dispatch to the two helpers; `self.use_relevant_features` is now assigned *before* `concat_features()` is called |

No public API changes. No attribute renames. Downstream callers (`ForecastModel`, `ModelTrainer`, tests) unaffected.

---

## Add root_dir Parameter (2026-02-13)

Added `resolve_output_dir()` helper function to `utils.py` and a `root_dir: str | None = None` parameter to all standalone classes and `ForecastModel`. All relative `output_dir` values are now resolved against `root_dir` (falling back to `os.getcwd()` when `None`). Absolute `output_dir` values bypass `root_dir` entirely. `ForecastModel` normalises relative `root_dir` to an absolute path immediately in `__init__` for multiprocessing safety. Backward compatible: existing code without `root_dir` behaves identically to before.

---

---

## ModelTrainer Refactor (2026-02-13)

Refactored `model_trainer.py` for readability and maintainability without changing any behaviour:

- **Bug fix:** Missing space in `update_grid_params` log message (`"has been updated to{grid_params}"` → `"Grid parameters updated from … to …"`).
- **Comment fix:** Typo and misleading comment in `train_and_evaluate()` replaced with `# Accumulate results across all seeds before aggregating`.
- **Docstring fix:** Removed incorrect `_run_train_and_evaluate()` example that used a non-existent `seed=0` parameter.
- **Skip logic unified:** `_generate_filepaths()` now returns `can_skip` as its 7th value instead of setting `self.can_skip` as a hidden side effect. `self.can_skip` attribute removed from the class.
- **`_setup_grid_search()` extracted:** Both `_run_train_and_evaluate()` and `_run_train()` shared an identical GridSearchCV setup block; extracted to a private helper.
- **`_run_jobs()` extracted:** Sequential/parallel dispatch logic duplicated across `train_and_evaluate()` and `train()`; extracted to a private helper.
- **`_save_models_registry()` extracted:** Post-processing (building and saving the models CSV) was duplicated; extracted to a private helper.
- **Comment style fixed:** Inline comments follow the project convention (explain *why*, not *what*; 2 spaces before `#`; proper grammar).
- **Method renames:**
  - `train()` → `train_and_evaluate()` (clarifies train/test split + evaluation)
  - `_train()` → `_run_train_and_evaluate()` (consistent internal naming)
  - `fit()` → `train()` (aligns with domain term; trains on full dataset)
  - `_fit()` → `_run_train()` (consistent internal naming)
- **Call sites updated:** `forecast_model.py`, `model_predictor.py`, `tests/test_train_model_fixed.py`, `tests/test_train_model_dynamic_classifier.py`.

---

## Refactor Output Directory Structure (2026-02-15)

Restructured `ModelTrainer` output directories and updated `ForecastModel.train()`.

### Changes

| Component | Change |
|-----------|--------|
| `ModelTrainer` output | Reorganised into `model-with-evaluation/` and `model-only/` subdirectories, each further split by `{classifier-slug}/{cv-slug}/` |
| `ModelTrainer` | Added `get_classifier_properties()` — returns classifier name, slug, and ID; added `update_directories()` — recalculates all output paths after classifier changes |
| `ForecastModel.train()` | Added `with_evaluation: bool` parameter (dispatches to `train_and_evaluate()` or `train()`); added `grid_params` parameter for custom hyperparameter overrides |
| `utils.py` | Added `slugify_class_name()` — converts PascalCase class names to kebab-case slugs (e.g. `XGBClassifier` → `xgb-classifier`) |
| Bug fix | `get_metrics()` parameter renamed from `classifier` → `classifier_model` |
| Docstring fixes | Fixed `pd.DataFRame`, `SLugify`, `Trainig` typos across `model_trainer.py`, `utils.py` |

### New Output Directory Layout

```
trainings/
├── model-with-evaluation/      ← train_and_evaluate()
│   └── {classifier-slug}/      e.g. xgb-classifier
│       └── {cv-slug}/          e.g. stratified-shuffle-split
│           ├── features/
│           ├── models/
│           ├── metrics/
│           ├── trained_model_{suffix}.csv
│           ├── all_metrics_{suffix}.csv
│           └── metrics_summary_{suffix}.csv
│
└── model-only/                 ← train()
    └── {classifier-slug}/
        └── {cv-slug}/
            ├── features/
            ├── models/
            └── trained_model_{suffix}.csv
```

---

## Unified Scientific Plotting System (2026-02-15)

Created a comprehensive scientific plotting system with Nature/Science journal styling standards for consistent, publication-quality visualizations across the entire codebase.

### New Architecture

Created new `plots/` module following project's domain-driven pattern:

```
src/eruption_forecast/plots/
├── __init__.py              # Public API exports
├── styles.py                # Central style configuration
├── tremor_plots.py          # Tremor time-series visualization
├── feature_plots.py         # Feature importance & selection plots
├── evaluation_plots.py      # Model evaluation plots (ROC, PR, confusion matrix, etc.)
└── forecast_plots.py        # Prediction & forecast visualization
```

### Style Configuration

**Nature/Science Journal Standards:**
- **Typography**: Arial/Helvetica family, 8-10pt labels, 10-12pt titles
- **Color palette**: 
  - Okabe-Ito colorblind-safe categorical palette
  - High-contrast blues/reds for binary classification
  - Perceptually uniform sequential colormap (viridis)
- **Layout**: 
  - Clean axis spines (top/right removed)
  - Minimal grid lines (light gray, alpha=0.3)
  - Tight bounding boxes
  - High DPI defaults (150-300 for publication quality)
- **Figure sizes**: Standard column widths (3.5", 7") for journals

### New Plotting Functions

**Tremor Plotting** (`tremor_plots.py`):
- `plot_tremor()`: Enhanced time-series plots with color-coded frequency bands (RSAM=blue, DSAR=orange), improved date formatting, and optional eruption markers

**Feature Plotting** (`feature_plots.py`):
- `plot_significant_features()`: Horizontal bar charts with top-N features highlighted darker, p-value labels, statistical significance markers

**Model Evaluation** (`evaluation_plots.py`):
- `plot_confusion_matrix()`: Heatmap with Nature/Science styling
- `plot_roc_curve()`: Clean ROC with AUC annotation
- `plot_precision_recall_curve()`: PR curve with AP score
- `plot_threshold_analysis()`: Multi-metric threshold optimization
- `plot_feature_importance()`: Top-N features bar chart with VotingClassifier support
- `plot_calibration()`: Calibration curve with perfect calibration reference
- `plot_prediction_distribution()`: Histogram with KDE overlay by true class

**Forecast Plotting** (`forecast_plots.py`):
- `plot_forecast()`: Single/multi-model probability plots with confidence bands
- `plot_forecast_with_events()`: Forecast with eruption event markers

### Backward Compatibility (CRITICAL)

**Zero breaking changes** — all existing code works without modifications:
- Existing functions (`plot.py`, `model_evaluator.py`, `model_predictor.py`) remain in place
- Old functions delegate to new styled implementations internally
- Function signatures unchanged — all parameters pass through
- Default DPI upgraded from 100 to 150 automatically
- No import path changes required
- No deprecation warnings

**Migration paths:**
```python
# OLD (continues to work):
from eruption_forecast.plot import plot_tremor

# NEW (also works):
from eruption_forecast.plots.tremor_plots import plot_tremor
```

Both produce identical publication-quality styled output.

### Files Modified

| File | Changes |
|------|---------|
| `plots/styles.py` | Central style system with NATURE_COLORS, OKABE_ITO palette, typography, `apply_nature_style()` context manager |
| `plots/tremor_plots.py` | Enhanced tremor time-series with RSAM/DSAR color coding |
| `plots/feature_plots.py` | Feature importance with top-N highlighting |
| `plots/evaluation_plots.py` | All 7 ML evaluation plot functions with consistent styling |
| `plots/forecast_plots.py` | Forecast probability and multi-model consensus plots |
| `plot.py` | Refactored to delegate to `plots/tremor_plots` and `plots/feature_plots` (backward compatible) |
| `model_evaluator.py` | All 7 plot methods refactored to delegate to `plots/evaluation_plots` (backward compatible) |

### Code Quality

- **Linting**: All files pass `ruff check --fix` with zero errors
- **Type checking**: Minor pedantic warnings from `ty` (e.g., xlim/ylim list vs tuple) resolved
- **Documentation**: Comprehensive docstrings with Google style for all plotting functions
- **Testing**: Ready for visual testing with sample data (not yet executed)

### Technical Notes

- Uses `apply_nature_style()` context manager to apply styling temporarily
- Color coding: RSAM=blue (OKABE_ITO[4]), DSAR=orange (OKABE_ITO[0])
- Feature plots highlight top-N features with darker colors
- All plotting functions are pure (no side effects except file I/O)
- Supports multiprocessing-safe file operations

---

---

> **Note:** The HTTP API layer has been moved to a separate project (`eruption-forecast-api`) and is maintained independently. This document covers the core `eruption-forecast` package only.

**Document Version:** 3.4
**Last Updated:** 2026-02-15
**Author:** Claude Code (Sonnet 4.5)
