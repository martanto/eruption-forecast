# Eruption Forecast Package — Technical Summary

**Project:** eruption-forecast — Volcanic Eruption Forecasting using Seismic Data Analysis
**Repository:** D:\Projects\eruption-forecast
**Branch:** `dev/predictions`
**Last Updated:** 2026-02-11

---

## Table of Contents

1. [Package Overview](#package-overview)
2. [Architecture](#architecture)
3. [Machine Learning Workflow Analysis](#machine-learning-workflow-analysis)
4. [Model Comparison & Recommendations](#model-comparison--recommendations)
5. [Grid Parameter Analysis](#grid-parameter-analysis)
6. [Code Quality Summary](#code-quality-summary)
7. [Future Recommendations](#future-recommendations)

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
│   ├── train_model.py       # Multi-seed training
│   ├── classifier_model.py  # Classifier management
│   └── model_evaluator.py   # Evaluation, export, and plotting
├── utils.py             # Shared utilities
├── sds.py               # SDS file handling
├── plot.py              # Visualization
└── logger.py            # Centralized logging
```

---

## Architecture

### Data Pipeline

```
Raw Seismic Data (SDS/FDSN)
         ↓
   [CalculateTremor]
         ↓
   Tremor CSV (RSAM + DSAR metrics, 10-min intervals)
         ↓
   [LabelBuilder]
         ↓
   Label CSV (binary erupted/not labels per time window)
         ↓
   [TremorMatrixBuilder]
         ↓
   Tremor Matrix (id, datetime, tremor columns aligned to label windows)
         ↓
   [FeaturesBuilder + tsfresh]
         ↓
   Extracted Features (700+ time-series features)
         ↓
   [FeatureSelector (optional)]
         ↓
   Selected Features (tsfresh, RandomForest, or combined)
         ↓
   [TrainModel]
         ↓
   Significant Features (top-N per seed across N seeds)
         ↓
   [ClassifierModel + GridSearchCV]
         ↓
   Trained Model → Eruption Predictions
```

### Key Design Patterns

1. **Method Chaining**: Fluent API for pipeline configuration (`ForecastModel`)
2. **Multi-seed Training**: Robust feature selection across many random seeds
3. **Class Balancing**: RandomUnderSampler for imbalanced eruption data
4. **Separation of Concerns**: Clear module boundaries
5. **Data Leakage Prevention**: Strict train/test split before any resampling or feature selection

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
| Multi-model Support | 9 classifiers available | Flexibility for different data characteristics |
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
| `voting` | VotingClassifier Ensemble | **Best accuracy** | Combines multiple models, robust | Slower training, more complex |

### Cross-Validation Strategies

| Strategy | Class | Best For |
|----------|-------|----------|
| `shuffle` | StratifiedShuffleSplit | General classification |
| `stratified` | StratifiedKFold | Preserves class distribution |
| `timeseries` | TimeSeriesSplit | **Temporal data — prevents leakage** |

---

## Grid Parameter Analysis

### 1. Random Forest (`rf`)

**Grid Size:** 54 combinations (3 × 3 × 2 × 3)

| Parameter | Values |
|-----------|--------|
| `n_estimators` | [10, 30, 100] |
| `max_depth` | [3, 5, 7] |
| `criterion` | ["gini", "entropy"] |
| `max_features` | ["sqrt", "log2", None] |

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

**Grid Size:** 15 combinations (3 × 5)

| Parameter | Values |
|-----------|--------|
| `penalty` | ["l2", "l1", "elasticnet"] |
| `C` | [0.001, 0.01, 0.1, 1, 10] |

### 5. Neural Network / MLP (`nn`)

**Grid Size:** 8 combinations (4 × 2)

| Parameter | Values |
|-----------|--------|
| `activation` | ["identity", "logistic", "tanh", "relu"] |
| `hidden_layer_sizes` | [10, 100] |

> **Note:** `hidden_layer_sizes` grid is too limited. Expanding to multi-layer architectures is recommended (see TODO.md).

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
| Pre-2026-02-09 | **VotingClassifier** | Added `"voting"` ensemble combining RF, GB, LR, SVM |

### Code Quality Metrics

| Metric | Status |
|--------|--------|
| Type Checking (pyrefly) | 0 errors |
| Import Sorting (isort) | Applied |
| Test Coverage | 81+ tests passing |
| Docstring Coverage | All public methods documented |

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

**Document Version:** 2.3
**Last Updated:** 2026-02-11
**Author:** Claude Code (Sonnet 4.5)
