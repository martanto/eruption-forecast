# Eruption Forecast Package - Technical Summary

**Project:** eruption-forecast - Volcanic Eruption Forecasting using Seismic Data Analysis
**Repository:** D:\Projects\eruption-forecast
**Branch:** `claude/add-timeseries-ensemble`
**Last Updated:** 2026-02-09

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
├── features/            # Feature extraction
│   ├── features_builder.py
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
   [FeaturesBuilder]
         ↓
   Feature Matrix (id, datetime, tremor columns)
         ↓
   [tsfresh extract_features]
         ↓
   Extracted Features (700+ time-series features)
         ↓
   [TrainModel]
         ↓
   Significant Features (top-N per seed across 500 seeds)
         ↓
   [ClassifierModel + GridSearchCV]
         ↓
   Trained Model → Eruption Predictions
```

### Key Design Patterns

1. **Method Chaining**: Fluent API for pipeline configuration
2. **Multi-seed Training**: Robust feature selection across 500 random seeds
3. **Class Balancing**: RandomUnderSampler for imbalanced eruption data
4. **Separation of Concerns**: Clear module boundaries

---

## Critical Fix: Data Leakage in Model Training (2026-02-09)

### Problem Identified

The original `TrainModel._train()` implementation contained **critical data leakage** that violated fundamental machine learning best practices:

1. **Resampling applied to entire dataset** before train/test split
2. **Feature selection seeing test data** through p-value calculations on full dataset
3. **No held-out test set** for generalization validation
4. **Overly optimistic metrics** due to information leakage

### Original (Incorrect) Workflow

```python
# ❌ WRONG: This was the old implementation
def _train(self, seed, random_state, ...):
    # Resample entire dataset (LEAKAGE!)
    features, labels = random_under_sampler(
        features=self.df_features,      # ENTIRE dataset
        labels=self.df_labels,          # ENTIRE dataset
        sampling_strategy=sampling_strategy,
        random_state=state,
    )

    # Feature selection on entire dataset (LEAKAGE!)
    significant_features = get_significant_features(
        features=features,
        labels=labels,
    )

    # Save features but NO model training!
    significant_features.head(number_of_significant_features).to_csv(...)
```

### Corrected Workflow

```python
# ✅ CORRECT: New implementation
def _train(self, seed, random_state, ...):
    # 1. Split FIRST (80/20, stratified)
    features_train, features_test, labels_train, labels_test = train_test_split(
        self.df_features, self.df_labels,
        test_size=0.2, random_state=state, stratify=self.df_labels,
    )

    # 2. Resample ONLY training data
    features_train_resampled, labels_train_resampled = random_under_sampler(
        features=features_train,  # ONLY training data
        labels=labels_train,
        sampling_strategy=sampling_strategy,
        random_state=state,
    )

    # 3. Feature selection ONLY on training data
    significant_features = get_significant_features(
        features=features_train_resampled,  # ONLY training data
        labels=labels_train_resampled,
    )

    # 4. Apply selected features to both sets
    top_features = significant_features.head(number_of_significant_features).index.tolist()
    features_train_selected = features_train_resampled[top_features]
    features_test_selected = features_test[top_features]

    # 5. Cross-validation with StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=state)
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=state),
        param_grid=param_grid,
        cv=cv,
        scoring="balanced_accuracy",
    )
    grid_search.fit(features_train_selected, labels_train_resampled)

    # 6. Evaluate on held-out test set
    y_pred = grid_search.best_estimator_.predict(features_test_selected)
    metrics = {
        'accuracy': accuracy_score(labels_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(labels_test, y_pred),
        'f1_score': f1_score(labels_test, y_pred),
        'precision': precision_score(labels_test, y_pred),
        'recall': recall_score(labels_test, y_pred),
        'best_params': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
    }

    # 7. Save model, metrics, and features
    joblib.dump(grid_search.best_estimator_, model_filepath)
    with open(metrics_filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
```

### Key Changes

| Aspect | Before | After |
|--------|--------|-------|
| Train/Test Split | ❌ None | ✅ 80/20 stratified split FIRST |
| Resampling | ❌ Applied to entire dataset | ✅ Applied only to training set |
| Feature Selection | ❌ Saw entire dataset | ✅ Only sees training data |
| Model Training | ❌ Not implemented | ✅ RandomForest + GridSearchCV |
| Evaluation | ❌ No test set | ✅ Held-out test set evaluation |
| Outputs | Features only | Features + trained models + metrics |

### New Output Structure

```
output/trainings/
├── significant_features/    # Top-N features per seed (existing)
│   ├── 00000.csv
│   └── ...
├── models/                  # NEW: Trained RandomForest models
│   ├── 00000.pkl
│   └── ...
├── metrics/                 # NEW: Per-seed evaluation metrics
│   ├── 00000.json
│   └── ...
├── significant_features.csv           # Aggregated features
├── all_metrics.csv                    # NEW: All seed metrics
└── metrics_summary.csv                # NEW: Mean/std statistics
```

### Impact on Results

The corrected implementation now provides:

1. **Realistic performance estimates** through proper train/test isolation
2. **Unbiased feature selection** that doesn't see test data
3. **Trained models** ready for deployment
4. **Comprehensive metrics** across multiple seeds for robustness
5. **True generalization performance** on held-out data

### Testing

Created integration test (`tests/test_train_model_fixed.py`) that verifies:
- Train + test samples < total (confirming undersampling)
- All outputs created (models, metrics, features)
- Metrics computed correctly
- Models can be loaded and used for prediction

---

## Machine Learning Workflow Analysis

### Current Workflow

The current workflow follows a well-structured approach for binary classification:

1. **Feature Extraction**: Uses tsfresh's ComprehensiveFCParameters (700+ features per tremor column)
2. **Feature Selection**: Multi-seed approach with under-sampling for stability
3. **Model Training**: GridSearchCV with StratifiedShuffleSplit cross-validation
4. **Class Handling**: Balanced class weights + RandomUnderSampler

### Strengths

| Aspect | Implementation | Benefit |
|--------|----------------|---------|
| Feature Robustness | 500-seed feature selection | Reduces overfitting to single split |
| Class Imbalance | Under-sampling + balanced weights | Handles rare eruption events |
| Hyperparameter Tuning | GridSearchCV with stratified CV | Optimal parameter selection |
| Multi-model Support | 9 classifiers available | Flexibility for different data characteristics |

### Current Limitations & Recommendations

| Limitation | Status | Recommendation |
|------------|--------|----------------|
| ~~No ensemble methods~~ | ✅ RESOLVED | VotingClassifier now available (`"voting"`) |
| Fixed feature selection | Pending | Consider Recursive Feature Elimination (RFE) |
| ~~No temporal validation~~ | ✅ RESOLVED | TimeSeriesSplit now available (`cv_strategy="timeseries"`) |
| No probability calibration | Pending | Add CalibratedClassifierCV for better probability estimates |
| ~~No model persistence~~ | ✅ RESOLVED | ModelEvaluator.export_model() with joblib |
| ~~No evaluation metrics~~ | ✅ RESOLVED | ModelEvaluator with ROC-AUC, PR-AUC, confusion matrix, plots |
| ~~Data leakage in TrainModel~~ | ✅ RESOLVED | Fixed workflow: split → resample → select features → train → evaluate |

---

## Model Comparison & Recommendations

### Available Classifiers

The package supports 9 classifiers through `ClassifierModel`:

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
| `voting` | VotingClassifier Ensemble | **Best accuracy** | Combines multiple models, robust predictions | Slower training, more complex |

### Recommended Model Selection Strategy

For volcanic eruption prediction with imbalanced time-series data:

#### 1. **Primary Choice: Random Forest (`rf`)**
- Handles class imbalance well with `class_weight="balanced"`
- Provides feature importance for scientific interpretation
- Robust to outliers common in seismic data
- No feature scaling required

```python
clf = ClassifierModel("rf")
model, grid = clf.model_and_grid
```

#### 2. **Alternative: Gradient Boosting (`gb`)**
Now available in the package:
- Better handling of imbalanced data
- Often outperforms RF on tabular data
- Built-in feature importance

```python
clf = ClassifierModel("gb")
model, grid = clf.model_and_grid
```

#### 3. **For Interpretability: Logistic Regression (`lr`)**
- Provides probability scores
- Coefficients show feature importance direction
- Good baseline for comparison

```python
clf = ClassifierModel("lr")
```

#### 4. **For Complex Patterns: Neural Network (`nn`)**
- Can learn complex temporal patterns
- Use when linear models underperform
- Requires more data and tuning

```python
clf = ClassifierModel("nn")
```

#### 5. **For Best Accuracy: VotingClassifier Ensemble (`voting`)**
Now available in the package:
- Combines RF, GB, LR, and SVM into a soft-voting ensemble
- Leverages strengths of multiple models
- More robust predictions through model diversity

```python
clf = ClassifierModel("voting", random_state=42)
model, grid = clf.model_and_grid
```

### Cross-Validation Strategies

The package now supports two cross-validation strategies:

#### StratifiedKFold (Default)
- Preserves class distribution in each fold
- Best for general classification tasks
- Shuffles data for randomization

```python
clf = ClassifierModel("rf", cv_strategy="stratified", n_splits=5)
cv = clf.get_cv_splitter()
```

#### TimeSeriesSplit (Temporal Data)
- Ensures training data always precedes test data
- Prevents data leakage in time-series forecasting
- **Recommended for eruption prediction**

```python
clf = ClassifierModel("rf", cv_strategy="timeseries", n_splits=5)
cv = clf.get_cv_splitter()

# Use with GridSearchCV
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf.model, clf.grid, cv=cv, scoring="balanced_accuracy")
```

### Model Selection Workflow

```python
from sklearn.model_selection import cross_val_score

# Compare multiple models
models = ["rf", "lr", "svm", "nn"]
results = {}

for model_name in models:
    clf = ClassifierModel(model_name)
    model = clf.model
    scores = cross_val_score(model, X, y, cv=5, scoring="balanced_accuracy")
    results[model_name] = scores.mean()

# Select best performer
best_model = max(results, key=results.get)
```

---

## Grid Parameter Analysis

### 1. Random Forest (`rf`)

**Grid Size:** 54 combinations (3 × 3 × 2 × 3)

| Parameter | Values | Recommendation |
|-----------|--------|----------------|
| `n_estimators` | [10, 30, 100] | Increase to [50, 100, 200] for better performance |
| `max_depth` | [3, 5, 7] | Add [10, None] for complex patterns |
| `criterion` | ["gini", "entropy"] | Both viable; gini often faster |
| `max_features` | ["sqrt", "log2", None] | "sqrt" usually optimal |

**Recommended Grid Expansion:**
```python
{
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [5, 7, 10, 15, None],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
```

### 2. Gradient Boosting (`gb`)

**Grid Size:** 216 combinations (3 × 3 × 3 × 2 × 2 × 2)

| Parameter | Values | Recommendation |
|-----------|--------|----------------|
| `n_estimators` | [50, 100, 200] | Good range for most datasets |
| `max_depth` | [3, 5, 7] | Shallow trees prevent overfitting |
| `learning_rate` | [0.01, 0.1, 0.2] | Lower rates need more estimators |
| `subsample` | [0.8, 1.0] | 0.8 adds stochasticity, reduces overfit |
| `min_samples_split` | [2, 5] | Higher values prevent overfitting |
| `min_samples_leaf` | [1, 2] | Higher values create more generalized trees |

**Usage:**
```python
clf = ClassifierModel("gb")
model, grid = clf.model_and_grid
```

### 3. Support Vector Machine (`svm`)

**Grid Size:** 120 combinations (5 × 3 × 4 × 2)

| Parameter | Values | Recommendation |
|-----------|--------|----------------|
| `C` | [0.001, 0.01, 0.1, 1, 10] | Good range; add 100 for hard margins |
| `kernel` | ["poly", "rbf", "sigmoid"] | Add "linear" for high-dimensional data |
| `degree` | [2, 3, 4, 5] | Only affects poly kernel |
| `decision_function_shape` | ["ovo", "ovr"] | "ovr" faster for binary |

**Note:** SVM requires feature scaling (StandardScaler). Add to pipeline:
```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced"))
])
```

### 4. Logistic Regression (`lr`)

**Grid Size:** 15 combinations (3 × 5)

| Parameter | Values | Recommendation |
|-----------|--------|----------------|
| `penalty` | ["l2", "l1", "elasticnet"] | l1 for feature selection, l2 for stability |
| `C` | [0.001, 0.01, 0.1, 1, 10] | Good range |

**Recommended Grid Expansion:**
```python
{
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "solver": ["saga"],  # Required for l1/elasticnet
    "l1_ratio": [0.25, 0.5, 0.75],  # Only for elasticnet
    "max_iter": [1000],
}
```

### 5. Neural Network / MLP (`nn`)

**Grid Size:** 8 combinations (4 × 2)

| Parameter | Values | Recommendation |
|-----------|--------|----------------|
| `activation` | ["identity", "logistic", "tanh", "relu"] | relu usually best |
| `hidden_layer_sizes` | [10, 100] | Too limited; expand significantly |

**Recommended Grid Expansion:**
```python
{
    "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100), (200, 100)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [500, 1000],
    "early_stopping": [True],
    "validation_fraction": [0.1],
}
```

### 6. Decision Tree (`dt`)

**Grid Size:** 18 combinations (3 × 2 × 3)

| Parameter | Values | Recommendation |
|-----------|--------|----------------|
| `max_depth` | [3, 5, 7] | Shallow to prevent overfitting |
| `criterion` | ["gini", "entropy"] | Both viable |
| `max_features` | ["sqrt", "log2", None] | "sqrt" for high-dimensional |

**Recommended Grid Expansion:**
```python
{
    "max_depth": [3, 5, 7, 10],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2"],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
}
```

### 7. K-Nearest Neighbors (`knn`)

**Grid Size:** 24 combinations (4 × 2 × 3)

| Parameter | Values | Recommendation |
|-----------|--------|----------------|
| `n_neighbors` | [3, 6, 12, 24] | Odd numbers preferred |
| `weights` | ["uniform", "distance"] | distance often better |
| `p` | [1, 2, 3] | 2 (Euclidean) most common |

**Note:** KNN requires feature scaling and is slow on large datasets.

### 8. Gaussian Naive Bayes (`nb`)

**Grid Size:** 1 combination

| Parameter | Values | Recommendation |
|-----------|--------|----------------|
| `var_smoothing` | [1.0] | Expand to [1e-9, 1e-8, 1e-7, 1e-6] |

---

## Code Quality Summary

### Recent Improvements (2026-02-09)

| Category | Changes |
|----------|---------|
| **Grammar/Spelling** | Fixed "aftrer" → "after", "each significant features" → "each significant feature", "calulation" → "calculation", "filenmae" → "filename" |
| **Docstrings** | Added comprehensive examples to ForecastModel, ClassifierModel, TrainModel, FeaturesBuilder |
| **Docstrings (2026-02-09)** | Enhanced docstrings in FeaturesBuilder, TremorMatrixBuilder, ForecastModel, TrainModel with detailed descriptions, complete Args/Returns/Raises sections, and examples for all public methods |
| **Type Annotations** | Fixed `dict[str, any]` → `dict[str, Any]` |
| **Deprecated Parameters** | Removed `max_features="auto"` (deprecated in sklearn 1.4) |
| **Documentation** | Enhanced method descriptions with usage examples |
| **New Classifier** | Added GradientBoostingClassifier (`"gb"`) with 216-combination hyperparameter grid |
| **Random State** | Added configurable `random_state` parameter to ClassifierModel for reproducibility |
| **TimeSeriesSplit** | Added `cv_strategy` parameter and `get_cv_splitter()` method for temporal cross-validation |
| **VotingClassifier** | Added `"voting"` ensemble classifier combining RF, GB, LR, and SVM with soft voting |
| **ModelEvaluator** | New class for comprehensive model evaluation with metrics, export, and plotting capabilities |
| **Data Leakage Fix** | Completely rewrote `TrainModel._train()` to eliminate data leakage: (1) train/test split first, (2) resample only training data, (3) feature selection only on training data, (4) RandomForest training with GridSearchCV + StratifiedShuffleSplit, (5) proper test set evaluation. Now saves trained models, per-seed metrics, and aggregated statistics. |
| **Dynamic Classifier** | TrainModel now supports all classifier types through ClassifierModel integration: RF, GB, SVM, LR, NN, DT, KNN, NB, Voting. Users can select classifier type and CV strategy (shuffle, stratified, timeseries) during initialization. |

### Code Quality Metrics

| Metric | Status |
|--------|--------|
| Type Checking (pyrefly) | 0 errors |
| Import Sorting (isort) | Applied |
| Test Coverage | 81 tests passing |
| Docstring Coverage | All public methods documented |

---

## Future Recommendations

### Short-term Improvements

1. **~~Add Gradient Boosting Models~~** ✓ COMPLETED
   - Added `GradientBoostingClassifier` as `"gb"` classifier
   - Includes optimized hyperparameter grid (216 combinations)

2. **~~Implement TimeSeriesSplit~~** ✓ COMPLETED
   - Added `cv_strategy` parameter to ClassifierModel ("stratified" or "timeseries")
   - Added `get_cv_splitter()` method returning appropriate CV splitter
   - Prevents data leakage in temporal eruption forecasting
   ```python
   clf = ClassifierModel("rf", cv_strategy="timeseries", n_splits=5)
   cv = clf.get_cv_splitter()  # Returns TimeSeriesSplit
   ```

3. **~~Add Ensemble Voting~~** ✓ COMPLETED
   - Added `"voting"` classifier type to ClassifierModel
   - Combines RF, GB, LR, and SVM with soft voting
   - Includes hyperparameter grid for ensemble tuning
   ```python
   clf = ClassifierModel("voting", random_state=42)
   model, grid = clf.model_and_grid
   ```

4. **~~Add Model Evaluation & Export~~** ✓ COMPLETED
   - Added `ModelEvaluator` class with comprehensive evaluation capabilities
   - Metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, balanced accuracy
   - Export: model (joblib), metrics (CSV/JSON), reports, confusion matrix
   - Plots: ROC curve, PR curve, confusion matrix, feature importance, learning curve
   ```python
   from eruption_forecast.model.model_evaluator import ModelEvaluator
   evaluator = ModelEvaluator(model, X_test, y_test, X_train, y_train)
   metrics = evaluator.get_metrics()
   evaluator.plot_all()
   evaluator.export_all()
   ```

### Medium-term Improvements

1. **Probability Calibration**: Use CalibratedClassifierCV for reliable probability estimates
2. **Feature Pipeline**: Add StandardScaler for SVM/KNN/NN in pipeline

### Long-term Improvements

1. **Deep Learning**: Implement LSTM/Transformer for sequential patterns
2. **AutoML Integration**: Consider TPOT or auto-sklearn for automated model selection
3. **Real-time Inference**: Streaming prediction pipeline
4. **Explainability**: SHAP values for model interpretation

---

## Quick Reference

### Complete Pipeline Example

```python
from eruption_forecast.model.forecast_model import ForecastModel
from eruption_forecast.model.classifier_model import ClassifierModel

# 1. Setup and calculate tremor
model = ForecastModel(
    station="OJN",
    channel="EHZ",
    start_date="2024-01-01",
    end_date="2024-06-30",
    window_size=1,
    volcano_id="LEWOTOBI",
    n_jobs=4,
)

# 2. Process seismic data
model.calculate(source="sds", sds_dir="data/sds")

# 3. Build labels
model.build_label(
    window_step=12,
    window_step_unit="hours",
    day_to_forecast=2,
    eruption_dates=["2024-03-15", "2024-05-20"],
)

# 4. Extract features
model.build_features()
model.extract_features()

# 5. Train models
model.train(
    random_state=0,
    total_seed=100,
    number_of_significant_features=20,
)

# 6. Use classifier for prediction
clf = ClassifierModel("rf")
model, grid = clf.model_and_grid
```

### Classifier Quick Reference

```python
# Random Forest (recommended)
clf = ClassifierModel("rf")

# Gradient Boosting (imbalanced data)
clf = ClassifierModel("gb")

# VotingClassifier Ensemble (best accuracy)
clf = ClassifierModel("voting", random_state=42)

# With reproducible random state
clf = ClassifierModel("rf", random_state=42)

# With TimeSeriesSplit for temporal data (prevents data leakage)
clf = ClassifierModel("rf", cv_strategy="timeseries", n_splits=5)
cv = clf.get_cv_splitter()

# Logistic Regression (interpretable)
clf = ClassifierModel("lr")

# SVM (high-dimensional)
clf = ClassifierModel("svm")

# Neural Network (complex patterns)
clf = ClassifierModel("nn")

# Custom grid
clf.grid = {"n_estimators": [100, 200], "max_depth": [5, 10]}
```

### Model Evaluation Quick Reference

```python
from eruption_forecast.model.model_evaluator import ModelEvaluator

# Initialize evaluator with fitted model
evaluator = ModelEvaluator(
    model=fitted_model,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,  # Optional, for learning curves
    y_train=y_train,  # Optional, for learning curves
    model_name="rf_eruption",
    output_dir="output/evaluation"
)

# Get comprehensive metrics
metrics = evaluator.get_metrics()
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")

# Print summary
print(evaluator.summary())

# Generate all plots
evaluator.plot_all(feature_names=feature_names, top_n_features=20)

# Individual plots
evaluator.plot_roc_curve()
evaluator.plot_precision_recall_curve()
evaluator.plot_confusion_matrix(normalize="true")
evaluator.plot_feature_importances(top_n=15)
evaluator.plot_learning_curve()
evaluator.plot_metrics_summary()

# Export all results
paths = evaluator.export_all()
# Returns: model.joblib, metrics.csv, metrics.json, report.txt, etc.

# Export individually
evaluator.export_model()  # Save with joblib
evaluator.export_metrics(format="csv")
evaluator.export_classification_report()
evaluator.export_confusion_matrix()
evaluator.export_feature_importances()
```

---

**Document Version:** 2.2
**Last Updated:** 2026-02-09
**Author:** Claude Code (Sonnet 4.5)
