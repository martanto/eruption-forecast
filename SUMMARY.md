# Eruption Forecast Package - Technical Summary

**Project:** eruption-forecast - Volcanic Eruption Forecasting using Seismic Data Analysis
**Repository:** D:\Projects\eruption-forecast
**Branch:** `dev/predictions`
**Last Updated:** 2026-02-04

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
│   └── classifier_model.py  # Classifier management
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
| Multi-model Support | 7 classifiers available | Flexibility for different data characteristics |

### Current Limitations & Recommendations

| Limitation | Recommendation |
|------------|----------------|
| No ensemble methods | Add VotingClassifier combining top models |
| Fixed feature selection | Consider Recursive Feature Elimination (RFE) |
| No temporal validation | Implement TimeSeriesSplit for proper temporal evaluation |
| No probability calibration | Add CalibratedClassifierCV for better probability estimates |
| No model persistence | Add joblib serialization for production deployment |

---

## Model Comparison & Recommendations

### Available Classifiers

The package supports 8 classifiers through `ClassifierModel`:

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

### Recent Improvements (2026-02-04)

| Category | Changes |
|----------|---------|
| **Grammar/Spelling** | Fixed "aftrer" → "after", "each significant features" → "each significant feature" |
| **Docstrings** | Added comprehensive examples to ForecastModel, ClassifierModel, TrainModel, FeaturesBuilder |
| **Type Annotations** | Fixed `dict[str, any]` → `dict[str, Any]` |
| **Deprecated Parameters** | Removed `max_features="auto"` (deprecated in sklearn 1.4) |
| **Documentation** | Enhanced method descriptions with usage examples |
| **New Classifier** | Added GradientBoostingClassifier (`"gb"`) with 216-combination hyperparameter grid |
| **Random State** | Added configurable `random_state` parameter to ClassifierModel for reproducibility |

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

2. **Implement TimeSeriesSplit**
   ```python
   from sklearn.model_selection import TimeSeriesSplit
   cv = TimeSeriesSplit(n_splits=5)
   ```

3. **Add Ensemble Voting**
   ```python
   from sklearn.ensemble import VotingClassifier

   ensemble = VotingClassifier(
       estimators=[
           ("rf", RandomForestClassifier()),
           ("lr", LogisticRegression()),
           ("svm", SVC(probability=True)),
       ],
       voting="soft"
   )
   ```

### Medium-term Improvements

1. **Probability Calibration**: Use CalibratedClassifierCV for reliable probability estimates
2. **Feature Pipeline**: Add StandardScaler for SVM/KNN/NN in pipeline
3. **Model Persistence**: Save trained models with joblib
4. **Evaluation Metrics**: Add precision-recall curves, ROC-AUC, confusion matrices

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

# With reproducible random state
clf = ClassifierModel("rf", random_state=42)

# Logistic Regression (interpretable)
clf = ClassifierModel("lr")

# SVM (high-dimensional)
clf = ClassifierModel("svm")

# Neural Network (complex patterns)
clf = ClassifierModel("nn")

# Custom grid
clf.grid = {"n_estimators": [100, 200], "max_depth": [5, 10]}
```

---

**Document Version:** 2.0
**Last Updated:** 2026-02-04
**Author:** Claude Code (Opus 4.5)
