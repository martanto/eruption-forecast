# eruption-forecast

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-orange)

A comprehensive Python package for volcanic eruption forecasting using seismic data analysis. Process raw seismic tremor data, extract time-series features, train machine learning models, and predict volcanic eruptions based on seismic patterns.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start: Complete Pipeline](#quick-start-complete-pipeline)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [1. Calculate Tremor Metrics](#1-calculate-tremor-metrics)
  - [2. Build Training Labels](#2-build-training-labels)
  - [3. Build Tremor Matrix](#3-build-tremor-matrix)
  - [4. Extract Time-Series Features](#4-extract-time-series-features)
  - [5. Feature Selection](#5-feature-selection)
  - [6. Train Models with Multiple Seeds](#6-train-models-with-multiple-seeds)
  - [7. Supported Classifiers](#7-supported-classifiers)
  - [8. Hyperparameter Grids](#8-hyperparameter-grids)
  - [9. Cross-Validation Strategies](#9-cross-validation-strategies)
  - [10. Predict on Future Data with ModelPredictor](#10-predict-on-future-data-with-modelpredictor)
  - [11. Model Evaluation](#11-model-evaluation)
  - [12. Analyze Training Results](#12-analyze-training-results)
- [Advanced Usage](#advanced-usage)
  - [ForecastModel — Orchestrated Pipeline](#forecastmodel--orchestrated-pipeline)
  - [Comparing Multiple Classifiers](#comparing-multiple-classifiers)
  - [train() + ModelPredictor Workflow](#train--modelpredictor-workflow-recommended-for-operational-use)
  - [Custom Frequency Bands](#custom-frequency-bands)
- [Visualization & Plotting](#visualization--plotting)
  - [Tremor Time-Series Plots](#tremor-time-series-plots)
  - [Feature Importance Plots](#feature-importance-plots)
  - [Model Evaluation Plots](#model-evaluation-plots)
  - [Forecast Visualization](#forecast-visualization)
  - [Batch Replot Utilities](#batch-replot-utilities)
  - [Common Parameters](#common-parameters)
  - [Import Examples](#import-examples)
- [Output Directory Structure](#output-directory-structure)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Tremor Calculation**: Process raw seismic data (SDS/FDSN) to calculate RSAM and DSAR metrics across multiple frequency bands
- **Label Building**: Generate training labels from eruption dates with configurable forecast horizons
- **Feature Extraction**: Extract 700+ time-series features using tsfresh for machine learning
- **Enhanced Feature Selection**: Three-method feature selection — tsfresh statistical, RandomForest permutation importance, or combined two-stage
- **Model Training**: Train 10 classifier types (Random Forest, Gradient Boosting, XGBoost, SVM, Logistic Regression, Neural Networks, Ensembles) across multiple random seeds
- **Model Evaluation**: Comprehensive evaluation with ROC curves, precision-recall curves, confusion matrices, threshold analysis, calibration curves, and feature importance
- **Two Training Workflows**: `train_and_evaluate()` for in-sample evaluation (80/20 split), `train()` for full-dataset training with future-data evaluation via `ModelPredictor`; `fit()` as a unified entry point that dispatches between the two
- **Multi-processing**: Parallel processing for faster tremor calculations and model training
- **Logging**: Built-in logging with loguru for debugging and monitoring

## Pipeline Overview

```
Raw Seismic Data (SDS / FDSN)
         │
         ▼
┌─────────────────────┐
│   CalculateTremor   │  RSAM + DSAR metrics (10-min intervals)
└─────────┬───────────┘
          │  tremor.csv
          ▼
┌─────────────────────┐
│    LabelBuilder     │  Binary labels (1 = eruption, 0 = normal)
└─────────┬───────────┘
          │  label_*.csv
          ▼
┌─────────────────────┐
│ TremorMatrixBuilder │  Windowed tremor matrix aligned to labels
└─────────┬───────────┘
          │  tremor_matrix_*.csv
          ▼
┌─────────────────────┐
│   FeaturesBuilder   │  700+ tsfresh features per tremor column
└─────────┬───────────┘
          │  all_extracted_features_*.csv
          ▼
┌─────────────────────┐
│   ModelTrainer      │  Multi-seed GridSearchCV training
│  ┌───────────────┐  │
│  │FeatureSelector│  │  tsfresh / RandomForest / combined
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │ClassifierModel│  │  10 classifiers, 3 CV strategies
│  └───────────────┘  │
└─────────┬───────────┘
          │  trained_model_*.csv  +  *.pkl models
          ▼
┌─────────────────────┐
│   ModelPredictor    │  Evaluation or forecast on future data
│  ┌───────────────┐  │
│  │ModelEvaluator │  │  Metrics, plots, threshold analysis
│  └───────────────┘  │
└─────────────────────┘
```

## Installation

This project uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
# Clone the repository
git clone https://github.com/martanto/eruption-forecast.git
cd eruption-forecast

# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev
```

## Quick Start: Complete Pipeline

Here's a minimal end-to-end example from raw seismic data to trained models:

```python
from eruption_forecast import ForecastModel

fm = ForecastModel(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=2,
    volcano_id="LEWOTOBI",
    root_dir="/path/to/project",  # all output paths are anchored here
    n_jobs=4,
    verbose=True,
)

fm.calculate(
    source="sds",
    sds_dir="/path/to/sds/data",
).build_label(
    day_to_forecast=2,
    window_step=6,
    window_step_unit="hours",
    eruption_dates=[
        "2025-03-20",
        "2025-06-15",
        "2025-09-10",
    ],
).extract_features(
    select_tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4"],
).train(
    classifier="xgb",
    cv_strategy="stratified",
    random_state=0,
    total_seed=500,
    number_of_significant_features=20,
    sampling_strategy=0.75,
)
```

---

## Step-by-Step Usage Guide

### 1. Calculate Tremor Metrics

Process raw seismic data to calculate RSAM (amplitude) and DSAR (ratios) across frequency bands.

```python
from eruption_forecast import CalculateTremor

# Basic usage with SDS data
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
    n_jobs=4,
).from_sds(sds_dir="/data/sds").run()

# Custom frequency bands
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
).change_freq_bands([
    (0.1, 1.0),   # Low frequency
    (1.0, 5.0),   # Mid frequency
    (5.0, 10.0),  # High frequency
]).from_sds(sds_dir="/data/sds").run()

# Access the DataFrame
print(tremor.df.head())
# Output columns: rsam_f0, rsam_f1, rsam_f2, dsar_f0-f1, dsar_f1-f2

# Get file path
print(f"Saved to: {tremor.csv}")
```

**Output format:**
- DateTime index with 10-minute intervals
- RSAM columns: `rsam_f0`, `rsam_f1`, `rsam_f2`, `rsam_f3`, `rsam_f4`
- DSAR columns: `dsar_f0-f1`, `dsar_f1-f2`, `dsar_f2-f3`, `dsar_f3-f4`
- Default bands: (0.01–0.1), (0.1–2), (2–5), (4.5–8), (8–16) Hz

### 2. Build Training Labels

Create binary labels (erupted/not erupted) based on known eruption dates.

```python
from eruption_forecast import LabelBuilder

labels = LabelBuilder(
    start_date="2020-01-01",
    end_date="2020-12-31",
    window_step=12,          # Slide by 12 hours
    window_step_unit="hours",
    day_to_forecast=2,       # Label 2 days before eruption as positive
    eruption_dates=[
        "2020-03-15",
        "2020-06-20",
        "2020-09-10",
    ],
    volcano_id="VOLCANO_001",
).build()

# Access the DataFrame
print(labels.df.head())
# Columns: id, is_erupted

# Check label distribution
print(f"Positive labels: {(labels.df['is_erupted'] == 1).sum()}")
print(f"Negative labels: {(labels.df['is_erupted'] == 0).sum()}")
```

**Label logic:**
- Windows whose **end time** falls within `[eruption_date − day_to_forecast, eruption_date]` are labeled `is_erupted = 1`
- All other windows: `is_erupted = 0`
- Each window's datetime index is its **end time** (tremor data for the preceding `window_size` days)

#### Labeling Strategy Visualization

Example with `window_step=12h`, `day_to_forecast=2d`, `eruption=Jan 15`.

```
Timeline (each tick = 12 hours):

──── Jan10 ──────── Jan11 ──────── Jan12 ──────── Jan13 ──────── Jan14 ────── Jan15 ☄
 00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │  00
     │      │      │      │      │      │      │      │      │      │      │      │
     ← window_step: 12h →         │      │      │      │      │
                                  │      ◄──────────── day_to_forecast=2d ───────────►│
                                  │                       label = 1 zone              │
```

```
 ID  Window data span                     End time (index)    Label
 ──  ──────────────────────────────────── ──────────────────  ──────
  1  Jan09·12:00 ══════════ Jan10·12:00   Jan10 12:00           0
  2  Jan10·00:00 ══════════ Jan11·00:00   Jan11 00:00           0
  3  Jan10·12:00 ══════════ Jan11·12:00   Jan11 12:00           0
  4  Jan11·00:00 ══════════ Jan12·00:00   Jan12 00:00           0
  5  Jan11·12:00 ══════════ Jan12·12:00   Jan12 12:00           0
  6  Jan12·00:00 ══════════ Jan13·00:00   Jan13 00:00           1  ← label zone starts
  7  Jan12·12:00 ══════════ Jan13·12:00   Jan13 12:00           1
  8  Jan13·00:00 ══════════ Jan14·00:00   Jan14 00:00           1
  9  Jan13·12:00 ══════════ Jan14·12:00   Jan14 12:00           1
 10  Jan14·00:00 ══════════ Jan15·00:00   Jan15 00:00           1  ← eruption day
```

> **Key:** The window's datetime index = its **end time**. A window gets `label=1` when its
> end time falls within `[eruption_date − day_to_forecast, eruption_date]`.

**Parameter reference:**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `window_step` | `int` | How far to shift the window between consecutive labels | `12` |
| `window_step_unit` | `"minutes"` \| `"hours"` | Unit for `window_step` | `"hours"` |
| `day_to_forecast` | `int` (days) | How many days before the eruption to start labeling as positive (`is_erupted=1`) | `2` |
| `eruption_dates` | `list[str]` | Known eruption dates in `YYYY-MM-DD` format | `["2025-03-20"]` |
| `start_date` / `end_date` | `str` | Date range for generating all label windows | `"2025-01-01"` |
| `volcano_id` | `str` | Identifier used in output filenames | `"LEWOTOBI"` |

### 3. Build Tremor Matrix

Align tremor time-series with label windows to create a unified feature matrix.

```python
from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder

builder = TremorMatrixBuilder(
    tremor_df=tremor.df,
    label_df=labels.df,
    output_dir="output/features",
    window_size=1,
).build(
    select_tremor_columns=["rsam_f0", "rsam_f1", "dsar_f0-f1"],
    save_tremor_matrix_per_method=True,
)

print(builder.df.shape)  # (n_windows × n_samples_per_window, n_columns)
```

### 4. Extract Time-Series Features

Extract 700+ statistical features from tremor data using tsfresh.

```python
from eruption_forecast import FeaturesBuilder

features_builder = FeaturesBuilder(
    tremor_matrix_df=builder.df,
    label_df=labels.df,
    output_dir="output/features",
    n_jobs=4,
)

# Extract all features
features = features_builder.extract_features(
    select_tremor_columns=["rsam_f0", "rsam_f1"],
    exclude_features=["length", "has_duplicate"],
)

print(f"Features shape: {features.shape}")
# Example: (5000 windows, 1500 features)
```

**Feature types extracted:**
- Statistical: mean, median, std, variance, min, max, quantiles
- Time-domain: autocorrelation, partial autocorrelation
- Frequency-domain: FFT coefficients, spectral entropy
- Complexity: approximate entropy, sample entropy
- Peaks: number of peaks, peak positions

### 5. Feature Selection

Select the most informative features using one of three available methods.

```python
from eruption_forecast.features import FeatureSelector

# Two-stage combined selection (recommended)
selector = FeatureSelector(method="combined", n_jobs=4, verbose=True)
X_selected = selector.fit_transform(
    X_train, y_train,
    fdr_level=0.05,  # Stage 1: tsfresh FDR level
    top_n=30,        # Stage 2: final feature count
)
print(f"Reduced: {X_train.shape[1]} → {X_selected.shape[1]} features")

# tsfresh statistical selection only
selector = FeatureSelector(method="tsfresh", n_jobs=4)
X_selected = selector.fit_transform(X_train, y_train, fdr_level=0.05)

# RandomForest permutation importance only
selector = FeatureSelector(method="random_forest", n_jobs=4)
X_selected = selector.fit_transform(X_train, y_train, top_n=30)

# Get comprehensive feature scores
scores = selector.get_feature_scores()
print(scores.head(10))
```

**Selection methods:**

| Method | Reduces | Captures Interactions | Speed |
|--------|---------|-----------------------|-------|
| `tsfresh` | 1000s → 100s | No | Fast |
| `random_forest` | Direct → N | Yes | Slow |
| `combined` | 1000s → 100s → N | Yes | Fast |

### 6. Train Models with Multiple Seeds

Two training workflows are available depending on your evaluation strategy.

```
train_and_evaluate() workflow:         train() workflow:

Full Dataset                           Full Dataset
     │                                      │
     ▼                                      ▼
 80/20 Split                          RandomUnderSampler
     │                                 (full dataset)
  ┌──┴──┐                                   │
Train  Test                           Feature Selection
  │     │                              (full dataset)
RandomUnder                                 │
Sampler                               GridSearchCV + CV
  │                                         │
Feature                               ┌─────┴──────┐
Selection                          Save model  Save registry
  │                                (.pkl)      (.csv)
GridSearchCV + CV
  │
Evaluate on Test
  │
Save model + metrics
```

#### Which workflow should I use?

| Question | `train_and_evaluate()` | `train()` |
|---|---|---|
| Do I want to measure in-sample performance? | ✅ Yes — evaluates each seed on held-out 20% | ❌ No metrics computed |
| Do I have a separate future dataset to evaluate on? | — | ✅ Use with `ModelPredictor` |
| Am I exploring classifiers and hyperparameters? | ✅ Quick feedback per run | ❌ Not suitable |
| Am I training the final production model? | ❌ Wastes 20% of data | ✅ Uses 100% of data |
| Does order of data matter (time-series)? | ✅ Use `cv_strategy="timeseries"` | ✅ Use `cv_strategy="timeseries"` |

#### `train_and_evaluate()` — with held-out test set (80/20 split)

Splits data **before** resampling and feature selection to prevent data leakage.
Evaluates each seed on the held-out 20% and aggregates metrics across seeds.

```python
from eruption_forecast.model.model_trainer import ModelTrainer

trainer = ModelTrainer(
    extracted_features_csv="output/features/all_features.csv",
    label_features_csv="output/features/label_features.csv",
    output_dir="output/trainings",
    classifier="xgb",
    cv_strategy="stratified",
    number_of_significant_features=20,
    feature_selection_method="combined",
    n_jobs=4,
)

trainer.train_and_evaluate(
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)
```

#### `train()` — full dataset training (no split)

Trains on the **entire** dataset across multiple seeds. No metrics are computed here —
evaluation is deferred to `ModelPredictor` using a separate future dataset.

```python
trainer = ModelTrainer(
    extracted_features_csv="output/features/all_features.csv",
    label_features_csv="output/features/label_features.csv",
    output_dir="output/trainings",
    classifier="rf",
    n_jobs=4,
)

trainer.train(
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)
```

#### `fit()` — unified entry point

`fit()` dispatches to `train_and_evaluate()` or `train()` based on the
`with_evaluation` flag. Use it when the calling code needs a single method
regardless of which workflow is active.

```python
# Equivalent to train_and_evaluate()
trainer.fit(
    with_evaluation=True,
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)

# Equivalent to train()
trainer.fit(
    with_evaluation=False,
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)
```

`ForecastModel.train()` calls `fit()` internally and exposes `with_evaluation`
as a direct parameter, so you can control the workflow from the high-level API
without dropping down to `ModelTrainer`.

#### ModelTrainer constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extracted_features_csv` | `str` | — | Path to extracted features CSV (output of `FeaturesBuilder`) |
| `label_features_csv` | `str` | — | Path to aligned labels CSV (output of `FeaturesBuilder`) |
| `output_dir` | `str \| None` | `None` | Output directory; resolved against `root_dir` when relative. Defaults to `root_dir/output/trainings` |
| `root_dir` | `str \| None` | `None` | Anchor for resolving relative `output_dir`. Defaults to `os.getcwd()` |
| `prefix_filename` | `str \| None` | `None` | Optional prefix prepended to every output filename |
| `classifier` | `str` | `"rf"` | Classifier type — see [Supported Classifiers](#7-supported-classifiers) |
| `cv_strategy` | `str` | `"shuffle"` | Cross-validation strategy — `"shuffle"`, `"stratified"`, or `"timeseries"` |
| `cv_splits` | `int` | `5` | Number of CV folds |
| `number_of_significant_features` | `int` | `20` | Top-N features retained per seed and aggregated across seeds |
| `feature_selection_method` | `str` | `"tsfresh"` | Feature selection algorithm — `"tsfresh"`, `"random_forest"`, or `"combined"` |
| `overwrite` | `bool` | `False` | Re-run even if output files already exist |
| `n_jobs` | `int` | `1` | Parallel workers for multi-seed dispatch |
| `verbose` | `bool` | `False` | Print progress messages |
| `debug` | `bool` | `False` | Enable debug-level logging |

#### `train_and_evaluate()` method parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `0` | Starting random seed; seeds are `random_state, random_state+1, …, random_state+total_seed−1` |
| `total_seed` | `int` | `500` | Number of seeds (independent train/test splits) to run |
| `sampling_strategy` | `str \| float` | `0.75` | Under-sampling ratio for `RandomUnderSampler` on training data only |
| `save_all_features` | `bool` | `False` | Save all ranked features per seed (can produce many files) |
| `plot_significant_features` | `bool` | `False` | Save a feature-importance plot per seed |

#### `train()` method parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `0` | Starting random seed |
| `total_seed` | `int` | `500` | Number of seeds to run |
| `sampling_strategy` | `str \| float` | `0.75` | Under-sampling ratio for `RandomUnderSampler` on full dataset |
| `save_all_features` | `bool` | `False` | Save all ranked features per seed |
| `plot_significant_features` | `bool` | `False` | Save a feature-importance plot per seed |

#### `fit()` method parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `with_evaluation` | `bool` | `True` | `True` → `train_and_evaluate()` (80/20 split + metrics); `False` → `train()` (full dataset, no metrics) |
| `**kwargs` | — | — | Forwarded to `train_and_evaluate()` or `train()` (same parameters as those methods) |

### 7. Supported Classifiers

| Classifier | Description | Imbalance Handling |
|------------|-------------|---------------------|
| `rf` | Random Forest (balanced, robust, default) | `class_weight="balanced"` |
| `gb` | Gradient Boosting (handles imbalance natively) | None (natural) |
| `xgb` | XGBoost (excellent for imbalanced data) | `scale_pos_weight` grid search |
| `svm` | Support Vector Machine (high-dimensional) | `class_weight="balanced"` |
| `lr` | Logistic Regression (interpretable, fast) | `class_weight="balanced"` |
| `nn` | Neural Network MLP (complex patterns) | None |
| `dt` | Decision Tree (interpretable baseline) | `class_weight="balanced"` |
| `knn` | K-Nearest Neighbors (simple baseline) | None |
| `nb` | Gaussian Naive Bayes (fast baseline) | None |
| `voting` | Soft VotingClassifier (RF + XGBoost ensemble) | Combined |

### 8. Hyperparameter Grids

Each classifier comes with a built-in hyperparameter grid for `GridSearchCV`. You can override the grid via `ClassifierModel`:

```python
from eruption_forecast.model.classifier_model import ClassifierModel

clf = ClassifierModel("xgb", random_state=42)

# Override default grid
clf.grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "scale_pos_weight": [1, 5, 10, 20],
}
```

**Default grids by classifier:**

<details>
<summary>Random Forest (<code>rf</code>)</summary>

```python
{
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
```
</details>

<details>
<summary>Gradient Boosting (<code>gb</code>)</summary>

```python
{
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
```
</details>

<details>
<summary>XGBoost (<code>xgb</code>)</summary>

```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
    "scale_pos_weight": [1, 5, 10, 15],  # Tunes positive-class weighting
}
```

> `scale_pos_weight` controls how much extra weight positive (eruption) samples receive. Higher values increase sensitivity at the cost of more false positives.
</details>

<details>
<summary>Voting Ensemble (<code>voting</code>)</summary>

```python
{
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, None],
    "xgb__n_estimators": [50, 100],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__max_depth": [5, 7],
}
```

> Combines Random Forest and XGBoost with soft voting (probability averaging).
</details>

### 9. Cross-Validation Strategies

| Strategy | Class | Best For |
|----------|-------|----------|
| `shuffle` | `StratifiedShuffleSplit` | Random splits with stratification (default) |
| `stratified` | `StratifiedKFold` | Preserves class distribution across folds |
| `timeseries` | `TimeSeriesSplit` | Temporal data, strict no-future-leakage |

### 10. Predict on Future Data with ModelPredictor

`ModelPredictor` supports two modes after `train()`:

#### Single model — evaluation mode (labelled data)

Evaluates each seed model against known eruption labels and aggregates metrics across seeds.

```python
from eruption_forecast.model.model_predictor import ModelPredictor

predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models=trainer.csv,  # trained_model_*.csv from train()
    output_dir="output/predictions",
)

# One row per (classifier, seed)
df_metrics = predictor.predict(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
)
print(df_metrics[["balanced_accuracy", "f1_score"]].describe())

# Best (classifier, seed) overall
evaluator = predictor.predict_best(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
    criterion="balanced_accuracy",
)
print(evaluator.summary())
evaluator.plot_all()
```

`predict_best()` accepts any metric column as `criterion`:
`"accuracy"`, `"balanced_accuracy"`, `"f1_score"`, `"precision"`, `"recall"`, `"roc_auc"`, `"pr_auc"`.

#### ModelPredictor constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | `str \| datetime` | — | Start date for prediction period (format: YYYY-MM-DD) |
| `end_date` | `str \| datetime` | — | End date for prediction period (format: YYYY-MM-DD) |
| `trained_models` | `str \| dict[str, str]` | — | Single `trained_model_*.csv` path (from `train()`) or a `{name: path}` dict for multi-model consensus |
| `overwrite` | `bool` | `False` | Overwrite existing output files |
| `n_jobs` | `int` | `1` | Number of parallel jobs for feature extraction |
| `output_dir` | `str \| None` | `None` | Output directory; defaults to `<root_dir>/output/predictions` |
| `root_dir` | `str \| None` | `None` | Root directory for resolving output paths |
| `verbose` | `bool` | `False` | Enable verbose logging |

#### Single model — forecast mode (unlabelled data)

When no ground-truth labels are available, use `predict_proba()`.

```python
predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models=trainer.csv,
    output_dir="output/predictions",
)

df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",  # or pd.DataFrame
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

#### Multi-model consensus

Pass a dict of classifier registries.  `predict_proba()` aggregates within
each classifier (across seeds) and then across classifiers (consensus).

```python
predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models={
        "rf":  "output/VG.OJN.00.EHZ/trainings/model-only/random-forest-classifier/stratified-shuffle-split/trained_model_RandomForestClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv",
        "xgb": "output/VG.OJN.00.EHZ/trainings/model-only/xgb-classifier/stratified-shuffle-split/trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv",
    },
    output_dir="output/predictions",
)

df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

**Output columns (multi-model):**

| Column | Description |
|--------|-------------|
| `{name}_eruption_probability` | Mean P(eruption) across seeds of that classifier |
| `{name}_uncertainty` | Std across seeds of that classifier |
| `{name}_confidence` | Seed-level agreement fraction (0.5–1.0) |
| `{name}_prediction` | Hard label for that classifier |
| `consensus_eruption_probability` | Mean P(eruption) averaged across all classifiers |
| `consensus_uncertainty` | Std of per-classifier means (inter-model disagreement) |
| `consensus_confidence` | Fraction of classifiers voting with consensus majority |
| `consensus_prediction` | Hard label — `1` if `consensus_eruption_probability ≥ 0.5` |

Results are saved to `predictions.csv`.  The plot shows each classifier as a
dashed line and the consensus as a solid black line with a shaded uncertainty
band (`eruption_forecast.png` in `figures/`).

### 11. Model Evaluation

```python
from eruption_forecast.model.model_evaluator import ModelEvaluator

# From in-memory objects
evaluator = ModelEvaluator(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    model_name="xgb_42",
    output_dir="output/eval",
)

# Or load directly from files
evaluator = ModelEvaluator.from_files(
    model_path="output/trainings/classifier/XGBClassifier/stratified/models/00042.pkl",
    X_test="output/features/all_features.csv",
    y_test="output/features/label_features.csv",
    selected_features=["feat_a", "feat_b"],  # optional
    model_name="xgb_42",
    output_dir="output/eval",
)

# Print a formatted summary
print(evaluator.summary())

# Get metrics as a dict
metrics = evaluator.get_metrics()
# Keys: accuracy, balanced_accuracy, precision, recall, f1_score, roc_auc, pr_auc,
#       true_positives, true_negatives, false_positives, false_negatives,
#       sensitivity, specificity, optimal_threshold, f1_at_optimal,
#       recall_at_optimal, precision_at_optimal

# Generate all plots (saved to output_dir)
evaluator.plot_all()
# Produces: confusion_matrix, roc_curve, pr_curve, threshold_analysis,
#            feature_importance, calibration, prediction_distribution

# Find optimal decision threshold
threshold, threshold_metrics = evaluator.optimize_threshold(criterion="f1")
print(f"Optimal threshold: {threshold:.3f}")
print(f"F1 at threshold:   {threshold_metrics['f1']:.4f}")
```

### 12. Analyze Training Results

```python
import pandas as pd

# Suffix format: {ClassifierName}-{CVName}_rs-{random_state}_ts-{total_seed}_top-{n}
# e.g., XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20
base = "output/trainings/model-with-evaluation/xgb-classifier/stratified-shuffle-split"
suffix = "XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20"

# All per-seed metrics
metrics = pd.read_csv(f"{base}/all_metrics_{suffix}.csv")

# Summary statistics (mean ± std)
summary = pd.read_csv(f"{base}/metrics_summary_{suffix}.csv", index_col=0)
print(summary[["balanced_accuracy", "f1_score", "precision", "recall"]])

# Best seed
best_seed = metrics.loc[metrics["balanced_accuracy"].idxmax()]
print(f"Best seed:          {best_seed['random_state']}")
print(f"Balanced Accuracy:  {best_seed['balanced_accuracy']:.4f}")
print(f"F1 Score:           {best_seed['f1_score']:.4f}")

# Aggregated significant features
sig_features = pd.read_csv(f"{base}/features/significant_features.csv")
print(sig_features.head(10))
```

---

## Advanced Usage

### ForecastModel — Orchestrated Pipeline

`ForecastModel` chains the entire pipeline in a single fluent API. It internally delegates to `ModelTrainer.train_and_evaluate()` or `ModelTrainer.train()` (controlled by `with_evaluation`) when you call `.train()`.

#### ForecastModel constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `station` | `str` | — | Seismic station code (e.g. `"OJN"`) |
| `channel` | `str` | — | Seismic channel code (e.g. `"EHZ"`) |
| `start_date` | `str \| datetime` | — | Training period start date (`YYYY-MM-DD`) |
| `end_date` | `str \| datetime` | — | Training period end date (`YYYY-MM-DD`) |
| `window_size` | `int` | — | Duration (days) of each tremor window fed into tsfresh |
| `volcano_id` | `str` | — | Identifier used in output filenames (e.g. `"LEWOTOBI"`) |
| `network` | `str` | `"VG"` | Seismic network code |
| `location` | `str` | `"00"` | Seismic location code |
| `output_dir` | `str \| None` | `None` | Base output directory; relative paths are resolved against `root_dir`. Defaults to `root_dir/output` |
| `root_dir` | `str \| None` | `None` | Anchor for resolving relative `output_dir`. Relative values are normalised to an absolute path immediately. Defaults to `os.getcwd()` |
| `overwrite` | `bool` | `False` | Re-run and overwrite existing output files |
| `n_jobs` | `int` | `1` | Parallel workers propagated to all pipeline stages |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `debug` | `bool` | `False` | Enable debug-level logging |



```python
from eruption_forecast.model.forecast_model import ForecastModel

model = ForecastModel(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=2,
    volcano_id="LEWOTOBI",
    network="VG",
    location="00",
    n_jobs=4,
    verbose=True,
)

model.calculate(
    source="sds",
    sds_dir="/data/sds",
    plot_tmp=True,
    save_plot=True,
).build_label(
    window_step=6,
    window_step_unit="hours",
    day_to_forecast=2,
    eruption_dates=["2025-03-20", "2025-06-15"],
).extract_features(
    select_tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4"],
    use_relevant_features=True,
).train(
    classifier="xgb",
    cv_strategy="stratified",
    random_state=0,
    total_seed=500,
    number_of_significant_features=20,
    sampling_strategy=0.75,
)
```

If you already have pre-calculated tremor data, skip the `calculate()` step:

```python
model.load_tremor_data(
    tremor_csv="output/VG.OJN.00.EHZ/tremor/tremor_2025-01-01_2025-12-31.csv"
).build_label(...).extract_features(...).train(...)
```

Change feature selection method before training:

```python
model.set_feature_selection_method("combined").train(
    classifier="rf",
    cv_strategy="timeseries",
    total_seed=200,
)
```

`ForecastModel.train()` passes `with_evaluation` to `ModelTrainer.fit()` internally. Set `with_evaluation=False` to train on the full dataset without an 80/20 split. Alternatively, use `ModelTrainer` directly after calling `fm.extract_features()`:

```python
from eruption_forecast.model.model_trainer import ModelTrainer

# After running fm.extract_features(), features and labels are at:
#   fm.features_csv  — extracted features
#   fm.label_csv     — aligned labels

trainer = ModelTrainer(
    extracted_features_csv=fm.features_csv,
    label_features_csv=fm.label_csv,
    output_dir="output/trainings",
    classifier="xgb",
    cv_strategy="stratified",
    n_jobs=4,
)
trainer.train(random_state=0, total_seed=500)  # Full-dataset training
```

### Comparing Multiple Classifiers

```python
import pandas as pd

base = "output/trainings/model-with-evaluation"
suffix = "rs-0_ts-500_top-20"
classifiers = {
    "rf":     f"{base}/random-forest-classifier/stratified-shuffle-split/all_metrics_RandomForestClassifier-StratifiedShuffleSplit_{suffix}.csv",
    "gb":     f"{base}/gradient-boosting-classifier/stratified-shuffle-split/all_metrics_GradientBoostingClassifier-StratifiedShuffleSplit_{suffix}.csv",
    "xgb":    f"{base}/xgb-classifier/stratified-shuffle-split/all_metrics_XGBClassifier-StratifiedShuffleSplit_{suffix}.csv",
    "voting": f"{base}/voting-classifier/stratified-shuffle-split/all_metrics_VotingClassifier-StratifiedShuffleSplit_{suffix}.csv",
}

results = []
for name, path in classifiers.items():
    df = pd.read_csv(path)
    results.append({
        "classifier": name,
        "mean_balanced_acc": df["balanced_accuracy"].mean(),
        "std_balanced_acc":  df["balanced_accuracy"].std(),
        "mean_f1":           df["f1_score"].mean(),
        "mean_roc_auc":      df["roc_auc"].mean(),
    })

comparison = pd.DataFrame(results).sort_values("mean_balanced_acc", ascending=False)
print(comparison.to_string(index=False))
```

### train() + ModelPredictor Workflow (Recommended for Operational Use)

Use `train()` to train on all available historical data, then evaluate on future events
or forecast on unlabelled data.

```python
from eruption_forecast.model.model_trainer import ModelTrainer
from eruption_forecast.model.model_predictor import ModelPredictor

# --- Stage 1: Train on historical data ---
trainer = ModelTrainer(
    extracted_features_csv="data/historical/all_features.csv",
    label_features_csv="data/historical/label_features.csv",
    output_dir="output/trainings",
    classifier="xgb",
    cv_strategy="stratified",
    number_of_significant_features=20,
    n_jobs=4,
)

trainer.train(
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)

# --- Stage 2a: Evaluate on future/held-out labelled data ---
predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models=trainer.csv,
    output_dir="output/predictions",
)

df_metrics = predictor.predict(
    future_features_csv="data/future/all_features.csv",
    future_labels_csv="data/future/label_features.csv",  # known labels
    plot=True,
)
print(df_metrics[["classifier", "balanced_accuracy", "f1_score", "roc_auc"]])

best_evaluator = predictor.predict_best(
    future_features_csv="data/future/all_features.csv",
    future_labels_csv="data/future/label_features.csv",
    criterion="balanced_accuracy",
)
print(best_evaluator.summary())
best_evaluator.plot_all()

# --- Stage 2b: Forecast on unlabelled data (single model) ---
predictor = ModelPredictor(
    start_date="2025-03-23",
    end_date="2025-03-30",
    trained_models=trainer.csv,
    output_dir="output/forecast",
)
df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
print(df_forecast[df_forecast["model_prediction"] == 1])

# --- Stage 2c: Multi-model consensus forecast ---
predictor = ModelPredictor(
    start_date="2025-03-23",
    end_date="2025-03-30",
    trained_models={
        "rf":  "output/VG.OJN.00.EHZ/trainings/model-only/random-forest-classifier/stratified-shuffle-split/trained_model_RandomForestClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv",
        "xgb": "output/VG.OJN.00.EHZ/trainings/model-only/xgb-classifier/stratified-shuffle-split/trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv",
    },
    output_dir="output/consensus",
)
df_consensus = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
# Inspect windows where all classifiers agree on eruption
eruption_windows = df_consensus[df_consensus["consensus_confidence"] == 1.0]
print(eruption_windows[["consensus_eruption_probability", "consensus_confidence"]])
```

### Custom Frequency Bands

```python
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
).change_freq_bands([
    (0.1, 0.5),    # Very low frequency
    (0.5, 2.0),    # Low frequency
    (2.0, 5.0),    # Mid frequency
    (5.0, 10.0),   # High frequency
    (10.0, 20.0),  # Very high frequency
]).from_sds(sds_dir="/data/sds").run()
```

---

## Visualization & Plotting

The `plots/` module provides publication-quality visualization functions with consistent Nature/Science journal styling. All plots use colorblind-safe palettes, clean typography, and high DPI output suitable for papers and presentations.

### Tremor Time-Series Plots

#### plot_tremor()

Visualize tremor metrics (RSAM/DSAR) as multi-panel time-series plots.

```python
from eruption_forecast.plots.tremor_plots import plot_tremor
import pandas as pd

# Load tremor data
df = pd.read_csv("output/VG.OJN.00.EHZ/tremor/tremor.csv", index_col=0, parse_dates=True)

# Basic tremor plot
plot_tremor(
    df=df,
    figure_dir="output/figures",
    dpi=150,
)

# Custom interval and selected columns
plot_tremor(
    df=df,
    interval=6,
    interval_unit="hours",
    selected_columns=["rsam_f2", "rsam_f3", "dsar_f2-f3"],
    figure_dir="output/figures",
    filename="tremor_selected",
    dpi=300,
    verbose=True,
)
```

**Key Parameters:**
- `df` - Tremor DataFrame with datetime index
- `interval` - X-axis tick interval (default: 1)
- `interval_unit` - "hours" or "days" (default: "hours")
- `selected_columns` - Subset of columns to plot (default: all)
- `dpi` - Resolution (default: 150)

### Feature Importance Plots

#### plot_significant_features()

Visualize feature importance or p-values as horizontal bar charts.

```python
from eruption_forecast.plots.feature_plots import plot_significant_features
import pandas as pd

# From DataFrame
df_features = pd.read_csv("features.csv")
plot_significant_features(
    features=df_features,
    number_of_features=50,
    top_features=20,
    output_dir="output/figures",
    filename="feature_importance",
    dpi=150,
)

# From CSV file directly
plot_significant_features(
    features="path/to/features.csv",
    number_of_features=30,
    top_features=10,
    values_column="importance",  # or "p_values"
    output_dir="output/figures",
)
```

**Key Parameters:**
- `features` - DataFrame or CSV path with feature data
- `number_of_features` - Total features to display (default: 50)
- `top_features` - Number highlighted with darker color (default: 20)
- `values_column` - Column name for values (auto-detected if None)

### Model Evaluation Plots

The `ModelEvaluator` class provides 7 evaluation plot types for comprehensive model analysis.

#### Available Plot Types

1. **Confusion Matrix** - Classification performance breakdown
2. **ROC Curve** - True/false positive rate tradeoff with AUC score
3. **Precision-Recall Curve** - Precision/recall tradeoff with Average Precision
4. **Threshold Analysis** - Metrics vs decision threshold
5. **Feature Importance** - Top contributing features to predictions
6. **Calibration Curve** - Predicted vs actual probabilities (reliability diagram)
7. **Prediction Distribution** - Score distributions by class (histogram + KDE)

#### Usage Example

```python
from eruption_forecast.model.model_evaluator import ModelEvaluator

# Create evaluator from trained model
evaluator = ModelEvaluator.from_files(
    model_path="output/trainings/model.pkl",
    features_csv="output/features.csv",
    label_csv="output/label.csv",
    output_dir="output/figures",
)

# Generate all 7 plots at once
evaluator.plot_all()

# Or generate individual plots
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curve()
evaluator.plot_precision_recall_curve()
evaluator.plot_threshold_analysis()
evaluator.plot_feature_importance()
evaluator.plot_calibration()
evaluator.plot_prediction_distribution()
```

All plots are saved to `output_dir` with publication-quality styling.

### Forecast Visualization

#### Plotting with ModelPredictor

Visualize eruption probability forecasts as time-series plots.

```python
from eruption_forecast.model.model_predictor import ModelPredictor

predictor = ModelPredictor(
    start_date="2025-03-23",
    end_date="2025-03-30",
    trained_models="output/trainings/model-only/random-forest-classifier/stratified-k-fold/trained_model_*.csv",
    output_dir="output/forecast",
)

# With labeled data (evaluation mode)
df_metrics = predictor.predict(
    future_features_csv="output/features_forecast.csv",
    future_labels_csv="output/label_forecast.csv",
    plot=True,
)

# Without labels (forecasting mode)
df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

**Plot Features:**
- Probability time-series for each classifier
- Consensus probability (mean across classifiers)
- Confidence bands (standard deviation)
- Eruption event markers (if labels provided)
- Saved as `eruption_forecast.png` in `figures/` subdirectory

### Batch Replot Utilities

Regenerate plots for multiple files in parallel — useful for applying style updates or reprocessing after data changes.

#### replot_tremor()

Batch replot daily tremor CSV files.

```python
from eruption_forecast.plots.tremor_plots import replot_tremor

# Sequential processing
results = replot_tremor(
    daily_dir="output/VG.OJN.00.EHZ/tremor/daily",
    output_dir="output/VG.OJN.00.EHZ/tremor/figures",
    overwrite=True,
)
print(f"Created: {results['created']}, Skipped: {results['skipped']}")

# Parallel processing with custom parameters
results = replot_tremor(
    daily_dir="output/VG.OJN.00.EHZ/tremor/daily",
    n_jobs=4,
    interval=6,
    interval_unit="hours",
    dpi=300,
    selected_columns=["rsam_f2", "rsam_f3"],
    overwrite=False,
)
```

#### replot_significant_features()

Batch replot feature importance across multiple seeds.

```python
from eruption_forecast.plots.feature_plots import replot_significant_features

# Sequential processing
results = replot_significant_features(
    all_features_dir="output/trainings/features/all_features",
    output_dir="output/trainings/features/figures/significant",
    overwrite=True,
)

# Parallel processing
results = replot_significant_features(
    all_features_dir="output/trainings/features/all_features",
    n_jobs=4,
    number_of_features=50,
    top_features=20,
    dpi=300,
    overwrite=False,
)
print(f"Processed: Created {results['created']}, Skipped {results['skipped']}, Failed {results['failed']}")
```

**Return Value:**
Both functions return a dict with counts:
- `'created'` - Plots successfully generated
- `'skipped'` - Plots skipped (file exists, overwrite=False)
- `'failed'` - Plots that encountered errors

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dpi` | `int` | `150` | Resolution in dots per inch (use 300 for publication) |
| `overwrite` | `bool` | `True` | Replace existing plots if they exist |
| `n_jobs` | `int` | `1` | Parallel workers for batch utilities (use >1 for large datasets) |
| `output_dir` / `figure_dir` | `str` | varies | Directory where plots are saved |
| `verbose` | `bool` | `False` | Enable logging of plot generation |
| `filename` | `str` | auto | Custom filename stem (extension added automatically) |

### Import Examples

```python
# Tremor plots
from eruption_forecast.plots.tremor_plots import plot_tremor, replot_tremor

# Feature plots
from eruption_forecast.plots.feature_plots import (
    plot_significant_features,
    replot_significant_features,
)

# Evaluation plots (via ModelEvaluator)
from eruption_forecast.model.model_evaluator import ModelEvaluator

# Forecast plots (via ModelPredictor)
from eruption_forecast.model.model_predictor import ModelPredictor

# All evaluation plot functions available individually:
from eruption_forecast.plots.evaluation_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_threshold_analysis,
    plot_feature_importance,
    plot_calibration,
    plot_prediction_distribution,
)
```

---

## Output Directory Structure

All outputs are organized under `{output_dir}/{network}.{station}.{location}.{channel}/`
(e.g., `output/VG.OJN.00.EHZ/`).

```
output/
└── VG.OJN.00.EHZ/
    ├── tremor/
    │   ├── tmp/                              # Temporary daily files
    │   └── tremor_*.csv                      # Final merged tremor data
    │
    ├── features/
    │   ├── tremor_matrix_*.csv               # Aligned tremor matrix (all columns)
    │   ├── tremor_matrix_per_method/         # Per-column tremor matrices (optional)
    │   ├── all_extracted_features_*.csv      # tsfresh output per tremor column
    │   └── label_features_*.csv             # Labels aligned with features
    │
    └── trainings/
        ├── model-with-evaluation/        # Output of train_and_evaluate()
        │   └── {classifier-slug}/        # e.g., random-forest-classifier
        │       └── {cv-slug}/            # e.g., stratified-shuffle-split
        │           ├── features/
        │           │   ├── significant_features/     # Per-seed top-N features
        │           │   │   ├── 00000.csv
        │           │   │   └── ...
        │           │   ├── all_features/             # All ranked features (optional)
        │           │   │   ├── 00000.csv
        │           │   │   └── ...
        │           │   ├── figures/significant/      # Feature plots (optional)
        │           │   │   └── 00000.jpg
        │           │   ├── significant_features.csv  # Aggregated features (all seeds)
        │           │   └── top_20_significant_features.csv
        │           ├── models/
        │           │   ├── 00000.pkl     # Trained model (seed 0)
        │           │   ├── 00001.pkl
        │           │   └── ...
        │           ├── metrics/
        │           │   ├── 00000.json    # Per-seed metrics
        │           │   └── ...
        │           ├── trained_model_{suffix}.csv    # Registry of all trained models
        │           ├── all_metrics_{suffix}.csv      # All seed metrics
        │           └── metrics_summary_{suffix}.csv  # Mean ± std summary
        │
        └── model-only/                   # Output of train()
            └── {classifier-slug}/
                └── {cv-slug}/
                    ├── features/
                    │   ├── significant_features/
                    │   └── ...
                    ├── models/
                    └── trained_model_{suffix}.csv
```

**ModelPredictor output** (`output_dir/predictions/`):

*Evaluation mode (`predict()` / `predict_best()`):*

```
predictions/
├── metrics/
│   ├── all_metrics.csv
│   └── metrics_summary.csv
└── seed_00000/                    # Only created when plot=True
    ├── seed_00000_confusion_matrix.png
    ├── seed_00000_roc_curve.png
    ├── seed_00000_pr_curve.png
    ├── seed_00000_threshold_analysis.png
    ├── seed_00000_feature_importance.png
    ├── seed_00000_calibration.png
    └── seed_00000_prediction_distribution.png
```

*Forecast mode (`predict_proba()`):*

```
predictions/
├── predictions.csv                # eruption_probability, uncertainty, confidence, prediction
└── figures/
    └── eruption_forecast.png      # Probability + confidence time-series plot
```

---

## Configuration

### Logging

```python
from eruption_forecast.logger import set_log_level, set_log_directory

set_log_level("DEBUG")  # Options: DEBUG, INFO, WARNING, ERROR
set_log_directory("/custom/logs")
```

---

## Requirements

### Core Dependencies

- Python >= 3.11
- pandas >= 3.0.0
- numpy
- scipy
- obspy (seismic data processing)
- tsfresh (time-series feature extraction)
- scikit-learn
- imbalanced-learn
- xgboost
- joblib
- matplotlib
- seaborn
- loguru

### Development Dependencies

- ruff (linting)
- ty (type checking)
- pytest (testing)

---

## Development

### Code Quality Tools

```bash
# Lint and auto-fix
uv run ruff check --fix src/

# Type checking
uvx ty check src/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/eruption_forecast tests/

# Run specific test
pytest tests/test_train_model.py
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make changes with tests
4. Ensure code passes linting and type checks (`uv run ruff check --fix src/`)
5. Update documentation
6. Submit a pull request

### Code Style

- Follow PEP 8 guidelines
- Use Google-style docstrings
- Include type hints for all functions
- Write unit tests for new features

---

## License

MIT License — see LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{eruption_forecast,
  author = {Martanto},
  title = {eruption-forecast: Volcanic Eruption Forecasting with Seismic Data},
  year = {2025},
  url = {https://github.com/martanto/eruption-forecast}
}
```

## Support

- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed architecture
- **Issues**: Report bugs and request features on GitHub
- **Email**: martanto@live.com

## Acknowledgments

This project uses:
- [ObsPy](https://github.com/obspy/obspy) for seismic data processing
- [tsfresh](https://github.com/blue-yonder/tsfresh) for feature extraction
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
- [uv](https://docs.astral.sh/uv/) for package management

---

**Version:** 0.2.1
**Status:** Active Development
**Last Updated:** 2026-02-15
