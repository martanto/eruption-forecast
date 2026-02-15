# Copilot Instructions for eruption-forecast

This Python package implements volcanic eruption forecasting using seismic tremor analysis and machine learning.

## Package Manager and Commands

This project uses **uv** as the package manager.

### Setup
```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --group dev
```

### Code Quality
```bash
# Lint (auto-fix enabled)
uv run ruff check --fix src/

# Type check
uvx ty check src/

# Run tests
uv run pytest tests/
```

### Running the Pipeline
```bash
# Execute the full pipeline
python main.py

# Or with uv
uv run python main.py

# Run specific test file
pytest tests/test_train_model.py
```

## Architecture

The package implements a chained pipeline for seismic data analysis and eruption forecasting:

```
CalculateTremor → LabelBuilder → TremorMatrixBuilder → FeaturesBuilder → ModelTrainer → ModelPredictor
```

### Core Pipeline Flow

**ForecastModel** orchestrates all components via method chaining:

```python
fm = ForecastModel(root_dir, station, channel, start_date, end_date, window_size, volcano_id, n_jobs)
fm.calculate(source, sds_dir) \
  .build_label(eruption_dates, day_to_forecast, window_step, window_step_unit) \
  .extract_features(select_tremor_columns) \
  .train(classifier, cv_strategy, random_state, total_seed)
```

All output paths are anchored to `root_dir`. The pipeline generates this directory structure:

```
output/{network}.{station}.{location}.{channel}/
├── tremor/
│   ├── tmp/              # Daily CSVs (multiprocessing)
│   └── tremor.csv        # Merged tremor metrics
├── forecast/             # Model predictions
├── figures/              # Evaluation plots
└── logs/                 # Debug logs
```

### Key Components

#### 1. Tremor Calculation (`src/eruption_forecast/tremor/`)

**CalculateTremor** converts raw seismic waveforms into tremor metrics:
- Reads SDS (SeisComP Data Structure) files or FDSN web services
- Calculates **RSAM** (mean amplitude) and **DSAR** (amplitude ratio) across frequency bands
- Default bands: `(0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz`
- Outputs CSV with 10-minute sampling interval
- Supports multiprocessing via `n_jobs` (each day processed independently)

**Column naming:** Frequency bands are aliased as `f0`, `f1`, `f2`, etc.:
- `rsam_f0`: RSAM for band 0
- `dsar_f0-f1`: DSAR ratio between bands f0 and f1

**Key classes:**
- `CalculateTremor`: Main orchestrator
- `RSAM`, `DSAR`: Metric calculators
- `TremorData`: Loads/validates tremor CSV files
- `SDS`: Reads SeisComP data structure

#### 2. Label Building (`src/eruption_forecast/label/`)

**LabelBuilder** generates binary labels for supervised learning:
- Creates sliding time windows over the date range
- Labels windows as erupted (1) or not (0) based on `day_to_forecast`
- Filename encodes parameters: `label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv`

**Window parameters:**
- `window_size`: Window duration in days
- `window_step` + `window_step_unit`: Step size ("hours" or "minutes")
- `day_to_forecast`: Days before eruption to start positive labeling

**Key classes:**
- `LabelBuilder`: Creates labeled windows
- `LabelData`: Loads label CSV and parses parameters from filename

#### 3. Tremor Matrix Building (`src/eruption_forecast/features/`)

**TremorMatrixBuilder** slices tremor time-series into windows:
- Aligns tremor data with label windows
- Validates sample counts per window
- Outputs matrix with `id`, `datetime`, and tremor columns

#### 4. Feature Extraction (`src/eruption_forecast/features/`)

**FeaturesBuilder** extracts time-series features using tsfresh:
- **Training mode** (with labels): Filters windows, saves aligned label CSV
- **Prediction mode** (no labels): Extracts all features, no relevance filtering
- Processes each tremor column independently

**FeatureSelector** implements two-stage feature selection:
1. Statistical filtering via tsfresh (FDR-based)
2. Random Forest importance ranking
- Methods: `"tsfresh"`, `"random_forest"`, `"combined"`

#### 5. Model Training (`src/eruption_forecast/model/`)

**ModelTrainer** trains classifiers with multiple random seeds:
- Supports 10 classifiers: `rf`, `gb`, `xgb`, `svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting`
- CV strategies: `shuffle` (StratifiedShuffleSplit), `stratified` (StratifiedKFold), `timeseries` (TimeSeriesSplit)
- Handles class imbalance via `RandomUnderSampler`
- Each classifier has built-in hyperparameter grids for GridSearchCV

**Two training workflows:**

```
train_and_evaluate():              train():
Full Dataset                       Full Dataset
     │                                  │
     ▼                                  ▼
 80/20 Split                       RandomUnderSampler
     │                              (full dataset)
  ┌──┴──┐                                │
Train  Test                         Feature Selection
  │     │                            (full dataset)
RandomUnder                               │
Sampler                              GridSearchCV + CV
  │                                       │
Feature                              Save model + registry
Selection                            (.pkl + .csv)
  │
GridSearchCV + CV
  │
Evaluate on Test
  │
Save model + metrics
```

**When to use which:**
- `train_and_evaluate()`: Quick feedback on in-sample performance (80/20 split), suitable for hyperparameter exploration
- `train()`: Production training on 100% of data, evaluate later on future dataset via `ModelPredictor`

**Unified entry point:**
- `fit(with_evaluation=True)` → calls `train_and_evaluate()`
- `fit(with_evaluation=False)` → calls `train()`

**Key classes:**
- `ModelTrainer`: Multi-seed training loop with feature selection
- `ClassifierModel`: Manages classifier instances and hyperparameter grids
- `ModelEvaluator`: Computes metrics (accuracy, balanced_accuracy, precision, recall, f1, AUC, ROC, PR) and 7 plot types
- `ModelPredictor`: Runs inference in two modes:
  - **Evaluation mode** (with labels): `predict()` / `predict_best()` — evaluates each seed, aggregates metrics
  - **Forecast mode** (no labels): `predict_proba()` — returns probability, uncertainty, confidence per window

### Data Classes

**TremorData** and **LabelData** wrap CSV files with metadata:
- Provide `start_date_str`, `end_date_str` properties
- Validate data formats and sampling rates
- Use `@cached_property` for lazy evaluation

### Utility Functions (`src/eruption_forecast/utils.py`)

- `construct_windows()`: Generates sliding time windows
- `calculate_window_metrics()`: Window-based metrics with outlier removal
- `detect_maximum_outlier()` / `remove_outliers()`: Z-score outlier detection
- `to_datetime()`: Date string parsing with validation
- `validate_date_ranges()`: Ensures start < end
- `validate_window_step()`: Validates step parameters
- `random_under_sampler()`: Balances classes
- `get_significant_features()`: Retrieves tsfresh-filtered features
- `resolve_output_dir()`: Resolves paths relative to `root_dir`

## Key Conventions

### Method Chaining Pattern

All builder classes return `self` to enable fluent interfaces:

```python
CalculateTremor(...) \
    .from_sds(sds_dir) \
    .change_freq_bands([(0.1, 1.0), (1.0, 5.0)]) \
    .run()
```

### Data Format Standards

- **Tremor CSV**: DateTime index, columns `rsam_f0`, `rsam_f1`, `dsar_f0-f1`, etc.
- **Label CSV**: DateTime index, columns `id` (int) and `is_erupted` (0 or 1)
- **Date format**: Always `YYYY-MM-DD` string format
- **Sampling interval**: Default 10 minutes for tremor data
- **Label logic**: Window labeled `is_erupted=1` when its **end time** (index) falls within `[eruption_date − day_to_forecast, eruption_date]`

### Path Handling

This project is developed on Windows. Use `os.path.join()` for cross-platform compatibility (the codebase already follows this pattern).

### Multiprocessing

`CalculateTremor` processes each day independently when `n_jobs > 1`:
- Generates temporary daily CSVs in `tmp/` directory
- Merges into final `tremor.csv` after completion

`ModelTrainer` and `FeaturesBuilder` also support `n_jobs` for parallel seed training and feature extraction

## Output Directory Structure

All outputs are organized under `{output_dir}/{network}.{station}.{location}.{channel}/` (e.g., `output/VG.OJN.00.EHZ/`).

```
output/VG.OJN.00.EHZ/
├── tremor/
│   ├── tmp/                              # Temporary daily files
│   └── tremor_*.csv                      # Final merged tremor data
│
├── features/
│   ├── tremor_matrix_*.csv               # Aligned tremor matrix
│   ├── tremor_matrix_per_method/         # Per-column matrices (optional)
│   ├── all_extracted_features_*.csv      # tsfresh output
│   └── label_features_*.csv              # Aligned labels
│
└── trainings/
    ├── model-with-evaluation/            # train_and_evaluate() output
    │   └── {classifier-slug}/            # e.g., xgb-classifier
    │       └── {cv-slug}/                # e.g., stratified-shuffle-split
    │           ├── features/
    │           │   ├── significant_features/     # Per-seed top-N
    │           │   ├── significant_features.csv  # Aggregated (all seeds)
    │           │   └── top_20_significant_features.csv
    │           ├── models/               # .pkl files (00000.pkl, ...)
    │           ├── metrics/              # .json per seed
    │           ├── trained_model_{suffix}.csv    # Model registry
    │           ├── all_metrics_{suffix}.csv      # All seed metrics
    │           └── metrics_summary_{suffix}.csv  # Mean ± std
    │
    └── model-only/                       # train() output
        └── {classifier-slug}/
            └── {cv-slug}/
                ├── features/significant_features/
                ├── models/
                └── trained_model_{suffix}.csv
```

**File naming suffix format:** `{ClassifierName}-{CVName}_rs-{random_state}_ts-{total_seed}_top-{n}`  
Example: `XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20`

**ModelPredictor output:**
- Evaluation mode: `predictions/metrics/` with CSV summaries + optional per-seed plots
- Forecast mode: `predictions/predictions.csv` + `figures/eruption_forecast.png`

## Dependencies

**Scientific stack:**
- **obspy**: Seismic data processing (Stream, Trace objects)
- **pandas**: Time-series manipulation (requires ≥ 3.0.0)
- **tsfresh**: Automated time-series feature extraction (700+ features per column)
- **numpy**, **numba**: Numerical computation and JIT compilation
- **scikit-learn** (≥ 1.4): ML classifiers and cross-validation
- **xgboost** (≥ 3.2), **lightgbm** (≥ 4.6): Gradient boosting classifiers
- **imbalanced-learn** (≥ 0.14.1): RandomUnderSampler for class imbalance
- **seaborn**, **matplotlib**: Visualization
- **loguru**: Structured logging

**Data sources:**
- SDS (SeisComP Data Structure): Primary format for seismic waveforms
- FDSN web services: Alternative data source

**Python version:** ≥ 3.11

## Testing Data

For development and testing, use:
- **SDS directory:** `D:\Data\OJN`
- **Date range:** 2025-03-16 to 2025-03-22
- **Eruption date:** 2025-03-20

See `main.py` for a complete pipeline example.

## Workflow Conventions

1. **Create branch before modifications:** Always create a new branch with `copilot/` prefix before making any commits or code modifications (e.g., `copilot/fix-docstrings`, `copilot/add-feature-x`)
2. **Update SUMMARY.md after each task:** Document all completed work (bug fixes, features, tests, docs)
3. **Track pending work in TODO.md:** Check off completed items, add new tasks
4. **Use `tests/` directory:** All test outputs go here
5. **Do not edit CLAUDE.md:** Never modify CLAUDE.md without explicit user permission

## Model Evaluation Metrics

`ModelEvaluator` provides comprehensive metrics and 7 plot types:

**Metrics:**
- Classification: accuracy, balanced_accuracy, precision, recall, f1_score, sensitivity, specificity
- Curves: roc_auc, pr_auc
- Confusion matrix: true_positives, true_negatives, false_positives, false_negatives
- Threshold optimization: optimal_threshold, f1_at_optimal, recall_at_optimal, precision_at_optimal

**Plots:**
1. Confusion matrix
2. ROC curve
3. Precision-Recall curve
4. Threshold analysis (precision/recall vs threshold)
5. Feature importance (from model)
6. Calibration curve
7. Prediction distribution

Access via `evaluator.get_metrics()`, `evaluator.summary()`, `evaluator.plot_all()`, or `evaluator.from_files()`

## Multi-Model Consensus

`ModelPredictor` supports multi-model consensus forecasting by passing a dict of trained model registries:

```python
predictor = ModelPredictor(
    trained_models={
        "rf": "path/to/rf_trained_model.csv",
        "xgb": "path/to/xgb_trained_model.csv",
    },
    future_features_csv="...",
)
df = predictor.predict_proba(plot=True)
```

**Output columns:**
- Per-classifier: `{name}_eruption_probability`, `{name}_uncertainty`, `{name}_confidence`, `{name}_prediction`
- Consensus: `consensus_eruption_probability`, `consensus_uncertainty`, `consensus_confidence`, `consensus_prediction`
- Uncertainty = std across seeds (intra-model) or classifier means (inter-model)
- Confidence = agreement fraction (0.5–1.0)
