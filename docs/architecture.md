# Package Architecture

> Back to [README](../README.md)

## Table of Contents

- [Directory Structure](#directory-structure)
- [Design Principles](#design-principles)
- [Pipeline Overview](#pipeline-overview)
- [Component Details](#component-details)
  - [Model Class Relationships](#51-model-class-relationships)

---

## Directory Structure

```
eruption-forecast/
├── src/eruption_forecast/
│   ├── data_container.py    # BaseDataContainer — shared ABC for TremorData & LabelData
│   ├── tremor/              # Seismic tremor processing
│   │   ├── calculate_tremor.py
│   │   ├── rsam.py          # Real Seismic Amplitude Measurement
│   │   ├── dsar.py          # Displacement Seismic Amplitude Ratio
│   │   ├── shanon_entropy.py  # Shannon Entropy metric
│   │   └── tremor_data.py
│   ├── label/               # Training label generation
│   │   ├── label_builder.py
│   │   ├── dynamic_label_builder.py
│   │   └── label_data.py
│   ├── features/            # Feature extraction & selection
│   │   ├── features_builder.py
│   │   ├── feature_selector.py
│   │   └── tremor_matrix_builder.py
│   ├── model/               # ML model training & prediction
│   │   ├── forecast_model.py
│   │   ├── model_trainer.py
│   │   ├── model_predictor.py
│   │   ├── model_evaluator.py       # Single-seed evaluation
│   │   ├── multi_model_evaluator.py # Multi-seed aggregate evaluation
│   │   ├── classifier_model.py
│   │   ├── base_ensemble.py         # Shared save/load mixin
│   │   ├── seed_ensemble.py         # All seeds, 1 classifier
│   │   └── classifier_ensemble.py   # All seeds, N classifiers
│   ├── sources/             # Seismic data source adapters
│   │   ├── base.py          # SeismicDataSource — abstract base class
│   │   ├── sds.py           # SDS (SeisComP Data Structure) reader
│   │   └── fdsn.py          # FDSN web service client with local caching
│   ├── plots/               # Visualization utilities
│   │   └── ...
│   ├── utils/               # Focused utility modules
│   │   ├── array.py         # Array operations, outlier detection
│   │   ├── window.py        # Time window operations
│   │   ├── date_utils.py    # Date/time conversion and filename parsing
│   │   ├── dataframe.py     # DataFrame helpers
│   │   ├── ml.py            # ML utilities (resampling, feature loading)
│   │   ├── validation.py    # Centralised validation (dates, random state, columns, sampling)
│   │   ├── pathutils.py     # Path resolution
│   │   └── formatting.py    # Text formatting
│   └── decorators/          # Function decorators
└── tests/                   # Unit tests
```

## Design Principles

- **Single Responsibility**: Each module has one clear purpose
- **DRY (Don't Repeat Yourself)**: Shared behaviour extracted into base classes (`BaseDataContainer`, `SeismicDataSource`) and shared utilities (`validate_random_state`, `load_labels_from_csv`)
- **Explicit Imports**: No hidden re-exports (e.g., `from eruption_forecast.utils.date_utils import to_datetime`)
- **Minimal Dependencies**: Each utils module imports only what it needs
- **Clean Architecture**: Reduced coupling, easier testing and maintenance

---

## Pipeline Overview

```
Raw Seismic Data (SDS / FDSN)
         │
         ▼
┌─────────────────────┐
│   CalculateTremor   │  RSAM + DSAR + Entropy → tremor.csv
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
│         ↓  evaluate()  ↓ train()  │
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

---

## Research Workflow (`main.py`)

`main.py` is the top-level research script. It runs the full pipeline in two
sequential branches — train-with-evaluation and train-for-prediction — both
operating on the same `ForecastModel` instance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             main.py  —  Stage Flow                          │
└─────────────────────────────────────────────────────────────────────────────┘

  fm = ForecastModel(root_dir, station, channel, …, n_jobs=6)
       │
       ▼
┌─────────────────┐
│  fm.calculate() │  CalculateTremor
│                 │  SDS → RSAM / DSAR / Entropy → tremor_*.csv
│                 │  dates: 2025-01-01 → 2025-12-31
└────────┬────────┘
         │
         ├──────────────────────────────────────────────────────────────┐
         │                                                              │
         │  evaluate(fm)                              predict(fm)       │
         │                                                              │
         ▼                                                              ▼
┌─────────────────┐                                            ┌─────────────────┐
│  build_label()  │  2025-01-01 → 2025-08-24                   │  build_label()  │  2025-07-28 → 2025-08-20
│                 │  window_step=6h, dtf=2                     │                 │  window_step=6h, dtf=2
└────────┬────────┘                                            └────────┬────────┘
         │                                                              │
         ▼                                                              ▼
┌─────────────────┐                                            ┌─────────────────┐
│ extract_        │  FeaturesBuilder                           │ extract_        │  FeaturesBuilder
│ features()      │  rsam_f2/f3/f4, dsar_f3-f4                 │ features()      │  (same kwargs)
│                 │  700+ tsfresh features → CSV               │                 │
└────────┬────────┘                                            └────────┬────────┘
         │                                                              │
         ▼                                                              ▼
┌─────────────────┐                                            ┌─────────────────┐
│  train()        │  ModelTrainer                              │  train()        │  ModelTrainer
│  with_eval=True │  classifiers: lite-rf, rf, gb, xgb         │  with_eval=False│  (same classifiers)
│                 │  cv: stratified, seeds: 500                │                 │  cv: stratified, seeds: 500
│                 │  80/20 split → metrics JSON per seed       │                 │  full dataset → no metrics
└────────┬────────┘                                            └────────┬────────┘
         │                                                              │
         ▼                                                              ▼
┌─────────────────┐                                            ┌─────────────────┐
│ MultiModel      │  per-classifier aggregate plots            │  forecast()     │  ModelPredictor
│ Evaluator       │  ROC, PR, calibration, confusion,          │                 │  predict_proba
│ (loop per clf)  │  SHAP, seed stability, …                   │                 │  2025-07-28 → 2025-08-20
└────────┬────────┘                                            └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Classifier      │  cross-classifier comparison
│ Comparator      │  metric bar, ROC overlay,
│ (≥ 2 clf)       │  ranking CSV
└─────────────────┘


  Runtime flags (top-level constants in main.py):
  ┌──────────────────────────┬─────────────────────────────────────────────┐
  │ DEBUG                    │ Read from .env; reduces seeds to 10,        │
  │                          │ classifiers to [lite-rf, rf]                │
  │ N_JOBS                   │ 6 (outer parallelism)                       │
  │ TRAINING_SEEDS           │ 500 (or 10 in DEBUG mode)                   │
  │ CLASSIFIER               │ ["lite-rf", "rf", "gb", "xgb"]              │
  └──────────────────────────┴─────────────────────────────────────────────┘
```

---

## Component Details

### 1. Tremor Calculation (`src/eruption_forecast/tremor/`)

**`CalculateTremor`** processes raw seismic data into tremor metrics:
- Reads seismic data from SDS (SeisComP Data Structure) format or FDSN web services
- Calculates three metrics across multiple frequency bands in parallel:
  - **RSAM** (Real Seismic Amplitude Measurement): Mean amplitude per band
  - **DSAR** (Displacement Seismic Amplitude Ratio): Ratio between consecutive bands
  - **Shannon Entropy**: Signal complexity, single broadband column
- Default frequency bands: `(0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz`
- Supports multiprocessing via `n_jobs`; outputs 10-minute interval CSVs

**Key classes:**
- `CalculateTremor`: Main orchestrator (`calculate_tremor.py`)
- `RSAM`: Mean amplitude metrics (`rsam.py`)
- `DSAR`: Amplitude ratios between bands (`dsar.py`)
- `ShanonEntropy`: Signal complexity metric (`shanon_entropy.py`)
- `TremorData`: Loads and validates tremor CSV files (`tremor_data.py`)
- `SDS`: Reads SeisComP Data Structure files (`src/eruption_forecast/sources/sds.py`)
- `FDSN`: Downloads seismic data from FDSN web services with local SDS caching (`src/eruption_forecast/sources/fdsn.py`)

**Workflow:**
```python
from eruption_forecast.tremor.calculate_tremor import CalculateTremor

# From SDS archive
calculate = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-03",
    n_jobs=4
).from_sds(sds_dir="/path/to/sds").run()
# Output CSV columns: rsam_f0, rsam_f1, dsar_f0-f1, entropy, etc.

# From FDSN web service
calculate = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-03",
).from_fdsn(client_url="https://service.iris.edu").run()
```

### 2. Label Building (`src/eruption_forecast/label/`)

**`LabelBuilder`** generates binary labels for supervised learning:
- Creates sliding time windows and labels them erupted (1) or not (0)
- Uses `day_to_forecast` to look ahead N days before eruptions
- `include_eruption_date` (default `False`): when `True`, the eruption date itself is included in the positive window; when `False`, the window ends the day before the eruption
- Label filenames follow: `label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv`

**`DynamicLabelBuilder`** (extends `LabelBuilder`) generates one window per eruption:
- Each window spans `days_before_eruption` days ending on the eruption date
- Handles overlapping eruptions: when a secondary eruption's `day_to_forecast` period falls inside another eruption's window, both positive regions are marked (a warning is logged)
- All per-eruption windows are concatenated into one DataFrame with globally unique IDs

```
LabelBuilder — one global window over the full date range
──────────────────────────────────────────────────────────────────
 start_date                                              end_date
 │                                                          │
 ├──────────────────────── window ──────────────────────────┤
 │           0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 E              │
 │                               ↑           ↑              │
 │                           dtf start   eruption           │
 └──────────────────────────────────────────────────────────┘

 include_eruption_date=False (default)
 │  0 0 0 0 0 0 0 0 0 0  1  1  1  1  1  1  0                │
 │                       ↑              ↑  ↑                │
 │                   dtf start    day before | eruption     │
 │                                eruption   | (excluded)   │

 include_eruption_date=True
 │  0 0 0 0 0 0 0 0 0 0  1  1  1  1  1  1  1                │
 │                       ↑                 ↑                │
 │                   dtf start          eruption            │
 │                                      (included)          │

DynamicLabelBuilder — one window per eruption, overlaps handled
─────────────────────────────────────────────────────────────────────
 Eruption A  (2025-03-20)        Eruption B  (2025-03-23)
 ┌───────────────────────────┐   ┌───────────────────────────┐
 │ days_before_eruption = 10 │   │ days_before_eruption = 10 │
 │                           │   │                           │
 │ 0  0  0  0  0  0  1  1  E │   │ 0  1  1  1  1  1  1  1  E │
 │                   ↑       │   │    ↑           ↑          │
 │                dtf start  │   │ overlap from  dtf start   │
 │                (Mar 18)   │   │ Eruption A    (Mar 20)    │
 └───────────────────────────┘   └───────────────────────────┘
                                   ↑ secondary overlap → warning logged
   E = eruption date (is_erupted = 1)
   1 = positive label within day_to_forecast window
   0 = negative label
```

**Key classes:**
- `LabelBuilder`: Creates labeled windows over a global date range (`label_builder.py`)
- `DynamicLabelBuilder`: Per-eruption windows with overlap handling (`dynamic_label_builder.py`)
- `LabelData`: Loads label CSV and parses parameters from filename (`label_data.py`)

### 3. Tremor Matrix Building (`src/eruption_forecast/features/`)

**`TremorMatrixBuilder`** slices tremor time-series into windows aligned with labels:
- Takes tremor DataFrame and label DataFrame as input
- Validates sample counts per window
- Concatenates all windows into a unified matrix with `id`, `datetime`, and tremor columns

### 4. Feature Extraction (`src/eruption_forecast/features/`)

**`FeaturesBuilder`** extracts tsfresh features from the tremor matrix:
- Operates in two modes:
  - **Training mode** (labels provided): Filters windows to match labels, saves aligned label CSV
  - **Prediction mode** (no labels): Extracts all features, disables relevance filtering
- Runs tsfresh extraction per tremor column independently

**Key classes:**
- `FeaturesBuilder`: Orchestrates tsfresh feature extraction (`features_builder.py`)
- `FeatureSelector`: Two-stage selection — tsfresh (statistical FDR) → RandomForest (importance) (`feature_selector.py`)
  - Methods: `"tsfresh"`, `"random_forest"`, `"combined"`

### 5. Model Training (`src/eruption_forecast/model/`)

**`ModelTrainer`** trains classifiers across multiple random seeds:
- Supports 10 classifiers: `rf`, `gb`, `xgb`, `svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting`
- CV strategies: `shuffle`, `stratified`, `shuffle-stratified`, `timeseries`
- Uses `RandomUnderSampler` to handle class imbalance
- Feature selection runs once per seed and is shared across classifiers; results are written to `features/{cv-slug}/` rather than inside each classifier directory
- Two training modes:
  - `evaluate()`: 80/20 split → resample train → feature selection → CV → evaluate on test set → save
  - `train()`: Resample full dataset → feature selection → CV → save (no metrics)

**Key classes:**
- `ModelTrainer`: Multi-seed training and evaluation (`model_trainer.py`)
  - `fit(with_evaluation=True)`: Dispatches to `evaluate()` or `train()` based on flag
  - `n_jobs`: outer seed workers; `grid_search_n_jobs`: inner `GridSearchCV`/`FeatureSelector` workers
- `BaseEnsemble`: Shared persistence mixin (`base_ensemble.py`)
  - `save(path)`: Joblib-dumps the ensemble to `.pkl`, creating parent directories
  - `load(path)` (classmethod): Restores instance from `.pkl`; raises `FileNotFoundError` if absent
  - Inherited by both `SeedEnsemble` and `ClassifierEnsemble` — no duplicated boilerplate
- `ClassifierModel`: Manages classifier instances and hyperparameter grids (`classifier_model.py`)
- `ModelEvaluator`: Computes metrics and plots for a fitted model (`model_evaluator.py`)
  - Methods: `get_metrics()`, `summary()`, `plot_all()`, `from_files()`
  - `cv_name` parameter (default `"cv"`): slugified into the default output path `output/trainings/evaluations/classifiers/{clf-slug}/{cv-slug}/` when `output_dir` is `None`
- `MultiModelEvaluator`: Aggregate evaluation across all seeds (`multi_model_evaluator.py`)
  - Methods: `plot_all()`, `plot_roc()`, `get_aggregate_metrics()`, `save_aggregate_metrics()`
- `ModelPredictor`: Runs inference in evaluation or forecast mode (`model_predictor.py`)
  - `predict()` / `predict_best()`: Requires labels (evaluation mode)
  - `predict_proba()`: Unlabelled forecasting with per-classifier + consensus output
- `PipelineConfig`: Serialisable pipeline configuration (`src/eruption_forecast/config/pipeline_config.py`)

### 5.1 Model Class Relationships

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         TRAINING PHASE                                              │
│                                                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                              ModelTrainer  (one classifier)                                  │  │
│   │                                                                                              │  │
│   │   .fit(with_evaluation=True)                     .fit(with_evaluation=False)                 │  │
│   │           │                                                   │                              │  │
│   │       evaluate()                                            train()                          │  │
│   │   80/20 split → resample                            full dataset → resample                  │  │
│   │   → feature select → CV                              → feature select → CV                   │  │
│   │   → eval on test set                                    (no evaluation)                      │  │
│   └───────────────────┬──────────────────────────────────────────────────────────────────────────┘  │
│                       │ produces (per seed)                                                         │
│                       ▼                                                                             │
│          ┌────────────────────────┐                                                                 │
│          │  trained_model_*.pkl   │   metrics/*.json   features/*.csv   registry.csv                │
│          └────────────┬───────────┘                                                                 │
│                       │                                                                             │
│          ┌────────────┴─────────────────────────────────────────────┐                               │
│          │ .merge_models()                                          │ .merge_classifier_models()    │
│          ▼                                                          ▼                               │
└──────────┼──────────────────────────────────────────────────────────────────────────────────────────┘  
           │                                                          │
           ▼                                                          ▼
┌─────────────────────────────┐                       ┌───────────────────────────────────────────────┐
│        SeedEnsemble         │                       │              ClassifierEnsemble               │
│  (all seeds, 1 classifier)  │◄───────────────────── │  (multiple SeedEnsembles, N classifiers)      │
│  inherits BaseEnsemble      │                       │  inherits BaseEnsemble                        │
│                             │  contains 1..N seeds  │                                               │
│  .predict_proba(X)          │                       │  .from_seed_ensembles(dict)                   │
│    → (n_samples, 2)         │                       │  .from_registry_dict(dict)                    │
│                             │                       │                                               │
│  .predict_with_uncertainty  │                       │  .predict_proba(X) → consensus (n_samples, 2) │
│    → (mean, std, conf, pred)│                       │                                               │
│                             │                       │  .predict_with_uncertainty(X)                 │
│  .save() / .load()          │                       │    → (mean, std, conf, pred, per_clf_dict)    │
└─────────────────────────────┘                       │                                               │
                                                      │  .classifiers  .__getitem__  .__len__         │
                                                      │  .save() / .load()                            │
                                                      └───────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        EVALUATION PHASE                                             │
│                                                                                                     │
│  ┌──────────────────────────┐    ┌────────────────────────────────┐    ┌────────────────────────┐   │
│  │      ModelEvaluator      │    │      MultiModelEvaluator       │    │  ClassifierComparator  │   │
│  │  (1 fitted model/seed)   │    │  (all seeds, 1 classifier)     │    │  (N classifiers)       │   │
│  │                          │    │                                │    │                        │   │
│  │  .get_metrics()          │    │  .get_aggregate_metrics()      │    │  .get_ranking_table()  │   │
│  │  .summary()              │    │  .save_aggregate_metrics()     │    │  .save_ranking_table() │   │
│  │  .plot_all()   (8 plots) │    │  .plot_all()    (10 plots)     │    │  .plot_all()           │   │
│  │  .from_files()           │    │  ───────────────────────────   │    │  ────────────────────  │   │
│  │  ─────────────────────── │    │  reads: metrics/*.json         │    │  wraps N instances of  │   │
│  │  inputs:                 │    │         registry.csv           │    │  MultiModelEvaluator   │   │
│  │    fitted model          │    │                                │    │  (one per classifier)  │   │
│  │    X_test, y_test        │    │  outputs:                      │    │                        │   │
│  │    selected_features     │    │    aggregate_metrics.csv       │    │  outputs:              │   │
│  │                          │    │    aggregate_*.png/.csv        │    │    ranking_table.csv   │   │
│  │  called internally by    │    │    seed_stability_*.png        │    │    comparison plots    │   │
│  │  ModelTrainer per seed   │    │    freq_band_contribution.png  │    │                        │   │
│  └──────────────────────────┘    └────────────────────────────────┘    └────────────────────────┘   │
│               ▲                                  ▲                                  ▲               │
│               │                                  │                                  │               │
│         called per seed                 reads per-seed metrics                reads per-clf         │
│         during training                     & registry CSV                  metrics/registries      │
└─────────────────────────────────────────────────────────────────────────────────────────────────────┘


  SCOPE SUMMARY
  ─────────────────────────────────────────────────────────────────────────────
  ModelEvaluator        → 1 model,  1 seed,   1 classifier    (micro)
  MultiModelEvaluator   → N models, N seeds,  1 classifier    (per-classifier)
  ClassifierComparator  → N models, N seeds,  N classifiers   (cross-classifier)
  SeedEnsemble          → N seeds,  1 classifier              (inference)
  ClassifierEnsemble    → N seeds,  N classifiers             (inference, consensus)
  ─────────────────────────────────────────────────────────────────────────────
```

### 6. Data Classes

- **`BaseDataContainer`** (`data_container.py`): Abstract base class declaring the `start_date_str`, `end_date_str`, and `data` interface shared by all CSV-backed data wrappers.
- **`TremorData`** (`tremor/tremor_data.py`): Inherits from `BaseDataContainer`; wraps tremor CSV, validates sampling rate.
- **`LabelData`** (`label/label_data.py`): Inherits from `BaseDataContainer`; parses all parameters from the label filename via the module-level `parse_label_filename()` helper.

Both use `@cached_property` for efficient attribute access.

### 7. Utility Modules (`src/eruption_forecast/utils/`)

| Module | Contents |
|--------|----------|
| `window.py` | `construct_windows()`, `calculate_window_metrics()` |
| `array.py` | `detect_maximum_outlier()`, `remove_outliers()` — Z-score based |
| `date_utils.py` | `to_datetime()`, `normalize_dates()`, `parse_label_filename()` |
| `ml.py` | `random_under_sampler()`, `get_significant_features()`, `load_labels_from_csv()` |
| `validation.py` | `validate_random_state()`, `validate_date_ranges()`, `validate_window_step()`, `validate_columns()`, `check_sampling_consistency()` |
| `pathutils.py` | `resolve_output_dir()` — resolves paths relative to `root_dir` |
| `dataframe.py` | DataFrame shape/column validation helpers |
| `formatting.py` | Text formatting utilities |

### 8. Data Source Adapters (`src/eruption_forecast/sources/`)

- **`SeismicDataSource`** (`sources/base.py`): Abstract base class declaring the `get(date)` interface and `_make_log_prefix(date)` helper shared by all adapters.
- **`SDS`** (`sds.py`): Reads SeisComP Data Structure files directly from a local archive. Inherits from `SeismicDataSource`.
- **`FDSN`** (`fdsn.py`): Downloads from any FDSN web service with transparent local SDS caching. Inherits from `SeismicDataSource`.
  - `download_dir` is created automatically if absent
  - Downloaded files are cached as SDS miniSEED so subsequent runs skip the network
