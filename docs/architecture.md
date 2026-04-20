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
│       train() — full dataset, no split      │
└─────────┬───────────────────────────────────┘
          │  trained_model_*.csv  +  *.pkl
          ▼
┌─────────────────────────────────────────────┐
│               ModelPredictor                │
│  ┌──────────────────────────────────────┐   │
│  │ predict_proba()                      │   │
│  │ (forecast mode — no labels needed)   │   │
│  │ → auto-evaluates when eruption_dates │   │
│  │   provided (temporal out-of-sample)  │   │
│  └──────────────────────────────────────┘   │
│  Single model or multi-model consensus      │
└─────────────────────────────────────────────┘
```

---

## Research Workflow (`main.py`)

`main.py` is the top-level research script. It runs three independent branches
— evaluate, predict, and scenarios — all operating on the same `ForecastModel`
instance.

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
│  build_label()  │  2025-01-01 → 2025-07-26                   │  build_label()  │  2025-01-01 → 2025-07-26
│                 │  window_step=6h, dtf=2                     │                 │  window_step=6h, dtf=2
│                 │  eruption_dates=[…]                        │                 │  eruption_dates=[…]
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
│                 │  classifiers: lite-rf, rf, gb, xgb         │                 │  (same classifiers)
│                 │  cv: stratified, seeds: 500                │                 │  cv: stratified, seeds: 500
│                 │  full dataset → models/*.pkl               │                 │  full dataset → models/*.pkl
└────────┬────────┘                                            └────────┬────────┘
         │                                                              │
         ▼                                                              ▼
┌─────────────────┐                                            ┌─────────────────┐
│ MultiModel      │  per-classifier aggregate plots            │  forecast()     │  ModelPredictor
│ Evaluator       │  ROC, PR, calibration, confusion,          │                 │  predict_proba
│ (loop per clf)  │  SHAP, seed stability, …                   │                 │  2025-07-27 → 2025-08-22
└────────┬────────┘                                            │                 │  → auto-evaluates on
         │                                                     │                 │    forecast period
         ▼  (when ≥ 2 classifiers)                            └─────────────────┘
┌─────────────────┐
│ Classifier      │  cross-classifier comparison
│ Comparator      │  metric bar, ROC overlay,
│ (≥ 2 clf)       │  ranking CSV
└─────────────────┘


  Runtime flags (top-level constants in main.py):
  ┌──────────────────────────┬─────────────────────────────────────────────┐
  │ DEBUG                    │ Read from .env; reduces seeds to 25,        │
  │                          │ classifiers to [lite-rf, rf]                │
  │ N_JOBS                   │ 8 (outer parallelism)                       │
  │ TRAINING_SEEDS           │ 500 (or 25 in DEBUG mode)                   │
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
- `ShannonEntropy`: Signal complexity metric (`shannon_entropy.py`)
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
- `include_eruption_date` (default `False`): controls whether the eruption date counts toward the `day_to_forecast` window. When `True`, the window spans exactly `day_to_forecast` days ending on the eruption date. When `False`, the window covers `day_to_forecast` days strictly before the eruption date and the eruption day itself is additionally marked positive (`day_to_forecast + 1` positive days total)
- Label filenames follow: `label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv`

**`DynamicLabelBuilder`** (extends `LabelBuilder`) generates one window per eruption:
- Each window spans `days_before_eruption` days ending on the eruption date
- Build proceeds in three phases:
  1. **Initiate** — create one all-zero label DataFrame per eruption window
  2. **Deduplicate** — concat all windows, drop duplicate datetimes from overlapping windows, sort by datetime
  3. **Label** — iterate eruption dates and mark the `day_to_forecast` positive region on the unified frame
- Overlapping windows are handled cleanly: duplicate datetimes are removed before labeling, so positive labels from multiple eruptions accumulate correctly in the shared frame
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
 │  0 0 0 0 0 0 0 0 0 0  1  1  1  1  1  1  1                │
 │                       ↑              ↑  ↑                │
 │                   dtf start    day before  eruption      │
 │                                eruption    (also = 1)    │
 │  → day_to_forecast days strictly before eruption         │
 │    + eruption day = day_to_forecast + 1 positive days    │

 include_eruption_date=True
 │  0 0 0 0 0 0 0 0 0 0  0  1  1  1  1  1  1                │
 │                          ↑              ↑                │
 │                      dtf start      eruption             │
 │                                    (counted in dtf)      │
 │  → exactly day_to_forecast days ending on eruption day   │

DynamicLabelBuilder — three-phase build, overlapping windows deduped
─────────────────────────────────────────────────────────────────────
 Phase 1: initiate (all zeros)
   Eruption A window           Eruption B window
   [0 0 0 0 0 0 0 0 0 0]      [0 0 0 0 0 0 0 0 0 0]

 Phase 2: concat + deduplicate datetimes
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]  ← unified, sorted, no duplicates

 Phase 3: mark positive labels per eruption
   Eruption A (2025-03-20, dtf=2):  mark Mar 18–20 → 1
   Eruption B (2025-03-23, dtf=2):  mark Mar 21–23 → 1
   [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1]
                              ↑     ↑
                          Erup A  Erup B

   dtf = days_to_forecast
   E   = eruption date (is_erupted = 1)
   1   = positive label within day_to_forecast window
   0   = negative label
```

**Key classes:**
- `LabelBuilder`: Creates labeled windows over a global date range (`label_builder.py`)
- `DynamicLabelBuilder`: Per-eruption windows with three-phase build and overlap deduplication (`dynamic_label_builder.py`)
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

**`ModelTrainer`** trains classifiers across multiple random seeds on the full dataset:
- Supports 10 classifiers: `rf`, `gb`, `xgb`, `svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting`
- CV strategies: `shuffle`, `stratified`, `shuffle-stratified`, `timeseries`
- Uses `RandomUnderSampler` to handle class imbalance
- Feature selection and resampled training data are cached per seed to `features/{cv-slug}/` (resampled data in `features/{cv-slug}/resampled/`) for deterministic two-phase parallel dispatch
- One training mode: `train()` — resample full dataset → feature selection → CV → save models; no internal 80/20 split
- Out-of-sample evaluation is handled by `ModelPredictor.evaluate()` on the forecast period

**Key classes:**
- `ModelTrainer`: Multi-seed full-dataset training (`model_trainer.py`)
  - `train()`: Resample full dataset → feature selection → GridSearchCV → save models
  - `fit(**kwargs)`: Thin wrapper around `train()` for method chaining
  - `n_jobs`: outer seed workers; `grid_search_n_jobs`: inner `GridSearchCV`/`FeatureSelector` workers
- `BaseModelTrainer`: Constructor, validation, directory management, registry utilities (`base_model_trainer.py`)
  - Registry CSV columns: `random_state` (index), `significant_features_csv`, `trained_model_filepath`
- `BaseEnsemble`: Shared persistence mixin (`base_ensemble.py`)
  - `save(path)`: Joblib-dumps the ensemble to `.pkl`, creating parent directories
  - `load(path)` (classmethod): Restores instance from `.pkl`; raises `FileNotFoundError` if absent
  - Inherited by both `SeedEnsemble` and `ClassifierEnsemble` — no duplicated boilerplate
- `ClassifierModel`: Manages classifier instances and hyperparameter grids (`classifier_model.py`)
- `ModelEvaluator`: Computes metrics and plots for a single fitted model (`model_evaluator.py`)
  - Methods: `get_metrics()`, `summary()`, `plot_all()`, `from_files()`, `plot_shap_summary()`, `plot_shap_waterfall()`
  - `cv_name` parameter (default `"cv"`): slugified into the default output path when `output_dir` is `None`
  - `plot_shap=True` required to enable SHAP plots in `plot_all()`
- `MultiModelEvaluator`: Aggregate evaluation across all seeds (`multi_model_evaluator.py`)
  - Two modes: (1) from per-seed JSON metrics files; (2) from registry CSV + `X_test`/`y_test` for temporal out-of-sample evaluation
  - Methods: `plot_all()`, `plot_roc()`, `plot_shap_summary()`, `get_aggregate_metrics()`, `save_aggregate_metrics()`
- `ModelPredictor`: Runs forecast inference and temporal evaluation (`model_predictor.py`)
  - `predict_proba()`: Forecast mode — no labels needed; auto-evaluates when `eruption_dates` provided
  - `build_forecast_labels()`: Build labeled windows for the forecast period from known eruption dates
  - `evaluate(X_forecast, y_forecast)`: Per-seed evaluation on forecast-period features and labels
- `PipelineConfig`: Serialisable pipeline configuration (`src/eruption_forecast/config/pipeline_config.py`)

### 5.1 Model Class Relationships

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         TRAINING PHASE                                              │
│                                                                                                     │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────────┐  │
│   │                              ModelTrainer  (one classifier)                                  │  │
│   │                                                                                              │  │
│   │                        .fit(**kwargs)  →  .train(**kwargs)                                   │  │
│   │                                                   │                                          │  │
│   │                                               train()                                        │  │
│   │                                        full dataset → resample                               │  │
│   │                                         → feature select → CV                               │  │
│   │                                            (no internal split)                               │  │
│   └───────────────────┬──────────────────────────────────────────────────────────────────────────┘  │
│                       │ produces (per seed)                                                         │
│                       ▼                                                                             │
│          ┌────────────────────────┐                                                                 │
│          │  trained_model_*.pkl   │   features/*.csv   registry.csv (3 cols: rs, sig_feat, model)  │
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
│                        (temporal out-of-sample — forecast period is the test set)                  │
│                                                                                                     │
│  ┌──────────────────────────┐    ┌────────────────────────────────┐    ┌────────────────────────┐   │
│  │      ModelEvaluator      │    │      MultiModelEvaluator       │    │  ClassifierComparator  │   │
│  │  (1 fitted model/seed)   │    │  (all seeds, 1 classifier)     │    │  (N classifiers)       │   │
│  │                          │    │                                │    │                        │   │
│  │  .get_metrics()          │    │  .get_aggregate_metrics()      │    │  .get_ranking_table()  │   │
│  │  .summary()              │    │  .save_aggregate_metrics()     │    │  .save_ranking_table() │   │
│  │  .plot_all()   (9 plots) │    │  .plot_all()    (11 plots)     │    │  .plot_all()           │   │
│  │  .from_files()           │    │  ───────────────────────────   │    │  ────────────────────  │   │
│  │  ─────────────────────── │    │  inputs:                       │    │  wraps N instances of  │   │
│  │  inputs:                 │    │    registry.csv                │    │  MultiModelEvaluator   │   │
│  │    fitted model          │    │    X_test (forecast features)  │    │  (one per classifier)  │   │
│  │    X_test, y_test        │    │    y_test (forecast labels)    │    │                        │   │
│  │    selected_features     │    │    — OR — metrics/*.json       │    │  outputs:              │   │
│  │                          │    │                                │    │    ranking_table.csv   │   │
│  │  called by               │    │  outputs:                      │    │    comparison plots    │   │
│  │  ModelPredictor.evaluate │    │    aggregate_metrics.csv       │    │                        │   │
│  └──────────────────────────┘    │    aggregate_*.png/.csv        │    └────────────────────────┘   │
│               ▲                  │    seed_stability_*.png        │                  ▲               │
│               │                  │    freq_band_contribution.png  │                  │               │
│   called per seed by             └────────────────────────────────┘          reads per-clf          │
│   ModelPredictor.evaluate()                   ▲                              metrics/registries     │
│   on forecast-period data          called by ModelPredictor.evaluate()                              │
│   (no training split needed)       for temporal out-of-sample metrics                               │
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
| `array.py` | `detect_maximum_outlier()`, `remove_outliers()`, `detect_anomalies_zscore()`, `predict_proba_from_estimator()`, `aggregate_seed_probabilities()` |
| `date_utils.py` | `to_datetime()`, `normalize_dates()`, `sort_dates()`, `parse_label_filename()`, `set_datetime_index()` |
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
