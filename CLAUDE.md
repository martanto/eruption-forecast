# CLAUDE.md

_Last updated: 2026-06-01_

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Package Manager
This package is using UV (https://docs.astral.sh/uv/) as package manager.

## Claude Code Guidelines

- Always use available skills whenever possible when executing commands (e.g., use the scikit-learn skill for ML tasks, matplotlib/seaborn skills for plotting, etc.)

## Rules

1. **Log every completed task in the daily changelog.** Any finished task — bug fix, refactor, new feature, test, documentation change — must have its outcome recorded in `changelogs/YYYY-mm-dd.md` (using today's date) before moving on. Create the file if it does not exist. Append new entries; never overwrite previous entries in the same file. The `changelogs/` directory is git-ignored (local only).
2. **Type checker is `ty`.** Use `uvx ty check src/` for type checking. Always use forward slashes: `uvx ty check src/` (not `.\src`).
3. **Lint with ruff.** Use `uv run ruff check --fix src/` for linting.
4. **All `uv`, `uvx`, and `python` commands are permitted.** `uv sync`, `uv run`, `uv pip install/uninstall`, `uv lock`, `uvx ty check`, `python main.py`, etc. — no need to ask. User has granted permission to run these commands without approval.
5. **ALWAYS create a new branch before any commits or modifications.** Use `git checkout -b <prefix>/<branch-name>` to create a new branch before making ANY commits or code modifications. Choose the prefix based on the type of work: `fix/` for bug fixes (e.g., `fix/docstring-errors`), `ft/` for new features (e.g., `ft/add-fdsn-source`), and `dev/` as the default for everything else (e.g., `dev/refactor-utils`). Never work directly on main or master branches.
6. **Always use `tests` directory when running testing.** Create the tests output inside `tests` directory.
7. **Do not commit temporary test files.** Files starting with `test` in the root directory (e.g., `test_*.py`, `test.py`) should not be committed. These are temporary test scripts and are excluded via `.gitignore`.
8. **All module imports must be at the top of the file.** Never place `import` statements inside functions, methods, or conditional blocks. All stdlib, third-party, and local imports belong at the module level, grouped and sorted by ruff.
9. **No signatures.** Never add author attributions, `Co-Authored-By`, `🤖 Generated with [Claude Code]`, or any name to any file.
10. **Run circular import test after every module change.** After adding, removing, or reorganising any module or import, run `uv run pytest tests/test_imports.py -v` to confirm no circular imports were introduced.
11. **Keep `config.example.yaml` in sync.** Whenever `src/eruption_forecast/model/forecast_model.py` or `src/eruption_forecast/config/pipeline_config.py` is modified — new parameter added, removed, renamed, or default changed — update `config.example.yaml` to reflect the change. The YAML keys, defaults, and inline comments must match the current dataclass fields and `forecast_model.py` constructor calls exactly.
12. **Documentation updates are comprehensive.** When asked to add/update docs, update ALL of: `wiki/*.md`, `README.md`. Also update `CLAUDE.md` for architecture/design changes. The `docs/` directory has been removed — `wiki/` is the single source of truth.
13. **Always `git checkout <branch>` before creating or working on a branch.** Never create a branch from the wrong base — verify current branch first with `git status` or `git branch`.
14. **Ask when unsure.** If the intent, scope, or correct approach is unclear, ask before proceeding.
15. **Keep `forecast_config.py` in sync with the new `ForecastModel`.** Whenever `src/eruption_forecast/model/forecast.py` or `src/eruption_forecast/config/forecast_config.py` is modified — new parameter added, removed, renamed, default changed — run `uvx ty check tests/test_forecast_config.py` to confirm the test file (which asserts every dataclass field and default) still type-checks against the current API. If `ty` flags drift, update both `tests/test_forecast_config.py` and `config.example.yaml` so the example YAML, the dataclass, and the constructor surface all match.

## Testing

Testing is not mandatory, but if needed, you can use the following test data:

- **Input directory (SDS format):** `D:\Data\OJN`
- **Start date:** `2025-03-16`
- **End date:** `2025-03-22`
- **Eruption date:** `2025-03-20`

See `main.py` for a complete usage example of the pipeline.

## Project Overview

`eruption-forecast` is a Python package for volcanic eruption forecasting using seismic data analysis. The package processes seismic tremor data, extracts features, builds labels, and creates forecast models to predict volcanic eruptions based on time-series seismic measurements.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Install with dev dependencies
uv sync --group dev
```

### Code Quality
```bash
# Lint with ruff (--fix auto-corrects fixable issues)
uv run ruff check --fix src/

# Type checking with ty
uvx ty check src/
```

### Running Tests
```bash
uv run pytest tests/
```

### Running the Application
```bash
# Run main script
python main.py

# Run with uv
uv run python main.py
```


## Architecture Overview

### Pipeline

`ForecastModel` orchestrates the full pipeline with method chaining:

```
CalculateTremor → LabelBuilder → TremorMatrixBuilder → FeaturesBuilder → ModelTrainer → ModelPredictor
```

### Package Layout

```
src/eruption_forecast/
├── data_container.py      # BaseDataContainer — shared ABC for TremorData & LabelData
├── logger.py              # Loguru-based logging with enable/disable/set_level/set_directory
├── tremor/                # Seismic tremor processing
│   ├── calculate_tremor.py    # CalculateTremor — main orchestrator
│   ├── rsam.py                # Real Seismic Amplitude Measurement
│   ├── dsar.py                # Displacement Seismic Amplitude Ratio
│   ├── shannon_entropy.py     # Shannon Entropy metric
│   └── tremor_data.py         # TremorData — wraps tremor CSV
├── label/                 # Training label generation
│   ├── label_builder.py           # LabelBuilder — global sliding window labelling
│   ├── dynamic_label_builder.py   # DynamicLabelBuilder — per-eruption windows
│   └── label_data.py              # LabelData — wraps label CSV
├── features/              # Feature extraction & selection
│   ├── features_builder.py    # FeaturesBuilder — tsfresh extraction
│   ├── feature_selector.py    # FeatureSelector — 3-method selection
│   └── tremor_matrix_builder.py  # TremorMatrixBuilder — windowed alignment
├── model/                 # ML model training & prediction (see "Refactor status" below)
│   │
│   │ ── NEW METHOD (current; to become canonical) ──
│   ├── forecast.py            # ForecastModel — refactored orchestrator (will replace forecast_model.py)
│   ├── base_model.py          # BaseModel — ABC for Training/Prediction/EvaluationModel; provides save()/load()
│   ├── cache_model.py         # CacheModel — ABC for content-addressable param-based model caches
│   ├── training_model.py      # TrainingModel — label, features, multi-seed fit; inherits BaseModel + CacheModel
│   ├── prediction_model.py    # PredictionModel — forecast inference; inherits BaseModel + CacheModel
│   ├── evaluation_model.py    # EvaluationModel — 3 modes: training/prediction reuse + standalone
│   │
│   │ ── OLD METHOD (legacy; pending removal — do NOT modify unless asked) ──
│   ├── forecast_model.py      # ForecastModel (legacy) — replaced by forecast.py when ready
│   ├── base_model_trainer.py  # BaseModelTrainer — constructor, validation, utilities
│   ├── evaluation_trainer.py  # EvaluationTrainer — adds evaluate(), learning curves, SHAP
│   ├── model_predictor.py     # ModelPredictor — inference & forecasting
│   ├── model_evaluator.py     # ModelEvaluator — single-seed evaluation
│   ├── model_trainer.py       # ModelTrainer — paired with the new-method classes
│   │
│   │ ── SHARED (used by both methods) ──
│   ├── multi_model_evaluator.py  # MultiModelEvaluator — aggregate evaluation
│   ├── classifier_comparator.py  # ClassifierComparator — cross-classifier comparison
│   ├── metrics_computer.py    # MetricsComputer — metrics logic
│   └── classifier_model.py    # ClassifierModel — classifier + grid management
├── ensemble/              # Serialisable ensemble classes (moved from model/)
│   ├── base_ensemble.py       # BaseEnsemble — shared joblib save/load mixin
│   ├── seed_ensemble.py       # SeedEnsemble — all seeds, 1 classifier
│   └── classifier_ensemble.py # ClassifierEnsemble — N SeedEnsembles; from_any() / from_json() / from_dict() / from_seed_ensembles()
├── sources/               # Seismic data source adapters
│   ├── base.py                # SeismicDataSource — abstract base class
│   ├── sds.py                 # SDS reader (SeisComP Data Structure)
│   └── fdsn.py                # FDSN web service client with local caching
├── config/                # Pipeline configuration
│   └── pipeline_config.py     # PipelineConfig + sub-config dataclasses
├── dataclass/             # Data containers
│   └── station_data.py        # StationData — immutable station identity container
├── plots/                 # Visualization utilities
│   ├── styles.py              # Nature style configuration
│   ├── tremor_plots.py
│   ├── feature_plots.py
│   ├── evaluation_plots.py
│   ├── shap_plots.py          # XGBoost ≥3.x compatible via Independent masker
│   └── forecast_plots.py
├── report/                # Report generation
│   ├── pipeline_report.py
│   ├── training_report.py
│   ├── comparator_report.py
│   ├── features_report.py
│   ├── label_report.py
│   ├── prediction_report.py
│   └── tremor_report.py
├── decorators/            # Function decorators
│   ├── decorator_class.py
│   └── notify.py              # Telegram notification decorator + direct send function
└── utils/                 # Focused utility modules
    ├── array.py               # Z-score outlier detection, anomaly detection
    ├── window.py              # Sliding window construction
    ├── date_utils.py          # Date conversion, filename parsing, label ID conversion
    ├── dataframe.py           # DataFrame helpers
    ├── ml.py                  # Resampling, feature utilities, ensemble merging
    ├── validation.py          # Centralised validation (dates, random state, columns, sampling)
    ├── pathutils.py           # Path resolution, ensure_dir, save_figure, save_data, load_json
    └── formatting.py          # Text formatting
```

### Refactor Status: New Method vs Old Method

The `model/` package is mid-migration. Two parallel stacks currently coexist:

**New method** (canonical going forward — prefer for new work):
- `base_model.py` — shared ABC providing `save()` / `load()` and tremor loading
- `cache_model.py` — content-addressable parameter cache mixin (`build_cache_identity`, `compute_hash`, `save_to_cache`, `load_from_cache`)
- `training_model.py` — `TrainingModel` (label → features → multi-seed fit)
- `prediction_model.py` — `PredictionModel` (forecast over unlabelled window grid)
- `evaluation_model.py` — `EvaluationModel` (training/prediction reuse + standalone)
- `forecast.py` — `ForecastModel` orchestrator (`calculate` → `train` → `predict` → `evaluate`); will replace `forecast_model.py` when ready

**Old method** (legacy — being replaced; do NOT modify unless explicitly asked):
- `forecast_model.py` — superseded by `forecast.py`
- `base_model_trainer.py`, `evaluation_trainer.py`, `model_evaluator.py`, `model_predictor.py`, `model_trainer.py`

When user asks for changes to the model pipeline, default to editing the new-method files. Only touch the old-method files when the user names them explicitly or asks for a legacy bug-fix.

### Ensemble Package (`src/eruption_forecast/ensemble/`)

Ensemble classes moved out of `model/` into a dedicated `ensemble/` package:
- `BaseEnsemble` (`base_ensemble.py`) — joblib `save(path)` / `load(path)` mixin
- `SeedEnsemble` (`seed_ensemble.py`) — bundles all seeds for one classifier
- `ClassifierEnsemble` (`classifier_ensemble.py`) — wraps multiple `SeedEnsemble`s; factories: `from_any()`, `from_json()`, `from_dict()`, `from_seed_ensembles()`

`ensemble/__init__.py` re-exports only `SeedEnsemble` and `ClassifierEnsemble`. Import `BaseEnsemble` directly from `eruption_forecast.ensemble.base_ensemble`. Always import from `eruption_forecast.ensemble.*`, never from `eruption_forecast.model.*` (the old paths no longer exist).

### Key Components

#### 1. Tremor Calculation (`src/eruption_forecast/tremor/`)

**`CalculateTremor`** processes raw seismic data into tremor metrics:
- Reads seismic data from SDS or FDSN web services
- Calculates three metrics across multiple frequency bands in parallel:
  - **RSAM** (Real Seismic Amplitude Measurement): Mean amplitude per band
  - **DSAR** (Displacement Seismic Amplitude Ratio): Ratio between consecutive bands
  - **Shannon Entropy**: Single broadband signal complexity column (`entropy`)
- Default frequency bands: `(0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz`
- Supports multiprocessing via `n_jobs`; outputs 10-minute interval CSVs
- Optional params: `remove_outlier_method`, `remove_tremor_anomalies`, `interpolate`, `value_multiplier`, `cleanup_daily_dir`, `plot_daily`, `save_plot`, `filename_prefix`, `methods`

**Key classes:**
- `CalculateTremor`: Main orchestrator (`calculate_tremor.py`)
- `RSAM`: Mean amplitude metrics (`rsam.py`)
- `DSAR`: Amplitude ratios between bands (`dsar.py`)
- `ShannonEntropy`: Signal complexity metric — `.filter(freqmin, freqmax).calculate()` (`shannon_entropy.py`)
- `TremorData`: Loads and validates tremor CSV files (`tremor_data.py`)
- `SDS`: Reads SeisComP Data Structure files (`sources/sds.py`)
- `FDSN`: Downloads seismic data from FDSN web services with local SDS caching (`sources/fdsn.py`)

**Workflow:**
```python
# From SDS archive
calculate = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-03",
    n_jobs=4
).from_sds(sds_dir="/path/to/sds").run()
# Output CSV columns: rsam_f0..f4, dsar_f0-f1..f3-f4, entropy

# From FDSN web service
calculate = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-03",
).from_fdsn(client_url="https://service.iris.edu").run()
```

#### 2. Label Building (`src/eruption_forecast/label/`)

**`LabelBuilder`** generates binary labels for supervised learning:
- Creates sliding time windows and labels them erupted (1) or not (0)
- Uses `day_to_forecast` to look ahead N days before eruptions
- `include_eruption_date` (default `False`): when `False`, eruption day itself is additionally marked positive (`day_to_forecast + 1` total positive days)
- Label filenames follow: `label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv`

**`DynamicLabelBuilder`** (extends `LabelBuilder`): Three-phase build — (1) initiate all-zero frames per eruption, (2) concat + deduplicate datetimes from overlapping windows, (3) mark positive labels per eruption date on unified frame.

**Key classes:**
- `LabelBuilder`: Creates labeled windows (`label_builder.py`)
- `DynamicLabelBuilder`: Per-eruption windows with overlap deduplication (`dynamic_label_builder.py`)
- `LabelData`: Loads label CSV and parses parameters from filename (`label_data.py`)

**Window configuration:**
- `window_size`: Window size in days
- `window_step` + `window_step_unit`: Step between windows (minutes or hours)
- `day_to_forecast`: Days before eruption to start labeling positive

#### 3. Tremor Matrix Building (`src/eruption_forecast/features/`)

**`TremorMatrixBuilder`** slices tremor time-series into windows aligned with labels:
- Takes tremor DataFrame and label DataFrame as input
- Validates sample counts per window; start/end dates derived from label range
- Concatenates all windows into a unified matrix with `id`, `datetime`, and tremor columns
- `save_tremor_matrix_per_method=True`: saves per-column matrices to `per_method/` subdirectory

#### 4. Feature Extraction (`src/eruption_forecast/features/`)

**`FeaturesBuilder`** extracts tsfresh features from the tremor matrix:
- **Training mode** (labels provided): Filters windows, saves aligned label CSV
- **Prediction mode** (no labels): Extracts all features, disables relevance filtering
- Runs tsfresh extraction per tremor column independently

**Key classes:**
- `FeaturesBuilder`: Orchestrates tsfresh feature extraction (`features_builder.py`)
- `FeatureSelector`: Two-stage selection — tsfresh (statistical FDR) → RandomForest (importance) (`feature_selector.py`)
  - Methods: `"tsfresh"`, `"random_forest"`, `"combined"`

#### 5. Model Training (`src/eruption_forecast/model/`)

**Three-class hierarchy:** `BaseModelTrainer` → `EvaluationTrainer` → `ModelTrainer`
- `BaseModelTrainer`: Constructor, validation, directory management, feature selection, grid-search setup, merge utilities
- `EvaluationTrainer`: Adds `evaluate()`, learning curves, SHAP, metric aggregation
- `ModelTrainer`: Adds `train()` and `fit(with_evaluation=True)` dispatcher

**`ModelTrainer`** trains classifiers across multiple random seeds:
- Supports 11 classifiers: `rf`, `gb`, `xgb`, `svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting`, `lite-rf`
- CV strategies: `shuffle`, `stratified`, `shuffle-stratified` (default), `timeseries`
- Uses `RandomUnderSampler` to handle class imbalance
- `classifier` accepts `str` or `list[str]` for multi-classifier training
- **Two-phase parallel dispatch**: Phase 1 = feature selection per seed (parallel); Phase 2 = (seed × classifier) GridSearchCV pairs (parallel)
- Two training modes:
  - `evaluate()`: 80/20 stratified split → resample train → feature selection → CV → evaluate on test set → save
  - `train()`: Resample full dataset → feature selection → CV → save (no metrics)
- `fit(with_evaluation=True)`: Dispatches to `evaluate()` or `train()` based on flag
- `n_jobs`: outer seed workers; `grid_search_n_jobs`: inner `GridSearchCV`/`FeatureSelector` workers. Enforced: `n_jobs × grid_search_n_jobs ≤ cpu_count`. Uses `joblib.Parallel(backend="loky")` for nested-parallelism safety.
- `use_gpu=True`: GPU acceleration for XGBoost; forces `n_jobs=1`

**Key classes:**
- `ModelTrainer`: Multi-seed training (`model_trainer.py`)
- `ClassifierModel`: Manages classifier instances and hyperparameter grids (`classifier_model.py`)
- `MetricsComputer`: `compute_all_metrics()`, `optimize_threshold()` — includes `g_mean` (G-mean at 0.5 threshold) (`metrics_computer.py`)
- `ModelEvaluator`: Computes metrics and plots for a fitted model (`model_evaluator.py`)
  - Methods: `get_metrics()`, `summary()`, `plot_all()`, `from_files()`, `plot_shap_summary()`, `plot_shap_waterfall()`, `save_metrics()`, `optimize_threshold()`
  - `cv_name: str = "cv"` param: auto-builds `output/trainings/evaluations/classifiers/{clf-slug}/{cv-slug}/` when `output_dir=None`
  - `plot_shap=True` required to enable SHAP plots in `plot_all()`
- `MultiModelEvaluator`: Aggregates per-seed results; 11 aggregate plots + stats (`multi_model_evaluator.py`)
- `ClassifierComparator`: Wraps N `MultiModelEvaluator` instances; comparison tables + plots (`classifier_comparator.py`)
  - `from_json(json_path, ...)`: Load classifiers from JSON file
  - Methods: `get_metrics_table()`, `get_ranking()`, `plot_metric_bar()`, `plot_seed_stability()`, `plot_comparison_grid()`, `plot_roc()`, `plot_all()`
- `ModelPredictor`: Runs inference (`model_predictor.py`)
  - Constructor: `trained_models: str | dict[str, str]`, accepts CSV registry, `SeedEnsemble.pkl`, `ClassifierEnsemble.pkl`, or dict thereof
  - `predict()`: Evaluation mode — returns metrics DataFrame per (classifier, seed)
  - `predict_best(criterion="balanced_accuracy")`: Returns `ModelEvaluator` for best seed
  - `predict_proba()`: Forecast mode — unlabelled forecasting with per-classifier + consensus output
  - Output columns: `{name}_eruption_probability/uncertainty/confidence/prediction` + `consensus_*`

**Persistence (`BaseModel`):**
- `BaseModel.save(path=None) -> str`: Dumps instance to `{output_dir}/{ClassName}_{basename}.pkl` by default; returns the written path.
- `BaseModel.load(path) -> Self` (classmethod): Restores a previously saved instance via joblib. `FileNotFoundError` if path is missing.
- Inherited by `TrainingModel`, `PredictionModel`, `EvaluationModel`. Serialises everything including `features_df`, `ClassifierEnsemble`, `LabelBuilder`.

**`EvaluationModel` (`evaluation_model.py`):**
- Three modes: **training reuse** (pass a `TrainingModel` — truth labels are already embedded), **prediction reuse** (pass a `PredictionModel` — `eruption_dates` required, truth rebuilt on prediction grid), **standalone** (bare ensemble or file path + tremor data).
- Detection: both sources expose `ClassifierEnsemble` + `features_df`; kind resolved by `labels` type (`Series` = training, `DataFrame` = prediction).
- Output namespaced to `evaluation/{training|prediction|standalone}/` to prevent metric-file collisions.
- `eruption_dates` is optional for training reuse; required for prediction reuse and standalone.
- Saved `.pkl` files accepted wherever a live instance is expected.

**Ensemble classes:**
- `BaseEnsemble`: Mixin providing `save(path)` / `load(path)` via joblib; inherited by SeedEnsemble + ClassifierEnsemble
- `SeedEnsemble`: Bundles all seed models + feature lists for one classifier; sklearn BaseEstimator+ClassifierMixin
  - `from_registry(registry_csv)`, `predict_proba(X)`, `predict(X)`, `predict_with_uncertainty(X, threshold)`, `save()`, `load()`
- `ClassifierEnsemble`: Wraps multiple `SeedEnsemble` objects; `from_seed_ensembles()`, `from_registry_dict()`, `predict_proba()`, `predict_with_uncertainty()`, `classifiers`, `__getitem__`, `__len__`, `save()`, `load()`
- `merge_seed_models(registry_csv)` in `utils/ml.py` → saves `SeedEnsemble_{suffix}.pkl`
- `merge_all_classifiers(trained_models)` → produces `ClassifierEnsemble`
- `ForecastModel.train()` auto-saves `ClassifierEnsemble.pkl` to `{output_dir}` after merging all classifier seed models

**`ForecastModel` constructor parameters:** `station`, `channel`, `network`, `window_size`, `volcano_id` (required); `location=None`, `output_dir=None`, `root_dir=None`, `overwrite=False`, `n_jobs=1`, `verbose=False`
- Note: `start_date` and `end_date` go in `calculate()`, not the constructor

**`ForecastModel` additional methods** (beyond pipeline stages):
- `load_tremor_data(tremor_csv)`: Load pre-calculated tremor data instead of calling `calculate()`
- `set_feature_selection_method(using)`: Change feature selection method before `train()`
- `save_config(path=None, fmt="yaml")` / `from_config(path)`: Persist and replay pipeline configuration
- `save_model(path=None)` / `load_model(path)`: Serialise/restore full `ForecastModel` via joblib (legacy `forecast_model.py`)
- `run()`: Replay all pipeline stages from a loaded config
- `generate_report()`: Generate pipeline report

**`forecast.py` `ForecastModel.evaluate()` — extended signature:**
- `model` (instance or `str` path): `TrainingModel` → training-window eval; `PredictionModel` or path → forecast-window eval; `None` → falls back to `self.PredictionModel` or `self.TrainingModel`.
- `eruption_dates`, `label_builder`, `days_before_eruption`: override the values captured from `train()`. Not needed for `TrainingModel` reuse.
- Call `fm.TrainingModel.save(path)` / `fm.PredictionModel.save(path)` explicitly; no auto-save in `train()`/`predict()`.

**`PipelineConfig`** sub-configs: `ModelConfig`, `CalculateConfig`, `BuildLabelConfig`, `ExtractFeaturesConfig`, `TrainConfig`, `ForecastConfig`

#### 6. Data Classes

- **`BaseDataContainer`** (`data_container.py`): Abstract base for TremorData/LabelData; provides `filename`, `basename`, `filetype` cached properties
- **`TremorData`**: Wraps tremor CSV; provides start/end dates, validates sampling rate; uses `@cached_property`
- **`LabelData`**: Wraps label CSV; parses parameters from filename; uses `@cached_property`
- **`StationData`** (`dataclass/station_data.py`): Immutable container for station identity codes — fields: `station`, `channel`, `network`, `location` (str, default `""`), `channel_type` (default `"D"`); derived: `nslc` (N.S.L.C) and `nslct` (N.S.L.C.Type); all uppercased in `__post_init__`

#### 7. Reporting (`src/eruption_forecast/report/`)

- `LabelReport`, `TremorReport`, `FeaturesReport`, `TrainingReport`, `ComparatorReport`, `PredictionReport`, `PipelineReport`
- `generate_report()` function exported from package root

#### 8. Utility Functions (`src/eruption_forecast/utils/`)

| Module | Key Functions |
|--------|---------------|
| `array.py` | `detect_maximum_outlier()`, `remove_outliers()`, `detect_anomalies_zscore()`, `aggregate_seed_probabilities()`, `predict_proba_from_estimator()` |
| `window.py` | `construct_windows()`, `calculate_window_metrics()` |
| `date_utils.py` | `to_datetime()`, `normalize_dates()`, `sort_dates()`, `parse_label_filename()`, `set_datetime_index()`, `label_id_to_datetime()` |
| `ml.py` | `random_under_sampler()`, `get_significant_features()`, `load_labels_from_csv()`, `merge_seed_models()`, `merge_all_classifiers()`, `compute_seed_eruption_probability()`, `compute_model_probabilities()`, `get_classifier_models()`, `compute_g_mean()` |
| `validation.py` | `validate_random_state()`, `validate_date_ranges()`, `validate_window_step()`, `validate_columns()`, `check_sampling_consistency()` |
| `pathutils.py` | `resolve_output_dir()`, `ensure_dir()`, `save_figure()`, `save_data()`, `load_json()` |
| `dataframe.py` | `load_label_csv()`, DataFrame shape/column validation helpers |
| `formatting.py` | Human-readable text formatting (elapsed time, file sizes) |

#### 9. Decorators (`src/eruption_forecast/decorators/`)

- `notify`: Decorator for Telegram success/error notifications — credentials from `.env` (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) or passed explicitly
- `send_telegram_notification(message, files, file_caption)`: Direct one-off notification with optional file attachments

### Data Classes

Both `TremorData` and `LabelData` use `@cached_property` for efficient attribute access. Both inherit `BaseDataContainer`.

## Important Patterns

### Method Chaining
Most builder classes support fluent interfaces:
```python
CalculateTremor(...)
    .from_sds(sds_dir)
    .change_freq_bands([(0.1, 1.0), (1.0, 5.0)])
    .run()
```

### Multiprocessing
`CalculateTremor` supports parallel processing via `n_jobs`. Each day's calculation runs independently.

### Data Leakage Prevention
Train/test split MUST happen before resampling and feature selection. The split is always performed inside the trainer, never externally.

### Data Format Standards
- **Tremor CSV**: DateTime index, columns like `rsam_f0..f4`, `dsar_f0-f1..f3-f4`, `entropy`
- **Label CSV**: DateTime index, columns `id` (int) and `is_erupted` (0 or 1)
- **Date format**: Always `YYYY-MM-DD`
- **Sampling interval**: Default 10 minutes for tremor data

### Frequency Band Naming
| Alias | Band (Hz) |
|-------|-----------|
| `f0`  | 0.01 – 0.1 |
| `f1`  | 0.1 – 2    |
| `f2`  | 2 – 5      |
| `f3`  | 4.5 – 8    |
| `f4`  | 8 – 16     |

Columns: `rsam_f0`–`rsam_f4`, `dsar_f0-f1`–`dsar_f3-f4`, `entropy`

### Output Directory Structure
```
output/
└── {network}.{station}.{location}.{channel}/     e.g., VG.OJN.00.EHZ/
    ├── tremor/
    │   ├── daily/                    # Per-day CSVs (removed if cleanup_daily_dir=True)
    │   ├── figures/                  # Daily tremor plots (if plot_daily=True)
    │   ├── tremor_*.csv              # Final merged tremor data
    │   └── matrix/                   # Tremor matrix outputs
    │       ├── tremor_matrix_unified_*.csv
    │       └── per_method/           # Per-column matrices (optional)
    ├── features/
    │   ├── extracted/train/ and forecast/
    │   ├── all_features_*.csv
    │   └── label-features_*.csv
    ├── trainings/
    │   ├── evaluations/              # Output of train(with_evaluation=True)
    │   │   ├── evaluations_trained_models.json
    │   │   ├── features/{cv-slug}/significant_features/ tests/ ...
    │   │   └── classifiers/{clf-slug}/{cv-slug}/
    │   │       ├── models/00000.pkl  ...
    │   │       ├── metrics/00000.json  ...
    │   │       ├── figures/aggregate_*.png  ...
    │   │       ├── trained_model_{suffix}.csv
    │   │       └── SeedEnsemble_{suffix}.pkl
    │   └── predictions/              # Output of train(with_evaluation=False)
    │       ├── predictions_trained_models.json
    │       ├── ClassifierEnsemble.pkl   # Auto-saved by ForecastModel.train()
    │       ├── features/{cv-slug}/
    │       └── classifiers/{clf-slug}/{cv-slug}/
    │           ├── models/00000.pkl  ...
    │           ├── trained_model_{suffix}.csv
    │           └── SeedEnsemble_{suffix}.pkl
    ├── forecast/
    │   ├── predictions.csv
    │   └── figures/eruption_forecast.png
    ├── config_forecast.yaml
    └── forecast_model.pkl
```

**File name suffix pattern:** `{ClassifierName}-{CVName}_rs-{random_state}_ts-{total_seed}_top-{n}`
e.g., `XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20`

**Classifier folder slugs:** `rf` → `random-forest-classifier`, `xgb` → `xgb-classifier`, `gb` → `gradient-boosting-classifier`, etc.
**CV folder slugs:** `shuffle` → `shuffle-split`, `stratified` → `stratified-k-fold`, `shuffle-stratified` → `stratified-shuffle-split`, `timeseries` → `time-series-split`

## Docstring Guidelines

- Style: Google style
- **Always use the `google-python-docstrings` skill when asked to add or update docstrings.**
- **Preserve double backticks** around variable names, parameter names, values (``None``, ``True``, ``False``), method calls, and attribute references in docstring prose. Never strip them during rewrites.

## Comment Guidelines

- Inline/block comments start at least 2 spaces from code, `#` followed by at least one space
- Never describe the code — explain *why*, not *what*
- Use proper capitalization, punctuation, and grammar; comments should read like narrative text

## SHAP Plotting

Always pass `plot_size=None` to `shap.plots.beeswarm` — prevents SHAP from overriding pre-created `figsize`.

## Dependencies

### Key Scientific Libraries
- **obspy**: Seismic data processing (Stream, Trace objects)
- **pandas**: Time-series data manipulation (requires pandas >= 3.0.0)
- **tsfresh**: Automated time-series feature extraction
- **numpy**: Numerical computations
- **numba**: Performance optimization via JIT compilation
- **xgboost**: Gradient boosting with GPU support (≥3.x)
- **shap**: SHAP feature importance (≥0.46); uses Independent masker for XGBoost ≥3.x

### Data Sources

Both data-source adapters live in `src/eruption_forecast/sources/`:

- **`SeismicDataSource`** (`sources/base.py`): Abstract base class with `get(date)` interface
- **`SDS`** (`sds.py`): Reads SeisComP Data Structure files from local archive
- **`FDSN`** (`fdsn.py`): Downloads from any FDSN web service with transparent local SDS caching


## Common Workflows

### Complete Pipeline Example

```python
from eruption_forecast import ForecastModel

fm = ForecastModel(
    root_dir="/path/to/project",
    station="OJN",
    channel="EHZ",
    network="VG",
    location="00",
    window_size=2,
    volcano_id="VOLCANO_001",
    n_jobs=4,
)

fm.calculate(
    source="sds",
    sds_dir="/path/to/sds",
    methods=["rsam", "dsar", "entropy"],
).build_label(
    start_date="2020-01-01",
    end_date="2020-12-31",
    day_to_forecast=2,
    window_step=6,
    window_step_unit="hours",
    eruption_dates=["2020-06-15"],
).extract_features(
    select_tremor_columns=["rsam_f2", "rsam_f3", "dsar_f3-f4", "entropy"],
).train(
    classifier=["rf", "xgb"],          # list for multi-classifier
    cv_strategy="shuffle-stratified",  # default
    random_state=0,
    total_seed=500,
    with_evaluation=False,             # train on full dataset; True for 80/20 + metrics
    number_of_significant_features=20,
).forecast(
    start_date="2020-07-01",
    end_date="2020-07-31",
    window_step=10,
    window_step_unit="minutes",
)
```

See `main.py` for a full working example.

### Working with Existing Data
```python
# Load pre-calculated tremor data
tremor_data = TremorData(tremor_csv="path/to/tremor.csv")
print(tremor_data.start_date_str, tremor_data.end_date_str)

# Load existing labels
label_data = LabelData(label_csv="path/to/label.csv")
print(label_data.parameters)  # Extracts all params from filename
```

### Merging Seed Models
```python
# After train() completes:
merged_path = trainer.merge_models()
# → SeedEnsemble_{suffix}.pkl alongside the registry CSV

# Multi-classifier bundle
bundle_path = trainer.merge_classifier_models(
    {"rf": rf_trainer.csv, "xgb": xgb_trainer.csv}
)

# ForecastModel.train() with multiple classifiers auto-saves ClassifierEnsemble.pkl
```

### Saving and Replaying Config
```python
fm.save_config()                  # YAML → {station_dir}/config.yaml
fm.save_model()                   # joblib → forecast_model.pkl

# Replay
fm2 = ForecastModel.from_config("output/VG.OJN.00.EHZ/config.yaml")
fm2.run()
```

## GitHub Wiki

- Local wiki pages: `D:\Projects\eruption-forecast\wiki\`
- 13 pages: Home, Installation, Quick-Start, Architecture, Data-Sources, Pipeline-Walkthrough, Training-Workflows, Classifiers-and-CV, Evaluation-and-Forecasting, Visualization, Configuration, Output-Structure, API-Reference
- Wiki remote: `https://github.com/martanto/eruption-forecast.wiki.git`

## Notes on Windows Development

This project is developed on Windows. When working with file paths:
- Use `os.path.join()` for cross-platform compatibility
- Use forward slashes for `uvx ty check src/` (not `.\src` — causes a path error on Windows)
