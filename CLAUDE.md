# CLAUDE.md

_Last updated: 2026-02-25 (added SeedEnsemble; updated utils/ml.py entries)_

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Package Manager
This package is using UV (https://docs.astral.sh/uv/) as package manager.

## Claude Code Guidelines

- Always use available skills whenever possible when executing commands (e.g., use the scikit-learn skill for ML tasks, matplotlib/seaborn skills for plotting, etc.)

## Rules

1. **Log every completed task in the daily changelog.** Any finished task — bug fix, refactor, new feature, test, documentation change — must have its outcome recorded in `changelogs/YYYY-mm-dd.md` (using today's date) before moving on. Create the file if it does not exist. Append new entries; never overwrite previous entries in the same file. The `changelogs/` directory is git-ignored (local only).
2. **TODO.md tracks pending work.** Use TODO.md for next things to do. Check off items when complete and add new items as they arise.
3. **Type checker is `ty`.** Use `uvx ty check src/` for type checking.
4. **Lint with ruff.** Use `uv run ruff check --fix src/` for linting.
5. **All `uv`, `uvx`, and `python` commands are permitted.** `uv sync`, `uv run`, `uv pip install/uninstall`, `uv lock`, `uvx ty check`, `python main.py`, etc. — no need to ask. User has granted permission to run these commands without approval.
6. **ALWAYS create a new branch before any commits or modifications.** Use `git checkout -b <prefix>/<branch-name>` to create a new branch before making ANY commits or code modifications. Choose the prefix based on the type of work: `fix/` for bug fixes (e.g., `fix/docstring-errors`), `ft/` for new features (e.g., `ft/add-fdsn-source`), and `dev/` as the default for everything else (e.g., `dev/refactor-utils`). Never work directly on main or master branches.
7. **Always use `tests` directory when running testing.** Create the tests output inside `tests` directory.
8. **Do not commit temporary test files.** Files starting with `test` in the root directory (e.g., `test_*.py`, `test.py`) should not be committed. These are temporary test scripts and are excluded via `.gitignore`.
9. **All module imports must be at the top of the file.** Never place `import` statements inside functions, methods, or conditional blocks. All stdlib, third-party, and local imports belong at the module level, grouped and sorted by ruff.
11. **Run circular import test after every module change.** After adding, removing, or reorganising any module or import, run `uv run pytest tests/test_imports.py -v` to confirm no circular imports were introduced.
10. **Always use `martanto` for author contributions.** Use the GitHub username `martanto` for all author attributions in any file.
12. **Keep `config.example.yaml` in sync.** Whenever `src/eruption_forecast/model/forecast_model.py` or `src/eruption_forecast/config/pipeline_config.py` is modified — new parameter added, removed, renamed, or default changed — update `config.example.yaml` to reflect the change. The YAML keys, defaults, and inline comments must match the current dataclass fields and `forecast_model.py` constructor calls exactly.

## Testing

Testing is not mandatory, but if needed, you can use the following test data:

- **Input directory (SDS format):** `D:\Data\OJN`
- **Start date:** `2025-03-16`
- **End date:** `2025-03-28`
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

### Key Components

#### 1. Tremor Calculation (`src/eruption_forecast/tremor/`)

**`CalculateTremor`** processes raw seismic data into tremor metrics:
- Reads seismic data from SDS (SeisComP Data Structure) format or FDSN web services
- Calculates two metrics across multiple frequency bands in parallel:
  - **RSAM** (Real Seismic Amplitude Measurement): Mean amplitude per band
  - **DSAR** (Displacement Seismic Amplitude Ratio): Ratio between consecutive bands
- Default frequency bands: `(0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz`
- Supports multiprocessing via `n_jobs`; outputs 10-minute interval CSVs

**Key classes:**
- `CalculateTremor`: Main orchestrator (`calculate_tremor.py`)
- `RSAM`: Mean amplitude metrics (`rsam.py`)
- `DSAR`: Amplitude ratios between bands (`dsar.py`)
- `TremorData`: Loads and validates tremor CSV files (`tremor_data.py`)
- `SDS`: Reads SeisComP Data Structure files (`src/eruption_forecast/sources/sds.py`)
- `FDSN`: Downloads seismic data from FDSN web services with local SDS caching (`src/eruption_forecast/sources/fdsn.py`)

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
# Output CSV columns: rsam_f0, rsam_f1, dsar_f0-f1, etc.

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
- Label filenames follow: `label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv`

**Key classes:**
- `LabelBuilder`: Creates labeled windows (`label_builder.py`)
- `LabelData`: Loads label CSV and parses parameters from filename (`label_data.py`)

**Window configuration:**
- `window_size`: Window size in days
- `window_step` + `window_step_unit`: Step between windows (minutes or hours)
- `day_to_forecast`: Days before eruption to start labeling positive

**Workflow:**
```python
label_builder = LabelBuilder(
    start_date="2020-01-01",
    end_date="2020-12-31",
    window_size=1,
    window_step=12,
    window_step_unit="hours",
    day_to_forecast=2,
    eruption_dates=["2020-06-15", "2020-09-20"],
    volcano_id="VOLCANO_ID"
).build()
```

#### 3. Tremor Matrix Building (`src/eruption_forecast/features/`)

**`TremorMatrixBuilder`** slices tremor time-series into windows aligned with labels:
- Takes tremor DataFrame and label DataFrame as input
- Validates sample counts per window
- Concatenates all windows into a unified matrix with `id`, `datetime`, and tremor columns

**Key classes:**
- `TremorMatrixBuilder`: Builds the windowed tremor matrix (`tremor_matrix_builder.py`)

#### 4. Feature Extraction (`src/eruption_forecast/features/`)

**`FeaturesBuilder`** extracts tsfresh features from the tremor matrix:
- Operates in two modes:
  - **Training mode** (labels provided): Filters windows to match labels, saves aligned label CSV
  - **Prediction mode** (no labels): Extracts all features, disables relevance filtering
- Runs tsfresh extraction per tremor column independently

**Key classes:**
- `FeaturesBuilder`: Orchestrates tsfresh feature extraction (`features_builder.py`)
- `FeatureSelector`: Two-stage selection — tsfresh (statistical FDR) → RandomForest (importance) (`feature_selector.py`)
  - Methods: `"tsfresh"`, `"random_forest"`, `"combined"`

#### 5. Model Training (`src/eruption_forecast/model/`)

**`ModelTrainer`** trains classifiers across multiple random seeds:
- Supports 10 classifiers: `rf`, `gb`, `xgb`, `svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting`
- CV strategies: `shuffle`, `stratified`, `timeseries`
- Uses `RandomUnderSampler` to handle class imbalance
- Two training modes:
  - `train_and_evaluate()`: 80/20 split → resample train → feature selection → CV → evaluate on test set → save
  - `train()`: Resample full dataset → feature selection → CV → save (no metrics)

**Key classes:**
- `ModelTrainer`: Multi-seed training and evaluation (`model_trainer.py`)
  - `fit(with_evaluation=True)`: Dispatches to `train_and_evaluate()` or `train()` based on flag
  - `merge_models(output_path)`: Bundle all seed models → `SeedEnsemble` `.pkl`
  - `merge_classifier_models(trained_models, output_path)`: Bundle multiple classifier registries → multi-classifier `.pkl`
  - `n_jobs`: outer seed workers; `grid_search_n_jobs`: inner `GridSearchCV`/`FeatureSelector` workers. Enforced: `n_jobs × grid_search_n_jobs ≤ cpu_count`. Uses `joblib.Parallel(backend="loky")` for nested-parallelism safety.
- `SeedEnsemble`: Bundles all seed models for one classifier (`seed_ensemble.py`); sklearn-compatible (`BaseEstimator` + `ClassifierMixin`)
  - `from_registry(registry_csv)`: Load from registry CSV
  - `predict_proba(X)` → `(n_samples, 2)`; `predict_with_uncertainty(X)` → `(mean, std, conf, pred)`
  - `save(path)` / `load(path)`: joblib serialisation
- `ClassifierModel`: Manages classifier instances and hyperparameter grids (`classifier_model.py`)
- `ModelEvaluator`: Computes metrics and plots for a fitted model (`model_evaluator.py`)
  - Methods: `get_metrics()`, `summary()`, `plot_all()`, `from_files()`
  - `cv_name` parameter (default `"cv"`): slugified into the default output path `output/trainings/evaluations/{clf-slug}/{cv-slug}/` when `output_dir` is `None`
- `ModelPredictor`: Runs inference in evaluation or forecast mode (`model_predictor.py`)
  - `predict()` / `predict_best()`: Requires labels (evaluation mode)
  - `predict_proba()`: Unlabelled forecasting with per-classifier + consensus output
- `PipelineConfig`: Serialisable pipeline configuration (`src/eruption_forecast/config/pipeline_config.py`)
  - Sub-configs: `ModelConfig`, `CalculateConfig`, `BuildLabelConfig`, `ExtractFeaturesConfig`, `TrainConfig`, `ForecastConfig`

**`ForecastModel` additional methods** (beyond the pipeline stages):
- `load_tremor_data(tremor_csv)`: Load pre-calculated tremor data instead of calling `calculate()`
- `set_feature_selection_method(using)`: Change feature selection method before `train()`
- `save_config(path, fmt)` / `from_config(path)`: Persist and replay pipeline configuration
- `save_model(path)` / `load_model(path)`: Serialise/restore full `ForecastModel` via joblib
- `run()`: Replay all pipeline stages from a loaded config (use after `from_config()`)

### Data Classes

- **`TremorData`**: Wraps tremor CSV; provides start/end dates, validates sampling rate
- **`LabelData`**: Wraps label CSV; parses parameters from filename

Both use `@cached_property` for efficient attribute access.

### Utility Functions (`src/eruption_forecast/utils/`)

`utils.py` has been split into a package with focused sub-modules:

- **`window.py`**: `construct_windows()`, `calculate_window_metrics()`
- **`array.py`**: `detect_maximum_outlier()`, `remove_outliers()` — Z-score based outlier detection
- **`date_utils.py`**: `to_datetime()`, `validate_date_ranges()`, `validate_window_step()`
- **`ml.py`**: `random_under_sampler()`, `get_significant_features()`, `merge_seed_models()`, `merge_all_classifiers()`
- **`pathutils.py`**: `resolve_output_dir()` — resolves paths relative to `root_dir`
- **`dataframe.py`**: DataFrame validation utilities
- **`formatting.py`**: Text formatting utilities

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
`CalculateTremor` supports parallel processing via `n_jobs` parameter. Each day's calculation runs independently, generating temporary CSV files that are merged at the end.

### Data Format Standards
- **Tremor CSV**: DateTime index, columns like `rsam_f0`, `dsar_f0-f1`
- **Label CSV**: DateTime index, columns `id` (int) and `is_erupted` (0 or 1)
- **Date format**: Always `YYYY-MM-DD`
- **Sampling interval**: Default 10 minutes for tremor data

### Frequency Band Naming
Frequency bands are aliased as `f0`, `f1`, `f2`, etc. in column names:
- `rsam_f0`: RSAM for band 0 (e.g., 0.01-0.1 Hz)
- `dsar_f0-f1`: DSAR ratio between bands f0 and f1

### Output Directory Structure
```
output/
└── {network}.{station}.{location}.{channel}/
    ├── features/         # Extracted tsfresh features + aligned label CSV
    ├── tremor/
    │   ├── daily/        # Daily CSVs before merging (cleaned up if cleanup_daily_dir=True)
    │   ├── figures/      # Daily tremor plots (created if plot_daily=True)
    │   └── {nslc}_{start}_{end}.csv  # Final merged tremor data
    ├── forecast/         # Forecast/prediction outputs
    └── config.yaml       # Saved pipeline config (written by save_config())
```

## Dependencies

### Key Scientific Libraries
- **obspy**: Seismic data processing (Stream, Trace objects)
- **pandas**: Time-series data manipulation (requires pandas >= 3.0.0)
- **tsfresh**: Automated time-series feature extraction
- **numpy**: Numerical computations
- **numba**: Performance optimization via JIT compilation

### Data Sources

Both data-source adapters live in `src/eruption_forecast/sources/`:

- **SDS (SeisComP Data Structure)**: Primary seismic data format — `sds.py`
- **FDSN**: Downloads from any FDSN web service with transparent local SDS caching — `fdsn.py`
  - `download_dir` is created automatically if absent
  - Downloaded files are cached as SDS miniSEED so subsequent runs skip the network


## Common Workflows

### Complete Pipeline Example

`ForecastModel` orchestrates the full pipeline via method chaining. `root_dir` anchors all output paths.

```python
from eruption_forecast import ForecastModel

fm = ForecastModel(
    root_dir="/path/to/project",
    station="OJN",
    channel="EHZ",
    start_date="2020-01-01",
    end_date="2020-12-31",
    window_size=2,
    volcano_id="VOLCANO_001",
    n_jobs=4,
)

fm.calculate(
    source="sds",
    sds_dir="/path/to/sds",
).build_label(
    start_date="2020-01-01",
    end_date="2020-12-31",
    day_to_forecast=2,
    window_step=6,
    window_step_unit="hours",
    eruption_dates=["2020-06-15"],
).extract_features(
    select_tremor_columns=["rsam_f2", "rsam_f3", "dsar_f3-f4"],
).train(
    classifier="xgb",
    cv_strategy="stratified",
    random_state=0,
    total_seed=500,
    with_evaluation=False,   # train on full dataset; set True for 80/20 split + metrics
    number_of_significant_features=20,
).forecast(
    start_date="2020-07-01",
    end_date="2020-07-31",
    window_size=2,
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

## Notes on Windows Development

This project is developed on Windows (see `win32` platform in env info). When working with file paths:
- Use `os.path.join()` for cross-platform compatibility
- The codebase already follows this pattern consistently
