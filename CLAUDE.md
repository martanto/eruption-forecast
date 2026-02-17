# CLAUDE.md

_Last updated: 2026-02-17_

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Package Manager
This package is using UV (https://docs.astral.sh/uv/) as package manager.

## Claude Code Guidelines

- Always use available skills whenever possible when executing commands (e.g., use the scikit-learn skill for ML tasks, matplotlib/seaborn skills for plotting, etc.)

## Rules

1. **SUMMARY.md must be updated after every completed task.** Any finished task — bug fix, refactor, new feature, test, documentation change — must have its outcome recorded in SUMMARY.md before moving on.
2. **TODO.md tracks pending work.** Use TODO.md for next things to do. Check off items when complete and add new items as they arise.
3. **Type checker is `ty`.** Use `uvx ty check src/` for type checking.
4. **Lint with ruff.** Use `uv run ruff check --fix src/` for linting.
5. **All `uv`, `uvx`, and `python` commands are permitted.** `uv sync`, `uv run`, `uv pip install/uninstall`, `uv lock`, `uvx ty check`, `python main.py`, etc. — no need to ask. User has granted permission to run these commands without approval.
6. **ALWAYS create a new branch before any commits or modifications.** Use `git checkout -b copilot/<branch-name>` to create a new branch with `copilot/` prefix before making ANY commits or code modifications (e.g., `copilot/fix-docstrings`, `copilot/add-feature-x`). Never work directly on main or dev branches.
7. **Always use `tests` directory when running testing.** Create the tests output inside `tests` directory.
8. **Do not commit temporary test files.** Files starting with `test` in the root directory (e.g., `test_*.py`, `test.py`) should not be committed. These are temporary test scripts and are excluded via `.gitignore`.

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
- `SDS`: Reads SeisComP Data Structure files (`sds.py`)

**Workflow:**
```python
calculate = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-03",
    n_jobs=4
).from_sds(sds_dir="/path/to/sds").run()
# Output CSV columns: rsam_f0, rsam_f1, dsar_f0-f1, etc.
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
- `ClassifierModel`: Manages classifier instances and hyperparameter grids (`classifier_model.py`)
- `ModelEvaluator`: Computes metrics and plots for a fitted model (`model_evaluator.py`)
  - Methods: `get_metrics()`, `summary()`, `plot_all()`, `from_files()`
- `ModelPredictor`: Runs inference in evaluation or forecast mode (`model_predictor.py`)
  - `predict()` / `predict_best()`: Requires labels (evaluation mode)
  - `predict_proba()`: Unlabelled forecasting with per-classifier + consensus output

### Data Classes

- **`TremorData`**: Wraps tremor CSV; provides start/end dates, validates sampling rate
- **`LabelData`**: Wraps label CSV; parses parameters from filename

Both use `@cached_property` for efficient attribute access.

### Utility Functions (`src/eruption_forecast/utils.py`)

- `construct_windows()`: Creates sliding time windows
- `calculate_window_metrics()`: Metrics over windows with outlier removal
- `detect_maximum_outlier()` / `remove_outliers()`: Z-score based outlier detection
- `to_datetime()`: Date string validation and conversion
- `validate_date_ranges()`: Ensures start_date < end_date
- `validate_window_step()`: Validates window step parameters
- `random_under_sampler()`: Balances classes via random undersampling
- `get_significant_features()`: Retrieves statistically significant tsfresh features
- `resolve_output_dir()`: Resolves paths relative to `root_dir`

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
    ├── tremor/
    │   ├── tmp/          # Daily CSVs before merging
    │   └── tremor.csv    # Final merged tremor data
    ├── forecast/         # Model predictions
    ├── figures/          # Plots
    └── logs/             # Debug logs
```

## Dependencies

### Key Scientific Libraries
- **obspy**: Seismic data processing (Stream, Trace objects)
- **pandas**: Time-series data manipulation (requires pandas >= 3.0.0)
- **tsfresh**: Automated time-series feature extraction
- **numpy**: Numerical computations
- **numba**: Performance optimization via JIT compilation

### Data Sources
- **SDS (SeisComP Data Structure)**: Primary seismic data format
- **FDSN**: Alternative data source via web services


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
    number_of_significant_features=20,
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
