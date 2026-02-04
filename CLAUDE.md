# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Python Package Manager
This package is using UV (https://docs.astral.sh/uv/) as package manager.

## Claude Code Guidelines

- Always use available skills whenever possible when executing commands (e.g., use the scikit-learn skill for ML tasks, matplotlib/seaborn skills for plotting, etc.)

## Rules

1. **SUMMARY.md must be updated after every completed task.** Any finished task — bug fix, refactor, new feature, test, documentation change — must have its outcome recorded in SUMMARY.md before moving on.
2. **TODO.md tracks pending work.** Use TODO.md for next things to do. Check off items when complete and add new items as they arise.
3. **Run `uv run isort src/` before every commit.** Imports must be sorted before staging and committing.
4. **Type checker is pyrefly, not mypy.** Use `uv run pyrefly check src/` for type checking. mypy has been removed from the project.
5. **All `uv` commands are permitted.** `uv sync`, `uv run`, `uv pip install/uninstall`, `uv lock`, etc. — no need to ask.
6. **Create a new branch before commits or modifications.** Always create a new branch with `claude/` prefix before making any commits or code modifications (e.g., `claude/fix-docstrings`, `claude/add-feature-x`).

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
# Format code with black
uv run black src/

# Lint with ruff
uv run ruff check src/

# Type checking with pyrefly
uv run pyrefly check src/

# Sort imports with isort
uv run isort src/
```

### Running the Application
```bash
# Run main script
python main.py

# Run with uv
uv run python main.py
```

### Jupyter Notebooks
The `examples/` directory contains Jupyter notebooks demonstrating various workflows. Run notebooks using:
```bash
jupyter notebook examples/
```

## Architecture Overview

### Core Pipeline (Three-Stage Process)

The package follows a sequential three-stage pipeline for eruption forecasting:

1. **Tremor Calculation** (`CalculateTremor`): Process raw seismic data from SDS format to calculate tremor metrics (RSAM and DSAR)
2. **Feature Engineering** (`FeaturesBuilder`): Extract time-series features from tremor data and build labeled training datasets
3. **Forecasting** (`ForecastModel`): Train models and predict eruptions (work in progress)

### Key Components

#### 1. Tremor Calculation (`src/eruption_forecast/tremor/`)

**`CalculateTremor`** is the entry point for processing raw seismic data:
- Reads seismic data from SDS (SeisComP Data Structure) format or FDSN web services
- Calculates two types of tremor metrics in parallel across multiple frequency bands:
  - **RSAM** (Real Seismic Amplitude Measurement): Mean amplitude in frequency bands
  - **DSAR** (Displacement Seismic Amplitude Ratio): Ratio between consecutive frequency bands
- Default frequency bands: `(0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz`
- Supports multiprocessing for parallel daily calculations
- Outputs time-series CSV files with 10-minute sampling intervals

**Key classes:**
- `CalculateTremor`: Main orchestrator (in `calculate_tremor.py`)
- `RSAM`: Calculates mean amplitude metrics (in `rsam.py`)
- `DSAR`: Calculates amplitude ratios between bands (in `dsar.py`)
- `TremorData`: Wrapper for loading and validating tremor CSV files (in `tremor_data.py`)
- `SDS`: Handles SeisComP Data Structure file reading (in `sds.py`)

**Workflow:**
```python
# Configure and run tremor calculation
calculate = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-03",
    n_jobs=4  # Parallel processing
).from_sds(sds_dir="/path/to/sds").run()

# Output: CSV with columns like rsam_f0, rsam_f1, dsar_f0-f1, etc.
```

#### 2. Label Building (`src/eruption_forecast/label/`)

**`LabelBuilder`** generates binary labels for supervised learning:
- Creates sliding time windows from tremor data based on configurable parameters
- Labels windows as erupted (1) or not erupted (0) based on known eruption dates
- Uses a "day_to_forecast" parameter to look ahead N days before eruptions
- Validates label filenames follow the pattern: `label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv`

**Key classes:**
- `LabelBuilder`: Creates labeled windows (in `label_builder.py`)
- `LabelData`: Loads and parses label CSV files, extracting parameters from filename (in `label_data.py`)

**Window configuration:**
- `window_size`: Size of each training window in days
- `window_step` + `window_step_unit`: Step size between windows (minutes or hours)
- `day_to_forecast`: Number of days before eruption to start labeling as positive

**Workflow:**
```python
label_builder = LabelBuilder(
    start_date="2020-01-01",
    end_date="2020-12-31",
    window_size=1,  # 1-day windows
    window_step=12,
    window_step_unit="hours",
    day_to_forecast=2,  # Label 2 days before eruption
    eruption_dates=["2020-06-15", "2020-09-20"],
    volcano_id="VOLCANO_ID"
).build()
```

#### 3. Feature Extraction (`src/eruption_forecast/features/`)

**`FeaturesBuilder`** combines tremor data with labels:
- Loads tremor CSV (from `TremorData`) and label CSV (from `LabelData`)
- Synchronizes time windows between tremor metrics and labels
- Prepares feature matrix for machine learning (uses tsfresh for time-series features)

**Key classes:**
- `FeaturesBuilder`: Orchestrates feature extraction (in `features_builder.py`)

**Workflow:**
```python
features = FeaturesBuilder(
    tremor_csv="path/to/tremor.csv",
    label_csv="path/to/label.csv",
    output_dir="output/"
)
```

#### 4. Forecast Model (`src/eruption_forecast/model/`)

**`ForecastModel`** (work in progress) will handle:
- Model training on extracted features
- Eruption prediction
- Model evaluation

### Data Classes

The package uses dataclasses to wrap CSV files with metadata:
- **`TremorData`**: Wraps tremor CSV, provides start/end dates, validates sampling rate
- **`LabelData`**: Wraps label CSV, parses parameters from filename (window_size, window_step, day_to_forecast)

These data classes use `@cached_property` for efficient access to derived attributes.

### Utility Functions (`src/eruption_forecast/utils.py`)

Core utilities used throughout the package:
- `calculate_window_metrics()`: Calculates metrics over time windows with outlier removal
- `construct_windows()`: Creates time window DataFrames for labeling
- `detect_maximum_outlier()` / `delete_outliers()`: Z-score based outlier detection
- `to_datetime()`: Date string validation and conversion
- `validate_date_ranges()`: Ensures start_date < end_date
- `validate_window_step()`: Validates window step parameters

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

## Type Checking Configuration

The project uses pyrefly for type checking (mypy has been removed):
- Run with: `uv run pyrefly check src/`
- All functions must have complete type annotations

## Common Workflows

### Complete Pipeline Example
```python
# 1. Calculate tremor from seismic data
tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2020-01-01",
    end_date="2020-12-31",
    n_jobs=4
).from_sds(sds_dir="/data/sds").run()

# 2. Build labels
labels = LabelBuilder(
    start_date="2020-01-01",
    end_date="2020-12-31",
    window_size=1,
    window_step=12,
    window_step_unit="hours",
    day_to_forecast=2,
    eruption_dates=["2020-06-15"],
    volcano_id="VOLCANO_001"
).build()

# 3. Extract features
features = FeaturesBuilder(
    tremor_csv=tremor.filepath,
    label_csv=labels.filepath,
    output_dir="output/"
)

# 4. Train model (work in progress)
# model = ForecastModel(...).train()
```

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
