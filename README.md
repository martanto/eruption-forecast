# eruption-forecast

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active%20development-orange)

A Python package for volcanic eruption forecasting using seismic data analysis. Process raw seismic tremor data, extract time-series features, train machine learning models, and predict volcanic eruptions probability based on seismic patterns.

## Important Disclaimers

**This software is intended for research purposes only.**

1. **Probabilistic Predictions**: This eruption forecast model provides probabilistic predictions of future volcanic activity, NOT deterministic guarantees. Predictions should be interpreted as likelihood estimates based on historical seismic patterns.

2. **No Guarantee of Accuracy**: This model is **not guaranteed to predict every future eruption**. Volcanic systems are complex and can exhibit unexpected behavior. False negatives (missed eruptions) and false positives (false alarms) are possible.

3. **Software Limitations**: This software is **not guaranteed to be free of bugs or errors**. Users should validate results independently and use this tool as one component of a comprehensive volcano monitoring strategy.

4. **Not for Operational Use**: This package is a research tool and should not be used as the sole basis for public safety decisions, evacuation orders, or emergency response without expert volcanological assessment.

5. **Expert Interpretation Required**: Results should always be interpreted by qualified volcanologists familiar with the specific volcano being monitored.

**Always consult with local volcano observatories and follow official warnings from government agencies.**

---

## Table of Contents

- [Features](#features)
- [Package Architecture](#package-architecture)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Data Sources](#data-sources)
- [Quick Start: Complete Pipeline](#quick-start-complete-pipeline)
- [Reports](#reports)
- [Advanced Usage](#advanced-usage)
- [Supported Classifiers](#supported-classifiers)
- [Cross-Validation Strategies](#cross-validation-strategies)
- [Output Directory Structure](#output-directory-structure)
- [Requirements](#requirements)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

**Detailed documentation:**
- [Step-by-Step Usage Guide](docs/step-by-step-guide.md) — Sections 1–12, per-stage code examples
- [API Reference](docs/api-reference.md) — Constructor and method parameter tables
- [Visualization & Plotting](docs/visualization.md) — All plot types and usage
- [Configuration](docs/configuration.md) — notify decorator, pipeline config save/replay, logging
- [Output Directory Structure](docs/output-structure.md) — Full directory tree
- [Architecture](docs/architecture.md) — Component details, design principles, key classes

---

## Features

- **Tremor Calculation**: Process raw seismic data (SDS/FDSN) to calculate RSAM, DSAR, and Shannon Entropy metrics across multiple frequency bands
- **Label Building**: Generate training labels from eruption dates with configurable forecast horizons
- **Feature Extraction**: Extract 700+ time-series features using tsfresh for machine learning
- **Enhanced Feature Selection**: Three-method feature selection — tsfresh statistical, RandomForest permutation importance, or combined two-stage
- **Model Training**: Train 10 classifier types (Random Forest, Gradient Boosting, XGBoost, SVM, Logistic Regression, Neural Networks, Ensembles) across multiple random seeds
- **Model Evaluation**: Comprehensive evaluation with ROC curves, precision-recall curves, confusion matrices, threshold analysis, calibration curves, feature importance, SHAP explainability, seed stability violin plots, frequency band contribution charts, and **learning curve plots** (`plot_learning_curve_grid`) via `ModelEvaluator` and `MultiModelEvaluator`; cross-classifier comparison plots and ranking tables via `ClassifierComparator`
- **Two Training Workflows**: `train_and_evaluate()` for in-sample evaluation (80/20 split), `train()` for full-dataset training with future-data evaluation via `ModelPredictor`; `fit()` as a unified entry point that dispatches between the two
- **Seed Ensemble Merging**: Combine all 500 seed models + their feature lists into a single `.pkl` file via `BaseEnsemble.save()` / `SeedEnsemble` / `ClassifierEnsemble` / `merge_seed_models()` / `merge_all_classifiers()` — eliminates per-seed I/O at prediction time and enables sklearn-compatible `predict_proba()` / `predict()` calls directly on the ensemble
- **Multi-processing**: Parallel processing for faster tremor calculations and model training
- **Interactive HTML Reports**: (beta, not fully functional yet) Generate self-contained Plotly-powered reports for every pipeline stage via `ForecastModel.generate_report()` or the standalone `generate_report()` function — no external dependencies except an optional `weasyprint` for PDF export
- **Telegram Notifications**: `notify` decorator sends structured Telegram messages (success/error, elapsed time, file attachments) on function completion
- **Modular Architecture**: Clean separation of concerns with focused utility modules

## Package Architecture

```
eruption-forecast/
├── src/eruption_forecast/
│   ├── data_container.py    # BaseDataContainer — shared ABC for TremorData & LabelData
│   ├── tremor/              # Seismic tremor processing
│   ├── label/               # Training label generation
│   ├── features/            # Feature extraction & selection
│   ├── model/               # ML model training & prediction
│   ├── sources/             # SDS and FDSN data source adapters
│   ├── plots/               # Visualization utilities
│   ├── report/              # (beta) Interactive HTML report generation
│   ├── utils/               # Focused utility modules
│   └── decorators/          # Function decorators
└── tests/                   # Unit tests
```

> Full directory tree, design principles, and per-component details: [docs/architecture.md](docs/architecture.md)

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
│         ↓  train_and_evaluate()  ↓ train()  │
│    80/20 split + metrics   Full dataset     │
└─────────┬───────────────────────────────────┘
          │  trained_model_*.csv  +  *.pkl
          │
          │  (optional) trainer.merge_models()
          │  → merged_model_*.pkl  (SeedEnsemble)
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
│  Single model, merged pkl, or multi-model   │
│  consensus                                  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│              Report (optional)              │
│   generate_report() / fm.generate_report()  │
│   → self-contained HTML (Plotly, CDN JS)    │
│   Tremor · Labels · Features · Training     │
│   Comparator · Prediction · Pipeline        │
└─────────────────────────────────────────────┘
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

## Data Sources

The package reads seismic data from two sources, both routed through `CalculateTremor`.

### SDS — SeisComP Data Structure

SDS is a standardized directory and file layout used by [SeisComP](https://www.seiscomp.de/) to store waveform data portably across data servers and analysis tools. See the [official SDS specification](https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html) for full details.

**Directory layout:**

```
<sds_dir>/
└── YEAR/
    └── NET/
        └── STA/
            └── CHAN.TYPE/
                └── NET.STA.LOC.CHAN.TYPE.YEAR.DAY
```

**Example** for network `VG`, station `OJN`, channel `EHZ`, day 075 of 2025:

```
/data/
└── 2025/
    └── VG/
        └── OJN/
            └── EHZ.D/
                └── VG.OJN.00.EHZ.D.2025.075
```

**Field reference:**

| Field | Description | Example |
|-------|-------------|---------|
| `YEAR` | Four-digit year | `2025` |
| `NET` | Network code (≤ 8 chars) | `VG` |
| `STA` | Station code (≤ 8 chars) | `OJN` |
| `CHAN` | Channel code (≤ 8 chars) | `EHZ` |
| `LOC` | Location code (≤ 8 chars, may be empty) | `00` |
| `TYPE` | Data type — `D` = waveform data (most common) | `D` |
| `DAY` | Three-digit day-of-year, zero-padded | `075` |

Files are miniSEED format. Periods in filenames are always present even when a field is empty.

**Usage:**

```python
from eruption_forecast import CalculateTremor

tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-31",
    n_jobs=4,
).from_sds(sds_dir="/data/sds").run()
```

### FDSN — Web Service

FDSN downloads waveform data from any FDSN-compatible web service (IRIS, GEOFON, etc.) and caches it locally as SDS miniSEED so subsequent runs skip the network.

```python
tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-31",
).from_fdsn(client_url="https://service.iris.edu").run()
```

---

## Quick Start: Complete Pipeline

Here's a complete end-to-end example from raw seismic data to trained models and eruption forecasting (adapted from `main.py`):

```python
from eruption_forecast import ForecastModel

# Initialize the forecast model with station and time range
fm = ForecastModel(
    root_dir="output",
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=2,
    volcano_id="Lewotobi Laki-laki",
    n_jobs=4,
    verbose=True,
)

# Complete pipeline with method chaining
fm.calculate(
    source="sds",
    sds_dir="/path/to/sds/data",
    methods=["rsam", "dsar", "entropy"],
    plot_daily=True,
    save_plot=True,
    remove_outlier_method="maximum",
).build_label(
    start_date="2025-01-01",
    end_date="2025-07-24",
    day_to_forecast=2,
    window_step=6,
    window_step_unit="hours",
    eruption_dates=[
        "2025-03-20",
        "2025-04-22",
        "2025-05-18",
        "2025-06-17",
        "2025-07-07",
    ],
).extract_features(
    select_tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4", "entropy"],
    save_tremor_matrix_per_method=True,
    exclude_features=["agg_linear_trend", "linear_trend_timewise", "length"],
    use_relevant_features=True,
).train(
    classifier="rf",
    cv_strategy="stratified",
    random_state=0,
    total_seed=500,
    with_evaluation=False,
    number_of_significant_features=20,
    sampling_strategy=0.75,
    save_all_features=True,
    plot_significant_features=True,
).forecast(
    start_date="2025-07-28",
    end_date="2025-08-04",
    window_step=10,
    window_step_unit="minutes",
)
```

**What this pipeline does:**

1. **Calculate tremor** (RSAM, DSAR, Shannon Entropy) from raw seismic data with outlier removal
2. **Build labels** from known eruption dates (training period: Jan 1 – Jul 24)
3. **Extract features** using tsfresh (700+ features) and select top 20
4. **Train models** using Random Forest with 500 random seeds for robust predictions
5. **Forecast** future eruptions (Jul 28 – Aug 4) using the trained ensemble

See `main.py` in the repository for the complete working example.

> Full per-stage guide with code examples: [docs/step-by-step-guide.md](docs/step-by-step-guide.md)

---

## Reports (beta)

The `report/` package generates self-contained, interactive HTML reports (powered by Plotly, loaded from CDN) for every pipeline stage. No image files are produced — each report is a single `.html` file you can open in any browser or share by email.

### Integrated — chain after any pipeline stage

```python
fm.calculate(...).build_label(...).train(...).generate_report()
# → output/VG.OJN.00.EHZ/reports/pipeline_report.html

# Select specific sections only
fm.generate_report(sections=["tremor", "label"])

# Export to PDF (requires: uv add weasyprint)
fm.generate_report(fmt="pdf")
```

### Standalone — from an existing output directory

```python
from eruption_forecast.report import generate_report

path = generate_report("output/VG.OJN.00.EHZ")
print(f"Report saved to {path}")

# Specific sections
path = generate_report("output/VG.OJN.00.EHZ", sections=["tremor", "training"])
```

### Individual section reports

Each report class can be used directly for a focused view:

```python
from eruption_forecast.report import (
    TremorReport,
    LabelReport,
    FeaturesReport,
    TrainingReport,
    ComparatorReport,
    PredictionReport,
    PipelineReport,
)

# Tremor: data completeness, band stats, full-range + daily detail chart
path = TremorReport("output/.../tremor.csv", station="OJN").save()

# Labels: window config, class distribution, eruption timeline
path = LabelReport(
    "output/.../label_2025-01-01_....csv",
    eruption_dates=["2025-03-20", "2025-04-22"],
).save()

# Features: feature counts, top-N bar, band contribution
path = FeaturesReport(
    features_csv="output/.../all_extracted_features.csv",
    significant_features_dir="output/.../significant_features/",
    selection_method="combined",
).save()

# Training: per-seed <details>, aggregate mean±std, stability, threshold analysis
path = TrainingReport(
    metrics_dir="output/.../metrics/",
    classifier_name="RandomForestClassifier",
).save()

# Classifier comparison: grouped bar + aggregate table
from eruption_forecast.report import ComparatorReport
path = ComparatorReport(
    classifier_registry={
        "rf":  "output/.../trainings/evaluations/rf/.../trained_model_rf.csv",
        "xgb": "output/.../trainings/evaluations/xgb/.../trained_model_xgb.csv",
    }
).save()

# Forecast probabilities: consensus line, uncertainty band, eruption markers
path = PredictionReport(
    prediction_df=fm.prediction_df,
    eruption_dates=["2025-03-20"],
    threshold=0.5,
).save()
```

**What each report contains:**

| Report | Charts | Tables |
|--------|--------|--------|
| `TremorReport` | Full-range overview, daily detail with date dropdown | Completeness, band stats |
| `LabelReport` | Class distribution bar, eruption timeline | Window config, class counts |
| `FeaturesReport` | Top-N features (horizontal bar), band contribution | Feature counts summary |
| `TrainingReport` | Seed stability, threshold analysis | Per-seed rows + aggregate mean±std |
| `ComparatorReport` | Grouped bar per metric × classifier | Aggregate metrics table |
| `PredictionReport` | Probability lines + uncertainty band + eruption markers | Forecast config |
| `PipelineReport` | All of the above + executive summary | Pipeline stage availability |

---

## Advanced Usage

### Use FDSN instead of a local SDS archive

```python
fm.calculate(
    source="fdsn",
    client_url="https://service.iris.edu",
).build_label(...).train(...)
# Downloaded miniSEED files are cached locally so subsequent runs skip the network.
```

### Skip tremor calculation if data already exists

```python
fm.load_tremor_data(
    tremor_csv="output/VG.OJN.00.EHZ/tremor/tremor_2025-01-01_2025-12-31.csv"
).build_label(...).extract_features(...).train(...).forecast(...)
```

### Change feature selection method

```python
fm.set_feature_selection_method("combined").train(
    classifier="rf",
    cv_strategy="timeseries",
    total_seed=200,
)
```

### Train with evaluation (80/20 split) for in-sample testing

```python
fm.train(
    classifier="xgb",
    with_evaluation=True,
    total_seed=100,
)
```

### GPU acceleration for XGBoost

XGBoost (`xgb`) and the voting ensemble (`voting`) support GPU training via `use_gpu=True`. Use `gpu_id` to select a specific device on multi-GPU machines.

```python
# Train on the first GPU (default)
fm.train(
    classifier="xgb",
    cv_strategy="stratified",
    total_seed=500,
    use_gpu=True,          # enable CUDA
    gpu_id=0,              # first GPU (default)
)

# Train on the second GPU
fm.train(
    classifier="xgb",
    use_gpu=True,
    gpu_id=1,              # second GPU
)
```

**Parallelism architecture:**

```
ModelTrainer.fit()
│
├── [outer] n_jobs  → Parallel(loky backend)
│   Each worker runs one full seed independently:
│   resample → feature selection → GridSearchCV → evaluate → save
│   GPU: forced to 1 (seeds share one GPU device)
│
└── [inner, per seed] grid_search_n_jobs
    │
    ├── FeatureSelector  → tsfresh/RandomForest, CPU-only
    │   GPU: unchanged — safe to parallelise
    │
    └── GridSearchCV  → runs XGBoost CV folds
        GPU: forced to 1 (fold workers share one GPU device)
```

**Parallelism rules when `use_gpu=True`:**

| Parameter | Normal (CPU) | GPU (`use_gpu=True`) |
|---|---|---|
| `n_jobs` (outer seed workers) | Configurable | Forced to `1` — multiple seeds sharing one GPU causes VRAM contention |
| `grid_search_n_jobs` in `GridSearchCV` | Configurable | Forced to `1` — parallel CV fold workers each try to use the GPU simultaneously |
| `grid_search_n_jobs` in `FeatureSelector` | Configurable | **Unchanged** — feature selection is CPU-only (tsfresh/RandomForest) and is safe to parallelise |

> `use_gpu=True` has no effect on non-XGBoost classifiers (`rf`, `gb`, `svm`, etc.). Passing it with those classifiers emits a warning and training proceeds on CPU as normal.

### Train multiple classifiers and run consensus forecast

Pass a `list[str]` or comma-separated string to `classifier`. Each classifier is trained sequentially; all trained model registries are available for multi-model consensus forecasting.

```python
fm.train(
    classifier=["rf", "xgb", "gb"],
    cv_strategy="stratified",
    total_seed=500,
    with_evaluation=False,
).forecast(
    start_date="2025-07-28",
    end_date="2025-08-04",
    window_step=10,
    window_step_unit="minutes",
)
```

### Compare multiple classifiers side-by-side

After training several classifiers with `train_and_evaluate()`, use `ClassifierComparator`
to rank them and produce comparison plots:

```python
from eruption_forecast.model import ClassifierComparator

# From a dict
comparator = ClassifierComparator(
    classifiers={
        "rf": "output/.../trainings/evaluations/rf/stratified/trained_model_rf_...csv",
        "xgb": "output/.../trainings/evaluations/xgb/stratified/trained_model_xgb_...csv",
        "gb": "output/.../trainings/evaluations/gb/stratified/trained_model_gb_...csv",
    },
    metrics=["f1_score", "roc_auc", "recall"],  # or None for all DEFAULT_METRICS
)

# Or from a JSON file  {"ClassifierName": "/path/to/trained_model_*.csv", ...}
comparator = ClassifierComparator.from_json(
    "output/VG.OJN.00.EHZ/evaluations_trained_models.json",
    metrics=["f1_score", "roc_auc", "recall"],
)

# Ranked by recall (default), saved to metrics/ranking_recall.csv
ranking = comparator.get_ranking()

# All plots — saved to figures/
results = comparator.plot_all()
# results["metric_bar"]      → dict[str, Figure]  (one per metric + "all" overview)
# results["seed_stability"]  → dict[str, Figure]  (one per metric + "all" overview)
# results["comparison_grid"] → Figure
# results["roc"]             → Figure
# results["ranking"]         → DataFrame
```

### Merge 500 seed models into one file

After training, collapse all seed models into a single `.pkl` to remove per-seed I/O overhead:

```python
from eruption_forecast.model.seed_ensemble import SeedEnsemble
from eruption_forecast.utils.ml import merge_seed_models, merge_all_classifiers

# Single classifier
merged_path = trainer.merge_models()
# → .../merged_model_RandomForestClassifier-StratifiedKFold_rs-0_ts-500_top-20.pkl

# Load and predict directly
ensemble = SeedEnsemble.load(merged_path)
mean_p, std, conf, pred = ensemble.predict_with_uncertainty(features_df)

# sklearn-compatible interface
proba = ensemble.predict_proba(features_df)   # shape (n_samples, 2)

# Multi-classifier bundle
bundle_path = trainer.merge_classifier_models({"rf": rf_csv, "xgb": xgb_csv})

# Pass merged pkl directly to ModelPredictor
predictor = ModelPredictor(
    start_date="2025-07-28", end_date="2025-08-04",
    trained_models=merged_path,    # single merged pkl
    # or: trained_models=bundle_path  (multi-classifier bundle)
)
```

### Save and replay pipeline configuration

```python
fm.save_config()          # YAML → {station_dir}/config.yaml
fm.save_model()           # joblib → {station_dir}/forecast_model.pkl

# Replay the full pipeline from a saved config
fm2 = ForecastModel.from_config("output/VG.OJN.00.EHZ/config.yaml")
fm2.run()

# Resume from a saved model (skip re-training)
fm3 = ForecastModel.load_model("output/VG.OJN.00.EHZ/forecast_model.pkl")
fm3.forecast(start_date="2025-04-01", end_date="2025-04-07",
             window_step=12, window_step_unit="hours")
```

> Full configuration reference: [docs/configuration.md](docs/configuration.md)

---

## Supported Classifiers

| Classifier | Description | Imbalance Handling |
|------------|-------------|---------------------|
| `rf` | Random Forest (balanced, robust, default) | `class_weight="balanced"` |
| `gb` | Gradient Boosting (handles imbalance natively) | None (natural) |
| `xgb` | XGBoost (excellent for imbalanced data, GPU-capable) | `scale_pos_weight` grid search |
| `svm` | Support Vector Machine (high-dimensional) | `class_weight="balanced"` |
| `lr` | Logistic Regression (interpretable, fast) | `class_weight="balanced"` |
| `nn` | Neural Network MLP (complex patterns) | None |
| `dt` | Decision Tree (interpretable baseline) | `class_weight="balanced"` |
| `knn` | K-Nearest Neighbors (simple baseline) | None |
| `nb` | Gaussian Naive Bayes (fast baseline) | None |
| `voting` | Soft VotingClassifier (RF + XGBoost ensemble, GPU-capable) | Combined |
| `lite-rf` | Random Forest with a smaller grid for faster training | `class_weight="balanced"` |

> Hyperparameter grids and overriding them: [docs/step-by-step-guide.md#8-hyperparameter-grids](docs/step-by-step-guide.md#8-hyperparameter-grids)

## Cross-Validation Strategies

| Strategy | Class | Best For |
|----------|-------|----------|
| `shuffle` | `StratifiedShuffleSplit` | Random splits with stratification (default) |
| `stratified` | `StratifiedKFold` | Preserves class distribution across folds |
| `timeseries` | `TimeSeriesSplit` | Temporal data, strict no-future-leakage |

---

## Output Directory Structure

All outputs are organized under `{output_dir}/{network}.{station}.{location}.{channel}/`
(e.g., `output/VG.OJN.00.EHZ/`).

```
output/
└── VG.OJN.00.EHZ/
    ├── tremor/          # Merged tremor CSVs + daily plots
    ├── features/        # Extracted features + aligned labels
    └── trainings/
        ├── evaluations/
        │   ├── features/          # Shared feature selection outputs
        │   └── classifiers/       # Per-classifier model outputs (evaluations)
        └── predictions/
            ├── features/          # Shared feature selection outputs
            └── classifiers/       # Per-classifier model outputs (predictions)
```

> Full directory tree with all sub-paths: [docs/output-structure.md](docs/output-structure.md)

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
- python-dotenv (Telegram notify decorator credential loading)

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

**Commit convention:** Use `fix/` for bug fixes, `ft/` for new features, `dev/` as default.

**Code style:** PEP 8, Google-style docstrings with explicit types, type hints on all functions.

---

## License

MIT License — see LICENSE file for details.

**Disclaimer of Liability**: This software is provided "as is" without warranty of any kind, express or implied. The authors and contributors shall not be liable for any damages or losses arising from the use of this software. Volcanic eruption forecasting is inherently uncertain, and this software should be used only as a research tool, not for operational volcano monitoring or public safety decisions.

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

---

**Version:** 0.1.0
**Status:** Active Development
**Last Updated:** 2026-03-09
