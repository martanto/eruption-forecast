# eruption-forecast

A comprehensive Python package for volcanic eruption forecasting using seismic data analysis. Process raw seismic tremor data, extract time-series features, train machine learning models, and predict volcanic eruptions based on seismic patterns.

## Features

- **Tremor Calculation**: Process raw seismic data (SDS/FDSN) to calculate RSAM and DSAR metrics across multiple frequency bands
- **Label Building**: Generate training labels from eruption dates with configurable forecast horizons
- **Feature Extraction**: Extract 700+ time-series features using tsfresh for machine learning
- **Enhanced Feature Selection**: Three-method feature selection — tsfresh statistical, RandomForest permutation importance, or combined two-stage
- **Model Training**: Train multiple classifier types (Random Forest, Gradient Boosting, SVM, Logistic Regression, Neural Networks, Ensembles) across multiple random seeds
- **Model Evaluation**: Comprehensive evaluation with ROC curves, precision-recall curves, confusion matrices, and feature importance
- **Multi-processing**: Parallel processing for faster tremor calculations and model training
- **Logging**: Built-in logging with loguru for debugging and monitoring

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

Here's a complete example from raw seismic data to eruption predictions:

```python
from eruption_forecast import ForecastModel
from eruption_forecast.model.model_evaluator import ModelEvaluator
import joblib
import pandas as pd

# ========== Complete Pipeline via ForecastModel ==========
fm = ForecastModel(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=2,
    volcano_id="LEWOTOBI",
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
    classifier="rf",
    random_state=0,
    total_seed=100,
    number_of_significant_features=20,
    sampling_strategy=0.75,
    verbose=True,
)
```

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
- Default bands: (0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz

### 2. Build Training Labels

Create binary labels (erupted/not erupted) based on known eruption dates.

```python
from eruption_forecast import LabelBuilder

labels = LabelBuilder(
    start_date="2020-01-01",
    end_date="2020-12-31",
    window_size=1,           # 1-day windows
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
- Windows within `day_to_forecast` days before an eruption: `is_erupted = 1`
- All other windows: `is_erupted = 0`
- Example: `day_to_forecast=2` with an eruption on 2020-03-15 labels windows from 2020-03-13 to 2020-03-15 as positive

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

**Feature types:**
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

### 6. Train Models with Multiple Classifiers

Train machine learning models with automatic feature selection across multiple random seeds.

```python
from eruption_forecast import TrainModel

# Random Forest (default)
trainer = TrainModel(
    extracted_features_csv="output/features/all_features.csv",
    label_features_csv="output/features/label_features.csv",
    output_dir="output/trainings_rf",
    classifier="rf",
    cv_strategy="timeseries",  # Prevents data leakage in temporal data
    cv_splits=5,
    n_jobs=4,
    verbose=True,
)

trainer.train(
    random_state=0,
    total_seed=100,
    number_of_significant_features=20,
    sampling_strategy=0.75,  # Under-sample majority class
    overwrite=False,
)

# Gradient Boosting (recommended for imbalanced data)
trainer = TrainModel(
    extracted_features_csv="output/features/all_features.csv",
    label_features_csv="output/features/label_features.csv",
    output_dir="output/trainings_gb",
    classifier="gb",
    cv_strategy="stratified",
    n_jobs=4,
)
trainer.train(random_state=0, total_seed=100, number_of_significant_features=20)

# VotingClassifier ensemble (best accuracy)
trainer = TrainModel(
    extracted_features_csv="output/features/all_features.csv",
    label_features_csv="output/features/label_features.csv",
    output_dir="output/trainings_voting",
    classifier="voting",
    n_jobs=4,
)
trainer.train(random_state=0, total_seed=50, number_of_significant_features=20)
```

**Supported classifiers:**
- `rf`: Random Forest (balanced, robust, default)
- `gb`: Gradient Boosting (excellent for imbalanced data)
- `svm`: Support Vector Machine (high-dimensional data)
- `lr`: Logistic Regression (interpretable, fast)
- `nn`: Neural Network (MLP, complex patterns)
- `dt`: Decision Tree (interpretable)
- `knn`: K-Nearest Neighbors (simple baseline)
- `nb`: Gaussian Naive Bayes (fast baseline)
- `voting`: VotingClassifier ensemble (best accuracy)

**Cross-validation strategies:**
- `shuffle`: StratifiedShuffleSplit (random splits with stratification)
- `stratified`: StratifiedKFold (preserves class distribution)
- `timeseries`: TimeSeriesSplit (for temporal data, prevents data leakage)

### 7. Analyze Training Results

```python
import pandas as pd

# Load aggregated metrics
metrics = pd.read_csv("output/trainings/all_metrics.csv")

# View summary statistics
summary = pd.read_csv("output/trainings/metrics_summary.csv", index_col=0)
print(summary[["balanced_accuracy", "f1_score", "precision", "recall"]])

# Find best performing seed
best_seed = metrics.loc[metrics["balanced_accuracy"].idxmax()]
print(f"Best seed: {best_seed['random_state']}")
print(f"Balanced Accuracy: {best_seed['balanced_accuracy']:.4f}")

# Load significant features
sig_features = pd.read_csv("output/trainings/significant_features.csv")
print(sig_features.head(10))
```

### 8. Model Evaluation and Visualization

```python
from eruption_forecast.model.model_evaluator import ModelEvaluator
import joblib
import pandas as pd

# Load trained model and test data
model = joblib.load("output/trainings/models/00042.pkl")
X_test = pd.read_csv("output/features/all_features.csv", index_col=0)
y_test = pd.read_csv("output/features/label_features.csv")["is_erupted"]

# Initialize evaluator
evaluator = ModelEvaluator(
    model=model,
    X_test=X_test,
    y_test=y_test,
    model_name="gradient_boosting_eruption",
    output_dir="output/evaluation",
)

# Get metrics
metrics = evaluator.get_metrics()
print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")
print(evaluator.summary())

# Generate all plots
evaluator.plot_all(feature_names=X_test.columns.tolist(), top_n_features=15)

# Export everything
paths = evaluator.export_all()
```

**ModelEvaluator outputs:**

Plots: `roc_curve.png`, `precision_recall_curve.png`, `confusion_matrix.png`, `feature_importances.png`, `learning_curve.png`, `metrics_summary.png`

Files: `{model_name}.joblib`, `metrics.csv`, `metrics.json`, `classification_report.txt`, `confusion_matrix.csv`, `feature_importances.csv`

## Advanced Features

### Using ForecastModel (Complete Pipeline)

The `ForecastModel` class orchestrates the entire pipeline in one go:

```python
from eruption_forecast.model.forecast_model import ForecastModel

model = ForecastModel(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=1,
    volcano_id="LEWOTOBI",
    n_jobs=4,
)

model.calculate(source="sds", sds_dir="/data/sds")
model.build_label(
    window_step=12,
    window_step_unit="hours",
    day_to_forecast=2,
    eruption_dates=["2025-03-20", "2025-06-15"],
)
model.extract_features()
model.train(
    random_state=0,
    total_seed=100,
    number_of_significant_features=20,
    classifier="gb",
    cv_strategy="timeseries",
)
```

### Comparing Multiple Classifiers

```python
import pandas as pd

classifiers = {
    "rf": "output/trainings_rf/all_metrics.csv",
    "gb": "output/trainings_gb/all_metrics.csv",
    "lr": "output/trainings_lr/all_metrics.csv",
    "voting": "output/trainings_voting/all_metrics.csv",
}

results = []
for name, path in classifiers.items():
    df = pd.read_csv(path)
    results.append({
        "classifier": name,
        "mean_balanced_acc": df["balanced_accuracy"].mean(),
        "std_balanced_acc": df["balanced_accuracy"].std(),
        "mean_f1": df["f1_score"].mean(),
    })

comparison = pd.DataFrame(results).sort_values("mean_balanced_acc", ascending=False)
print(comparison)
```

### Custom Hyperparameter Grids

```python
from eruption_forecast.model.classifier_model import ClassifierModel

clf = ClassifierModel("rf", random_state=42)

# Override default grid
clf.grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [15, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2"],
}
```

## Configuration

### Logging

```python
from eruption_forecast.logger import set_log_level, set_log_directory

set_log_level("DEBUG")  # Options: DEBUG, INFO, WARNING, ERROR
set_log_directory("/custom/logs")
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

## Output Directory Structure

```
output/
└── {network}.{station}.{location}.{channel}/
    ├── tremor/
    │   ├── tmp/                           # Temporary daily files
    │   └── tremor_*.csv                   # Final tremor data
    ├── features/
    │   ├── extracted/
    │   │   ├── all_features_*.csv         # All tsfresh features per column
    │   │   └── relevant_features_*.csv    # Relevant features per column
    │   ├── tremor_matrix_unified_*.csv    # Aligned tremor matrix
    │   ├── tremor_matrix_per_method/      # Per-column tremor matrices
    │   ├── all_features_*.csv             # Concatenated all features
    │   └── label_features_*.csv           # Labels aligned with features
    └── trainings/
        ├── significant_features/          # Top-N features per seed
        │   ├── 00000.csv
        │   └── ...
        ├── models/                        # Trained models
        │   ├── 00000.pkl
        │   └── ...
        ├── metrics/                       # Per-seed metrics
        │   ├── 00000.json
        │   └── ...
        ├── significant_features.csv       # Aggregated features
        ├── all_metrics.csv                # All seed metrics
        └── metrics_summary.csv            # Mean/std statistics
```

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
- joblib
- matplotlib
- seaborn
- loguru

### Development Dependencies
- black (code formatting)
- ruff (linting)
- pyrefly (type checking)
- isort (import sorting)
- pytest (testing)

## Development

### Code Quality Tools

```bash
# Format code
uv run black src/

# Lint code
uv run ruff check src/

# Type checking
uv run pyrefly check src/

# Sort imports
uv run isort src/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/eruption_forecast

# Run specific test
pytest tests/test_train_model_fixed.py
```

## Documentation

- **CLAUDE.md**: Architecture documentation and development guidelines
- **SUMMARY.md**: Technical summary with ML workflow analysis and model comparison

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Complete pipeline workflows
- Custom frequency band configurations
- Feature extraction strategies
- Model training and comparison
- Evaluation and visualization techniques

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b claude/my-feature`)
3. Make changes with tests
4. Ensure code passes linting and type checks
5. Update documentation
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use Google-style docstrings
- Include type hints for all functions
- Write unit tests for new features

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
- [uv](https://docs.astral.sh/uv/) for package management

---

**Version:** 0.2.0
**Status:** Active Development
**Last Updated:** 2026-02-11
