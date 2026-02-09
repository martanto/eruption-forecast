# eruption-forecast

A comprehensive Python package for volcanic eruption forecasting using seismic data analysis. Process seismic tremor data, extract time-series features, train machine learning models, and predict volcanic eruptions based on seismic patterns.

## Features

- **🌊 Tremor Calculation**: Process raw seismic data (SDS/FDSN) to calculate RSAM and DSAR metrics across multiple frequency bands
- **🏷️ Label Building**: Generate training labels from eruption dates with configurable forecast horizons
- **🔬 Feature Extraction**: Extract 700+ time-series features using tsfresh for machine learning
- **🤖 Model Training**: Train multiple classifier types (Random Forest, Gradient Boosting, SVM, Logistic Regression, Neural Networks, Ensembles)
- **📊 Model Evaluation**: Comprehensive evaluation with ROC curves, precision-recall curves, confusion matrices, and feature importance
- **⚡ Multi-processing**: Parallel processing for faster tremor calculations and model training
- **📝 Logging**: Built-in logging with loguru for debugging and monitoring

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
from eruption_forecast import (
    CalculateTremor,
    LabelBuilder,
    FeaturesBuilder,
    TrainModel,
)
from eruption_forecast.model.model_evaluator import ModelEvaluator
import joblib

# ========== 1. Calculate Tremor from Seismic Data ==========
print("Step 1: Calculating tremor metrics...")
tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    n_jobs=4  # Use 4 CPU cores for parallel processing
).from_sds(sds_dir="/path/to/sds/data").run()

print(f"✓ Tremor data saved to: {tremor.csv}")
# Output: tremor_VG.OJN.00.EHZ_2025-01-01_2025-12-31.csv
# Contains: rsam_f0, rsam_f1, rsam_f2, rsam_f3, rsam_f4, dsar_f0-f1, ...

# ========== 2. Build Labels for Training ==========
print("\nStep 2: Building training labels...")
labels = LabelBuilder(
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=1,           # 1-day windows
    window_step=12,          # Step by 12 hours
    window_step_unit="hours",
    day_to_forecast=2,       # Label 2 days before eruption as positive
    eruption_dates=[
        "2025-03-20",
        "2025-06-15",
        "2025-09-10"
    ],
    volcano_id="LEWOTOBI"
).build()

print(f"✓ Labels saved to: {labels.csv}")
# Output: label_2025-01-01_2025-12-31_ws-1_step-12-hours_dtf-2.csv
# Contains: id, is_erupted (0 or 1)

# ========== 3. Extract Features ==========
print("\nStep 3: Extracting time-series features...")
features = FeaturesBuilder(
    df_tremor=tremor.df,
    df_label=labels.df,
    output_dir="output/features",
    window_size=1
).build()

print(f"✓ Features saved to: {features.csv}")
# Extracts 700+ features per tremor column using tsfresh

# ========== 4. Train Model with Multiple Seeds ==========
print("\nStep 4: Training classifier (Gradient Boosting)...")
trainer = TrainModel(
    features_csv=features.csv,
    label_csv=labels.csv,
    output_dir="output/trainings",
    classifier="gb",              # Gradient Boosting
    cv_strategy="timeseries",     # TimeSeriesSplit for temporal data
    cv_splits=5,
    n_jobs=4,
    verbose=True
)

# Train across 100 random seeds for robust feature selection
trainer.train(
    random_state=42,
    total_seed=100,
    number_of_significant_features=20,
    sampling_strategy=0.75,  # Under-sample majority class
    overwrite=False
)

print(f"✓ Training complete!")
print(f"  - Trained models: output/trainings/models/")
print(f"  - Metrics: output/trainings/metrics/")
print(f"  - Aggregated metrics: output/trainings/all_metrics.csv")

# ========== 5. Load Best Model and Evaluate ==========
print("\nStep 5: Evaluating best model...")

# Load the best performing model (e.g., seed 00042)
best_model = joblib.load("output/trainings/models/00042.pkl")

# Load test data (in practice, use a held-out test set)
import pandas as pd
X_test = pd.read_csv(features.csv, index_col=0)
y_test = pd.read_csv(labels.csv)["is_erupted"]

# Evaluate with comprehensive metrics
evaluator = ModelEvaluator(
    model=best_model,
    X_test=X_test,
    y_test=y_test,
    model_name="gradient_boosting_eruption",
    output_dir="output/evaluation"
)

# Get metrics
metrics = evaluator.get_metrics()
print(f"\n📊 Model Performance:")
print(f"  - ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"  - Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
print(f"  - F1 Score: {metrics['f1']:.3f}")
print(f"  - Precision: {metrics['precision']:.3f}")
print(f"  - Recall: {metrics['recall']:.3f}")

# Generate all plots
print("\nGenerating evaluation plots...")
evaluator.plot_all(feature_names=X_test.columns.tolist(), top_n_features=15)
print(f"✓ Plots saved to: output/evaluation/")

# Export everything
evaluator.export_all()
print(f"✓ Model and metrics exported!")
```

## Step-by-Step Usage Guide

### 1. Calculate Tremor Metrics

Process raw seismic data to calculate RSAM (amplitude) and DSAR (ratios) across frequency bands.

```python
from eruption_forecast import CalculateTremor

# Basic usage with SDS data
tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-31",
    n_jobs=4
).from_sds(sds_dir="/data/sds").run()

# Custom frequency bands
tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-31"
).change_freq_bands([
    (0.1, 1.0),   # Low frequency
    (1.0, 5.0),   # Mid frequency
    (5.0, 10.0)   # High frequency
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
        "2020-09-10"
    ],
    volcano_id="VOLCANO_001"
).build()

# Access the DataFrame
print(labels.df.head())
# Columns: id, is_erupted

# Check label distribution
print(f"Positive labels: {(labels.df['is_erupted'] == 1).sum()}")
print(f"Negative labels: {(labels.df['is_erupted'] == 0).sum()}")
```

**Label Logic:**
- Windows within `day_to_forecast` days before an eruption: `is_erupted = 1`
- All other windows: `is_erupted = 0`
- Example: If `day_to_forecast=2` and eruption on 2020-03-15, windows from 2020-03-13 to 2020-03-15 are labeled as 1

### 3. Extract Time-Series Features

Extract 700+ statistical features from tremor data using tsfresh.

```python
from eruption_forecast import FeaturesBuilder

# Build feature matrix
features = FeaturesBuilder(
    df_tremor=tremor.df,
    df_label=labels.df,
    output_dir="output/features",
    window_size=1,
    n_jobs=4
).build()

# Extract features using tsfresh
extracted = features.extract_features(
    n_jobs=4,
    overwrite=False
)

print(f"Features shape: {extracted.df.shape}")
# Example: (5000 windows, 7000+ features)

# Get relevant features only (optional)
relevant = features.extract_relevant_features(
    n_jobs=4,
    overwrite=False
)

print(f"Relevant features: {relevant.df.shape}")
# Example: (5000 windows, 200 features)
```

**Feature Types:**
- Statistical: mean, median, std, variance, min, max, quantiles
- Time-domain: autocorrelation, partial autocorrelation
- Frequency-domain: FFT coefficients, spectral entropy
- Complexity: approximate entropy, sample entropy
- Peaks: number of peaks, peak positions

### 4. Train Models with Multiple Classifiers

Train machine learning models with automatic feature selection across multiple random seeds.

#### Random Forest (Default)

```python
from eruption_forecast import TrainModel

trainer = TrainModel(
    features_csv="output/features/extracted_features.csv",
    label_csv="output/features/labels.csv",
    output_dir="output/trainings_rf",
    classifier="rf",              # Random Forest
    cv_strategy="shuffle",        # StratifiedShuffleSplit
    cv_splits=5,
    n_jobs=4,
    verbose=True
)

trainer.train(
    random_state=42,
    total_seed=100,               # 100 random seeds
    number_of_significant_features=20,
    sampling_strategy=0.75,       # 75% under-sampling
    overwrite=False
)
```

#### Gradient Boosting (Recommended for Imbalanced Data)

```python
trainer = TrainModel(
    features_csv="output/features/extracted_features.csv",
    label_csv="output/features/labels.csv",
    output_dir="output/trainings_gb",
    classifier="gb",              # Gradient Boosting
    cv_strategy="stratified",     # StratifiedKFold
    cv_splits=5,
    n_jobs=4
)

trainer.train(
    random_state=42,
    total_seed=100,
    number_of_significant_features=20,
    sampling_strategy=0.75
)
```

#### Logistic Regression with TimeSeriesSplit

```python
trainer = TrainModel(
    features_csv="output/features/extracted_features.csv",
    label_csv="output/features/labels.csv",
    output_dir="output/trainings_lr",
    classifier="lr",              # Logistic Regression
    cv_strategy="timeseries",     # TimeSeriesSplit (prevents data leakage)
    cv_splits=5,
    n_jobs=4
)

trainer.train(
    random_state=42,
    total_seed=100,
    number_of_significant_features=20,
    sampling_strategy=0.75
)
```

#### VotingClassifier Ensemble (Best Accuracy)

```python
trainer = TrainModel(
    features_csv="output/features/extracted_features.csv",
    label_csv="output/features/labels.csv",
    output_dir="output/trainings_voting",
    classifier="voting",          # Ensemble of RF + GB + LR + SVM
    cv_strategy="stratified",
    cv_splits=5,
    n_jobs=4
)

trainer.train(
    random_state=42,
    total_seed=50,                # Fewer seeds (ensemble is slower)
    number_of_significant_features=20,
    sampling_strategy=0.75
)
```

**Supported Classifiers:**
- `rf`: Random Forest (balanced, robust, default)
- `gb`: Gradient Boosting (excellent for imbalanced data)
- `svm`: Support Vector Machine (high-dimensional data)
- `lr`: Logistic Regression (interpretable, fast)
- `nn`: Neural Network (MLP, complex patterns)
- `dt`: Decision Tree (interpretable)
- `knn`: K-Nearest Neighbors (simple)
- `nb`: Gaussian Naive Bayes (fast baseline)
- `voting`: VotingClassifier ensemble (best accuracy)

**Cross-Validation Strategies:**
- `shuffle`: StratifiedShuffleSplit (default, random splits with stratification)
- `stratified`: StratifiedKFold (preserves class distribution)
- `timeseries`: TimeSeriesSplit (for temporal data, prevents data leakage)

### 5. Analyze Training Results

```python
import pandas as pd

# Load aggregated metrics
metrics = pd.read_csv("output/trainings/all_metrics.csv")

# View summary statistics
summary = pd.read_csv("output/trainings/metrics_summary.csv", index_col=0)
print(summary[["balanced_accuracy", "f1_score", "precision", "recall"]])

# Find best performing seed
best_seed = metrics.loc[metrics["balanced_accuracy"].idxmax()]
print(f"\nBest seed: {best_seed['random_state']}")
print(f"Balanced Accuracy: {best_seed['balanced_accuracy']:.4f}")
print(f"Best params: {best_seed['best_params']}")

# Load significant features
sig_features = pd.read_csv("output/trainings/significant_features.csv")
print(f"\nTop 10 features:")
print(sig_features.head(10))
```

### 6. Model Evaluation and Visualization

Comprehensive model evaluation with metrics, plots, and export capabilities.

```python
from eruption_forecast.model.model_evaluator import ModelEvaluator
import joblib
import pandas as pd

# Load trained model
model = joblib.load("output/trainings/models/00042.pkl")

# Load test data
X_test = pd.read_csv("output/features/extracted_features.csv", index_col=0)
y_test = pd.read_csv("output/features/labels.csv")["is_erupted"]

# Optional: Load training data for learning curves
X_train = pd.read_csv("output/features/extracted_features_train.csv", index_col=0)
y_train = pd.read_csv("output/features/labels_train.csv")["is_erupted"]

# Initialize evaluator
evaluator = ModelEvaluator(
    model=model,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,      # Optional
    y_train=y_train,      # Optional
    model_name="gradient_boosting",
    output_dir="output/evaluation"
)

# ========== Get Metrics ==========
metrics = evaluator.get_metrics()
print("\n📊 Model Performance Metrics:")
print(f"  Accuracy: {metrics['accuracy']:.3f}")
print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.3f}")
print(f"  Precision: {metrics['precision']:.3f}")
print(f"  Recall: {metrics['recall']:.3f}")
print(f"  F1 Score: {metrics['f1']:.3f}")
print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"  PR-AUC: {metrics['pr_auc']:.3f}")

# Print full summary
print(evaluator.summary())

# ========== Generate Individual Plots ==========

# 1. ROC Curve
evaluator.plot_roc_curve()
# Saved to: output/evaluation/roc_curve.png

# 2. Precision-Recall Curve
evaluator.plot_precision_recall_curve()
# Saved to: output/evaluation/precision_recall_curve.png

# 3. Confusion Matrix
evaluator.plot_confusion_matrix(normalize="true")
# Saved to: output/evaluation/confusion_matrix.png

# 4. Feature Importances (top 15)
evaluator.plot_feature_importances(
    feature_names=X_test.columns.tolist(),
    top_n=15
)
# Saved to: output/evaluation/feature_importances.png

# 5. Learning Curve (if X_train/y_train provided)
evaluator.plot_learning_curve(cv=5)
# Saved to: output/evaluation/learning_curve.png

# 6. Metrics Summary Bar Chart
evaluator.plot_metrics_summary()
# Saved to: output/evaluation/metrics_summary.png

# ========== Generate All Plots at Once ==========
evaluator.plot_all(
    feature_names=X_test.columns.tolist(),
    top_n_features=20,
    cv=5
)
print("✓ All plots saved to: output/evaluation/")

# ========== Export Results ==========

# Export model
evaluator.export_model()
# Saved to: output/evaluation/gradient_boosting.joblib

# Export metrics as CSV
evaluator.export_metrics(format="csv")
# Saved to: output/evaluation/metrics.csv

# Export metrics as JSON
evaluator.export_metrics(format="json")
# Saved to: output/evaluation/metrics.json

# Export classification report
evaluator.export_classification_report()
# Saved to: output/evaluation/classification_report.txt

# Export confusion matrix
evaluator.export_confusion_matrix()
# Saved to: output/evaluation/confusion_matrix.csv

# Export feature importances
evaluator.export_feature_importances(feature_names=X_test.columns.tolist())
# Saved to: output/evaluation/feature_importances.csv

# ========== Export Everything at Once ==========
paths = evaluator.export_all()
print("\n✓ Exported files:")
for key, path in paths.items():
    print(f"  - {key}: {path}")
```

**ModelEvaluator Outputs:**

Plots:
- `roc_curve.png`: ROC curve with AUC score
- `precision_recall_curve.png`: Precision-Recall curve with AP score
- `confusion_matrix.png`: Confusion matrix heatmap
- `feature_importances.png`: Top-N feature importance bar chart
- `learning_curve.png`: Training/validation score vs. training size
- `metrics_summary.png`: Bar chart of all metrics

Files:
- `{model_name}.joblib`: Trained model (joblib format)
- `metrics.csv`/`metrics.json`: All evaluation metrics
- `classification_report.txt`: Detailed classification report
- `confusion_matrix.csv`: Confusion matrix values
- `feature_importances.csv`: Feature importance scores

## Advanced Features

### Using ForecastModel (Complete Pipeline)

The `ForecastModel` class orchestrates the entire pipeline in one go:

```python
from eruption_forecast.model.forecast_model import ForecastModel

# Initialize pipeline
model = ForecastModel(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=1,
    volcano_id="LEWOTOBI",
    n_jobs=4
)

# 1. Calculate tremor
model.calculate(source="sds", sds_dir="/data/sds")

# 2. Build labels
model.build_label(
    window_step=12,
    window_step_unit="hours",
    day_to_forecast=2,
    eruption_dates=["2025-03-20", "2025-06-15"]
)

# 3. Build and extract features
model.build_features()
model.extract_features()

# 4. Train model (with dynamic classifier)
model.train(
    random_state=42,
    total_seed=100,
    number_of_significant_features=20,
    classifier="gb",           # Use Gradient Boosting
    cv_strategy="timeseries"
)

print("✓ Complete pipeline finished!")
```

### Comparing Multiple Classifiers

```python
import pandas as pd

classifiers = {
    "rf": "output/trainings_rf/all_metrics.csv",
    "gb": "output/trainings_gb/all_metrics.csv",
    "lr": "output/trainings_lr/all_metrics.csv",
    "voting": "output/trainings_voting/all_metrics.csv"
}

results = []
for name, path in classifiers.items():
    df = pd.read_csv(path)
    results.append({
        "classifier": name,
        "mean_balanced_acc": df["balanced_accuracy"].mean(),
        "std_balanced_acc": df["balanced_accuracy"].std(),
        "mean_f1": df["f1_score"].mean(),
        "std_f1": df["f1_score"].std()
    })

comparison = pd.DataFrame(results).sort_values("mean_balanced_acc", ascending=False)
print("\n📊 Classifier Comparison:")
print(comparison)

# Output:
#   classifier  mean_balanced_acc  std_balanced_acc  mean_f1  std_f1
# 0     voting              0.872             0.051    0.623   0.089
# 1         gb              0.854             0.062    0.598   0.102
# 2         rf              0.841             0.071    0.571   0.115
# 3         lr              0.792             0.083    0.512   0.128
```

### Custom Hyperparameter Grids

```python
from eruption_forecast.model.classifier_model import ClassifierModel

# Create classifier with custom grid
clf = ClassifierModel("rf", random_state=42)

# Override default grid
clf.grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [15, 20, 30, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

# Use in TrainModel (would need manual integration)
# Or use directly with GridSearchCV
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf.model, clf.grid, cv=5)
```

## Configuration

### Logging

```python
from eruption_forecast.logger import set_log_level, set_log_directory

# Set log level
set_log_level("DEBUG")  # Options: DEBUG, INFO, WARNING, ERROR

# Change log directory
set_log_directory("/custom/logs")
```

### Custom Frequency Bands

```python
tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-31"
).change_freq_bands([
    (0.1, 0.5),   # Very low frequency
    (0.5, 2.0),   # Low frequency
    (2.0, 5.0),   # Mid frequency
    (5.0, 10.0),  # High frequency
    (10.0, 20.0)  # Very high frequency
]).from_sds(sds_dir="/data/sds").run()
```

## Output Directory Structure

```
output/
├── VG.OJN.00.EHZ/
│   ├── tremor/
│   │   ├── tmp/                           # Temporary daily files
│   │   └── tremor_*.csv                   # Final tremor data
│   ├── features/
│   │   ├── extracted_features_*.csv       # All tsfresh features
│   │   ├── relevant_features_*.csv        # Relevant features only
│   │   └── label_features_*.csv           # Labels
│   └── trainings/
│       ├── significant_features/          # Top-N features per seed
│       │   ├── 00000.csv
│       │   ├── 00001.csv
│       │   └── ...
│       ├── models/                        # Trained models
│       │   ├── 00000.pkl
│       │   ├── 00001.pkl
│       │   └── ...
│       ├── metrics/                       # Per-seed metrics
│       │   ├── 00000.json
│       │   ├── 00001.json
│       │   └── ...
│       ├── significant_features.csv       # Aggregated features
│       ├── all_metrics.csv                # All seed metrics
│       └── metrics_summary.csv            # Mean/std statistics
├── evaluation/
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importances.png
│   ├── learning_curve.png
│   ├── metrics_summary.png
│   ├── model_name.joblib
│   ├── metrics.csv
│   ├── metrics.json
│   ├── classification_report.txt
│   ├── confusion_matrix.csv
│   └── feature_importances.csv
└── figures/                               # Custom plots
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

- **CLAUDE.md**: Comprehensive architecture documentation and development guidelines
- **SUMMARY.md**: Technical summary with ML workflow analysis and model comparison
- **DOCSTRING_ANALYSIS.md**: Documentation quality report

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Complete pipeline workflows
- Custom frequency band configurations
- Feature extraction strategies
- Model training and comparison
- Evaluation and visualization techniques

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure code passes linting and type checks
5. Update documentation
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use Google-style docstrings
- Include type hints for all functions
- Write unit tests for new features
- Maintain > 90% test coverage

## License

MIT License - see LICENSE file for details.

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
**Last Updated:** 2026-02-09
