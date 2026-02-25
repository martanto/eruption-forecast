# Visualization & Plotting

> Back to [README](../README.md)

The `plots/` module provides publication-quality visualization functions with consistent Nature/Science journal styling. All plots use colorblind-safe palettes, clean typography, and high DPI output suitable for papers and presentations.

### Tremor Time-Series Plots

#### plot_tremor()

Visualize tremor metrics (RSAM/DSAR) as multi-panel time-series plots.

```python
from eruption_forecast.plots.tremor_plots import plot_tremor
import pandas as pd

# Load tremor data
df = pd.read_csv("output/VG.OJN.00.EHZ/tremor/tremor.csv", index_col=0, parse_dates=True)

# Basic tremor plot
plot_tremor(
    df=df,
    figure_dir="output/figures",
    dpi=150,
)

# Custom interval and selected columns
plot_tremor(
    df=df,
    interval=6,
    interval_unit="hours",
    selected_columns=["rsam_f2", "rsam_f3", "dsar_f2-f3"],
    figure_dir="output/figures",
    filename="tremor_selected",
    dpi=300,
    verbose=True,
)
```

**Key Parameters:**
- `df` - Tremor DataFrame with datetime index
- `interval` - X-axis tick interval (default: 1)
- `interval_unit` - "hours" or "days" (default: "hours")
- `selected_columns` - Subset of columns to plot (default: all)
- `dpi` - Resolution (default: 150)

### Feature Importance Plots

#### plot_significant_features()

Visualize feature importance or p-values as horizontal bar charts.

```python
from eruption_forecast.plots.feature_plots import plot_significant_features
import pandas as pd

# From DataFrame
df_features = pd.read_csv("features.csv")
plot_significant_features(
    features=df_features,
    number_of_features=50,
    top_features=20,
    output_dir="output/figures",
    filename="feature_importance",
    dpi=150,
)

# From CSV file directly
plot_significant_features(
    features="path/to/features.csv",
    number_of_features=30,
    top_features=10,
    values_column="importance",  # or "p_values"
    output_dir="output/figures",
)
```

**Key Parameters:**
- `features` - DataFrame or CSV path with feature data
- `number_of_features` - Total features to display (default: 50)
- `top_features` - Number highlighted with darker color (default: 20)
- `values_column` - Column name for values (auto-detected if None)

### Model Evaluation Plots

The `ModelEvaluator` class provides 7 evaluation plot types for single-seed and aggregate (ensemble-level) analysis.

#### Available Plot Types

**Single-seed** (instance methods on `ModelEvaluator`):
1. **Confusion Matrix** - Classification performance breakdown
2. **ROC Curve** - True/false positive rate tradeoff with AUC score
3. **Precision-Recall Curve** - Precision/recall tradeoff with Average Precision
4. **Threshold Analysis** - Metrics vs decision threshold
5. **Feature Importance** - Top contributing features to predictions
6. **Calibration Curve** - Predicted vs actual probabilities (reliability diagram)
7. **Prediction Distribution** - Score distributions by class (histogram + KDE)
8. **SHAP Summary** - Beeswarm plot showing feature impact direction and magnitude

**Aggregate across all seeds** (`MultiModelEvaluator`, requires `train_and_evaluate()`):
- `evaluator.plot_all()` — runs all 10 aggregate plots at once
- `evaluator.plot_roc()` — mean ROC ± std band across seeds
- `evaluator.plot_precision_recall()` — mean PR curve ± std band
- `evaluator.plot_calibration()` — mean calibration ± std band
- `evaluator.plot_prediction_distribution()` — pooled KDE by class
- `evaluator.plot_confusion_matrix()` — summed confusion matrix
- `evaluator.plot_threshold_analysis()` — mean metrics vs threshold ± std
- `evaluator.plot_feature_importance()` — mean importance ± std error bars
- `evaluator.plot_shap_summary()` — mean |SHAP| bar chart across seeds
- `evaluator.plot_seed_stability()` — violin plot of a metric across seeds
- `evaluator.plot_frequency_band_contribution()` — feature counts per seismic band

**Aggregate metrics (from JSON files, no plots)**:
- `evaluator.get_aggregate_metrics()` — returns mean/std/min/max per metric
- `evaluator.get_metrics_list()` — returns raw per-seed metrics as list of dicts
- `evaluator.save_aggregate_metrics()` — saves summary to CSV

#### How Aggregation Works

Each seed produces its own 80/20 train/test split; the held-out `X_test` and `y_test` are saved to `tests/` during training. Aggregate plots load every seed's test data and model, then apply the following strategy per plot type:

| Plot | Aggregation method |
|---|---|
| **ROC Curve** | Each seed's TPR is interpolated onto a shared FPR grid (200 points, 0→1). All curves are stacked into an `(n_seeds × 200)` matrix. **Mean** TPR is the bold line; **±1 std** is the shaded band. Mean AUC ± std is shown in the legend. |
| **Precision-Recall Curve** | Each seed's precision is interpolated onto a shared recall grid (200 points). **Mean** precision is the bold line; **±1 std** is the shaded band. Mean AP ± std in legend. |
| **Threshold Analysis** | For each of 101 thresholds (0→1), F1, precision, recall, and balanced accuracy are computed per seed. Each metric is stacked `(n_seeds × 101)` and the **mean** curve is plotted bold with **±1 std** shaded bands. |
| **Calibration Curve** | Each seed's calibration curve is interpolated onto a shared probability grid (n_bins points). **Mean** fraction of positives is the bold line; **±1 std** is the shaded band. |
| **Prediction Distribution** | Predicted probabilities from all seeds are **pooled** (concatenated) separately for class 0 and class 1, then a single KDE is computed over the full pool for each class. |
| **Confusion Matrix** | Raw confusion matrices are **summed** across all seeds (total TP/TN/FP/FN across the entire ensemble). Optional normalization is applied after summing. |
| **Feature Importance** | Importances are stacked `(n_seeds × n_features)`. **Mean** importance per feature is the bar length; **±1 std** is the error bar. Top-N features selected by mean importance. |

#### Single-Seed Usage Example

```python
from eruption_forecast import ModelEvaluator

# Load from files
evaluator = ModelEvaluator.from_files(
    model_path="output/trainings/models/00042.pkl",
    X_test="output/trainings/tests/00042_X_test.csv",
    y_test="output/trainings/tests/00042_y_test.csv",
    selected_features=["feat_a", "feat_b"],  # optional
    model_name="xgb_seed_42",
    output_dir="output/eval",
)

# Generate all 7 plots at once
evaluator.plot_all()

# Or generate individual plots
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curve()
evaluator.plot_precision_recall_curve()
evaluator.plot_threshold_analysis()
evaluator.plot_feature_importance()
evaluator.plot_calibration()
evaluator.plot_prediction_distribution()
evaluator.plot_shap_summary(max_display=20)

# Save metrics to JSON
path = evaluator.save_metrics()
```

All plots are saved to `output_dir` with publication-quality styling.

#### Multi-Seed Usage Example

```python
from eruption_forecast import MultiModelEvaluator

base = "output/trainings/evaluations/xgb-classifier/stratified-shuffle-split"
trained_model_csv = f"{base}/trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv"

# Plots from registry CSV
evaluator = MultiModelEvaluator(trained_model_csv=trained_model_csv)
figs = evaluator.plot_all(dpi=150, show_individual=True)

# Aggregate stats from JSON metrics files
evaluator = MultiModelEvaluator(metrics_dir=f"{base}/metrics")
summary = evaluator.get_aggregate_metrics()
evaluator.save_aggregate_metrics()
print(summary.loc[["f1_score", "roc_auc", "balanced_accuracy"]])
```

### Classifier Comparison Heatmap

Compare performance metrics across multiple classifiers in a single viridis heatmap. Each cell shows `mean ± std` over seeds; rows are sorted by mean F1 descending.

```python
from eruption_forecast import MultiModelEvaluator
from eruption_forecast.plots import plot_classifier_comparison

base = "output/trainings/evaluations"

# Load metrics from each classifier's evaluator
metrics_by_clf = {}
for clf in ["xgb", "rf", "gb"]:
    ev = MultiModelEvaluator(
        metrics_dir=f"{base}/{clf}-classifier/stratified-shuffle-split/metrics"
    )
    metrics_by_clf[clf] = ev.get_metrics_list()

# Plot comparison heatmap
fig, summary_df = plot_classifier_comparison(
    metrics_by_classifier=metrics_by_clf,
    metrics_to_show=["balanced_accuracy", "f1_score", "precision", "recall", "roc_auc", "pr_auc"],
    figsize=(12, 5),
    dpi=150,
)
fig.savefig("classifier_comparison.png", bbox_inches="tight")

# summary_df has MultiIndex (classifier, metric) with columns mean/std
print(summary_df.loc[("xgb", "f1_score")])
```

### SHAP Explainability Plots

Understand *why* the model makes predictions using SHAP (SHapley Additive exPlanations). The beeswarm plot shows both the direction (positive/negative) and magnitude of each feature's contribution.

```python
from eruption_forecast import ModelEvaluator, MultiModelEvaluator

# Single-seed beeswarm plot
evaluator = ModelEvaluator.from_files(
    model_path="output/.../models/00042.pkl",
    X_test="output/.../tests/00042_X_test.csv",
    y_test="output/.../tests/00042_y_test.csv",
    model_name="xgb_seed_42",
)
fig = evaluator.plot_shap_summary(max_display=20)
fig.savefig("shap_single.png", bbox_inches="tight")

# Aggregate mean |SHAP| bar chart across all seeds
ev = MultiModelEvaluator(trained_model_csv="output/.../trained_model_registry.csv")
fig = ev.plot_shap_summary(max_display=20)  # saves automatically

# Low-level standalone functions
from eruption_forecast.plots.shap_plots import plot_shap_summary, plot_aggregate_shap_summary

fig = plot_shap_summary(model, X_test, feature_names=features, max_display=15)

fig, df = plot_aggregate_shap_summary(
    models=trained_models,
    X_tests=x_test_list,
    feature_names=feature_names,
    max_display=20,
)
# df has columns: feature, mean_shap, std_shap
```

### Seed Stability Plot

Visualize how stable a metric is across random seeds using a violin + strip plot. Useful for detecting high-variance classifiers or seed-sensitive configurations.

```python
from eruption_forecast import MultiModelEvaluator
from eruption_forecast.plots import plot_seed_stability

# From a single classifier's evaluator (single violin)
ev = MultiModelEvaluator(
    metrics_dir="output/.../xgb-classifier/stratified-shuffle-split/metrics",
    trained_model_csv="output/.../trained_model_registry.csv",
)
fig = ev.plot_seed_stability(metric="f1_score")  # saves automatically

# Compare multiple classifiers side-by-side using standalone function
metrics_by_clf = {}
for clf in ["xgb", "rf", "gb", "svm"]:
    ev = MultiModelEvaluator(
        metrics_dir=f"output/.../trainings/{clf}/metrics"
    )
    metrics_by_clf[clf] = ev.get_metrics_list()

fig, df = plot_seed_stability(
    metrics_by_classifier=metrics_by_clf,
    metric="balanced_accuracy",
    dpi=150,
)
fig.savefig("seed_stability.png", bbox_inches="tight")
# df has columns: classifier, seed_idx, value
```

### Frequency Band Contribution

See which seismic frequency bands dominate the selected features. RSAM bands are coloured blue; DSAR bands are orange. Supports single-seed (exact counts) and multi-seed (mean ± std).

```python
from eruption_forecast import MultiModelEvaluator
from eruption_forecast.plots import plot_frequency_band_contribution

# From a registry (multi-seed, mean ± std per band)
ev = MultiModelEvaluator(trained_model_csv="output/.../trained_model_registry.csv")
fig = ev.plot_frequency_band_contribution()  # saves automatically

# Standalone — single seed
from eruption_forecast.plots.feature_plots import plot_frequency_band_contribution
import pandas as pd

features = pd.read_csv("output/.../significant_features.csv", index_col=0).index.tolist()
fig, df = plot_frequency_band_contribution(feature_names=features)
# df has columns: band, count

# Standalone — multi-seed (pass list of lists)
per_seed = [
    pd.read_csv(p, index_col=0).index.tolist()
    for p in significant_feature_csvs
]
fig, df = plot_frequency_band_contribution(feature_names=per_seed)
# df has columns: band, mean_count, std_count
```

### Forecast Visualization

#### Plotting with ModelPredictor

Visualize eruption probability forecasts as time-series plots.

```python
from eruption_forecast.model.model_predictor import ModelPredictor

predictor = ModelPredictor(
    start_date="2025-03-23",
    end_date="2025-03-30",
    trained_models="output/trainings/predictions/random-forest-classifier/stratified-k-fold/trained_model_*.csv",
    output_dir="output/forecast",
)

# With labeled data (evaluation mode)
df_metrics = predictor.predict(
    future_features_csv="output/features_forecast.csv",
    future_labels_csv="output/label_forecast.csv",
    plot=True,
)

# Without labels (forecasting mode)
df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

**Plot Features:**
- Probability time-series for each classifier
- Consensus probability (mean across classifiers)
- Confidence bands (standard deviation)
- Eruption event markers (if labels provided)
- Saved as `eruption_forecast.png` in `figures/` subdirectory

### Batch Replot Utilities

Regenerate plots for multiple files in parallel — useful for applying style updates or reprocessing after data changes.

#### replot_tremor()

Batch replot daily tremor CSV files.

```python
from eruption_forecast.plots.tremor_plots import replot_tremor

# Sequential processing
results = replot_tremor(
    daily_dir="output/VG.OJN.00.EHZ/tremor/daily",
    output_dir="output/VG.OJN.00.EHZ/tremor/figures",
    overwrite=True,
)
print(f"Created: {results['created']}, Skipped: {results['skipped']}")

# Parallel processing with custom parameters
results = replot_tremor(
    daily_dir="output/VG.OJN.00.EHZ/tremor/daily",
    n_jobs=4,
    interval=6,
    interval_unit="hours",
    dpi=300,
    selected_columns=["rsam_f2", "rsam_f3"],
    overwrite=False,
)
```

#### replot_significant_features()

Batch replot feature importance across multiple seeds.

```python
from eruption_forecast.plots.feature_plots import replot_significant_features

# Sequential processing
results = replot_significant_features(
    all_features_dir="output/trainings/features/all_features",
    output_dir="output/trainings/features/figures/significant",
    overwrite=True,
)

# Parallel processing
results = replot_significant_features(
    all_features_dir="output/trainings/features/all_features",
    n_jobs=4,
    number_of_features=50,
    top_features=20,
    dpi=300,
    overwrite=False,
)
print(f"Processed: Created {results['created']}, Skipped {results['skipped']}, Failed {results['failed']}")
```

**Return Value:**
Both functions return a dict with counts:
- `'created'` - Plots successfully generated
- `'skipped'` - Plots skipped (file exists, overwrite=False)
- `'failed'` - Plots that encountered errors

### Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dpi` | `int` | `150` | Resolution in dots per inch (use 300 for publication) |
| `overwrite` | `bool` | `True` | Replace existing plots if they exist |
| `n_jobs` | `int` | `1` | Parallel workers for batch utilities (use >1 for large datasets) |
| `output_dir` / `figure_dir` | `str` | varies | Directory where plots are saved |
| `verbose` | `bool` | `False` | Enable logging of plot generation |
| `filename` | `str` | auto | Custom filename stem (extension added automatically) |

### Import Examples

```python
# Tremor plots
from eruption_forecast.plots.tremor_plots import plot_tremor, replot_tremor

# Feature plots
from eruption_forecast.plots.feature_plots import (
    plot_significant_features,
    replot_significant_features,
    plot_frequency_band_contribution,
)

# Single-seed evaluation (top-level shortcut)
from eruption_forecast import ModelEvaluator

# Multi-seed aggregate evaluation (top-level shortcut)
from eruption_forecast import MultiModelEvaluator

# Or import from their modules directly
from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator

# Forecast plots (via ModelPredictor)
from eruption_forecast.model.model_predictor import ModelPredictor

# Low-level single-seed plot functions (used internally by ModelEvaluator):
from eruption_forecast.plots.evaluation_plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_threshold_analysis,
    plot_feature_importance,
    plot_calibration,
    plot_prediction_distribution,
)

# Cross-classifier and stability plots
from eruption_forecast.plots import (
    plot_classifier_comparison,
    plot_seed_stability,
)

# SHAP explainability plots
from eruption_forecast.plots.shap_plots import (
    plot_shap_summary,
    plot_aggregate_shap_summary,
)
```
