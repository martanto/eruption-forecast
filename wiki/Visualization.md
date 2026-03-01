# Visualization

All plots use colorblind-safe palettes and high-DPI output suitable for publications and presentations.

---

## Tremor Time-Series Plots

### `plot_tremor()`

Visualize RSAM, DSAR, and Shannon Entropy as multi-panel time-series plots.

```python
from eruption_forecast.plots.tremor_plots import plot_tremor
import pandas as pd

df = pd.read_csv("output/VG.OJN.00.EHZ/tremor/tremor.csv", index_col=0, parse_dates=True)

# Basic plot — all columns
plot_tremor(df=df, figure_dir="output/figures", dpi=150)

# Custom interval and selected columns
plot_tremor(
    df=df,
    interval=6,
    interval_unit="hours",
    selected_columns=["rsam_f2", "rsam_f3", "dsar_f2-f3"],
    figure_dir="output/figures",
    filename="tremor_selected",
    dpi=300,
)
```

### `replot_tremor()` — Batch Replot

Regenerate plots for all daily tremor CSV files (sequential or parallel).

```python
from eruption_forecast.plots.tremor_plots import replot_tremor

results = replot_tremor(
    daily_dir="output/VG.OJN.00.EHZ/tremor/daily",
    output_dir="output/VG.OJN.00.EHZ/tremor/figures",
    n_jobs=4,
    dpi=300,
    overwrite=False,
)
print(f"Created: {results['created']}, Skipped: {results['skipped']}, Failed: {results['failed']}")
```

---

## Feature Importance Plots

### `plot_significant_features()`

Horizontal bar chart of feature importance or p-values.

```python
from eruption_forecast.plots.feature_plots import plot_significant_features

plot_significant_features(
    features="path/to/features.csv",  # or a DataFrame
    number_of_features=50,
    top_features=20,                   # Highlighted with darker colour
    values_column="importance",        # or "p_values"
    output_dir="output/figures",
    filename="feature_importance",
    dpi=150,
)
```

### `replot_significant_features()` — Batch Replot

```python
from eruption_forecast.plots.feature_plots import replot_significant_features

results = replot_significant_features(
    all_features_dir="output/trainings/features/all_features",
    output_dir="output/trainings/features/figures/significant",
    n_jobs=4,
    number_of_features=50,
    top_features=20,
    dpi=300,
    overwrite=False,
)
```

---

## Model Evaluation Plots

### Single-Seed (`ModelEvaluator`)

`plot_all()` generates all 7–8 plots at once. Individual methods:

```python
from eruption_forecast import ModelEvaluator

evaluator = ModelEvaluator.from_files(
    model_path="output/.../models/00042.pkl",
    X_test="output/.../tests/00042_X_test.csv",
    y_test="output/.../tests/00042_y_test.csv",
    model_name="xgb_seed_42",
    output_dir="output/eval",
)

evaluator.plot_all()                      # All plots
evaluator.plot_confusion_matrix()
evaluator.plot_roc_curve()
evaluator.plot_precision_recall_curve()
evaluator.plot_threshold_analysis()
evaluator.plot_feature_importance()
evaluator.plot_calibration()
evaluator.plot_prediction_distribution()
evaluator.plot_shap_summary(max_display=20)  # Requires shap>=0.46
```

### Aggregate (`MultiModelEvaluator`)

Requires `train_and_evaluate()` output — loads test data per seed and aggregates.

```python
from eruption_forecast import MultiModelEvaluator

base = "output/trainings/evaluations/xgb-classifier/stratified-shuffle-split"
ev = MultiModelEvaluator(
    trained_model_csv=f"{base}/trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv"
)

figs = ev.plot_all(dpi=150, show_individual=True)
# Keys: roc_curve, pr_curve, calibration, prediction_distribution,
#       confusion_matrix, threshold_analysis, feature_importance,
#       shap_summary, seed_stability, frequency_band_contribution,
#       learning_curve
```

### Aggregation Strategy

| Plot | How seeds are combined |
|------|----------------------|
| ROC Curve | Each seed's TPR interpolated onto shared FPR grid → mean ± std band |
| Precision-Recall | Precision interpolated onto shared recall grid → mean ± std band |
| Threshold Analysis | Metrics computed per threshold per seed → mean ± std bands |
| Calibration | Fraction of positives interpolated → mean ± std band |
| Prediction Distribution | Predicted probabilities pooled across seeds → single KDE per class |
| Confusion Matrix | Raw confusion matrices summed across all seeds |
| Feature Importance | Importances stacked → mean bar + std error bar |

---

## Classifier Comparison Plots (`ClassifierComparator`)

`ClassifierComparator` provides four ready-made comparison plots across multiple classifiers.

```python
from eruption_forecast.model import ClassifierComparator

# From a dict
base = "output/trainings/evaluations"
comparator = ClassifierComparator(
    classifiers={
        "rf":  f"{base}/rf/stratified/trained_model_rf_...csv",
        "xgb": f"{base}/xgb/stratified/trained_model_xgb_...csv",
        "gb":  f"{base}/gb/stratified/trained_model_gb_...csv",
    },
    output_dir="output/comparison",
    metrics=["f1_score", "roc_auc", "recall"],  # or None for all DEFAULT_METRICS
)

# From a JSON file  {"ClassifierName": "/path/to/trained_model_*.csv", ...}
comparator = ClassifierComparator.from_json(
    "output/VG.OJN.00.EHZ/evaluations_trained_models.json",
    output_dir="output/comparison",
    metrics=["f1_score", "roc_auc", "recall"],
)

# Bar chart per metric (mean ± std) — plus a combined "all" overview figure
figs_bar = comparator.plot_metric_bar()
# {"f1_score": Figure, "roc_auc": Figure, "recall": Figure, "all": Figure}

# Violin + strip of per-seed distributions per metric — plus combined "all" figure
figs_violin = comparator.plot_seed_stability()
# {"f1_score": Figure, "roc_auc": Figure, "recall": Figure, "all": Figure}

# Grid of subplots: rows = classifiers, columns = metrics
fig_grid = comparator.plot_comparison_grid()

# Overlaid mean ROC curves with ± std bands
fig_roc = comparator.plot_roc()

# All plots + ranking in one call
results = comparator.plot_all()
# results["metric_bar"]      → dict[str, Figure]
# results["seed_stability"]  → dict[str, Figure]
# results["comparison_grid"] → Figure
# results["roc"]             → Figure
# results["ranking"]         → DataFrame (ranked by recall)
```

Output files are written to `output/comparison/figures/` and the ranking CSV to
`output/comparison/metrics/ranking_recall.csv`.

| Plot method | Saved files |
|-------------|-------------|
| `plot_metric_bar()` | `metric_bar_{metric}.png` per metric + `metric_bar_all.png` |
| `plot_seed_stability()` | `seed_stability_{metric}.png` per metric + `seed_stability_all.png` |
| `plot_comparison_grid()` | `comparison_grid.png` |
| `plot_roc()` | `comparison_roc.png` |

---

## SHAP Explainability

Understand which features drive predictions and in which direction.

```python
from eruption_forecast import ModelEvaluator, MultiModelEvaluator

# Single-seed beeswarm
evaluator = ModelEvaluator.from_files(
    model_path="output/.../models/00042.pkl",
    X_test="output/.../tests/00042_X_test.csv",
    y_test="output/.../tests/00042_y_test.csv",
    model_name="xgb_seed_42",
)
fig = evaluator.plot_shap_summary(max_display=20)

# Aggregate SHAP beeswarm across seeds
ev = MultiModelEvaluator(trained_model_csv="output/.../trained_model_registry.csv")
fig = ev.plot_shap_summary(max_display=20)
```

> Requires `shap >= 0.46`.

---

## Seed Stability Plot

Visualize metric variability across random seeds using a violin + strip plot.

```python
from eruption_forecast import MultiModelEvaluator
from eruption_forecast.plots import plot_seed_stability

# Single classifier
ev = MultiModelEvaluator(
    metrics_dir="output/.../xgb-classifier/stratified-shuffle-split/metrics"
)
fig = ev.plot_seed_stability(metric="f1_score")

# Compare multiple classifiers side-by-side
metrics_by_clf = {}
for clf in ["xgb", "rf", "gb", "svm"]:
    ev = MultiModelEvaluator(metrics_dir=f"output/.../{clf}/metrics")
    metrics_by_clf[clf] = ev.get_metrics_list()

fig, df = plot_seed_stability(
    metrics_by_classifier=metrics_by_clf,
    metric="balanced_accuracy",
    dpi=150,
)
```

---

## Frequency Band Contribution

Shows which seismic frequency bands dominate the selected features. RSAM bands are blue; DSAR bands are orange.

```python
from eruption_forecast import MultiModelEvaluator

# Multi-seed, mean ± std per band
ev = MultiModelEvaluator(trained_model_csv="output/.../trained_model_registry.csv")
fig = ev.plot_frequency_band_contribution()

# Single-seed standalone
from eruption_forecast.plots.feature_plots import plot_frequency_band_contribution
import pandas as pd

features = pd.read_csv("output/.../significant_features.csv", index_col=0).index.tolist()
fig, df = plot_frequency_band_contribution(feature_names=features)
# df columns: band, count
```

---

## Learning Curve Plots

Visualize how model performance scales with training-set size. Useful for diagnosing underfitting / overfitting and confirming that more data is beneficial.

Each seed's learning curve is stored as a JSON file with one key per scoring metric:

```json
{
  "balanced_accuracy": {
    "train_sizes": [50, 100, 200, 400],
    "train_scores": [[0.72, 0.73], [0.78, 0.79], ...],
    "test_scores":  [[0.65, 0.66], [0.70, 0.71], ...]
  },
  "f1_weighted": { ... }
}
```

The old flat single-metric format is still accepted for backward compatibility.

### Single-seed (`ModelEvaluator`)

```python
from eruption_forecast import ModelEvaluator

evaluator = ModelEvaluator.from_files(
    model_path="output/.../models/00042.pkl",
    X_test="output/.../tests/00042_X_test.csv",
    y_test="output/.../tests/00042_y_test.csv",
    learning_curve_path="output/.../learning_curves/00042_lc.json",
    model_name="xgb_seed_42",
    output_dir="output/eval",
)

# One subplot per scoring metric
fig = evaluator.plot_learning_curve(dpi=150)
```

### Aggregate across seeds (`MultiModelEvaluator`)

```python
from eruption_forecast import MultiModelEvaluator

ev = MultiModelEvaluator(
    trained_model_csv="output/.../trained_model_registry.csv"
)

# Bold mean line + ±1 std band per metric
fig = ev.plot_learning_curve(dpi=150)
```

### Standalone (`plot_learning_curve_grid`)

```python
from eruption_forecast.plots.evaluation_plots import plot_learning_curve_grid

fig = plot_learning_curve_grid(
    json_filepath="output/.../learning_curve_seed042.json",
    output_dir="output/figures",
    scorings=["balanced_accuracy", "f1_weighted"],
    dpi=150,
)
```

---

## Forecast Visualization

`ModelPredictor.predict_proba(plot=True)` automatically generates an eruption probability time-series plot saved to `figures/eruption_forecast.png`.

- Per-classifier lines (dashed)
- Consensus line (solid black)
- Uncertainty band (shaded ± std)
- Eruption event markers (if labels provided)

---

## Common Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dpi` | `int` | `150` | Resolution (use 300 for publication) |
| `overwrite` | `bool` | `True` | Replace existing plots |
| `n_jobs` | `int` | `1` | Parallel workers for batch utilities |
| `output_dir` / `figure_dir` | `str` | varies | Directory for saved plots |
| `verbose` | `bool` | `False` | Log plot generation |
| `filename` | `str` | auto | Custom filename stem (extension added automatically) |

---

## Import Reference

```python
from eruption_forecast.plots.tremor_plots import plot_tremor, replot_tremor
from eruption_forecast.plots.feature_plots import (
    plot_significant_features,
    replot_significant_features,
    plot_frequency_band_contribution,
)
from eruption_forecast.plots.shap_plots import plot_shap_summary, plot_aggregate_shap_summary
from eruption_forecast.plots import plot_classifier_comparison, plot_seed_stability
from eruption_forecast.plots.evaluation_plots import (
    plot_learning_curve,
    plot_aggregate_learning_curve,
    plot_learning_curve_grid,
)

# Top-level shortcuts
from eruption_forecast import ModelEvaluator, MultiModelEvaluator
```
