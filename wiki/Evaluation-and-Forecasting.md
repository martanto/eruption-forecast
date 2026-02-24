# Evaluation and Forecasting

This page covers how to evaluate trained models and run inference on new seismic data. Four classes are involved:

- **`ModelEvaluator`** — evaluates a single trained model (one seed) against a held-out test set.
- **`MultiModelEvaluator`** — aggregates metrics and plots across all seeds produced by `train_and_evaluate()`.
- **`ClassifierComparator`** — compares multiple classifiers side-by-side with ranking tables and comparison plots.
- **`ModelPredictor`** — runs inference using models trained by `ModelTrainer.train()`, in either evaluation or forecast mode.

---

## ModelEvaluator — Single-Seed Evaluation

`ModelEvaluator` computes classification metrics and generates diagnostic plots for a single trained model evaluated on a held-out test set. Use it after `train_and_evaluate()` (which saves per-seed test splits automatically) or after calling `ModelPredictor.predict_best()` (which returns a ready-made evaluator for the best-performing seed).

### Creating an evaluator

There are two ways to create a `ModelEvaluator`.

**From in-memory objects** — when you already have the model and test arrays in memory:

```python
from eruption_forecast.model.model_evaluator import ModelEvaluator

evaluator = ModelEvaluator(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    model_name="xgb_42",
    output_dir="output/eval",
)
```

**From files on disk** — using `ModelEvaluator.from_files()`:

```python
evaluator = ModelEvaluator.from_files(
    model_path="output/trainings/classifier/XGBClassifier/stratified/models/00042.pkl",
    X_test="output/features/all_features.csv",
    y_test="output/features/label_features.csv",
    selected_features=["feat_a", "feat_b"],  # optional: restrict to a feature subset
    model_name="xgb_42",
    output_dir="output/eval",
)
```

Once created, inspect the model with:

```python
print(evaluator.summary())
```

### Available metrics

`get_metrics()` returns a dict with the following keys:

| Key | Description |
|-----|-------------|
| `accuracy` | Overall fraction of correct predictions |
| `balanced_accuracy` | Mean recall per class (handles class imbalance) |
| `precision` | TP / (TP + FP) |
| `recall` | TP / (TP + FN) |
| `f1_score` | Harmonic mean of precision and recall |
| `roc_auc` | Area under the ROC curve |
| `pr_auc` | Area under the precision-recall curve |
| `true_positives` | Count of correctly predicted eruptions |
| `true_negatives` | Count of correctly predicted non-eruptions |
| `false_positives` | Count of non-eruptions predicted as eruptions |
| `false_negatives` | Count of eruptions predicted as non-eruptions |
| `sensitivity` | Same as recall (TP rate) |
| `specificity` | TN / (TN + FP) |
| `optimal_threshold` | Decision threshold that maximises F1 |
| `f1_at_optimal` | F1 score at the optimal threshold |
| `recall_at_optimal` | Recall at the optimal threshold |
| `precision_at_optimal` | Precision at the optimal threshold |

```python
metrics = evaluator.get_metrics()
print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
print(f"F1 score:          {metrics['f1_score']:.4f}")
```

You can also find the optimal decision threshold explicitly:

```python
threshold, threshold_metrics = evaluator.optimize_threshold(criterion="f1")
print(f"Optimal threshold: {threshold:.3f}")
print(f"F1 at threshold:   {threshold_metrics['f1']:.4f}")
```

### Available plots

`plot_all()` generates all 7 diagnostic plots at once (8 when SHAP is available) and saves them to `output_dir`:

```python
evaluator.plot_all()
```

Individual plot methods:

| Method | Description |
|--------|-------------|
| `plot_confusion_matrix()` | Confusion matrix heatmap |
| `plot_roc_curve()` | ROC curve with AUC annotation |
| `plot_precision_recall_curve()` | Precision-recall curve with AP annotation |
| `plot_threshold_analysis()` | F1, precision, recall, and balanced accuracy vs. threshold |
| `plot_feature_importance()` | Bar chart of feature importances from the classifier |
| `plot_calibration()` | Predicted probability vs. actual fraction of positives |
| `plot_prediction_distribution()` | KDE of predicted probabilities by true class |
| `plot_shap_summary(max_display=20)` | SHAP beeswarm plot — requires `shap>=0.46` |

### Saving metrics to JSON

Call `save_metrics()` to persist the metrics dict as a JSON file. `np.nan` values are serialised as `null`.

```python
# Save to the default path: {output_dir}/{model_name}_metrics.json
path = evaluator.save_metrics()
print(f"Saved: {path}")

# Or specify an explicit path
path = evaluator.save_metrics("results/xgb_42_metrics.json")
```

---

## MultiModelEvaluator — Aggregate Across All Seeds

`MultiModelEvaluator` aggregates evaluation results across all seeds produced by `train_and_evaluate()`. It can work from a model registry CSV (enabling aggregate plots), from per-seed JSON metrics files (enabling aggregate statistics), or from both at once.

### Creating an evaluator

Three construction modes are available:

```python
from eruption_forecast import MultiModelEvaluator

base = "output/trainings/evaluations/xgb-classifier/stratified-shuffle-split"
trained_model_csv = f"{base}/trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv"

# Mode 1: registry CSV only — enables aggregate plots
evaluator = MultiModelEvaluator(trained_model_csv=trained_model_csv)

# Mode 2: metrics directory only — enables aggregate statistics
evaluator = MultiModelEvaluator(metrics_dir=f"{base}/metrics")

# Mode 3: both combined — enables plots and statistics together
evaluator = MultiModelEvaluator(
    trained_model_csv=trained_model_csv,
    metrics_dir=f"{base}/metrics",
    output_dir="output/eval/aggregate",  # optional override
)
```

You can also provide an explicit list of JSON files instead of a directory:

```python
import glob

json_files = sorted(glob.glob(f"{base}/metrics/*.json"))
evaluator = MultiModelEvaluator(
    metrics_files=json_files,
    output_dir="output/eval/aggregate",
)
```

### Aggregate plots

`plot_all()` runs all 10 aggregate plots at once and saves them to `{output_dir}/figures/`:

```python
figs = evaluator.plot_all(dpi=150, show_individual=True)
```

Individual plot methods (each accepts `save=True`, `filename=None`, `dpi=150`, and `title=None`):

| Method | Description |
|--------|-------------|
| `plot_roc()` | Mean ROC curve with ±1 std shaded band |
| `plot_precision_recall()` | Mean PR curve with ±1 std shaded band |
| `plot_calibration(n_bins=10)` | Mean calibration curve with ±1 std shaded band |
| `plot_prediction_distribution()` | Pooled KDE of predicted probabilities by class |
| `plot_confusion_matrix(normalize=None)` | Summed confusion matrix across all seeds |
| `plot_threshold_analysis(show_individual=True)` | Mean metric curves vs. threshold with ±1 std bands |
| `plot_feature_importance(top_n=20)` | Mean importance per feature with ±1 std error bars |
| `plot_shap_summary(max_display=20)` | Mean absolute SHAP bar chart across seeds — requires `shap>=0.46` |
| `plot_seed_stability(metric="f1_score")` | Violin plot of a chosen metric across seeds |
| `plot_frequency_band_contribution()` | Feature counts per seismic frequency band |

**Aggregation strategy per plot:**

| Plot | Aggregation method |
|------|--------------------|
| ROC Curve | Each seed's TPR is interpolated onto a shared FPR grid (200 points). All curves are stacked into an `(n_seeds x 200)` matrix. Mean TPR is the bold line; ±1 std is the shaded band. Mean AUC ± std is shown in the legend. |
| Precision-Recall Curve | Each seed's precision is interpolated onto a shared recall grid (200 points). Mean precision is the bold line; ±1 std is the shaded band. Mean AP ± std is shown in the legend. |
| Threshold Analysis | For each of 101 thresholds (0 to 1), F1, precision, recall, and balanced accuracy are computed per seed, stacked `(n_seeds x 101)`, and plotted as mean with ±1 std bands. |
| Calibration Curve | Each seed's calibration curve is interpolated onto a shared probability grid. Mean fraction of positives is the bold line; ±1 std is the shaded band. |
| Prediction Distribution | Predicted probabilities from all seeds are pooled (concatenated) separately for class 0 and class 1, then a single KDE is computed over each pool. |
| Confusion Matrix | Raw confusion matrices are summed across all seeds (total TP/TN/FP/FN across the ensemble). Optional normalisation is applied after summing. |
| Feature Importance | Importances are stacked `(n_seeds x n_features)`. Mean importance is the bar length; ±1 std is the error bar. Top-N features selected by mean importance. |

### Aggregate metrics from JSON files

`get_aggregate_metrics()` summarises per-seed JSON metrics written by `save_metrics()`. It returns a DataFrame with metric names as the index and columns `mean`, `std`, `min`, `max`.

```python
evaluator = MultiModelEvaluator(metrics_dir=f"{base}/metrics")

summary = evaluator.get_aggregate_metrics()
print(summary.loc["f1_score"])
# mean     0.7842
# std      0.0321
# min      0.6900
# max      0.8500

# Save aggregate stats to CSV
path = evaluator.save_aggregate_metrics()
# Saved to: {metrics_dir}/figures/aggregate_metrics.csv

# Or specify an explicit output path
path = evaluator.save_aggregate_metrics("my_summary.csv")
```

---

## ClassifierComparator — Side-by-Side Comparison

`ClassifierComparator` accepts a mapping of classifier names to trained-model registry CSV paths, builds one `MultiModelEvaluator` per classifier, and generates cross-classifier comparison plots and a ranking table.

### Constructor

```python
from eruption_forecast.model import ClassifierComparator

comparator = ClassifierComparator(
    classifiers={
        "rf":  "output/.../trainings/evaluations/rf/stratified/trained_model_rf_...csv",
        "xgb": "output/.../trainings/evaluations/xgb/stratified/trained_model_xgb_...csv",
        "gb":  "output/.../trainings/evaluations/gb/stratified/trained_model_gb_...csv",
    },
    output_dir="output/comparison",   # optional — defaults to cwd/output/comparison
    metrics=["f1_score", "roc_auc"],  # optional — defaults to all DEFAULT_METRICS
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classifiers` | `dict[str, str]` | required | Classifier name → trained-model registry CSV path |
| `output_dir` | `str \| None` | `None` | Root output directory; defaults to `cwd/output/comparison/` |
| `metrics` | `str \| list[str] \| None` | `None` | Metrics for plots and ranking; `None` uses all `DEFAULT_METRICS` |

`DEFAULT_METRICS`: `f1_score`, `roc_auc`, `pr_auc`, `balanced_accuracy`, `precision`, `recall`, `specificity`, `sensitivity`.

Each CSV path must exist on disk. A `FileNotFoundError` is raised otherwise. The companion `metrics/` directory (sibling to each registry CSV) is detected automatically for per-seed JSON metrics.

Alternatively, load classifiers from a JSON file using the `from_json` classmethod:

```python
# evaluations_trained_models.json — {"ClassifierName": "/path/to/trained_model_*.csv", ...}
comparator = ClassifierComparator.from_json(
    "output/VG.OJN.00.EHZ/evaluations_trained_models.json",
    output_dir="output/comparison",
    metrics=["f1_score", "roc_auc"],
)
```

### Metrics table

```python
# Wide DataFrame: one row per classifier, columns like f1_score_mean / f1_score_std
table = comparator.get_metrics_table()
print(table[["f1_score_mean", "f1_score_std", "roc_auc_mean"]])
```

### Ranking

Ranks classifiers by a single metric (default: `recall`) and saves one CSV.

```python
# Sorted descending by mean recall; saved to metrics/ranking_recall.csv
ranking = comparator.get_ranking()

# Custom metric
ranking = comparator.get_ranking(metric="roc_auc", by="mean")
print(ranking.head())
```

### Comparison plots

Each plot method returns a `dict[str, plt.Figure]` keyed by metric name. When more than one metric is requested, an additional `"all"` key contains a combined subplot grid (max 4 columns per row).

| Method | Returns | Saved filenames |
|--------|---------|-----------------|
| `plot_metric_bar(metrics)` | `dict[str, Figure]` | `metric_bar_{metric}.png` per metric + `metric_bar_all.png` |
| `plot_seed_stability(metrics)` | `dict[str, Figure]` | `seed_stability_{metric}.png` per metric + `seed_stability_all.png` |
| `plot_comparison_grid(metrics)` | `Figure` | `comparison_grid.png` |
| `plot_roc()` | `Figure` | `comparison_roc.png` |
| `plot_all()` | `dict[str, Any]` | all of the above |

```python
# Individual plot methods
figs_bar = comparator.plot_metric_bar(metrics=["f1_score", "roc_auc"])
# figs_bar == {"f1_score": Figure, "roc_auc": Figure, "all": Figure}

figs_violin = comparator.plot_seed_stability(metrics=["f1_score", "roc_auc"])
# figs_violin == {"f1_score": Figure, "roc_auc": Figure, "all": Figure}

# All plots at once
results = comparator.plot_all(dpi=150)
# results == {
#     "metric_bar":      dict[str, Figure],   # one per metric + "all"
#     "seed_stability":  dict[str, Figure],   # one per metric + "all"
#     "comparison_grid": Figure,
#     "roc":             Figure,
#     "ranking":         DataFrame,           # ranked by recall
# }
```

Figures are saved to `{output_dir}/figures/` and the ranking CSV to `{output_dir}/metrics/`.

### Output structure

```
output/comparison/
├── figures/
│   ├── metric_bar_f1_score.png
│   ├── metric_bar_roc_auc.png
│   ├── metric_bar_all.png
│   ├── seed_stability_f1_score.png
│   ├── seed_stability_roc_auc.png
│   ├── seed_stability_all.png
│   ├── comparison_grid.png
│   └── comparison_roc.png
└── metrics/
    └── ranking_recall.csv
```

---

## ModelPredictor — Inference on New Data

`ModelPredictor` runs inference using models trained by `ModelTrainer.train()`. It supports two modes:

- **Evaluation mode** — ground-truth labels are available; useful for benchmarking on a future labelled dataset.
- **Forecast mode** — no labels; outputs eruption probability, uncertainty, and confidence per time window.

Both single-model and multi-model consensus workflows are supported.

```python
from eruption_forecast.model.model_predictor import ModelPredictor
```

### Constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | `str \| datetime` | — | Start of the prediction period (format: YYYY-MM-DD) |
| `end_date` | `str \| datetime` | — | End of the prediction period (format: YYYY-MM-DD) |
| `trained_models` | `str \| dict[str, str]` | — | Single `trained_model_*.csv` path or a `{name: path}` dict for multi-model consensus |
| `overwrite` | `bool` | `False` | Overwrite existing output files |
| `n_jobs` | `int` | `1` | Number of parallel jobs for feature extraction |
| `output_dir` | `str \| None` | `None` | Output directory; defaults to `<root_dir>/output/predictions` |
| `root_dir` | `str \| None` | `None` | Root directory for resolving output paths |
| `verbose` | `bool` | `False` | Enable verbose logging |

### Evaluation mode — predict() and predict_best()

Use evaluation mode when ground-truth eruption labels are available for the prediction period. Both methods require pre-extracted feature CSVs.

**`predict()`** evaluates every seed model against the labels and returns a DataFrame with one row per `(classifier, seed)`:

```python
predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models=trainer.csv,
    output_dir="output/predictions",
)

df_metrics = predictor.predict(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
)
print(df_metrics[["balanced_accuracy", "f1_score"]].describe())
```

**`predict_best()`** selects the single best-performing seed according to a chosen criterion and returns a `ModelEvaluator` for that seed:

```python
evaluator = predictor.predict_best(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
    criterion="balanced_accuracy",
)
print(evaluator.summary())
evaluator.plot_all()
```

The `criterion` parameter accepts any of: `"accuracy"`, `"balanced_accuracy"`, `"f1_score"`, `"precision"`, `"recall"`, `"roc_auc"`, `"pr_auc"`.

### Forecast mode — predict_proba()

Use forecast mode when no ground-truth labels are available. Pass raw tremor data directly; `ModelPredictor` handles windowing and feature extraction internally.

**Single model:**

```python
predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models=trainer.csv,
    output_dir="output/predictions",
)

df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",  # or a pd.DataFrame
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

Results are saved to `predictions.csv` in `output_dir`. When `plot=True`, an eruption forecast plot is saved to `figures/eruption_forecast.png`.

### Multi-model consensus

Pass a `{name: path}` dict to `trained_models` to enable multi-model consensus. `predict_proba()` first aggregates predicted probabilities within each classifier across its seeds, then computes a consensus across all classifiers.

```python
predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models={
        "rf":  "output/VG.OJN.00.EHZ/trainings/predictions/random-forest-classifier/stratified-shuffle-split/trained_model_RandomForestClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv",
        "xgb": "output/VG.OJN.00.EHZ/trainings/predictions/xgb-classifier/stratified-shuffle-split/trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv",
    },
    output_dir="output/predictions",
)

df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

The plot shows each classifier as a dashed line and the consensus as a solid black line with a shaded uncertainty band.

### Multi-model output columns

| Column | Description |
|--------|-------------|
| `{name}_eruption_probability` | Mean P(eruption) across seeds of that classifier |
| `{name}_uncertainty` | Std across seeds of that classifier |
| `{name}_confidence` | Seed-level agreement fraction (0.5–1.0) |
| `{name}_prediction` | Hard label for that classifier |
| `consensus_eruption_probability` | Mean P(eruption) averaged across all classifiers |
| `consensus_uncertainty` | Std of per-classifier means (inter-model disagreement) |
| `consensus_confidence` | Fraction of classifiers voting with the consensus majority |
| `consensus_prediction` | Hard label — `1` if `consensus_eruption_probability >= 0.5` |
