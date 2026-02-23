# API Reference

Full constructor and method parameter tables. For narrative usage examples, see [Pipeline Walkthrough](Pipeline-Walkthrough) and [Training Workflows](Training-Workflows).

---

## ForecastModel

High-level orchestrator. All pipeline stages available via method chaining.

```python
from eruption_forecast import ForecastModel
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `station` | `str` | — | Seismic station code (e.g. `"OJN"`) |
| `channel` | `str` | — | Seismic channel code (e.g. `"EHZ"`) |
| `start_date` | `str \| datetime` | — | Training period start (`YYYY-MM-DD`) |
| `end_date` | `str \| datetime` | — | Training period end (`YYYY-MM-DD`) |
| `window_size` | `int` | — | Duration (days) of each tremor window fed into tsfresh |
| `volcano_id` | `str` | — | Identifier used in output filenames |
| `network` | `str` | `"VG"` | Seismic network code |
| `location` | `str` | `"00"` | Seismic location code |
| `output_dir` | `str \| None` | `None` | Base output directory (relative → resolved against `root_dir`) |
| `root_dir` | `str \| None` | `None` | Anchor for path resolution. Defaults to `os.getcwd()` |
| `overwrite` | `bool` | `False` | Re-run and overwrite existing output files |
| `n_jobs` | `int` | `1` | Parallel workers propagated to all pipeline stages |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `debug` | `bool` | `False` | Enable debug-level logging |

### Additional Methods

| Method | Description |
|--------|-------------|
| `load_tremor_data(tremor_csv)` | Load pre-calculated tremor CSV instead of calling `calculate()` |
| `set_feature_selection_method(using)` | Override feature selection method (`"tsfresh"`, `"random_forest"`, `"combined"`) |
| `save_config(path=None, fmt="yaml")` | Save pipeline config to YAML/JSON |
| `save_model(path=None)` | Serialise full instance with joblib |
| `ForecastModel.from_config(path)` | Load config and construct instance |
| `ForecastModel.load_model(path)` | Restore a serialised instance |
| `run()` | Replay all stages from a loaded config |

---

## ModelTrainer

Multi-seed classifier training.

```python
from eruption_forecast.model.model_trainer import ModelTrainer
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extracted_features_csv` | `str` | — | Path to features CSV (output of `FeaturesBuilder`) |
| `label_features_csv` | `str` | — | Path to aligned labels CSV (output of `FeaturesBuilder`) |
| `output_dir` | `str \| None` | `None` | Output directory |
| `root_dir` | `str \| None` | `None` | Anchor for path resolution |
| `prefix_filename` | `str \| None` | `None` | Optional prefix on every output filename |
| `classifier` | `str` | `"rf"` | Classifier key (see [Classifiers and CV](Classifiers-and-CV)) |
| `cv_strategy` | `str` | `"shuffle"` | CV strategy — `"shuffle"`, `"stratified"`, `"timeseries"` |
| `cv_splits` | `int` | `5` | Number of CV folds |
| `number_of_significant_features` | `int` | `20` | Top-N features retained per seed |
| `feature_selection_method` | `str` | `"tsfresh"` | Feature selection — `"tsfresh"`, `"random_forest"`, `"combined"` |
| `overwrite` | `bool` | `False` | Re-run even if output already exists |
| `n_jobs` | `int` | `1` | Outer parallel workers (one per seed). `-1` = all cores. Enforced: `n_jobs × grid_search_n_jobs ≤ cpu_count` |
| `grid_search_n_jobs` | `int` | `1` | Parallel jobs inside `GridSearchCV` |
| `verbose` | `bool` | `False` | Print progress messages |
| `debug` | `bool` | `False` | Enable debug-level logging |

### `train_and_evaluate()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `0` | Starting seed; runs `random_state … random_state + total_seed − 1` |
| `total_seed` | `int` | `500` | Number of seeds to run |
| `sampling_strategy` | `str \| float` | `0.75` | `RandomUnderSampler` ratio (training data only) |
| `save_all_features` | `bool` | `False` | Save all ranked features per seed |
| `plot_significant_features` | `bool` | `False` | Save feature-importance plot per seed |

### `train()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `0` | Starting seed |
| `total_seed` | `int` | `500` | Number of seeds to run |
| `sampling_strategy` | `str \| float` | `0.75` | `RandomUnderSampler` ratio (full dataset) |
| `save_all_features` | `bool` | `False` | Save all ranked features per seed |
| `plot_significant_features` | `bool` | `False` | Save feature-importance plot per seed |

### `fit()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `with_evaluation` | `bool` | `True` | `True` → `train_and_evaluate()`; `False` → `train()` |
| `**kwargs` | — | — | Forwarded to the dispatched method |

---

## ModelPredictor

Inference on new data — evaluation mode or forecast mode.

```python
from eruption_forecast.model.model_predictor import ModelPredictor
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | `str \| datetime` | — | Start of prediction period |
| `end_date` | `str \| datetime` | — | End of prediction period |
| `trained_models` | `str \| dict[str, str]` | — | Single registry CSV path or `{name: path}` dict for multi-model consensus |
| `overwrite` | `bool` | `False` | Overwrite existing output files |
| `n_jobs` | `int` | `1` | Parallel jobs |
| `output_dir` | `str \| None` | `None` | Output directory |
| `root_dir` | `str \| None` | `None` | Root directory for path resolution |
| `verbose` | `bool` | `False` | Verbose logging |

### `predict()` — Evaluation Mode

```python
df_metrics = predictor.predict(
    future_features_csv="path/to/features.csv",
    future_labels_csv="path/to/labels.csv",
    plot=False,
)
```

Returns a DataFrame with one row per (classifier, seed) and columns: `accuracy`, `balanced_accuracy`, `f1_score`, `precision`, `recall`, `roc_auc`, `pr_auc`.

### `predict_best()` — Best Seed

```python
evaluator = predictor.predict_best(
    future_features_csv="path/to/features.csv",
    future_labels_csv="path/to/labels.csv",
    criterion="balanced_accuracy",   # any metric column
)
```

Returns a `ModelEvaluator` for the best-performing seed.

### `predict_proba()` — Forecast Mode

```python
df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",   # or pd.DataFrame
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

---

## ModelEvaluator

Single-seed evaluation.

```python
from eruption_forecast import ModelEvaluator
```

### Key Methods

| Method | Description |
|--------|-------------|
| `ModelEvaluator.from_files(model_path, X_test, y_test, ...)` | Load from disk |
| `get_metrics()` | Returns dict with all metrics |
| `summary()` | Prints formatted summary |
| `plot_all()` | Generates all 7–8 evaluation plots |
| `save_metrics(path=None)` | Saves metrics dict to JSON |
| `optimize_threshold(criterion="f1")` | Returns optimal threshold and metrics at that threshold |

### Available Metrics

`accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`, `pr_auc`, `true_positives`, `true_negatives`, `false_positives`, `false_negatives`, `sensitivity`, `specificity`, `optimal_threshold`, `f1_at_optimal`, `recall_at_optimal`, `precision_at_optimal`

---

## MultiModelEvaluator

Aggregate evaluation across all seeds.

```python
from eruption_forecast import MultiModelEvaluator
```

### Constructor

Pass `trained_model_csv` (for plots) and/or `metrics_dir` (for statistics) — or both.

```python
ev = MultiModelEvaluator(
    trained_model_csv="output/.../trained_model_*.csv",
    metrics_dir="output/.../metrics",
    output_dir="output/eval",
)
```

### Key Methods

| Method | Requires | Description |
|--------|----------|-------------|
| `plot_all(dpi, show_individual)` | `trained_model_csv` | Runs all 10 aggregate plots |
| `plot_roc()` | `trained_model_csv` | Mean ROC ± std band |
| `plot_precision_recall()` | `trained_model_csv` | Mean PR curve ± std band |
| `plot_calibration()` | `trained_model_csv` | Mean calibration ± std band |
| `plot_confusion_matrix()` | `trained_model_csv` | Summed confusion matrix |
| `plot_threshold_analysis()` | `trained_model_csv` | Mean metrics vs threshold ± std |
| `plot_feature_importance()` | `trained_model_csv` | Mean importance ± std error bars |
| `plot_shap_summary()` | `trained_model_csv` | Mean \|SHAP\| bar chart |
| `plot_seed_stability()` | `trained_model_csv` | Violin plot of a metric across seeds |
| `plot_frequency_band_contribution()` | `trained_model_csv` | Feature counts per seismic band |
| `get_aggregate_metrics()` | `metrics_dir` | DataFrame: metric × mean/std/min/max |
| `get_metrics_list()` | `metrics_dir` | Raw per-seed metrics as list of dicts |
| `save_aggregate_metrics(path=None)` | `metrics_dir` | Saves summary to CSV |

---

## FeatureSelector

```python
from eruption_forecast.features import FeatureSelector
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `"tsfresh"` | `"tsfresh"`, `"random_forest"`, or `"combined"` |
| `n_jobs` | `int` | `1` | Parallel workers |
| `verbose` | `bool` | `False` | Progress messages |

### Methods

```python
X_selected = selector.fit_transform(
    X_train, y_train,
    fdr_level=0.05,   # tsfresh stage (ignored for random_forest)
    top_n=20,         # RandomForest stage (ignored for tsfresh)
)
scores = selector.get_feature_scores()   # DataFrame with feature rankings
```
