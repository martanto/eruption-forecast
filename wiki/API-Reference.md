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
| `location` | `str \| None` | `None` | Seismic location code. `None` and `""` are both accepted and treated as an empty location code |
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
| `cv_strategy` | `str` | `"shuffle-stratified"` | CV strategy — `"shuffle"`, `"stratified"`, `"shuffle-stratified"`, `"timeseries"` |
| `cv_splits` | `int` | `5` | Number of CV folds |
| `number_of_significant_features` | `int` | `20` | Top-N features retained per seed |
| `feature_selection_method` | `str` | `"tsfresh"` | Feature selection — `"tsfresh"`, `"random_forest"`, `"combined"` |
| `overwrite` | `bool` | `False` | Re-run even if output already exists |
| `n_jobs` | `int` | `1` | Outer parallel workers (one per seed). `-1` = all cores. Enforced: `n_jobs × grid_search_n_jobs ≤ cpu_count` |
| `grid_search_n_jobs` | `int` | `1` | Parallel jobs inside `GridSearchCV` and `FeatureSelector`. When `use_gpu=True` this is still used by `FeatureSelector` (CPU-only), but `GridSearchCV` is forced to `1` |
| `use_gpu` | `bool` | `False` | Enable GPU acceleration for XGBoost via `device="cuda:<gpu_id>"`. Forces `n_jobs=1` and `GridSearchCV` `n_jobs=1` to prevent VRAM contention. Has no effect for non-XGBoost classifiers |
| `gpu_id` | `int` | `0` | GPU device index when `use_gpu=True`. Use `0` for the first GPU, `1` for the second, etc. |
| `verbose` | `bool` | `False` | Print progress messages |
| `debug` | `bool` | `False` | Enable debug-level logging |

### `evaluate()` Parameters

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
| `with_evaluation` | `bool` | `True` | `True` → `evaluate()`; `False` → `train()` |
| `**kwargs` | — | — | Forwarded to the dispatched method |

### Merge Methods

| Method | Description |
|--------|-------------|
| `merge_models(output_path=None)` | Bundle all seed models for this classifier into a single `SeedEnsemble` `.pkl`. Default output: same directory as the registry CSV, named `merged_model_{suffix}.pkl`. Requires training to have been run first (`self.csv` must be set). |
| `merge_classifier_models(trained_models, output_path=None)` | Bundle multiple classifier registry CSVs into one `dict[str, SeedEnsemble]` `.pkl`. Accepts `{"rf": "path/to/rf.csv", "xgb": "path/to/xgb.csv"}`. |

---

## SeedEnsemble

Bundles all seed models for a single classifier into one serialisable object. Subclasses `sklearn.base.BaseEstimator` and `ClassifierMixin`.

```python
from eruption_forecast.model.seed_ensemble import SeedEnsemble
# or
from eruption_forecast.model import SeedEnsemble
```

### Construction

```python
# Build from a trained-model registry CSV
ensemble = SeedEnsemble.from_registry("path/to/trained_model_*.csv")

# Load a previously saved ensemble
ensemble = SeedEnsemble.load("path/to/merged_model_*.pkl")
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `SeedEnsemble.from_registry(registry_csv)` | `SeedEnsemble` | Load all seeds from a registry CSV into memory |
| `SeedEnsemble.load(path)` | `SeedEnsemble` | Restore from a `.pkl` file written by `save()` |
| `predict_proba(X)` | `np.ndarray (n, 2)` | sklearn-compatible class probabilities — column 0: P(non-eruption), column 1: mean P(eruption) across seeds |
| `predict(X)` | `np.ndarray (n,)` | Binary predictions using 0.5 threshold on mean P(eruption) |
| `predict_with_uncertainty(X, threshold)` | `tuple[ndarray × 4]` | `(mean_probability, std, confidence, prediction)` — richer output than `predict_proba()` |
| `save(path)` | `None` | Serialise with joblib to a single `.pkl` |
| `len(ensemble)` | `int` | Number of seeds in the bundle |

### Standalone merge functions

```python
from eruption_forecast.utils.ml import merge_seed_models, merge_all_classifiers
```

| Function | Description |
|----------|-------------|
| `merge_seed_models(registry_csv, output_path=None)` | One classifier → one `SeedEnsemble` `.pkl`. Default name: `merged_model_{suffix}.pkl` alongside the registry CSV. |
| `merge_all_classifiers(trained_models, output_path=None)` | Multiple classifiers → one `dict[str, SeedEnsemble]` `.pkl`. `trained_models` is `{"rf": "rf_registry.csv", ...}`. Default name: `merged_classifiers_{suffix}.pkl` in the `trainings/` root. |

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
| `trained_models` | `str \| dict[str, str]` | — | Single registry CSV path, path to a merged `.pkl` (`SeedEnsemble` or `dict[str, SeedEnsemble]`), or `{name: path}` dict for multi-model consensus |
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
df_forecast = predictor.predict_proba(tremor_data="path/to/tremor.csv", window_size=2, window_step=12,
                                      window_step_unit="hours")
```

---

## ModelEvaluator

Single-seed evaluation.

```python
from eruption_forecast import ModelEvaluator
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `BaseEstimator \| GridSearchCV` | — | Fitted sklearn estimator |
| `X_test` | `pd.DataFrame` | — | Test feature DataFrame |
| `y_test` | `pd.Series` | — | True test labels |
| `model_name` | `str` | `"model"` | Identifier used in filenames and plot titles |
| `output_dir` | `str \| None` | `None` | Output directory; when `None`, auto-constructed as `output/trainings/evaluations/{clf-slug}/{cv-slug}/` |
| `cv_name` | `str` | `"cv"` | CV strategy name slugified into the default output path (e.g. `"StratifiedKFold"` → `stratified-k-fold`) |
| `selected_features` | `list[str] \| None` | `None` | Filter `X_test` to these columns before predicting |
| `random_state` | `int \| None` | `None` | Seed for filename prefix |
| `root_dir` | `str \| None` | `None` | Anchor for resolving relative `output_dir` |
| `plot_shap` | `bool` | `False` | Enable SHAP plots in `plot_all()` |
| `verbose` | `bool` | `False` | Emit progress log messages |

### Key Methods

| Method | Description |
|--------|-------------|
| `ModelEvaluator.from_files(model_path, X_test, y_test, ...)` | Load from disk |
| `get_metrics()` | Returns dict with all metrics |
| `summary()` | Prints formatted summary |
| `plot_all()` | Generates all evaluation plots (SHAP plots require `plot_shap=True`) |
| `plot_shap_summary(max_display)` | SHAP beeswarm for the test set |
| `plot_shap_waterfall(max_display)` | SHAP waterfall for the highest-probability sample |
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
| `plot_shap_summary()` | `trained_model_csv` | Aggregate SHAP beeswarm across seeds |
| `plot_shap_waterfall()` | `trained_model_csv` | Mean SHAP waterfall for highest-prob sample |
| `plot_seed_stability()` | `trained_model_csv` | Violin plot of a metric across seeds |
| `plot_frequency_band_contribution()` | `trained_model_csv` | Feature counts per seismic band |
| `get_aggregate_metrics()` | `metrics_dir` | DataFrame: metric × mean/std/min/max |
| `get_metrics_list()` | `metrics_dir` | Raw per-seed metrics as list of dicts |
| `save_aggregate_metrics(path=None)` | `metrics_dir` | Saves summary to CSV |

---

## ClassifierComparator

Compare multiple classifiers side-by-side with ranking tables and comparison plots.

```python
from eruption_forecast.model import ClassifierComparator
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `classifiers` | `dict[str, str]` | required | Mapping of classifier name → trained-model registry CSV path |
| `output_dir` | `str \| None` | `None` | Root output directory (defaults to `cwd/output/comparison/`) |
| `metrics` | `str \| list[str] \| None` | `None` | Metrics for plots; `None` uses all `DEFAULT_METRICS` |

```python
# From a dict
comparator = ClassifierComparator(
    classifiers={
        "rf":  "output/.../trained_model_rf_...csv",
        "xgb": "output/.../trained_model_xgb_...csv",
    },
    metrics=["f1_score", "roc_auc"],  # or None for all DEFAULT_METRICS
)

# From a JSON file  {name: csv_path, ...}
comparator = ClassifierComparator.from_json(
    "output/VG.OJN.00.EHZ/evaluations_trained_models.json",
    metrics=["f1_score", "roc_auc"],
)
```

### Key Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `ClassifierComparator.from_json(json_path, output_dir, metrics)` | `Self` | Load classifiers from a JSON file and construct instance |
| `get_metrics_table()` | `pd.DataFrame` | Aggregate metrics for all classifiers (one row per classifier) |
| `get_ranking(metric, by)` | `pd.DataFrame` | Sorted ranking with rank column; default metric `"recall"`; saved to `metrics/ranking_{metric}.csv` |
| `plot_metric_bar(metrics, save, filename, dpi)` | `dict[str, Figure]` | Bar chart per metric (mean ± std) + `"all"` overview; saved to `figures/metric_bar_{metric}.png` + `metric_bar_all.png` |
| `plot_seed_stability(metrics, save, filename, dpi)` | `dict[str, Figure]` | Violin + strip per metric + `"all"` overview; saved to `figures/seed_stability_{metric}.png` + `seed_stability_all.png` |
| `plot_comparison_grid(metrics, save, filename, dpi)` | `Figure` | Grid of subplots (rows = classifiers, cols = metrics); saved to `figures/comparison_grid.png` |
| `plot_roc(save, filename, dpi, show_individual)` | `Figure` | Overlaid mean ROC curves with ± std bands; saved to `figures/comparison_roc.png` |
| `plot_all(dpi)` | `dict[str, Any]` | All plots + ranking; keys: `metric_bar`, `seed_stability`, `comparison_grid`, `roc`, `ranking` |

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

---

## Logger Utilities

Functions for controlling package-wide logging output at runtime.

```python
from eruption_forecast import disable_logging, enable_logging
# or
from eruption_forecast.logger import disable_logging, enable_logging, set_log_level, set_log_directory
```

### `disable_logging()`

Removes all active loguru handlers so no messages are written to the console or log files.

```python
disable_logging()
fm.calculate(...)  # Silent — no output
```

### `enable_logging()`

Restores console and file handlers using the current log directory. Has no effect if logging is already enabled.

```python
enable_logging()
```

### `set_log_level(level)`

Changes the console log level dynamically. File handlers retain their original levels.

| Parameter | Type | Description |
|-----------|------|-------------|
| `level` | `str` | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, or `"CRITICAL"`. Case-insensitive. |

### `set_log_directory(log_dir)`

Changes the log file output directory. Creates the directory if it does not exist.

| Parameter | Type | Description |
|-----------|------|-------------|
| `log_dir` | `str` | Absolute or relative path to the new log directory. |
