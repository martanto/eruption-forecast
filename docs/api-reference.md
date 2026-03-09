# API Reference

> Back to [README](../README.md)

Full constructor and method parameter tables for the main pipeline classes.

## Table of Contents

- [ForecastModel](#forecastmodel)
- [ModelTrainer](#modeltrainer)
- [ModelPredictor](#modelpredictor)

---

## ForecastModel

`ForecastModel` orchestrates the full pipeline via method chaining.

```python
from eruption_forecast import ForecastModel
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `station` | `str` | — | Seismic station code (e.g. `"OJN"`) |
| `channel` | `str` | — | Seismic channel code (e.g. `"EHZ"`) |
| `start_date` | `str \| datetime` | — | Training period start date (`YYYY-MM-DD`) |
| `end_date` | `str \| datetime` | — | Training period end date (`YYYY-MM-DD`) |
| `window_size` | `int` | — | Duration (days) of each tremor window fed into tsfresh |
| `volcano_id` | `str` | — | Identifier used in output filenames (e.g. `"Lewotobi Laki-laki"`) |
| `network` | `str` | `"VG"` | Seismic network code |
| `location` | `str` | `"00"` | Seismic location code |
| `output_dir` | `str \| None` | `None` | Base output directory; relative paths are resolved against `root_dir`. Defaults to `root_dir/output` |
| `root_dir` | `str \| None` | `None` | Anchor for resolving relative `output_dir`. Relative values are normalised to an absolute path immediately. Defaults to `os.getcwd()` |
| `overwrite` | `bool` | `False` | Re-run and overwrite existing output files |
| `n_jobs` | `int` | `1` | Parallel workers propagated to all pipeline stages |
| `verbose` | `bool` | `False` | Enable verbose logging |
| `debug` | `bool` | `False` | Enable debug-level logging |

### Additional Methods

| Method | Description |
|--------|-------------|
| `load_tremor_data(tremor_csv)` | Load pre-calculated tremor data instead of calling `calculate()` |
| `set_feature_selection_method(using)` | Change feature selection method before `train()` |
| `save_config(path=None, fmt="yaml")` | Saves accumulated config; defaults to `{station_dir}/config.yaml` |
| `save_model(path=None)` | joblib-dumps the full instance; defaults to `{station_dir}/forecast_model.pkl` |
| `ForecastModel.from_config(path)` | Loads config, constructs instance, attaches `_loaded_config` |
| `ForecastModel.load_model(path)` | Restores a joblib-pickled instance with all pipeline state |
| `run()` | Replays all stages from `_loaded_config`; only valid after `from_config()` |

---

## ModelTrainer

`ModelTrainer` trains classifiers across multiple random seeds.

```python
from eruption_forecast.model.model_trainer import ModelTrainer
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extracted_features_csv` | `str` | — | Path to extracted features CSV (output of `FeaturesBuilder`) |
| `label_features_csv` | `str` | — | Path to aligned labels CSV (output of `FeaturesBuilder`) |
| `output_dir` | `str \| None` | `None` | Output directory; resolved against `root_dir` when relative. Defaults to `root_dir/output/trainings` |
| `root_dir` | `str \| None` | `None` | Anchor for resolving relative `output_dir`. Defaults to `os.getcwd()` |
| `prefix_filename` | `str \| None` | `None` | Optional prefix prepended to every output filename |
| `classifier` | `str` | `"rf"` | Classifier type — see [Supported Classifiers](../docs/step-by-step-guide.md#7-supported-classifiers). `ForecastModel.train()` also accepts a `list[str]` or comma-separated string to train multiple classifiers in sequence |
| `cv_strategy` | `str` | `"shuffle"` | Cross-validation strategy — `"shuffle"`, `"stratified"`, or `"timeseries"` |
| `cv_splits` | `int` | `5` | Number of CV folds |
| `number_of_significant_features` | `int` | `20` | Top-N features retained per seed and aggregated across seeds |
| `feature_selection_method` | `str` | `"tsfresh"` | Feature selection algorithm — `"tsfresh"`, `"random_forest"`, or `"combined"` |
| `overwrite` | `bool` | `False` | Re-run even if output files already exist |
| `n_jobs` | `int` | `1` | Parallel seed workers (outer loop). Pass `-1` to use all available cores. Enforced: `n_jobs × grid_search_n_jobs ≤ cpu_count` |
| `grid_search_n_jobs` | `int` | `1` | Parallel jobs inside each `GridSearchCV` call and `FeatureSelector` (inner loop). Uses `loky` backend — safe for Intel's scikit-learn extension. When `use_gpu=True`, `GridSearchCV` is forced to `1` but `FeatureSelector` keeps the configured value |
| `use_gpu` | `bool` | `False` | Enable GPU acceleration for XGBoost via `device="cuda:<gpu_id>"`. Forces outer `n_jobs=1` and `GridSearchCV` inner `n_jobs=1` to prevent VRAM contention. No effect on non-XGBoost classifiers |
| `gpu_id` | `int` | `0` | GPU device index when `use_gpu=True`. Use `0` for the first GPU, `1` for the second |
| `verbose` | `bool` | `False` | Print progress messages |
| `debug` | `bool` | `False` | Enable debug-level logging |

### `train_and_evaluate()` Parameters

Splits data **before** resampling and feature selection to prevent data leakage.
Evaluates each seed on the held-out 20% and aggregates metrics across seeds.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `0` | Starting random seed; seeds are `random_state, random_state+1, …, random_state+total_seed−1` |
| `total_seed` | `int` | `500` | Number of seeds (independent train/test splits) to run |
| `sampling_strategy` | `str \| float` | `0.75` | Under-sampling ratio for `RandomUnderSampler` on training data only |
| `save_all_features` | `bool` | `False` | Save all ranked features per seed (can produce many files) |
| `plot_significant_features` | `bool` | `False` | Save a feature-importance plot per seed |

### `train()` Parameters

Trains on the **entire current/present dataset** across multiple seeds — no internal 80/20 split.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `0` | Starting random seed |
| `total_seed` | `int` | `500` | Number of seeds to run |
| `sampling_strategy` | `str \| float` | `0.75` | Under-sampling ratio for `RandomUnderSampler` on full dataset |
| `save_all_features` | `bool` | `False` | Save all ranked features per seed |
| `plot_significant_features` | `bool` | `False` | Save a feature-importance plot per seed |

### `fit()` Parameters

`fit()` dispatches to `train_and_evaluate()` or `train()` based on the `with_evaluation` flag.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `with_evaluation` | `bool` | `True` | `True` → `train_and_evaluate()` (80/20 split + metrics); `False` → `train()` (full dataset, no metrics) |
| `**kwargs` | — | — | Forwarded to `train_and_evaluate()` or `train()` (same parameters as those methods) |

### `compute_learning_curve()` Parameters

Computes a scikit-learn learning curve for a single seed and saves the result as a JSON file. Called internally by `train_and_evaluate()` and `train()`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | `pd.DataFrame` | — | Feature matrix |
| `y` | `pd.Series` | — | Target labels |
| `classifier_model` | `ClassifierModel` | — | Fitted classifier whose estimator is evaluated |
| `scoring` | `str \| list[str]` | `LEARNING_CURVE_SCORINGS` | Scoring metric(s) — a single string or list of strings. Each metric is a key in the output JSON. |

**Returns:** `dict` mapping each scoring metric name to `{"train_sizes", "train_scores", "test_scores"}`.

```python
from eruption_forecast.model.model_trainer import ModelTrainer
from eruption_forecast.config.constants import LEARNING_CURVE_SCORINGS

# LEARNING_CURVE_SCORINGS = ["balanced_accuracy", "f1_weighted"]

trainer = ModelTrainer(
    extracted_features_csv="features.csv",
    label_features_csv="labels.csv",
)

# Called internally — but can be invoked directly:
lc = trainer.compute_learning_curve(X, y, classifier_model, scoring=LEARNING_CURVE_SCORINGS)
# lc == {"balanced_accuracy": {...}, "f1_weighted": {...}}
```

### Merge Methods

| Method | Description |
|--------|-------------|
| `merge_models(output_path=None)` | Bundle all seed models for this classifier into a single `SeedEnsemble` `.pkl`. Default output: alongside the registry CSV as `merged_model_{suffix}.pkl`. Must call `train()` or `train_and_evaluate()` first. |
| `merge_classifier_models(trained_models, output_path=None)` | Bundle multiple classifier registry CSVs into one `dict[str, SeedEnsemble]` `.pkl`. Accepts `{"rf": "path/to/rf.csv", "xgb": "path/to/xgb.csv"}`. |

---

## BaseEnsemble

Shared persistence mixin inherited by `SeedEnsemble` and `ClassifierEnsemble`.

```python
from eruption_forecast.model.base_ensemble import BaseEnsemble
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `save(path)` | `None` | Joblib-dumps the ensemble to `.pkl`; creates missing parent directories automatically |
| `BaseEnsemble.load(path)` (classmethod) | `BaseEnsemble` | Restores instance from `.pkl`; raises `FileNotFoundError` if the file does not exist |

```python
from eruption_forecast.model.seed_ensemble import SeedEnsemble

ensemble = SeedEnsemble.from_registry("path/to/trained_model_*.csv")
ensemble.save("output/merged.pkl")

# Reload later
ensemble = SeedEnsemble.load("output/merged.pkl")
```

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
| `predict_proba(X)` | `np.ndarray (n, 2)` | sklearn-compatible — column 0: P(non-eruption), column 1: mean P(eruption) across seeds |
| `predict(X)` | `np.ndarray (n,)` | Binary predictions using 0.5 threshold |
| `predict_with_uncertainty(X, threshold)` | `tuple[ndarray × 4]` | `(mean_probability, std, confidence, prediction)` |
| `save(path)` | `None` | Serialise to a single `.pkl` via joblib |
| `len(ensemble)` | `int` | Number of seeds |

### Standalone merge functions

```python
from eruption_forecast.utils.ml import merge_seed_models, merge_all_classifiers
```

| Function | Description |
|----------|-------------|
| `merge_seed_models(registry_csv, output_path=None)` | One classifier → one `SeedEnsemble` `.pkl`. Default: `merged_model_{suffix}.pkl` alongside the registry CSV. |
| `merge_all_classifiers(trained_models, output_path=None)` | Multiple classifiers → one `dict[str, SeedEnsemble]` `.pkl`. `trained_models` is `{"rf": "rf.csv", "xgb": "xgb.csv"}`. Default: `merged_classifiers_{suffix}.pkl` in the `trainings/` root. |

---

## ModelPredictor

`ModelPredictor` runs inference in evaluation mode (labelled data) or forecast mode (unlabelled data).

```python
from eruption_forecast.model.model_predictor import ModelPredictor
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | `str \| datetime` | — | Start date for prediction period (format: YYYY-MM-DD) |
| `end_date` | `str \| datetime` | — | End date for prediction period (format: YYYY-MM-DD) |
| `trained_models` | `str \| dict[str, str]` | — | Single `trained_model_*.csv` path, a merged `SeedEnsemble` `.pkl` path, a multi-classifier bundle `.pkl` path, or a `{name: path}` dict for multi-model consensus |
| `overwrite` | `bool` | `False` | Overwrite existing output files |
| `n_jobs` | `int` | `1` | Number of parallel jobs for feature extraction |
| `output_dir` | `str \| None` | `None` | Output directory; defaults to `<root_dir>/output/predictions` |
| `root_dir` | `str \| None` | `None` | Root directory for resolving output paths |
| `verbose` | `bool` | `False` | Enable verbose logging |

### Methods

#### `predict()` — Evaluation Mode (labelled data)

Evaluates each seed model against known eruption labels and aggregates metrics across seeds.

```python
df_metrics = predictor.predict(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
)
```

#### `predict_best()` — Best Seed Evaluation

Returns a `ModelEvaluator` for the best-performing seed. Accepts any metric column as `criterion`:
`"accuracy"`, `"balanced_accuracy"`, `"f1_score"`, `"precision"`, `"recall"`, `"roc_auc"`, `"pr_auc"`.

```python
evaluator = predictor.predict_best(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
    criterion="balanced_accuracy",
)
```

#### `predict_proba()` — Forecast Mode (unlabelled data)

When no ground-truth labels are available. Aggregates within each classifier (across seeds) and then across classifiers (consensus) for multi-model mode.

```python
df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",  # or pd.DataFrame
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

### Multi-Model Output Columns

| Column | Description |
|--------|-------------|
| `{name}_eruption_probability` | Mean P(eruption) across seeds of that classifier |
| `{name}_uncertainty` | Std across seeds of that classifier |
| `{name}_confidence` | Seed-level agreement fraction (0.5–1.0) |
| `{name}_prediction` | Hard label for that classifier |
| `consensus_eruption_probability` | Mean P(eruption) averaged across all classifiers |
| `consensus_uncertainty` | Std of per-classifier means (inter-model disagreement) |
| `consensus_confidence` | Fraction of classifiers voting with consensus majority |
| `consensus_prediction` | Hard label — `1` if `consensus_eruption_probability ≥ 0.5` |
