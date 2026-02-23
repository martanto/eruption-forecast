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
| `grid_search_n_jobs` | `int` | `1` | Parallel jobs inside each `GridSearchCV` call (inner loop). Uses `loky` backend — safe for Intel's scikit-learn extension |
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
| `trained_models` | `str \| dict[str, str]` | — | Single `trained_model_*.csv` path (from `train()`) or a `{name: path}` dict for multi-model consensus |
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
