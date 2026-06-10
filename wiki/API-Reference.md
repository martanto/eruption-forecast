# API Reference

Parameter tables and method signatures for every public class exported from `eruption_forecast`. Imports throughout this page:

```python
from eruption_forecast import (
    ForecastModel,
    TrainingModel,
    PredictionModel,
    EvaluationModel,
    CalculateTremor,
    LabelBuilder,
    DynamicLabelBuilder,
    FeaturesBuilder,
    TremorMatrixBuilder,
    TremorData,
    LabelData,
    enable_logging,
    disable_logging,
    notify,
    send_telegram_notification,
)
from eruption_forecast.ensemble import SeedEnsemble, ClassifierEnsemble
from eruption_forecast.ensemble.base_ensemble import BaseEnsemble
from eruption_forecast.ensemble.metrics_ensemble import MetricsEnsemble
from eruption_forecast.model.classifier_comparator import ClassifierComparator
from eruption_forecast.features.feature_selector import FeatureSelector
```

---

## ForecastModel

Top-level pipeline orchestrator. See [Pipeline Walkthrough](Pipeline-Walkthrough) for full examples.

### Constructor

```python
ForecastModel(
    station: str,
    channel: str,
    network: str,
    location: str = "",
    day_to_forecast: int = 2,
    output_dir: str | None = None,
    root_dir: str | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
    verbose: bool = False,
)
```

| Param | Type | Default | Notes |
|-------|------|---------|-------|
| `station` | `str` | — | Station code (uppercased) |
| `channel` | `str` | — | Channel code (uppercased) |
| `network` | `str` | — | FDSN network code |
| `location` | `str` | `""` | FDSN location code |
| `day_to_forecast` | `int` | `2` | Look-ahead window in days; threaded into `TrainingModel`/`PredictionModel` as `window_size` |
| `output_dir` | `str \| None` | `None` | Defaults to `{cwd}/output` (or `{root_dir}/output` when `root_dir` set) |
| `root_dir` | `str \| None` | `None` | Anchor for relative `output_dir` |
| `overwrite` | `bool` | `False` | Default for stage methods |
| `n_jobs` | `int` | `1` | Default for stage methods; clamped to `cpu_count - 2` |
| `verbose` | `bool` | `False` | Default for stage methods |

### `calculate(...)`

```python
fm.calculate(
    start_date: str | datetime,
    end_date: str | datetime,
    source: Literal["sds", "fdsn"] = "sds",
    methods: str | list[str] | None = None,
    remove_outlier_method: Literal["all", "maximum"] = "maximum",
    remove_tremor_anomalies: bool = False,
    interpolate: bool = True,
    value_multiplier: float | None = None,
    cleanup_daily_dir: bool = False,
    plot_daily: bool = False,
    save_plot: bool = False,
    overwrite_plot: bool = False,
    sds_dir: str | None = None,
    client_url: str = "https://service.iris.edu",
    minimum_completion_ratio: float = 0.3,
    overwrite: bool | None = None,
    n_jobs: int | None = None,
    verbose: bool | None = None,
) -> Self
```

`start_date` is internally pushed back by `day_to_forecast` days for full lead-in coverage. `sds_dir` is required when `source="sds"`. `None` for `overwrite`/`n_jobs`/`verbose` means "inherit from constructor". Sets `self.CalculateTremor`, `self.tremor_df`, `self.tremor_start_date`, `self.tremor_end_date`.

### `train(...)`

```python
fm.train(
    start_date: str | datetime,
    end_date: str | datetime,
    eruption_dates: list[str],
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
    label_builder: Literal["standard", "dynamic"] = "standard",
    days_before_eruption: int | None = None,
    classifiers: str | list[str] = "rf",
    cv_strategy: Literal["shuffle", "stratified", "shuffle-stratified"] = "shuffle-stratified",
    cv_splits: int = 5,
    scoring: str = "balanced_accuracy",
    top_n_features: int = 20,
    include_eruption_date: bool = True,
    select_tremor_columns: list[str] | None = None,
    save_tremor_matrix_per_method: bool = True,
    exclude_features: list[str] | None = None,
    select_features: str | list[str] | None = None,
    minimum_completion: float = 1.0,
    seeds: int = 10,
    resample_method: Literal["under", "over", "auto"] | None = "auto",
    minority_threshold: float = 0.15,
    sampling_strategy: str | float = 0.75,
    plot_features: bool = True,
    output_dir: str | None = None,
    overwrite: bool | None = None,
    n_jobs: int | None = None,
    n_grids: int = 1,
    use_cache: bool = True,
    verbose: bool | None = None,
) -> Self
```

Requires `calculate()` to have populated tremor data first. `label_builder="dynamic"` requires `days_before_eruption`. `use_cache=True` short-circuits via `TrainingModel.load_from_cache` when the cache identity matches. Sets `self.TrainingModel`, `self.ClassifierEnsemble`, and `self._training_cache_hash`.

### `predict(...)`

```python
fm.predict(
    start_date: str | datetime,
    end_date: str | datetime,
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
    save_seed_result: bool = True,
    plot_threshold: float = 0.5,
    plot_title: str | None = None,
    plot_pdf: bool = True,
    output_dir: str | None = None,
    overwrite: bool | None = None,
    n_jobs: int | None = None,
    use_cache: bool = True,
    verbose: bool | None = None,
    **plot_kwargs: Any,
) -> Self
```

Requires `train()` first. `**plot_kwargs` is forwarded to `plot_forecast` (see [Visualization](Visualization#forecast--plot_forecast) for keys) and is **not** captured in `ForecastConfig` because matplotlib objects do not round-trip through YAML. Sets `self.PredictionModel`, `self.results`.

### `evaluate(...)`

```python
fm.evaluate(
    model: Literal["training", "prediction"] = "prediction",
    eruption_dates: list[str] | None = None,
    plot_per_seed: bool = False,
    plot_aggregate: bool = True,
    output_dir: str | None = None,
    overwrite: bool | None = None,
    n_jobs: int | None = None,
    verbose: bool | None = None,
) -> Self
```

`eruption_dates=None` falls back to the dates captured during `train()`. Always auto-calls `save_config()` at the end. Sets `self.EvaluationModel`, `self.evaluation_results`.

### Config round-trip

| Method | Returns | Notes |
|--------|---------|-------|
| `fm.save_config(path=None, fmt="yaml")` | `str` | Defaults to `{station_dir}/forecast.config.{yaml,json}` |
| `ForecastModel.from_config(path)` | `ForecastModel` | Classmethod; restores the captured `ForecastConfig` |
| `fm.run()` | `Self` | Idempotent replay of every captured non-`None` stage |

---

## TrainingModel

`BaseModel + CacheModel` subclass. Use standalone when running outside `ForecastModel`; otherwise `fm.train(...)` constructs it for you.

### Constructor

```python
TrainingModel(
    tremor_data: str | pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    classifiers: str | list[str],
    eruption_dates: list[str],
    window_size: int = 2,
    cv_strategy: Literal["shuffle", "stratified", "shuffle-stratified"] = "shuffle-stratified",
    cv_splits: int = 5,
    top_n_features: int = 20,
    include_eruption_date: bool = False,
    output_dir: str | None = None,
    root_dir: str | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
    n_grids: int = 1,
    verbose: bool = False,
)
```

### Pipeline methods (all return `Self`)

```python
tm.build_label(
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
    builder: Literal["standard", "dynamic"] = "standard",
    days_before_eruption: int | None = None,
    verbose: bool | None = None,
)

tm.extract_features(
    select_tremor_columns: list[str] | None = None,
    save_tremor_matrix_per_method: bool = False,
    exclude_features: list[str] | None = None,
    select_features: str | list[str] | None = None,
    save_tremor_matrix_per_id: bool = False,
    minimum_completion: float = 1.0,
    overwrite: bool = False,
    n_jobs: int | None = None,
    verbose: bool | None = None,
)

tm.fit(
    seeds: int = 25,
    resample_method: Literal["under", "over", "auto"] | None = "auto",
    minority_threshold: float = 0.15,
    sampling_strategy: str | float = 0.75,
    plot_features: bool = False,
    scoring: str = "balanced_accuracy",
    compute_learning_curve: bool = False,
)
```

### Cache + persistence

| Method | Notes |
|--------|-------|
| `TrainingModel.build_cache_identity(**kwargs)` | Classmethod; returns canonical identity dict for hashing |
| `tm.save_to_cache(identity)` | Writes `{output_dir}/cache/TrainingModel/{hash}.pkl` (+ `.params.json`) |
| `TrainingModel.load_from_cache(output_dir, identity)` | Classmethod; returns the instance or `None` |
| `tm.save(path=None)` | `{output_dir}/TrainingModel_{basename}.pkl` |
| `TrainingModel.load(path)` | Classmethod |
| `tm.save_config(path=None)` | Standalone `training.config.yaml` |

Populated attributes after `fit()`: `tm.results` (per-classifier registry CSV paths), `tm.ClassifierEnsemble`, `tm.classifier_ensemble_path`, `tm.features_df`, `tm.labels`.

---

## PredictionModel

`BaseModel + CacheModel` subclass.

### Constructor

```python
PredictionModel(
    model: str | ClassifierEnsemble | SeedEnsemble,
    tremor_data: str | pd.DataFrame,
    start_date: str | datetime,
    end_date: str | datetime,
    window_size: int = 2,
    overwrite: bool = False,
    output_dir: str | None = None,
    root_dir: str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
)
```

`model` accepts a live `ClassifierEnsemble` / `SeedEnsemble`, a `ClassifierEnsemble.json` / `.pkl`, a `SeedEnsemble_*.pkl`, or a trained-model registry `.csv` — resolved via `ClassifierEnsemble.from_any(...)`.

### Pipeline methods

```python
pm.build_label(
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
) -> Self

pm.extract_features(
    select_tremor_columns: list[str] | None = None,
    save_tremor_matrix_per_method: bool = False,
    exclude_features: list[str] | None = None,
    overwrite: bool = False,
    n_jobs: int | None = None,
    verbose: bool | None = None,
) -> Self

pm.forecast(
    save_seed_result: bool = True,
    plot_threshold: float = 0.5,
    plot_title: str | None = None,
    plot_pdf: bool = True,
    **plot_kwargs,
) -> pd.DataFrame
```

`forecast()` returns the results DataFrame indexed by datetime with one column per `{classifier}_{eruption_probability|uncertainty|confidence|prediction}` plus the four `consensus_*` columns. Also sets `pm.results` and `pm.forecast_plot_path`.

### Cache + persistence

Same surface as `TrainingModel`: `build_cache_identity`, `save_to_cache`, `load_from_cache`, `save`, `load`. The cache identity embeds the upstream `training_hash`.

---

## EvaluationModel

`BaseModel` subclass (no cache).

### Constructor

```python
EvaluationModel(
    model: TrainingModel | PredictionModel,
    eruption_dates: list[str] | None = None,
    overwrite: bool = False,
    output_dir: str | None = None,
    root_dir: str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
)
```

Raises `ValueError` if `model` is a `PredictionModel` and `eruption_dates is None`, or if `model.ClassifierEnsemble is None`. Output is namespaced to `evaluation/{model.kind}/`.

### Methods

```python
em.evaluate(
    plot_aggregate: bool = True,
    plot_per_seed: bool = False,
    plot_shap: bool = False,
    compare_classifiers: bool = True,
) -> dict[str, pd.DataFrame]

em.compare(
    metrics: str | list[str] | None = None,
) -> ClassifierComparator
```

```python
EvaluationModel.from_file(
    filepath: str,
    eruption_dates: list[str] | None = None,
    overwrite: bool = False,
    output_dir: str | None = None,
    root_dir: str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> EvaluationModel
```

Classmethod. Loads a `.pkl` produced by `TrainingModel.save()` or `PredictionModel.save()` and dispatches on `kind`.

`plot_per_seed=True` and `plot_shap=True` currently emit a warning and are no-ops (reserved for a follow-up that rebuilds plots from `MetricsEnsemble`'s persisted matrices).

---

## CalculateTremor

### Constructor

```python
CalculateTremor(
    start_date: str | datetime,
    end_date: str | datetime,
    station: str,
    channel: str,
    network: str,
    location: str | None = None,
    channel_type: str = "D",
    methods: list[str] | None = None,
    output_dir: str | None = None,
    root_dir: str | None = None,
    overwrite: bool = False,
    remove_outlier_method: Literal["all", "maximum"] = "maximum",
    remove_tremor_anomalies: bool = False,
    interpolate: bool = False,
    value_multiplier: float | None = None,
    cleanup_daily_dir: bool = False,
    plot_daily: bool = False,
    save_plot: bool = False,
    overwrite_plot: bool = False,
    filename_prefix: str | None = None,
    minimum_completion_ratio: float = 0.3,
    n_jobs: int = 1,
    verbose: bool = False,
    debug: bool = False,
)
```

### Source binding + execution (all return `Self`)

```python
ct.from_sds(sds_dir: str)
ct.from_fdsn(client_url: str | None = None)
ct.change_freq_bands(freq_bands: list[tuple[float, float]])
ct.run()
```

After `run()`: `ct.df` (the tremor DataFrame), `ct.csv` (path to the merged file), `ct.daily_files`, `ct.daily_dir`.

---

## LabelBuilder / DynamicLabelBuilder

```python
LabelBuilder(
    start_date: str | datetime,
    end_date: str | datetime,
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
    day_to_forecast: int,
    eruption_dates: list[str] | list[datetime],
    volcano_id: str | None = None,
    include_eruption_date: bool = True,
    output_dir: str | None = None,
    root_dir: str | None = None,
    verbose: bool = False,
    debug: bool = False,
)

DynamicLabelBuilder(
    days_before_eruption: int,
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
    day_to_forecast: int,
    eruption_dates: list[str],
    volcano_id: str | None = None,
    output_dir: str | None = None,
    root_dir: str | None = None,
    prefix_filename: str | None = None,
    verbose: bool = False,
    debug: bool = False,
)
```

Both expose `.build() -> Self`. After `build()`: `lb.df` (DateTime-indexed `id`/`is_erupted` frame), `lb.csv` (path to the label CSV).

Note that within `TrainingModel`, `include_eruption_date` defaults to `False` (training pipeline behaviour); the `LabelBuilder` constructor's own default is `True`. The difference is intentional — see [Training Workflow](Training-Workflow).

---

## FeaturesBuilder

```python
FeaturesBuilder(
    tremor_matrix_df: pd.DataFrame,
    output_dir: str | None = None,
    label_df: pd.DataFrame | None = None,
    select_features: list[str] | None = None,
    root_dir: str | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
    verbose: bool = False,
)

fb.extract_features(
    select_tremor_columns: list[str] | None = None,
    exclude_features: list[str] | None = None,
) -> pd.DataFrame
```

`label_df=None` switches the builder to **prediction mode** (no tsfresh relevance filtering). `select_features` pre-filters tsfresh to the supplied fully-qualified feature names.

---

## TremorMatrixBuilder

```python
TremorMatrixBuilder(
    tremor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    output_dir: str | None = None,
    window_size: int = 1,
    root_dir: str | None = None,
    minimum_completion: float = 1.0,
    overwrite: bool = False,
    verbose: bool = False,
)

tmb.build(
    select_tremor_columns: list[str] | None = None,
    save_tremor_matrix_per_method: bool = False,
    save_tremor_matrix_per_id: bool = False,
) -> Self
```

After `build()`: `tmb.df` is the matrix with `id`, `datetime`, and tremor columns.

---

## FeatureSelector

```python
FeatureSelector(
    method: Literal["tsfresh", "random_forest", "combined"] = "tsfresh",
    random_state: int = 42,
    output_dir: str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
)
```

Used internally by `TrainingModel.fit()` per-seed; surfaced publicly for ad-hoc selection experiments. Populated after `fit(X, y)`: `selected_features_`, `p_values_`, `importance_scores_`, `n_features_tsfresh`, `n_features_rf`, `n_features`, `feature_names_`.

---

## SeedEnsemble

```python
class SeedEnsemble(BaseEnsemble, BaseEstimator, ClassifierMixin):
    classifier_name: str
    seeds: list[dict]
```

### Construction

```python
SeedEnsemble(classifier_name: str)

SeedEnsemble.from_registry(
    registry_csv: str,
    classifier_name: str | None = None,
    verbose: bool = False,
) -> SeedEnsemble
```

### Inference

```python
se.predict_proba(X: pd.DataFrame) -> np.ndarray   # (n_samples, 2)

se.predict_with_uncertainty(
    X: pd.DataFrame,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
# returns (mean_proba, std_proba, confidence, prediction)
```

`save(path)` / `load(path)` inherited from `BaseEnsemble`.

---

## ClassifierEnsemble

```python
class ClassifierEnsemble(BaseEnsemble, BaseEstimator, ClassifierMixin)
```

### Construction (classmethods)

| Factory | Accepts |
|---------|---------|
| `from_any(source, verbose=False)` | `ClassifierEnsemble.{json,pkl}`, `SeedEnsemble_*.pkl`, trained-model registry CSV, or a live `SeedEnsemble` |
| `from_seed_ensembles(seed_ensembles)` | Pre-built `SeedEnsemble` instances |
| `from_dict(registry_csvs, verbose=False)` | Dict mapping classifier name → registry CSV path |
| `from_json(json_path, verbose=False)` | JSON registry written by `TrainingModel.fit()` |

### Inference

```python
ce.predict_proba(X: pd.DataFrame) -> np.ndarray             # consensus, (n_samples, 2)
ce.predict_with_uncertainty(X: pd.DataFrame, threshold: float = 0.5)
# → (mean, std, confidence, prediction, per_classifier_dict)
```

### Inspection

```python
ce.classifiers          # list[str]: classifier class names in registration order
ce[name]                # SeedEnsemble for the named classifier
len(ce)                 # number of classifiers
```

`save(path)` / `load(path)` inherited from `BaseEnsemble`.

---

## MetricsEnsemble

```python
MetricsEnsemble(
    classifier_ensemble: ClassifierEnsemble,
    features_df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    kind: Literal["prediction", "training"] = "prediction",
    output_dir: str | None = None,
    root_dir: str | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
    verbose: bool = False,
)

MetricsEnsemble.from_file(
    model_filepath: str,
    features_csv: str,
    features_label_csv: str,
    eruption_dates: list[str] | None = None,
    kind: Literal["prediction", "training"] = "prediction",
    output_dir: str | None = None,
    root_dir: str | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
    verbose: bool = False,
) -> MetricsEnsemble
```

| Method | Notes |
|--------|-------|
| `me.compute() -> Self` | Per-seed metric loop; writes JSON + `y_proba/y_pred/y_true` CSV matrices |
| `me.plot_aggregate()` | Aggregate plots per classifier — ROC, PR, confusion, threshold, importance |
| `me.plot_seed()` | Per-seed plots (currently reserved) |
| `me.metrics` | `dict[str, pd.DataFrame]` — populated after `compute()` |

Imported from `eruption_forecast.ensemble.metrics_ensemble` (intentionally **not** in `ensemble/__init__.py` to avoid an import cycle).

---

## BaseEnsemble

```python
class BaseEnsemble:
    def save(self, path: str) -> None
    @classmethod
    def load(cls, path: str) -> Self
```

Joblib save/load mixin inherited by `SeedEnsemble` and `ClassifierEnsemble`. Imported from `eruption_forecast.ensemble.base_ensemble`.

---

## ClassifierComparator

```python
ClassifierComparator(
    metrics_ensemble: MetricsEnsemble,
    metrics: str | list[str] | None = None,
    output_dir: str | None = None,
)

ClassifierComparator.from_classifier_ensemble(
    classifier_ensemble: ClassifierEnsemble,
    features_df: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    kind: Literal["training", "prediction"] = "training",
    output_dir: str | None = None,
    root_dir: str | None = None,
    metrics: str | list[str] | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> ClassifierComparator
```

| Method | Returns |
|--------|---------|
| `cc.get_ranking()` | `pd.DataFrame` — cross-classifier ranking |
| `cc.plot_all()` | `None` — writes ranking plots under `{output_dir}/comparison/figures/` |

Imported from `eruption_forecast.model.classifier_comparator`. Usually instantiated indirectly via `em.compare()` or `fm.EvaluationModel.compare()`.

---

## TremorData

Thin wrapper around a tremor CSV. Imported as `TremorData` from the package root.

```python
TremorData(df: pd.DataFrame)                  # wrap an in-memory frame
TremorData.from_csv(path: str) -> TremorData  # classmethod
```

`@cached_property` accessors: `df`, `start_date`, `end_date`, `filename`, `basename`, `filetype`.

---

## LabelData

Thin wrapper around a label CSV. Imported as `LabelData` from the package root.

```python
LabelData(df: pd.DataFrame)
LabelData.from_csv(path: str) -> LabelData
```

`@cached_property` accessors: `df`, `parameters` (dict parsed from filename — `window_size`, `window_step`, `window_step_unit`, `day_to_forecast`), `filename`, `basename`.

---

## Logger helpers

```python
from eruption_forecast import enable_logging, disable_logging
from eruption_forecast.logger import set_log_level, set_log_directory

enable_logging()              # restore console + file handlers
disable_logging()             # remove every loguru handler
set_log_level(level: str)     # "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
set_log_directory(dir: str)   # move the log file to a new directory (created if absent)
```

---

## Telegram helpers

```python
from eruption_forecast import notify, send_telegram_notification

@notify(label: str)
def my_func(): ...

send_telegram_notification(
    message: str,
    files: list[str] | None = None,
    file_caption: str | None = None,
    send_as_document: bool = False,
)
```

Credentials are read from environment (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`). Both helpers degrade gracefully when the env vars are absent — they emit a warning and skip the network call instead of raising.

---

## Cross-references

- High-level pipeline overview: [Architecture](Architecture#2-pipeline-overview)
- End-to-end example: [Usage](Usage)
- Caching internals: [Architecture](Architecture#34-model-model) and [Training Workflow](Training-Workflow#caching)
- Output paths for every artefact: [Output Structure](Output-Structure)
