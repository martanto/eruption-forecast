# API Reference

Parameter tables and method signatures for every public class exported from `eruption_forecast`. Imports throughout this page:

```python
from eruption_forecast import (
    ForecastModel,
    TrainingModel,
    PredictionModel,
    EvaluationModel,
    ExplanationModel,
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
    timer,
    TelegramNotification,
)
from eruption_forecast.ensemble import SeedEnsemble, ClassifierEnsemble
from eruption_forecast.ensemble.base_ensemble import BaseEnsemble
from eruption_forecast.ensemble.metrics_ensemble import MetricsEnsemble
from eruption_forecast.ensemble.explainer_ensemble import ExplainerEnsemble
from eruption_forecast.dataclass import (
    StationData,
    SeedExplanation,
    ClassifierExplanation,
    ProbabilityPick,
    SeedSummary,
    EruptionWindow,
    ClassifierEnsembleSummary,
)
from eruption_forecast.model.classifier_comparator import ClassifierComparator
from eruption_forecast.features.feature_selector import FeatureSelector
# ⚠ Experimental — see the Feature Count Sweep wiki page:
from eruption_forecast.features import FeatureCountSweep, sweep_feature_count
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
| `station` | `str` | - | Station code (uppercased) |
| `channel` | `str` | - | Channel code (uppercased) |
| `network` | `str` | - | FDSN network code |
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
    plot_overwrite: bool = False,
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

Requires `calculate()` to have populated tremor data first. `label_builder="dynamic"` requires `days_before_eruption`. `use_cache=True` short-circuits via `TrainingModel.load(training_dir, identity)` when the cache identity matches. Sets `self.TrainingModel`, `self.ClassifierEnsemble`, and `self._training_cache_hash`.

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
    use_features_from: Literal["all", "files", "training"] = "all",
    features_matrix_path: str | None = None,
    label_features_csv: str | None = None,
    enable_segments_plot: bool = False,
    output_dir: str | None = None,
    overwrite: bool | None = None,
    n_jobs: int | None = None,
    use_cache: bool = True,
    verbose: bool | None = None,
    **plot_kwargs: Any,
) -> Self
```

Requires `train()` first. `**plot_kwargs` is forwarded to `plot_forecast` (see [Visualization](Visualization#forecast--plot_forecast) for keys) and is **not** captured in `ForecastConfig` because matplotlib objects do not round-trip through YAML. Sets `self.PredictionModel`, `self.results`.

`use_features_from` switches feature scoping between three modes — see [Prediction Workflow → Feature Scoping](Prediction-Workflow#feature-scoping-via-use_features_from) for the full mode table:

| Mode | Behaviour | Path kwargs |
|------|-----------|-------------|
| `"all"` (default) | Extract every tsfresh feature (`select_features=None`) | Ignored |
| `"files"` | Skip tsfresh and load `features_matrix_path` + `label_features_csv` via `PredictionModel.load_features(...)`. Both paths are required and must exist (raises `ValueError` / `FileNotFoundError` otherwise). `use_cache` is forced to `False` because `load_features()` bypasses the extract-features kwargs the cache identity depends on. | Both required |
| `"training"` | Narrow tsfresh to the union of features any seed picked during `train()` — pulled from `self.TrainingModel.features_selected_df.index`; falls back to `None` when the frame is empty. | Ignored |

`enable_segments_plot=True` forwards the training and prediction date ranges to `plot_forecast` so it renders the top Training → Gap → Prediction segment strip above the forecast panels. When the prediction start is on or before the training end, the strip helper snaps the prediction start forward to `training_end + 1 day` and logs a warning; when `False` (default) the strip is omitted.

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
    use_cache: bool = True,
    verbose: bool | None = None,
) -> Self
```

`eruption_dates=None` falls back to the dates captured during `train()`. `use_cache=True` (default) consults `{evaluation_dir}/{hash}.EvaluationModel.pkl` before running the per-classifier `predict_proba` pass; pass `use_cache=False` to force a fresh evaluation even when a cached pickle exists on disk. `use_cache` is threaded down to `EvaluationModel.evaluate(..., use_cache=...)` so it gates both the internal load and the write. Independent of `overwrite`, which additionally controls plot regeneration. Always auto-calls `save_config()` at the end. Sets `self.EvaluationModel`, `self.evaluation_results`.

### `explain(...)`

```python
fm.explain(
    model: Literal["training", "prediction"] = "prediction",
    eruption_dates: list[str] | None = None,
    save_per_seed: bool = True,
    plot_per_seed: bool = True,
    figsize: tuple[float, float] | None = None,
    max_display: int = 20,
    group_remaining_features: bool = False,
    dpi: int = 150,
    check_additivity: bool = False,
    overwrite_classifier_explanation: bool = False,
    output_dir: str | None = None,
    overwrite: bool | None = None,
    n_jobs: int | None = None,
    use_cache: bool = True,
    verbose: bool | None = None,
) -> Self
```

Requires the upstream `TrainingModel` or `PredictionModel` to exist on `self` (run `train()` and, for `model="prediction"`, `predict()` first). `eruption_dates=None` falls back to the dates captured during `train()`. `use_cache=True` (default) consults `{explanation_dir}/{hash}.ExplanationModel.pkl` before re-running SHAP; pass `use_cache=False` to force a fresh explanation even when a cached pickle exists on disk. `use_cache` is threaded down to `ExplanationModel.explain(..., use_cache=...)` so it gates both the internal load and the write. Independent of `overwrite`, which additionally controls per-classifier artefact regeneration. Internally constructs an `ExplanationModel`, runs `explain()` then `plot()`, and sets `self.ExplanationModel`. See [Explanation Workflow](Explanation-Workflow) for the TreeExplainer constraint (RF / `lite-rf` / GB / XGB only — other classifiers are skipped with a warning).

### Config round-trip

| Method | Returns | Notes |
|--------|---------|-------|
| `fm.save_config(path=None, fmt="yaml")` | `str` | Defaults to `{station_dir}/forecast.config.{yaml,json}` |
| `ForecastModel.from_config(path)` | `ForecastModel` | Classmethod; restores the captured `ForecastConfig` |
| `fm.run()` | `Self` | Idempotent replay of every captured non-`None` stage |

---

## TrainingModel

`BaseModel` subclass with content-addressable cache participation. Use standalone when running outside `ForecastModel`; otherwise `fm.train(...)` constructs it for you.

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
| `TrainingModel.build_identity(**kwargs)` | Classmethod; returns canonical identity dict for hashing. Called both by `ForecastModel.train()` (before the instance exists, for cache lookup) and inside `fit()` (with kwargs pulled from `self`, for the save). |
| `tm.save(identity)` | Cache mode: writes `{training_dir}/{hash}.TrainingModel.pkl` + `.params.json` sidecar |
| `tm.save()` | Legacy mode (no identity): `{output_dir}/TrainingModel_{basename}.pkl` joblib dump |
| `TrainingModel.load(stage_dir, identity)` | Classmethod; returns the instance on cache hit or `None` |
| `tm.save_config(path=None, fmt="yaml")` | `{training_dir}/training.config.{yaml,json}` — auto-called at end of `fit()` |

Populated attributes after `fit()`: `tm.results` (per-classifier trained-model JSON registry paths written by `save_model_json`), `tm.ClassifierEnsemble`, `tm.classifier_ensemble_path`, `tm.features_df`, `tm.labels`.

---

## PredictionModel

`BaseModel` subclass with content-addressable cache participation.

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

`model` accepts a live `ClassifierEnsemble` / `SeedEnsemble`, a `ClassifierEnsemble.json` / `.pkl`, a `SeedEnsemble_*.pkl`, or a trained-model registry `.csv` - resolved via `ClassifierEnsemble.from_any(...)`.

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

Same surface as `TrainingModel`: `build_identity(**kwargs)`, `save(identity)`, `load(stage_dir, identity)`. The cache identity embeds the upstream `training_hash` (constructor param).

| Method | Notes |
|--------|-------|
| `pm.save_config(path=None, fmt="yaml")` | `{prediction_dir}/prediction.config.{yaml,json}` — auto-called at end of `forecast()` |

`PredictionConfig` captures the user-supplied `model` and `tremor_data` as string handles (`null` when a live in-memory object was passed).

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

`plot_shap=True` is reserved on this surface and emits a warning — SHAP rendering is produced by the dedicated [Explanation Workflow](Explanation-Workflow) via `ExplanationModel.explain()`. `plot_per_seed=True` is plumbed through to `MetricsEnsemble.plot_seed()` for the metric plots (ROC, PR, confusion, etc.) — it does **not** render SHAP.

| Method | Notes |
|--------|-------|
| `em.save_config(path=None, fmt="yaml")` | `{evaluation_dir}/evaluation.config.{yaml,json}` — auto-called at end of `evaluate()` |

`EvaluationConfig` omits the upstream `model` parameter (live model instances are not serializable). Captured fields: `eruption_dates`, `overwrite`, `output_dir`, `root_dir`, `n_jobs`, `verbose`.

---

## ExplanationModel

`BaseModel` subclass with content-addressable cache participation. Per-seed SHAP explanations over a fitted `ClassifierEnsemble` — never re-fits. See [Explanation Workflow](Explanation-Workflow).

### Constructor

```python
ExplanationModel(
    model: TrainingModel | PredictionModel,
    eruption_dates: list[str] | None = None,
    overwrite: bool = False,
    output_dir: str | None = None,
    root_dir: str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
)
```

Output is namespaced to `explanation/{model.kind}/`. The constructor sets `self.kind="explanation"`, `self.model_kind` (mirrored from the upstream `TrainingModel.kind` / `PredictionModel.kind`), `self.ClassifierEnsemble`, `self.features_df`, `self.explanation_dir`, `self.classifiers_dir`, and `self.ExplainerEnsemble`.

### Methods

```python
em.explain(
    save_per_seed: bool = True,
    check_additivity: bool = False,
    overwrite_classifier_explanation: bool = False,
) -> Self

em.plot(
    figsize: tuple[float, float] | None = None,
    max_display: int = 20,
    group_remaining_features: bool = False,
    dpi: int = 150,
    plot_per_seed: bool = True,
)
```

`explain()` delegates to `ExplainerEnsemble.explain()` and caches the result via `BaseModel.save(identity)`. On a cache hit the stored `self.explanations` is restored without re-running SHAP. `plot()` renders the per-eruption waterfall (only when `eruption_dates` is available) and, optionally, per-seed bar + beeswarm plots.

```python
ExplanationModel.from_file(
    filepath: str,
    eruption_dates: list[str] | None = None,
    overwrite: bool = False,
    output_dir: str | None = None,
    root_dir: str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> ExplanationModel
```

Classmethod. Loads a `.pkl` from `TrainingModel.save()` or `PredictionModel.save()` and constructs an `ExplanationModel` against it. Raises `TypeError` if the pickle holds anything else.

Populated attributes after `explain()`: `em.explanations: list[ClassifierExplanation]`.

| Method | Notes |
|--------|-------|
| `em.save_config(path=None, fmt="yaml")` | `{explanation_dir}/explanation.config.{yaml,json}` — auto-called at end of `explain()` |

`ExplanationConfig` omits the upstream `model` parameter (live model instances are not serializable). Captured fields: `eruption_dates`, `overwrite`, `output_dir`, `root_dir`, `n_jobs`, `verbose`.

---

## ExplainerEnsemble

```python
ExplainerEnsemble(
    classifier_ensemble: ClassifierEnsemble,
    features_df: pd.DataFrame,
    kind: Literal["training", "prediction"] = "prediction",
    output_dir: str | None = None,
    explanation_dir: str | None = None,
    root_dir: str | None = None,
    overwrite: bool = False,
    n_jobs: int = 1,
    verbose: bool = False,
)
```

Per-seed SHAP engine driven by `shap.TreeExplainer`. Non-tree classifiers (`svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting`) are skipped at the per-classifier loop with a warning. `explanation_dir` is the sibling-of-`classifiers/` root used for per-eruption waterfall plots; when omitted it falls back to `dirname(output_dir)`.

### Methods

```python
ee.explain(
    save_per_seed: bool = True,
    check_additivity: bool = False,
    overwrite_classifier_explanation: bool = False,
) -> Self

ee.plot_seed(
    max_display: int = 20,
    group_remaining_features: bool = False,
    dpi: int = 150,
)   # per-classifier bar + beeswarm under classifiers/{ClfName}/figures/

ee.plot_waterfall(
    labels: pd.Series | pd.DataFrame,
    eruption_dates: list[str],
    figsize: tuple[float, float] | None = None,
    max_display: int = 20,
    dpi: int = 150,
)   # per-eruption waterfall under {explanation_dir}/eruptions/{date}/
```

### Static helpers

```python
ExplainerEnsemble.explain_seed(
    seed: dict,
    features_df: pd.DataFrame,
    save_per_seed: bool = False,
    check_additivity: bool = False,
    seed_explanation_filepath: str | None = None,
) -> shap.Explanation

ExplainerEnsemble.explain_classifier(
    seed_ensemble: SeedEnsemble,
    features_df: pd.DataFrame,
    save_per_seed: bool = False,
    kind: Literal["training", "prediction"] = "prediction",
    check_additivity: bool = False,
    output_dir: str | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> ClassifierExplanation

ExplainerEnsemble.normalise_shap_values(
    explanation: shap.Explanation,
) -> tuple[np.ndarray, np.ndarray]
```

Imported from `eruption_forecast.ensemble.explainer_ensemble` (intentionally **not** re-exported from `ensemble/__init__.py` to keep that subpackage cycle-free).

---

## SeedExplanation / ClassifierExplanation

```python
@dataclass(frozen=True, slots=True)
class SeedExplanation:
    random_state: int
    shap_values: shap.Explanation

@dataclass(slots=True)
class ClassifierExplanation:
    classifier_name: str
    seeds: list[SeedExplanation] = field(default_factory=list)
```

Both re-exported from `eruption_forecast.dataclass`. `SeedExplanation` is frozen; `ClassifierExplanation` is mutable so `ExplainerEnsemble.explain_classifier()` can append seeds incrementally. Produced by the explanation stage and consumed by every plot helper in `plots/explanation_plots.py`.

---

## ClassifierEnsembleSummary family

Per-classifier rollup of a `SeedEnsemble`'s probability matrix. Built by `eruption_forecast.utils.ml.build_classifier_ensemble_summary` and consumed by the SHAP waterfall renderer in `eruption_forecast.plots.explanation_plots.plot_classifier_waterfall`. All four types re-exported from `eruption_forecast.dataclass`.

```python
@dataclass(frozen=True)
class ProbabilityPick:
    random_state: int         # seed identifier the pick came from
    index: int                # row position in the per-seed probability matrix;
                              # aligns with the SHAP explanation row for the same seed
    datetime: pd.Timestamp    # timestamp of the sample
    value: float              # probability value at the picked row

@dataclass(frozen=True)
class SeedSummary:
    random_state: int         # seed identifier shared by both picks
    highest: ProbabilityPick  # top-probability pick in the window
    lowest: ProbabilityPick   # bottom-probability pick in the window

@dataclass(frozen=True)
class EruptionWindow:
    eruption_date: str        # eruption date in YYYY-MM-DD form
    highest: ProbabilityPick  # highest-probability pick across all seeds in this window
    lowest: ProbabilityPick   # lowest-probability pick across all seeds in this window
    seeds: list[SeedSummary]  # one SeedSummary per seed

@dataclass
class ClassifierEnsembleSummary:
    classifier_name: str
    eruption_windows: list[EruptionWindow] = field(default_factory=list)
    highest: ProbabilityPick | None = None   # None when no eruption window intersected the prediction grid
    lowest: ProbabilityPick | None = None    # None when no eruption window intersected the prediction grid
```

`ClassifierEnsembleSummary` is per-classifier (one instance per `SeedEnsemble`) — no cross-classifier collapse. The builder guarantees `EruptionWindow.highest` / `.lowest` are set (empty windows are skipped) but `ClassifierEnsembleSummary.highest` / `.lowest` may still be `None` when no eruption window intersected the prediction grid at all.

---

## StationData

Immutable container for the five codes that uniquely identify a seismic channel within a network archive (NSLC scheme). All codes are uppercased in `__post_init__`; `nslc` and `nslct` identifiers are derived automatically.

```python
@dataclass
class StationData:
    station: str              # e.g. "OJN"  — must be non-empty
    channel: str              # e.g. "EHZ"  — must be non-empty
    network: str              # e.g. "VG"   — must be non-empty
    location: str = ""        # e.g. "00"   — None coerced to ""
    channel_type: str = "D"   # SDS data-type suffix

    # Derived (init=False, repr=False):
    nslc: str                 # "{network}.{station}.{location}.{channel}"
    nslct: str                # "{nslc}.{channel_type}"
```

```python
>>> sd = StationData(station="OJN", channel="EHZ", network="VG", location="00")
>>> sd.nslc
'VG.OJN.00.EHZ'
>>> sd.nslct
'VG.OJN.00.EHZ.D'
```

Raises `ValueError` if `station` / `channel` / `network` is empty or non-`str`, or if `location` is neither a `str` nor `None`.

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
    plot_overwrite: bool = False,
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

Note that within `TrainingModel`, `include_eruption_date` defaults to `False` (training pipeline behaviour); the `LabelBuilder` constructor's own default is `True`. The difference is intentional - see [Training Workflow](Training-Workflow).

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
    method: Literal["tsfresh", "random_forest"] = "tsfresh",
    random_state: int = 42,
    output_dir: str | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
)
```

Used internally by `TrainingModel.fit()` per-seed (hardcoded `method="tsfresh"`); surfaced publicly for ad-hoc selection experiments. Pick `method="tsfresh"` for FDR-controlled p-value filtering (fast, model-agnostic) or `method="random_forest"` for permutation importance from a RandomForest probe. Populated after `fit(X, y)`: `selected_features_`, `p_values_`, `importance_scores_`, `n_features_tsfresh`, `n_features_rf`, `n_features`, `feature_names_`.

---

## FeatureCountSweep (Experimental)

> **⚠ Experimental — not production-ready.** Signature, defaults, and
> on-disk layout may change without deprecation. See
> [Feature Count Sweep](Feature-Count-Sweep) for the full contract,
> examples, and caveats.

Post-hoc `top_n_features` recommender. Consumes per-seed FDR rankings,
resampled ids, and tuned `best_model.pkl` files already written by
`TrainingModel.fit()`, then re-scores each seed at multiple candidate
`N` values.

### Constructor

```python
FeatureCountSweep(
    estimator: BaseEstimator | None = None,
    *,
    strategy: Literal["shared", "per-seed"] = "per-seed",
    n_candidates: list[int] | None = None,
    cv: BaseCrossValidator | int = 5,
    scoring: str = "average_precision",
    parsimony: bool = True,
    parsimony_tolerance: float | None = None,
    resample_method: Literal["under", "over", "auto"] | None = None,
    minority_threshold: float = 0.15,
    estimator_mode: Literal["default", "tuned"] = "default",
    mode: Literal["cv", "forecast"] = "forecast",
    n_jobs: int = 1,
    random_state: int = 42,
    verbose: bool = False,
)
```

| Param | Notes |
|-------|-------|
| `mode` | `"forecast"` (default) — score against a held-out prediction window; `"cv"` — RFECV-analog CV inside the trainer's resampled subset |
| `estimator_mode` | `"default"` (recommended) drops the tuned hyperparameters via `estimator.__class__()`; `"tuned"` keeps them and biases the curve toward the trained `N` |
| `parsimony_tolerance` | `None` → adaptive 1-SE rule; float in `(0, 1)` → fractional tolerance `peak × (1 − t)` |
| `n_jobs` | Applied over CV folds when `mode="cv"`; unused in `mode="forecast"` (one fit per `(seed, N)`) |

### `.fit(...)`

```python
sweep.fit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    per_seed_inputs: dict[int, dict[str, Any]] | None = None,
    shared_ranking: pd.Series | pd.Index | list[str] | None = None,
    per_seed_rankings: dict[int, ...] | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> Self
```

`per_seed_inputs` / `shared_ranking` / `per_seed_rankings` are mutually
exclusive input modes. `X_test` / `y_test` are required in
`mode="forecast"`; missing per-seed features in `X_test` raise `KeyError`
(never silently intersected).

Populated attributes: `cv_scores_` (aggregated N × [mean, std,
n_seeds]), `cv_scores_raw_` (full N × seed matrix), `n_features_`
(recommended `N*`), `seed_argmax_` (per-seed argmax `N`), `support_`
(per-seed top-`N*` feature lists).

### `sweep_feature_count(...)`

```python
sweep_feature_count(
    source: str | os.PathLike | TrainingModel,
    *,
    mode: Literal["cv", "forecast"] = "forecast",
    evaluation_source: EvaluationModel | None = None,
    classifier_name: str | None = None,
    n_candidates: list[int] | None = None,
    cv: BaseCrossValidator | int = 5,
    scoring: str = "average_precision",
    parsimony: bool = True,
    parsimony_tolerance: float | None = None,
    resample_method: Literal["under", "over", "auto"] | None = None,
    minority_threshold: float = 0.15,
    estimator_mode: Literal["default", "tuned"] = "default",
    n_jobs: int = 1,
    random_state: int = 42,
    output_dir: str | None = None,
    save: bool = True,
    verbose: bool = False,
) -> FeatureCountSweep | dict[str, FeatureCountSweep]
```

Convenience wrapper that harvests per-seed inputs from an existing
`training/` directory (or live `TrainingModel`). In `mode="forecast"`,
`evaluation_source` must be an `EvaluationModel` in prediction-reuse
mode with a populated `y_true` (call `em.build_label(...)` or
`em.evaluate()` first). Returns a single sweep when `classifier_name`
is given, otherwise a dict keyed by discovered classifier names.

### Persistence

| Method | Notes |
|--------|-------|
| `sweep.save(path)` | joblib-dump the full instance |
| `FeatureCountSweep.load(path)` | Classmethod; restore a saved sweep |

`sweep_feature_count(save=True)` also writes `cv_scores.csv`,
`cv_scores_raw.csv`, `seed_argmax_hist.csv`, `support.json`, and
`curve.png` alongside the pickle under
`{training_dir}/features/{cv-slug}/sweep/{mode}/{classifier}/`.

---

## SeedEnsemble

```python
class SeedEnsemble(BaseEnsemble, BaseEstimator, ClassifierMixin):
    classifier_name: str
    seeds: list[dict]      # per-seed records: {random_state, model, feature_names}
    features: list[str]    # sorted union of per-seed feature_names (set of features
                           # used by at least one seed in this ensemble); populated
                           # by every factory, empty on the bare constructor
```

### Construction

```python
SeedEnsemble(classifier_name: str)

# Recommended: dispatch on file extension (.json or .csv).
SeedEnsemble.from_any(
    trained_model_path: str,
    classifier_name: str | None = None,
    verbose: bool = False,
) -> SeedEnsemble

# New JSON trained-model registry written by utils.ml.save_model_json.
SeedEnsemble.from_json(
    trained_model_json: str,
    classifier_name: str | None = None,
    verbose: bool = False,
) -> SeedEnsemble

# Legacy CSV registry loader (kept for backwards compatibility).
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

### Inspection

```python
se.features             # list[str]: sorted union of per-seed feature_names
se.seeds                # list[dict]: per-seed records
se[i]                   # seed record at index i (dict)
len(se)                 # number of seeds
```

`save(path)` / `load(path)` inherited from `BaseEnsemble`.

---

## ClassifierEnsemble

```python
class ClassifierEnsemble(BaseEnsemble, BaseEstimator, ClassifierMixin):
    ensembles: dict[str, SeedEnsemble]   # classifier name → its SeedEnsemble
    features: list[str]                  # sorted union of SeedEnsemble.features
                                         # across every registered classifier;
                                         # populated by every factory, empty on
                                         # the bare constructor
```

### Construction (classmethods)

| Factory | Accepts |
|---------|---------|
| `from_any(source, verbose=False)` | `ClassifierEnsemble.{json,pkl}`, `SeedEnsemble_*.pkl`, trained-model registry `.json` (list) or `.csv`, or a live `SeedEnsemble` |
| `from_seed_ensembles(seed_ensembles)` | Pre-built `SeedEnsemble` instances |
| `from_dict(trained_model_paths, verbose=False)` | Dict mapping classifier name → trained-model registry path (`.json` or `.csv`); each value is dispatched through `SeedEnsemble.from_any` |
| `from_json(json_path, verbose=False)` | Top-level results map (`ClassifierEnsemble_{cv}.json`) written by `TrainingModel.fit()` |

### Inference

```python
ce.predict_proba(X: pd.DataFrame) -> np.ndarray             # consensus, (n_samples, 2)
ce.predict_with_uncertainty(X: pd.DataFrame, threshold: float = 0.5)
# → (mean, std, confidence, prediction, per_classifier_dict)
```

### Inspection

```python
ce.classifiers          # list[str]: classifier class names in registration order
ce.features             # list[str]: sorted union of SeedEnsemble.features across
                        #            every registered classifier
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
    features_path: str,
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
| `me.compute() -> Self` | Per-seed metric loop. Writes only the `(n_samples, n_seeds)` `y_proba.csv` / `y_pred.csv` matrices under `classifiers/{ClfName}/predictions/`. `metrics`, `y_probas`, `y_preds` stay in memory; no per-seed JSON is produced. Idempotent once `y_probas` is populated. |
| `me.plot_aggregate(include_plots=None, exclude_plots=None) -> list[str]` | Aggregate plots per classifier — ROC, PR, threshold analysis, g-mean curve, MCC curve. Writes `figures/aggregate/{plot_name}.{png,csv}` per classifier. |
| `me.plot_seed(include_plots=None, exclude_plots=None) -> list[str]` | Per-seed plots — same dispatcher catalogue. Writes `figures/{plot_name}/{seed:05d}.png` per classifier in parallel via `joblib`. |
| `me.metrics` | `dict[str, pd.DataFrame]` — populated after `compute()` |
| `me.save(path=None)` / `MetricsEnsemble.load(path)` | joblib round-trip of the full instance to `MetricsEnsemble.pkl`. |

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
| `cc.get_ranking()` | `pd.DataFrame` - cross-classifier ranking |
| `cc.plot_all()` | `None` - writes ranking plots under `{output_dir}/comparison/figures/` |

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

`@cached_property` accessors: `df`, `parameters` (dict parsed from filename - `window_size`, `window_step`, `window_step_unit`, `day_to_forecast`), `filename`, `basename`.

---

## Feature alias utilities

Aggregation across seeds ranks every significant tsfresh feature, and each
row of the ranked CSV is tagged with a short display alias (`ft_1`, `ft_2`,
…) plus a plain-English description. The alias is deterministic — it
follows the rank position — so regenerating the CSV with the same inputs
produces identical aliases. The utilities below let callers write, read,
and translate that mapping.

### Writers — `alias` + `description` columns

```python
from eruption_forecast.utils.feature_utils import (
    concat_significant_features,
    find_common_features,
)

concat_significant_features(
    features_csvs: list[str],
    features_dir: str,
    number_of_features: int | None = None,
    freq_bands: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame

find_common_features(
    top_features_csv: list[str],
    output_dir: str | None = None,
    freq_bands: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame
```

Both functions decorate the ranked frame with `alias` (`ft_1..ft_N` by
rank) and `description` (plain-English via
`humanize_feature_name`). `concat_significant_features` writes
`top_features.csv` and `top_{N}_features.csv` under `features_dir` when
`number_of_features > 0`; `find_common_features` writes
`common_top_features.csv` under `output_dir` (or `os.getcwd()`).

`freq_bands` is optional. When omitted, the humanizer uses the default
edges from `DEFAULT_FREQUENCY_BANDS` (`config/constants.py`), which
matches `CalculateTremor`'s defaults. Pass a matching dict — same shape
as `{"f0": (low, high), ...}` — when the tremor was calculated with
`CalculateTremor.change_freq_bands(...)` so descriptions in the CSVs
reflect the actual Hz values.

### Reader — `load_feature_aliases`

```python
from eruption_forecast.utils.feature_utils import load_feature_aliases

load_feature_aliases(
    source: str | pd.DataFrame,
    reverse: bool = True,
) -> dict[str, str]
```

- `source` accepts either a CSV path (`top_features.csv`,
  `top_{N}_features.csv`, `common_top_features.csv`) OR the DataFrame
  those functions return directly — the DataFrame path skips the disk
  round-trip when the frame is still in memory (natural right after
  `concat_significant_features(...)` in the same session).
- `reverse=True` (default) → `{alias: canonical_name}` so callers can
  translate a plot label back to the tsfresh name
  (`aliases["ft_5"] -> 'dsar_f3-f4__fft_coefficient__attr_"abs"__coeff_91'`).
- `reverse=False` → `{canonical_name: alias}`, useful for
  `features_df.rename(columns=mapping)`.

Raises `FileNotFoundError` for missing paths and `ValueError` when the
frame lacks the `alias` column (typical mistake: pointing at
`significant_features.csv`, the raw per-seed dump that carries no rank).

### Backfill — `update_top_features_csv`

```python
from eruption_forecast.utils.feature_utils import update_top_features_csv

update_top_features_csv(
    csv_path: str,
    output_path: str | None = None,
    freq_bands: dict[str, tuple[float, float]] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame
```

Adds `alias` and `description` columns to a legacy `top_features.csv`,
`top_{N}_features.csv`, or `common_top_features.csv` that predates the
alias rollout (or was hand-prepared). Row order is trusted as-is —
aliases follow the existing rank, no re-sorting — so a CSV that's
already sorted by `frequency` desc / `score_mean` asc lines up with the
same `ft_N` values `concat_significant_features` would produce today.

Also handles two column renames: `score` → `frequency` (the original
`score` column was actually a per-seed count) and `mean_score` →
`score_mean` (name-first, statistic-last convention, matching the
classifier-comparison tables). A missing `score_std` column is
backfilled with NaN on read so downstream selectors don't KeyError on
legacy CSVs. When any of these normalisations is outstanding the file
is rewritten on disk so downstream readers pick up the current schema
without further action.

- `output_path=None` (default) rewrites the file in place; pass a
  different path to keep the original for comparison.
- `freq_bands` forwards to `humanize_feature_name` so descriptions can
  reflect custom bands used for that scenario.
- `overwrite=False` (default) is idempotent — when `frequency`,
  `alias`, and `description` are all already present, the CSV is not
  touched. Pass `overwrite=True` to drop and regenerate the alias /
  description columns, useful after changing `freq_bands` or after
  the calculator humanization table is extended.

Raises `FileNotFoundError` if `csv_path` does not exist.

### Label formatters — `shorten_feature_name` / `humanize_feature_name`

```python
from eruption_forecast.utils.formatting import (
    shorten_feature_name,
    humanize_feature_name,
)

shorten_feature_name(name: str) -> str
humanize_feature_name(
    name: str,
    freq_bands: dict[str, tuple[float, float]] | None = None,
) -> str
```

- `shorten_feature_name` renders a compact axis-tick label
  (`dsar_f3-f4 | fft_coef(abs, 91)`), used by `ExplainerEnsemble` at
  ingest and by the two cross-scenario heatmaps in `label_style="short"`
  mode.
- `humanize_feature_name` renders a plain-English phrase (`Fourier
  coefficient (attr=abs, coeff=91) of DSAR ratio 4.5/8 Hz`), used to
  populate the `description` column of the ranked CSVs. The
  `freq_bands` kwarg has the same meaning as on
  `concat_significant_features` / `find_common_features` above.

Both return the input unchanged when it does not match the
`column__calculator[__key_value]*` shape.

### Heatmap label style

```python
from eruption_forecast.plots.feature_plots import (
    plot_common_features_heatmap,
    plot_common_features_correlation,
)

plot_common_features_heatmap(
    top_features_csv: dict[str, str],
    output_path: str | None = None,
    cmap: str = "viridis",
    label_style: Literal["short", "alias"] = "short",
) -> plt.Axes

plot_common_features_correlation(
    top_features_csv: dict[str, str],
    output_path: str | None = None,
    cmap: str = "RdBu_r",
    label_style: Literal["short", "alias"] = "short",
) -> plt.Axes
```

Both cross-scenario heatmaps accept `label_style`. `"short"` (default)
preserves the existing `shorten_feature_name` labels; `"alias"` uses the
`ft_1..ft_N` aliases from the merged ranking returned by
`find_common_features` — useful when the shortened names still crowd the
axis.

---

## Feature matrix loaders

Two thin readers rebuild an ``id``-indexed tsfresh features (or
probability) frame into a `DatetimeIndex`-ed frame by joining against
the sibling `features-label_*.csv` written by
`FeaturesBuilder`.

```python
from eruption_forecast.utils.dataframe import (
    load_datetime_indexed,
    load_features_matrix,
)

load_datetime_indexed(
    label_csv: str,
    features_path: str,
) -> pd.DataFrame

load_features_matrix(
    label_csv: str,
    features_path: str,
) -> pd.DataFrame
```

- `label_csv` — path to the aligned label CSV (`features-label_*.csv`)
  with a `DatetimeIndex` and an `id` column.
- `features_path` — path to the ``id``-indexed matrix. Dispatched by
  suffix: `.parquet` via `pd.read_parquet`, `.csv` via
  `pd.read_csv(..., index_col=0)`. Any other suffix raises `ValueError`.
- Returned frame: the input matrix's columns with a `DatetimeIndex`
  derived from `label_csv`; the `id` / `datetime` columns are absent.

`load_features_matrix` is the domain-named alias for
`load_datetime_indexed`. Reach for `load_features_matrix` at call sites
that specifically load a features matrix so intent is obvious; use
`load_datetime_indexed` when the payload is a probability matrix or any
other ``id``-indexed frame (e.g. `predictions/y_proba.csv`).

Both propagate `ValueError` from
`eruption_forecast.utils.date_utils.to_datetime_index` when the two
frames cannot be aligned (length mismatch, missing `id` column, missing
`datetime` column).

---

## Merging training + prediction feature matrices

```python
from eruption_forecast.utils.feature_utils import merge_features_matrix

merge_features_matrix(
    training_features_matrix: str | pd.DataFrame,
    prediction_features_matrix: str | pd.DataFrame,
    training_label_csv: str,
    prediction_label_csv: str,
    select_features: str | list[str],
    number_of_features: int | None = None,
    output_path: str | None = None,
) -> pd.DataFrame
```

Stitches a training-window features matrix and a prediction-window
features matrix into a single datetime-sorted frame, preserving each
stage's sampling cadence. Typical use is post-hoc analysis or plotting
that needs one continuous features frame across the training and
forecast ranges without collapsing the training cadence.

- Each matrix argument accepts a Parquet / CSV path OR a pre-loaded
  `pd.DataFrame`. When a DataFrame already has a `DatetimeIndex`, the
  matching `*_label_csv` is ignored; ``id``-indexed DataFrames are
  joined against the sibling label CSV the same way file paths are.
- The two matrices are usually sampled at different cadences —
  training at the coarse label window step (e.g. `6 hours`) and
  prediction at the fine forecast cadence (e.g. `10 minutes`). To
  keep the training cadence intact across the training range, the
  prediction frame is truncated to rows **strictly after**
  `training_end` (the maximum datetime in the training index) before
  the concatenation. No interleaving happens inside the training
  range.
- `select_features` is resolved via `load_select_features`, so any
  `top_features.csv` / `top_{N}_features.csv` path produced by
  `concat_significant_features` — or an explicit list of tsfresh
  feature names — is accepted. `number_of_features` is forwarded to
  the resolver and caps the resolved list to the top-N ranked
  entries; `None` (default) keeps every entry.
- Columns present in `select_features` but missing from either matrix
  are intersected away with an `info` log (the training matrix on disk
  is often already a strict subset).
- Duplicate datetime rows across the two frames are dropped
  (`keep="first"`) so the training row wins on overlap. The counts of
  dropped rows are logged at `info` level.
- `output_path` is optional. When set, the merged frame is also
  written to disk as Snappy-compressed Parquet via `pyarrow`.

Raises `ValueError` when the training matrix has no rows (so
`training_end` is undefined), when the resolved feature list has zero
matches in both matrices, or when a matrix path uses an unsupported
suffix. Propagates `FileNotFoundError` from `load_select_features` when
`select_features` is a path that does not exist.

```python
merged = merge_features_matrix(
    training_features_matrix="output/.../training/features/stratified-shuffle-split/features-matrix_2025-01-03_2025-03-31.parquet",
    prediction_features_matrix="output/.../prediction/features/features-matrix_2025-01-01-2025-08-22.parquet",
    training_label_csv="output/.../training/features/stratified-shuffle-split/features-label_2025-01-03_2025-03-31.csv",
    prediction_label_csv="output/.../prediction/features/features-label_2025-01-01_2025-08-22_ws-2_step-10-minutes.csv",
    select_features="output/.../training/features/stratified-shuffle-split/top_features.csv",
    number_of_features=20,
    output_path="output/.../analysis/features-matrix_merged.parquet",
)
```

---

## Logger helpers

```python
from eruption_forecast import enable_logging, disable_logging
from eruption_forecast.logger import (
    get_category_logger,
    register_error_category,
    set_log_directory,
    set_log_level,
)

enable_logging()              # restore console + file handlers
disable_logging()             # remove every loguru handler
set_log_level(level: str)     # "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL"
set_log_directory(dir: str)   # move the log file to a new directory (created if absent)

# Per-category error log files.
register_error_category(
    name: str,
    level: str = "WARNING",
    retention: str = "90 days",
) -> None                     # create/update logs/{name}_YYYY-MM-DD.log
get_category_logger(category: str)   # returns logger.bind(category=category)
```

The default installation registers a `"telegram"` category so warnings from
`TelegramNotification` land in `logs/telegram_YYYY-MM-DD.log` and are excluded
from `forecast_*.log` / `errors_*.log`. Re-registering an existing category is
idempotent — no duplicate sinks are created.

---

## Telegram helpers

```python
from eruption_forecast import notify, timer, TelegramNotification

@notify(
    task: str,
    message: str | None = None,
    to: Literal["telegram", "email"] = "telegram",
    on_success: bool = True,
    on_error: bool = True,
    timeout: float = 3.0,
    verbose: bool = False,
)
def my_func(): ...

@timer(name: str | None = None, send_to: Literal["telegram"] | None = None)
def my_func(): ...

tn = TelegramNotification(
    token: str | None = None,
    chat_id: str | int | None = None,
    verbose: bool = False,
)
tn.send_message(message: str, timeout: float = 3.0) -> Self
tn.send_document(file: str, timeout: float = 30.0, **kwargs) -> Self
tn.send_photo(file: str, timeout: float = 30.0, **kwargs) -> Self
tn.send_media_group(
    files: str | list[str],
    kind: Literal["photo", "document"] = "photo",
    caption: str | None = None,
    timeout: float = 30.0,
    disable_notification: bool = False,
) -> Self
```

Credentials are read from constructor arguments or the `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` environment variables (`.env` supported via `python-dotenv`). Every send method returns `self` for fluent chaining, and network failures are logged and swallowed — a dead network never blocks the caller.

---

## Cross-references

- High-level pipeline overview: [Architecture](Architecture#2-pipeline-overview)
- End-to-end example: [Usage](Usage)
- Caching internals: [Architecture](Architecture#34-model-model) and [Training Workflow](Training-Workflow#caching)
- Output paths for every artefact: [Output Structure](Output-Structure)
