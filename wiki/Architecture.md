# Architecture

This page is the structural reference for `eruption_forecast`: every module under `src/`, the top-level pipeline, how the model and ensemble classes relate, what flows between stages on disk, and the utility surface that holds the rest together.

---

## 1. Package Layout

```
src/eruption_forecast/
├── __init__.py            — public exports
├── logger.py              — loguru wrapper (enable/disable/set_level/set_directory)
├── data_container.py      — BaseDataContainer ABC for TremorData / LabelData
│
├── config/
│   ├── base_config.py         — shared config primitives
│   ├── constants.py           — ERUPTION_PROBABILITY_THRESHOLD, defaults
│   ├── forecast_config.py     — ForecastConfig + per-stage sub-configs
│   └── training_config.py     — TrainingConfig (standalone TrainingModel)
│
├── dataclass/
│   └── station_data.py        — StationData (immutable nslc identity)
│
├── decorators/
│   ├── decorator_class.py     — base decorator scaffolding
│   └── notify.py              — Telegram notify + send_telegram_notification
│
├── ensemble/
│   ├── base_ensemble.py       — BaseEnsemble (joblib save/load mixin)
│   ├── seed_ensemble.py       — SeedEnsemble (one classifier × N seeds)
│   ├── classifier_ensemble.py — ClassifierEnsemble (N classifiers)
│   └── metrics_ensemble.py    — MetricsEnsemble (metrics engine)
│
├── features/
│   ├── constants.py
│   ├── tremor_matrix_builder.py — TremorMatrixBuilder (windowed alignment)
│   ├── features_builder.py      — FeaturesBuilder (tsfresh extraction)
│   └── feature_selector.py      — FeatureSelector (tsfresh FDR + RF importance)
│
├── label/
│   ├── constants.py
│   ├── label_builder.py         — LabelBuilder (sliding window)
│   ├── dynamic_label_builder.py — DynamicLabelBuilder (per-eruption build)
│   ├── label_data.py            — LabelData (CSV wrapper)
│   └── label_plots.py           — plot_label_distribution
│
├── model/
│   ├── constants.py
│   ├── base_model.py            — BaseModel ABC (dates, I/O, save/load)
│   ├── cache_model.py           — CacheModel mixin (content-addressed cache)
│   ├── forecast_model.py        — ForecastModel orchestrator
│   ├── training_model.py        — TrainingModel(BaseModel, CacheModel)
│   ├── prediction_model.py      — PredictionModel(BaseModel, CacheModel)
│   ├── evaluation_model.py      — EvaluationModel(BaseModel)
│   ├── classifier_model.py      — ClassifierModel (estimator + grid)
│   └── classifier_comparator.py — ClassifierComparator (cross-classifier rank)
│
├── plots/
│   ├── styles.py
│   ├── tremor_plots.py          — plot_tremor
│   ├── feature_plots.py         — feature-importance plots
│   ├── forecast_plots.py        — plot_forecast, plot_forecast_from_file
│   └── evaluation_plots.py      — ROC, PR, confusion, threshold, importance
│
├── sources/
│   ├── base.py                  — SeismicDataSource ABC
│   ├── sds.py                   — Local SeisComP archive reader
│   └── fdsn.py                  — FDSN client with local SDS caching
│
├── tremor/
│   ├── calculate_tremor.py      — CalculateTremor (orchestrator)
│   ├── rsam.py, dsar.py, shannon_entropy.py — per-metric kernels
│   └── tremor_data.py           — TremorData (CSV wrapper)
│
└── utils/
    ├── array.py, dataframe.py, date_utils.py
    ├── formatting.py, ml.py, pathutils.py
    ├── validation.py, window.py
```

64 `.py` files in total.

---

## 2. Pipeline Overview

```
       ┌──────────────┐     ┌────────────────────┐    ┌─────────────────┐
       │  Seismic     │     │  CalculateTremor   │    │   TremorData    │
       │  archive     │ ──► │  (rsam/dsar/       │ ─► │   (CSV wrapper) │
       │  (SDS|FDSN)  │     │   entropy/bands)   │    │                 │
       └──────────────┘     └────────────────────┘    └────────┬────────┘
                                                               │
       ┌─────────────────────────── feature pipeline ──────────┴─────┐
       │   LabelBuilder            TremorMatrixBuilder               │
       │   DynamicLabelBuilder ──► FeaturesBuilder (tsfresh)         │
       │                           FeatureSelector (FDR+RF)          │
       └────────────────────────────┬────────────────────────────────┘
                                    ▼
                       ┌────────────────────────┐
                       │     TrainingModel      │
                       │   build_label →        │
                       │   extract_features →   │
                       │   fit (N seeds × M cv) │
                       └──┬───────────────────┬─┘
                          │ writes            │ assembles
                          ▼                   ▼
                ┌────────────────┐    ┌────────────────────────┐
                │ SeedEnsemble × │    │   ClassifierEnsemble   │
                │ N classifiers  │ ─► │ (all SeedEnsembles)    │
                └────────────────┘    └──────────┬─────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │      PredictionModel         │
                                  │   build_label →              │
                                  │   extract_features →         │
                                  │   forecast (per-seed proba)  │
                                  └──────────┬───────────────────┘
                                             │
            ┌────────────────────────────────┴──────────────────────┐
            ▼                                                       ▼
   ┌──────────────────────┐                          ┌────────────────────────┐
   │   EvaluationModel    │                          │  result_all_model_     │
   │  dispatch on .kind:  │                          │  predictions_*.csv +   │
   │  training | predict  │  ── MetricsEnsemble ──►  │  forecast PNG/PDF      │
   └──────────┬───────────┘                          └────────────────────────┘
              │ writes per-seed JSON + aggregate CSVs
              ▼
   ┌──────────────────────┐
   │ ClassifierComparator │   ranking_*.csv + comparison figures
   └──────────────────────┘
```

`ForecastModel` is the orchestrator that calls every box in sequence. The dashed arrows are also the **method-chain order**: `fm.calculate(...).train(...).predict(...).evaluate(...)`.

---

## 3. Component Details

### 3.1 Tremor (`tremor/`)

`CalculateTremor` reads seismic traces day-by-day from a `SeismicDataSource` and dispatches each day to the configured tremor kernels (`rsam.py`, `dsar.py`, `shannon_entropy.py`). Per-day CSVs are written to `tremor/daily/`, then concatenated into the merged tremor CSV at the station root. `TremorData` is a thin wrapper that exposes `df`, `start_date`, `end_date`, sampling-rate validation, and the CSV `filename` / `basename` / `filetype` triple.

### 3.2 Labels (`label/`)

Two builders share the same output shape (`id`, `is_erupted`) but differ in how positives are placed:

- **`LabelBuilder`** — sliding window over the full date range; `day_to_forecast` controls the look-ahead window. `include_eruption_date=False` (default) still marks the eruption day as positive, giving `day_to_forecast + 1` positive days per eruption.
- **`DynamicLabelBuilder`** — extends `LabelBuilder` with a per-eruption three-phase build: (1) zero frames per eruption, (2) concat + deduplicate datetimes, (3) mark positives per eruption. Solves the issue where overlapping look-ahead windows collide in `LabelBuilder`.

```
LabelBuilder — one global window over the full date range
─────────────────────────────────────────────────────────
 include_eruption_date=False  (default)
   0 0 0 0 0 0 0 0 0 0  1  1  1  1  1  1  1
                        ↑              ↑  ↑
                    dtf start       day-before eruption
                                       eruption (also 1)
   → dtf days strictly before eruption + eruption day = dtf+1 positives

 include_eruption_date=True
   0 0 0 0 0 0 0 0 0 0  0  1  1  1  1  1  1
                           ↑              ↑
                       dtf start      eruption (counted in dtf)
   → exactly dtf days ending on the eruption day


DynamicLabelBuilder — per-eruption build, overlapping windows deduped
─────────────────────────────────────────────────────────────────────
 Phase 1: initiate (all zeros)
   Eruption A window           Eruption B window
   [0 0 0 0 0 0 0 0 0 0]      [0 0 0 0 0 0 0 0 0 0]

 Phase 2: concat + deduplicate datetimes
   [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]   ← unified, sorted, unique

 Phase 3: mark positives per eruption
   Erup A (2025-03-20, dtf=2):  Mar 18–20 → 1
   Erup B (2025-03-23, dtf=2):  Mar 21–23 → 1
   [0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1]
                            ↑       ↑
                          Erup A  Erup B
```

`LabelData` parses parameters (`window_size`, `window_step`, `window_step_unit`, `day_to_forecast`) directly out of the label filename so a CSV alone is enough to rehydrate the build context.

### 3.3 Features (`features/`)

```
        labels (id, is_erupted)         tremor_df
                │                            │
                ▼                            ▼
        ┌────────────────────────────────────────┐
        │         TremorMatrixBuilder            │
        │  windowed slices aligned to labels     │
        └────────────────────┬───────────────────┘
                             ▼
        ┌────────────────────────────────────────┐
        │           FeaturesBuilder              │
        │  tsfresh extraction (per-column)       │
        │  training: relevance-filter on labels  │
        │  prediction: no filtering              │
        └────────────────────┬───────────────────┘
                             ▼
        ┌────────────────────────────────────────┐
        │           FeatureSelector              │
        │  1. tsfresh FDR filter                 │
        │  2. RandomForest importance            │
        │  → top-N feature names per seed        │
        └────────────────────────────────────────┘
```

`TremorMatrixBuilder.build()` validates sample counts per window against `minimum_completion` and skips short windows so tsfresh never sees ragged input. `FeaturesBuilder` runs per-column independent extractions so adding a new tremor band does not invalidate the cached results for the others.

### 3.4 Model (`model/`)

The model layer follows a **mixin** pattern:

- **`BaseModel`** — abstract base for every stage. Owns the date/window grid, the lazy `tremor_data` accessor, `output_dir` resolution, `n_jobs` clamping, and joblib `save()` / `load()`. Subclasses implement `set_directories`, `create_directories`, `validate`, `describe`, `to_dict`, `to_prompt`, `build_label`, `extract_features`.
- **`CacheModel`** — abstract mixin that adds a content-addressable cache layer. Implementers declare `build_cache_identity(**kwargs) -> dict`; the mixin canonicalises the dict, SHA-256-hashes it, and reads/writes `{output_dir}/cache/{ClassName}/{hash}.pkl` (with a `.params.json` sidecar for debugging).
- **`TrainingModel(BaseModel, CacheModel)`** — `build_label → extract_features → fit`. `fit()` runs per-seed `GridSearchCV` in `joblib.Parallel` over the selected classifiers, then auto-merges every seed via `merge_seed_models` and every classifier via `merge_all_classifiers`.
- **`PredictionModel(BaseModel, CacheModel)`** — `build_label → extract_features → forecast`. Cache identity embeds the upstream `training_hash`, so re-training automatically invalidates downstream forecasts.
- **`EvaluationModel(BaseModel)`** — no cache; dispatches on `model.kind` (`"training"` or `"prediction"`). Output is namespaced under `evaluation/{kind}/` so both modes can coexist.
- **`ForecastModel`** — the orchestrator. Not a `BaseModel` subclass — it owns `CalculateTremor`, builds the three stage classes lazily, and captures stage kwargs into a `ForecastConfig` for round-tripping.

`ClassifierModel` is the per-classifier descriptor (sklearn estimator + hyperparameter grid + slug). `ClassifierComparator` consumes the metric JSON tree written by `EvaluationModel` to rank classifiers head-to-head.

### 3.5 Ensemble (`ensemble/`)

```
            BaseEnsemble (joblib save/load mixin)
                │
       ┌────────┴────────┐
       ▼                 ▼
SeedEnsemble       ClassifierEnsemble
1 classifier ×     N classifiers ×
N fitted seeds     1 SeedEnsemble each
+ feature lists    + factories (from_any, from_json,
                     from_dict, from_seed_ensembles,
                     from_registry_dict)

           MetricsEnsemble  (standalone — not a BaseEnsemble subclass)
           wraps ClassifierEnsemble + features + y_true
           writes per-seed metrics JSON and y_proba / y_pred matrices
```

`MetricsEnsemble` is deliberately not exported from `ensemble/__init__.py` and is imported via `eruption_forecast.ensemble.metrics_ensemble` to break a `MetricsEnsemble → MetricsComputer → utils.ml → SeedEnsemble` cycle.

### 3.6 Sources (`sources/`)

`SeismicDataSource` is the read interface: `get(date) -> obspy.Stream`. Two concrete implementations:

- **`SDS`** — pure local read from `{root}/{year}/{network}/{station}/{channel}.D/{file}`.
- **`FDSN`** — pulls from a remote FDSN service, then caches the downloaded MSEED into a local SDS layout (`download_dir`). Repeat calls with the same date hit the local cache.

### 3.7 Plots (`plots/`)

`apply_nature_style()` normalises every figure to a Nature/Science-friendly palette and font stack. Each plot module is a thin functional wrapper around `matplotlib` (and `seaborn` where appropriate) — see [Visualization](Visualization) for the catalog and output paths.

### 3.8 Config (`config/`)

`ForecastConfig` is the round-trip record for `ForecastModel`. Its five sub-configs match the stage method signatures one-for-one:

```
ForecastConfig
├── model:     BaseForecastConfig
├── calculate: ForecastCalculateConfig | None
├── train:     ForecastTrainConfig     | None
├── predict:   ForecastPredictConfig   | None
└── evaluate:  ForecastEvaluateConfig  | None
```

`TrainingConfig` mirrors `TrainingModel.__init__` directly and is the standalone equivalent used when `TrainingModel` runs outside `ForecastModel`.

### 3.9 Decorators (`decorators/`)

`notify(label)` wraps a function with start/finish/error Telegram messages. `send_telegram_notification(message, files, file_caption)` is the one-off helper used inside `scenarios.py` to ship each per-scenario plot.

### 3.10 Utils (`utils/`)

Eight focused modules that the rest of the codebase pulls from — see the table in [§6](#6-utility-modules).

---

## 4. Model Class Relationships

```
                            ┌─────────────────────────┐
                            │       BaseModel         │
                            │   (ABC)                 │
                            │ • dates, output_dir     │
                            │ • tremor_data (lazy)    │
                            │ • n_jobs clamp          │
                            │ • save() / load()       │
                            └────────────┬────────────┘
                                         │ inherits
            ┌────────────────────────────┼────────────────────────────┐
            ▼                            ▼                            ▼
  ┌───────────────────┐       ┌────────────────────┐      ┌────────────────────┐
  │  TrainingModel    │       │  PredictionModel   │      │  EvaluationModel   │
  │ + CacheModel mix  │       │ + CacheModel mix   │      │ (BaseModel only)   │
  │                   │       │                    │      │                    │
  │ build_label →     │       │ build_label →      │      │ dispatch on        │
  │ extract_features →│       │ extract_features → │      │ model.kind:        │
  │ fit (N seeds)     │       │ forecast           │      │   training |       │
  └────────┬──────────┘       └─────────┬──────────┘      │   prediction       │
           │                            │                 │ evaluate / compare │
           │ produces                   │ consumes        └──────────┬─────────┘
           ▼                            │                            │ uses
  ┌────────────────────────┐            │                            ▼
  │  ClassifierEnsemble    │ ◄──────────┘             ┌────────────────────────┐
  │ ───────────────────    │                          │   MetricsEnsemble      │
  │  • from_any            │                          │ • per-seed metrics     │
  │  • from_json           │                          │ • y_proba / y_pred CSV │
  │  • from_seed_ensembles │                          └───────────┬────────────┘
  └────────┬───────────────┘                                      │ aggregates
           │ bundles                                              ▼
           ▼                                          ┌───────────────────────┐
  ┌────────────────────────┐                          │ ClassifierComparator  │
  │   SeedEnsemble × M     │                          │ • get_ranking()       │
  │ ───────────────────    │                          │ • plot_all()          │
  │ • predict_proba        │                          └───────────────────────┘
  │ • predict_with_        │
  │   uncertainty          │
  └──────────┬─────────────┘
             │ inherits
             ▼
       ┌──────────────────────┐
       │    BaseEnsemble      │
       │  (joblib save/load)  │
       └──────────────────────┘
```

**Scope cheat-sheet**:

| Class                  | Scope (per …)              | Mixin / Inheritance          | Cache |
|------------------------|----------------------------|------------------------------|-------|
| `BaseModel`            | —                          | ABC                          | ✗ |
| `CacheModel`           | —                          | ABC mixin                    | self |
| `BaseEnsemble`         | —                          | mixin                        | ✗ |
| `TrainingModel`        | One date span              | `BaseModel + CacheModel`     | ✓ |
| `PredictionModel`      | One forecast window grid   | `BaseModel + CacheModel`     | ✓ |
| `EvaluationModel`      | One trained model          | `BaseModel`                  | ✗ |
| `SeedEnsemble`         | 1 classifier × N seeds     | `BaseEnsemble`               | ✗ |
| `ClassifierEnsemble`   | M classifiers × N seeds    | `BaseEnsemble`               | ✗ |
| `MetricsEnsemble`      | 1 ensemble × 1 dataset     | standalone                   | ✗ |
| `ClassifierComparator` | M classifiers, post-eval   | standalone                   | ✗ |
| `ForecastModel`        | Full pipeline              | standalone orchestrator      | via stages |

---

## 5. Pipeline Data Flow

### 5.1 Per-stage I/O

| Stage              | Driver class               | Reads                          | Writes                                                                                       |
|--------------------|----------------------------|--------------------------------|----------------------------------------------------------------------------------------------|
| Tremor             | `CalculateTremor`          | `SeismicDataSource.get(date)`  | `tremor/daily/*.csv`, merged `{nslc}_{start}_{end}.csv`                                      |
| Label              | `LabelBuilder`             | Tremor index, eruption dates   | `training/features/{cv}/features-label_*.csv`                                                |
| Tremor matrix      | `TremorMatrixBuilder`      | Tremor CSV + labels            | `training/tremor/tremor_matrix_*.csv` (+ `per_method/`)                                      |
| Features           | `FeaturesBuilder`          | Tremor matrix                  | `training/features/{cv}/features-matrix_*.csv`                                               |
| Feature selection  | `FeatureSelector`          | Features + labels              | `training/features/{cv}/seed/{seed:05d}.csv` + `top_N_features.csv`                          |
| Training fit       | `TrainingModel`            | Selected features + labels     | `training/classifiers/{clf}/{cv}/models/*.pkl` + `SeedEnsemble_*.pkl` + `ClassifierEnsemble_*.{pkl,json}` |
| Prediction grid    | `PredictionModel`          | Tremor CSV + window grid       | `prediction/features/features-{matrix,label}_*.csv`                                          |
| Forecast           | `PredictionModel.forecast` | Forecast features + ensemble   | `prediction/results/{clf}/{seed:05d}.csv` + `result_all_model_predictions_*.csv` + `prediction/figures/forecast_*.{png,pdf}` |
| Evaluation         | `EvaluationModel.evaluate` | y_proba + y_true               | `evaluation/{kind}/classifiers/{Clf}/metrics/json/{seed:05d}.json` + `predictions/{y_proba,y_pred,y_true}.csv` + `metrics_summary_*.csv` + `figures/*.png` |
| Compare            | `ClassifierComparator`     | Metrics JSON tree              | `evaluation/{kind}/comparison/metrics/ranking_*.csv` + `comparison/figures/*.png`            |

### 5.2 On-disk artefact graph

```
            ┌────────────────────────────────────────┐
            │   tremor/{nslc}_{start}_{end}.csv      │  ← CalculateTremor
            └─────────┬──────────────────────────────┘
                      │ used by Training / Prediction / Evaluation
                      ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ training/                                                     │
    │  features/{cv}/                                               │
    │    features-matrix_*.csv ──► features-label_*.csv             │
    │       │                                                       │
    │       ▼                                                       │
    │    seed/{seed:05d}.csv  ──► resampled/{seed:05d}.csv          │
    │    top_{N}_features.csv  + .png                               │
    │                                                               │
    │  classifiers/                                                 │
    │    {clf}/{cv}/models/{seed:05d}.pkl                           │
    │    {clf}/{cv}/SeedEnsemble_*.pkl                              │
    │    ClassifierEnsemble_{cv}.{pkl,json}                         │
    └─────────┬─────────────────────────────────────────────────────┘
              │ ClassifierEnsemble bundle
              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ prediction/                                                   │
    │   features/{features-matrix,features-label}_*.csv             │
    │   results/{clf}/{seed:05d}.csv                                │
    │   figures/forecast_*.{png,pdf}                                │
    │ result_all_model_predictions_*.csv  (top-level dump)          │
    └─────────┬─────────────────────────────────────────────────────┘
              │ ClassifierEnsemble + features + y_true (rebuilt or training-derived)
              ▼
    ┌──────────────────────────────────────────────────────────────┐
    │ evaluation/{training|prediction}/                             │
    │   classifiers/{Clf}/                                          │
    │     predictions/{y_proba,y_pred,y_true}.csv                   │
    │     metrics/json/{seed:05d}.json                              │
    │     metrics_summary_*.csv  + all_metrics_*.csv                │
    │     figures/*.png                                             │
    │   comparison/                                                 │
    │     metrics/ranking_*.csv                                     │
    │     figures/*.png                                             │
    └──────────────────────────────────────────────────────────────┘

           ┌──────────────────────────────────────────────────┐
           │  cache/                                           │
           │    TrainingModel/{hash}.pkl + {hash}.params.json  │  ← CacheModel
           │    PredictionModel/{hash}.pkl + {hash}.params.json│
           └──────────────────────────────────────────────────┘
```

A cache hit on `TrainingModel` short-circuits everything in the `training/` box; a cache hit on `PredictionModel` short-circuits the `prediction/` box. Evaluation is **never cached** — the metric JSONs themselves act as the cache via the `overwrite` flag.

---

## 6. Utility Modules

| Module               | Key functions                                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------------------------|
| `utils/array.py`     | `detect_maximum_outlier`, `remove_outliers`, `detect_anomalies_zscore`, `aggregate_seed_probabilities`, `predict_proba_from_estimator` |
| `utils/window.py`    | `construct_windows`, `calculate_window_metrics`                                                                |
| `utils/date_utils.py`| `to_datetime`, `normalize_dates`, `sort_dates`, `parse_label_filename`, `set_datetime_index`, `label_id_to_datetime` |
| `utils/ml.py`        | `random_under_sampler`, `get_significant_features`, `load_labels_from_csv`, `merge_seed_models`, `merge_all_classifiers`, `compute_seed_eruption_probability`, `compute_model_probabilities`, `get_classifier_models`, `compute_g_mean`, `compute_seed`, `build_y_true` |
| `utils/validation.py`| `validate_random_state`, `validate_date_ranges`, `validate_window_step`, `validate_columns`, `check_sampling_consistency` |
| `utils/pathutils.py` | `resolve_output_dir`, `ensure_dir`, `save_figure`, `save_data`, `load_json`                                    |
| `utils/dataframe.py` | `load_label_csv`, DataFrame shape and column helpers                                                           |
| `utils/formatting.py`| `slugify`, human-readable elapsed time and file sizes                                                          |

`utils/ml.merge_seed_models` and `utils/ml.merge_all_classifiers` are the bridge between the per-seed `.pkl` outputs and the `SeedEnsemble` / `ClassifierEnsemble` packaging — both are invoked at the end of `TrainingModel.fit()`.

`utils/formatting.slugify` is what turns `"Scenario 1"` into `scenario-1` for the per-scenario `output_dir` used in `scenarios.py`.
