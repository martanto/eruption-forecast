# Architecture

This page is the structural reference for `eruption_forecast`: every module under `src/`, 
the top-level pipeline, how the model and ensemble classes relate, what flows between 
stages on disk, and the utility surface that holds the rest together.

---

## 1. Package Layout

```
src/eruption_forecast/
├── __init__.py            - public exports
├── logger.py              - loguru wrapper (enable/disable/set_level/set_directory) + per-category error files (register_error_category, get_category_logger; telegram category ships pre-registered)
├── data_container.py      - BaseDataContainer ABC for TremorData / LabelData
│
├── config/
│   ├── base_config.py         - shared config primitives
│   ├── constants.py           - ERUPTION_PROBABILITY_THRESHOLD, defaults
│   ├── forecast_config.py     - ForecastConfig + per-stage sub-configs
│   ├── training_config.py     - TrainingConfig    (standalone TrainingModel)
│   ├── prediction_config.py   - PredictionConfig  (standalone PredictionModel)
│   ├── evaluation_config.py   - EvaluationConfig  (standalone EvaluationModel)
│   └── explanation_config.py  - ExplanationConfig (standalone ExplanationModel)
│
├── dataclass/
│   ├── station_data.py                 - StationData (immutable nslc identity)
│   ├── classifier_ensemble_summary.py  - ClassifierEnsembleSummary, EruptionWindow, ProbabilityPick
│   └── classifier_explanation.py       - SeedExplanation, ClassifierExplanation (SHAP payloads)
│
├── decorators/
│   ├── notify.py              - @notify decorator (Telegram success/error notifications)
│   └── timer.py               - @timer decorator (elapsed-time logger, optional Telegram forward)
│
├── notification/
│   └── telegram.py            - TelegramNotification (send_message / send_document / send_photo / send_media_group)
│
├── ensemble/
│   ├── base_ensemble.py       - BaseEnsemble (joblib save/load mixin)
│   ├── seed_ensemble.py       - SeedEnsemble (one classifier × N seeds)
│   ├── classifier_ensemble.py - ClassifierEnsemble (N classifiers)
│   ├── metrics_ensemble.py    - MetricsEnsemble (metrics engine)
│   └── explainer_ensemble.py  - ExplainerEnsemble (per-seed SHAP engine)
│
├── features/
│   ├── constants.py
│   ├── tremor_matrix_builder.py - TremorMatrixBuilder (windowed alignment)
│   ├── features_builder.py      - FeaturesBuilder (tsfresh extraction)
│   ├── feature_selector.py      - FeatureSelector (tsfresh FDR or RF importance)
│   └── feature_count_sweep.py   - ⚠ Experimental. FeatureCountSweep + sweep_feature_count (post-hoc top_n_features recommender)
│
├── label/
│   ├── constants.py
│   ├── label_builder.py         - LabelBuilder (sliding window)
│   ├── dynamic_label_builder.py - DynamicLabelBuilder (per-eruption build)
│   ├── label_data.py            - LabelData (CSV wrapper)
│   └── label_plots.py           - plot_label_distribution
│
├── model/
│   ├── constants.py
│   ├── base_model.py            - BaseModel ABC (dates, I/O, dual-mode save/load + cache identity)
│   ├── forecast_model.py        - ForecastModel orchestrator
│   ├── training_model.py        - TrainingModel(BaseModel)
│   ├── prediction_model.py      - PredictionModel(BaseModel)
│   ├── evaluation_model.py      - EvaluationModel(BaseModel)
│   ├── explanation_model.py     - ExplanationModel(BaseModel)
│   ├── classifier_model.py      - ClassifierModel (estimator + grid)
│   └── classifier_comparator.py - ClassifierComparator (cross-classifier rank)
│
├── plots/
│   ├── styles.py
│   ├── tremor_plots.py          - plot_tremor
│   ├── feature_plots.py         - feature-importance plots
│   ├── forecast_plots.py        - plot_forecast, plot_forecast_from_file
│   ├── evaluation_plots.py      - ROC, PR, confusion, threshold, importance
│   └── explanation_plots.py     - SHAP waterfall / beeswarm / bar / aggregate
│
├── sources/
│   ├── base.py                  - SeismicDataSource ABC
│   ├── sds.py                   - Local SeisComP archive reader
│   └── fdsn.py                  - FDSN client with local SDS caching
│
├── tremor/
│   ├── calculate_tremor.py      - CalculateTremor (orchestrator)
│   ├── rsam.py, dsar.py, shannon_entropy.py - per-metric kernels
│   └── tremor_data.py           - TremorData (CSV wrapper)
│
└── utils/
    ├── array.py, benchmark.py, dataframe.py, date_utils.py
    ├── formatting.py, ml.py, pathutils.py
    ├── validation.py, window.py
```

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
       │                           FeatureSelector (FDR or RF)       │
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
                                  └──────────────┬───────────────┘
                                                 │
             ┌───────────────────────────────────┴──────────────────┐
             ▼                                                      ▼
   ┌──────────────────────┐                          ┌────────────────────────┐
   │   EvaluationModel    │                          │  forecast-results_     │
   │  dispatch on .kind:  │                          │  *.csv + forecast      │
   │  training | predict  │  ── MetricsEnsemble ──►  │  PNG/PDF               │
   └──────────┬───────────┘                          └────────────────────────┘
              │ writes (n_samples, n_seeds) y_proba / y_pred matrices
              ▼
   ┌──────────────────────┐
   │ ClassifierComparator │   ranking_*.csv + comparison figures
   └──────────────────────┘

   ┌────────────────────────────────────────────────────────────────┐
   │    ExplanationModel  (BaseModel)                               │
   │    dispatch on upstream model.kind: training | prediction      │
   │                                                                │
   │    ExplainerEnsemble                                           │
   │      ─ per-seed shap.TreeExplainer (RF / lite-rf / GB / XGB)   │
   │      ─ ClassifierExplanation.pkl per classifier                │
   │      ─ per-seed bar + beeswarm under classifiers/{Clf}/figures │
   │      ─ per-eruption waterfall under eruptions/{date}/          │
   └────────────────────────────────────────────────────────────────┘
```

`ForecastModel` is the orchestrator that calls every box in sequence. 
The dashed arrows are also the **method-chain order**: 
`fm.calculate(...).train(...).predict(...).evaluate(...).explain(...)`.

---

## 3. Component Details

### 3.1 Tremor (`tremor/`)

`CalculateTremor` reads seismic traces day-by-day from a `SeismicDataSource` and 
dispatches each day to the configured tremor kernels (`rsam.py`, `dsar.py`, `shannon_entropy.py`). 
Per-day CSVs are written to `tremor/daily/`, then concatenated into the merged tremor 
CSV at the station root. `TremorData` is a thin wrapper that exposes `df`, `start_date`, `end_date`, 
sampling-rate validation, and the CSV `filename` / `basename` / `filetype` triple.

### 3.2 Labels (`label/`)

Two builders share the same output shape (`id`, `is_erupted`) but differ in how positives are placed:

- **`LabelBuilder`** - sliding window over the full date range; `day_to_forecast` controls the look-ahead window. `include_eruption_date=False` (default) still marks the eruption day as positive, giving `day_to_forecast + 1` positive days per eruption.
- **`DynamicLabelBuilder`** - extends `LabelBuilder` with a per-eruption three-phase build: (1) zero frames per eruption, (2) concat + deduplicate datetimes, (3) mark positives per eruption. Solves the issue where overlapping look-ahead windows collide in `LabelBuilder`.

```
LabelBuilder - one global window over the full date range
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


DynamicLabelBuilder - per-eruption build, overlapping windows deduped
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

`LabelData` parses parameters (`window_size`, `window_step`, `window_step_unit`, `day_to_forecast`) 
directly out of the label filename so a CSV alone is enough to rehydrate the build context.

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
        │  method="tsfresh": FDR p-value filter  │
        │  method="random_forest": permutation   │
        │                          importance    │
        │  → top-N feature names per seed        │
        └────────────────────────────────────────┘
```

`TremorMatrixBuilder.build()` validates sample counts per window against `minimum_completion` 
and skips short windows so tsfresh never sees ragged input. 
`FeaturesBuilder` runs per-column independent extractions so adding a new tremor 
band does not invalidate the cached results for the others.

### 3.4 Model (`model/`)

The model layer follows a **mixin** pattern:

- **`BaseModel`** - abstract base for every stage. Owns the date/window grid, the lazy `tremor_data` accessor, `output_dir` resolution, `n_jobs` clamping, the content-addressable cache identity helpers (`build_identity`, `compute_hash`, `_canonicalize`, `tremor_fingerprint`, `cache_path`), and the dual-mode joblib `save(identity=None, path=None)` / cache-only `load(stage_dir, identity)`. When `identity` is supplied, `save()` writes to `{stage_dir}/{hash}.{ClassName}.pkl` (plus a `.params.json` sidecar). When `identity` is omitted, the legacy `{output_dir}/{ClassName}_{basename}.pkl` joblib dump is preserved for standalone manual saves. Subclasses implement `set_directories`, `create_directories`, `validate`, `describe`, `to_dict`, `to_prompt`, `build_label`, `extract_features`, and override `stage_dir` + `build_identity` when they participate in the cache.
- **`TrainingModel(BaseModel)`** - `build_label → extract_features → fit`. `fit()` runs per-seed `GridSearchCV` in `joblib.Parallel` over the selected classifiers, writes a per-classifier trained-model JSON registry via `save_model_json`, bundles every seed into a `SeedEnsemble` and every classifier into a `ClassifierEnsemble`, then calls `self.save(self.build_identity())` so the cache pickle lands at `{training_dir}/{hash}.TrainingModel.pkl` with a matching sidecar.
- **`PredictionModel(BaseModel)`** - `build_label → extract_features → forecast`. Cache identity embeds the upstream `training_hash` (a constructor param threaded by `ForecastModel.predict`), so re-training automatically invalidates downstream forecasts. `forecast()` calls `self.save(self.build_identity())`; cache files live at `{prediction_dir}/{hash}.PredictionModel.pkl`.
- **`EvaluationModel(BaseModel)`** - no cache; dispatches on `model.kind` (`"training"` or `"prediction"`). Output is namespaced under `evaluation/{kind}/` so both modes can coexist.
- **`ExplanationModel(BaseModel)`** - per-seed SHAP explanations over a fitted `ClassifierEnsemble`. Reuses the upstream `TrainingModel` or `PredictionModel` and dispatches on `model.kind`. Restricted to tree classifiers (RF / lite-rf / GB / XGB); non-tree classifiers are skipped at the `ExplainerEnsemble` loop with a warning. Output is namespaced under `explanation/{kind}/`; cache pickles land at `{explanation_dir}/{hash}.ExplanationModel.pkl` (already mode-namespaced so training-reuse and prediction-reuse caches never collide).
- **`ForecastModel`** - the orchestrator. Not a `BaseModel` subclass - it owns `CalculateTremor`, builds the four stage classes lazily, and captures stage kwargs into a `ForecastConfig` for round-tripping.

`ClassifierModel` is the per-classifier descriptor (sklearn estimator + hyperparameter grid + slug). 
`ClassifierComparator` consumes the in-memory `MetricsEnsemble` cached on `EvaluationModel` to rank classifiers head-to-head.

### 3.5 Ensemble (`ensemble/`)

```
            BaseEnsemble (joblib save/load mixin)
                │
       ┌────────┴────────┐
       ▼                 ▼
SeedEnsemble       ClassifierEnsemble
1 classifier ×     N classifiers ×
N fitted seeds     1 SeedEnsemble each
+ per-seed         + features (sorted union
  feature lists      across all SeedEnsembles)
+ features         + factories (from_any, from_json,
  (sorted union      from_dict, from_seed_ensembles)
  across all
  seeds)

           MetricsEnsemble  (standalone - not a BaseEnsemble subclass)
           wraps ClassifierEnsemble + features + y_true
           writes only (n_samples, n_seeds) y_proba / y_pred CSV matrices
           metrics / y_probas / y_preds stay in memory

           ExplainerEnsemble  (standalone - not a BaseEnsemble subclass)
           wraps ClassifierEnsemble + features
           writes per-classifier ClassifierExplanation.pkl
           + per-seed shap_values/{seed:05d}.pkl
           + per-seed bar / beeswarm + per-eruption waterfall plots
```

`MetricsEnsemble` and `ExplainerEnsemble` are both deliberately kept out of `ensemble/__init__.py` and imported via their full module paths (`eruption_forecast.ensemble.metrics_ensemble`, `eruption_forecast.ensemble.explainer_ensemble`) to keep the subpackage free of import cycles back through `utils.ml` and `plots/`.

### 3.6 Sources (`sources/`)

`SeismicDataSource` is the read interface: `get(date) -> obspy.Stream`. Two concrete implementations:

- **`SDS`** - pure local read from `{root}/{year}/{network}/{station}/{channel}.D/{file}`.
- **`FDSN`** - pulls from a remote FDSN service, then caches the downloaded MSEED into a local SDS layout (`download_dir`). Repeat calls with the same date hit the local cache.

### 3.7 Plots (`plots/`)

`apply_nature_style()` normalises every figure to a Nature/Science-friendly palette and font stack. 
Each plot module is a thin functional wrapper around `matplotlib` (and `seaborn` where appropriate) - see [Visualization](Visualization) for the catalog and output paths.

### 3.8 Config (`config/`)

`ForecastConfig` is the round-trip record for `ForecastModel`. Its six sub-configs match the stage method signatures one-for-one:

```
ForecastConfig
├── model:     BaseForecastConfig
├── calculate: ForecastCalculateConfig | None
├── train:     ForecastTrainConfig     | None
├── predict:   ForecastPredictConfig   | None
├── evaluate:  ForecastEvaluateConfig  | None
└── explain:   ForecastExplainConfig   | None
```

`TrainingConfig`, `PredictionConfig`, `EvaluationConfig`, and `ExplanationConfig` each mirror their stage model's `__init__` directly and are the standalone equivalents used when the model runs outside `ForecastModel`. Every stage model auto-calls `save_config()` at the end of its main run method (`fit()` / `forecast()` / `evaluate()` / `explain()`), so a standalone run always leaves a YAML snapshot next to its artefacts. The upstream `model` parameter on `EvaluationConfig` and `ExplanationConfig` is intentionally omitted because it is always a live model instance.

### 3.9 Decorators (`decorators/`) and Notification (`notification/`)

`notify(task)` wraps a function with success and error Telegram messages (MarkdownV2 body, hostname, elapsed time, exception details). `timer(name, send_to=None)` logs the wrapped function's elapsed wall-clock time via `loguru`; passing `send_to="telegram"` also mirrors the message to Telegram.

Both decorators delegate to `TelegramNotification` (`notification/telegram.py`), a fluent-chain client wrapping the Telegram Bot API. It exposes `send_message(...)`, `send_document(...)`, `send_photo(...)`, and `send_media_group(...)`; every send method returns `self` so calls can be chained (`tn.send_message(...).send_document(...)`). Credentials are resolved from constructor arguments or the `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` environment variables. Every network failure is logged and swallowed, so a dead network never blocks the caller. `scenarios.py` uses this class directly to ship each per-scenario forecast PNG next to a title message.

### 3.10 Utils (`utils/`)

Nine focused modules that the rest of the codebase pulls from - see the table in [6](#6-utility-modules).

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
            ┌────────────────┬───────────┼───────────────┬────────────────┐
            ▼                ▼           ▼               ▼                ▼
  ┌───────────────┐ ┌───────────────┐ ┌──────────────┐ ┌───────────────────┐
  │ TrainingModel │ │PredictionModel│ │EvaluationMdl │ │ ExplanationModel  │
  │ (BaseModel)   │ │ (BaseModel)   │ │(BaseModel)   │ │ (BaseModel)       │
  │               │ │               │ │              │ │                   │
  │ build_label → │ │ build_label → │ │ dispatch on  │ │ explain →         │
  │ extract_feat →│ │ extract_feat →│ │ model.kind   │ │   ExplainerEns.   │
  │ fit (N seeds) │ │ forecast      │ │ evaluate/    │ │ plot →            │
  │               │ │               │ │ compare      │ │   per-seed +      │
  │               │ │               │ │              │ │   waterfall       │
  └────────┬──────┘ └────────┬──────┘ └──────┬───────┘ └────────┬──────────┘
           │ produces        │ consumes      │ uses             │ reuses
           ▼                 │               ▼                  │
  ┌────────────────────────┐ │  ┌────────────────────────┐      │
  │  ClassifierEnsemble    │◄┘  │   MetricsEnsemble      │      │
  │ ───────────────────    │    │ • (n_samples × n_seeds)│      │
  │  • from_any / from_json│    │   y_proba / y_pred CSV │      │
  │  • from_seed_ensembles │    │ • metrics in memory    │      │
  └──────────┬─────────────┘    └───────────┬────────────┘      │
             │ bundles                      │ aggregates        │
             ▼                              ▼                   ▼
  ┌────────────────────────┐    ┌────────────────────────┐ ┌──────────────────┐
  │   SeedEnsemble × M     │    │ ClassifierComparator   │ │ ExplainerEnsemble│
  │ ─────────────────────  │    │ • get_ranking()        │ │ • TreeExplainer  │
  │ • predict_proba        │    │ • plot_all()           │ │   per seed       │
  │ • predict_with_        │    └────────────────────────┘ │ • ClassifierExpln│
  │   uncertainty          │                               │   per classifier │
  └──────────┬─────────────┘                               └──────────────────┘
             │ inherits
             ▼
 ┌─────────────────────────┐
 │      BaseEnsemble       │
 │    (joblib save/load)   │
 └─────────────────────────┘
```

**Scope cheat-sheet**:

| Class                  | Scope (per …)              | Mixin / Inheritance          | Cache |
|------------------------|----------------------------|------------------------------|-------|
| `BaseModel`            | -                          | ABC (cache identity + dual-mode save/load) | self |
| `BaseEnsemble`         | -                          | mixin                        | ✗ |
| `TrainingModel`        | One date span              | `BaseModel`                  | ✓ |
| `PredictionModel`      | One forecast window grid   | `BaseModel`                  | ✓ |
| `EvaluationModel`      | One trained model          | `BaseModel`                  | ✗ |
| `ExplanationModel`     | One trained ensemble       | `BaseModel`                  | ✓ |
| `SeedEnsemble`         | 1 classifier × N seeds     | `BaseEnsemble`               | ✗ |
| `ClassifierEnsemble`   | M classifiers × N seeds    | `BaseEnsemble`               | ✗ |
| `MetricsEnsemble`      | 1 ensemble × 1 dataset     | standalone                   | ✗ |
| `ExplainerEnsemble`    | 1 ensemble × 1 dataset     | standalone                   | ✗ |
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
| Features           | `FeaturesBuilder`          | Tremor matrix                  | `training/features/{cv}/features-matrix_*.parquet`                                           |
| Feature selection  | `FeatureSelector`          | Features + labels              | `training/features/{cv}/seed/{seed:05d}.csv` + `top_N_features.csv`                          |
| Training fit       | `TrainingModel`            | Selected features + labels     | `training/classifiers/{clf}/{cv}/models/*.pkl` + `SeedEnsemble_*.pkl` + `ClassifierEnsemble_*.{pkl,json}` |
| Prediction grid    | `PredictionModel`          | Tremor CSV + window grid       | `prediction/features/features-{matrix,label}_*.csv`                                          |
| Forecast           | `PredictionModel.forecast` | Forecast features + ensemble   | `prediction/results/{clf}/{seed:05d}.csv` + `forecast-results_*.csv` + `prediction/figures/forecast_*.{png,pdf}` |
| Evaluation         | `EvaluationModel.evaluate` | y_proba + y_true               | `evaluation/{kind}/classifiers/{Clf}/predictions/{y_proba,y_pred}.csv` + `figures/aggregate/{plot}.{png,csv}` + (when `plot_per_seed=True`) `figures/{plot}/{seed:05d}.png` |
| Compare            | `ClassifierComparator`     | Cached `MetricsEnsemble`       | `evaluation/{kind}/comparison/metrics/ranking_*.csv` + `comparison/figures/*.png`            |
| Explanation        | `ExplanationModel.explain` | `ClassifierEnsemble` + features| `explanation/{kind}/classifiers/{Clf}/ClassifierExplanation_*.pkl` + `shap_values/{seed:05d}.pkl` + `figures/{bar,beeswarm}/{seed:05d}.png` |
| Waterfalls         | `ExplainerEnsemble.plot_waterfall` | `ClassifierExplanation` + eruption dates | `explanation/{kind}/eruptions/{date}/{Clf}_{datetime}_seed=_index=.png` |

### 5.2 On-disk artefact graph

```
            ┌────────────────────────────────────────┐
            │   tremor/{nslc}_{start}_{end}.csv      │  ← CalculateTremor
            └─────────┬──────────────────────────────┘
                      │ used by Training / Prediction / Evaluation
                      ▼
    ┌───────────────────────────────────────────────────────────────┐
    │ training/                                                     │
    │  features/{cv}/                                               │
    │    features-matrix_*.parquet ──► features-label_*.csv         │
    │       │                                                       │
    │       ▼                                                       │
    │    seed/{seed:05d}.csv  ──► resampled/{seed:05d}.csv          │
    │    significant_features.csv  ──►  top_features.csv            │
    │                              ──►  top_{N}_features.csv + .png │
    │                                                               │
    │  classifiers/                                                 │
    │    {clf}/{cv}/models/{seed:05d}.pkl                           │
    │    {clf}/{cv}/SeedEnsemble_*.pkl                              │
    │    ClassifierEnsemble_{cv}.{pkl,json}                         │
    └─────────┬─────────────────────────────────────────────────────┘
              │ ClassifierEnsemble bundle
              ▼
    ┌───────────────────────────────────────────────────────────────┐
    │ prediction/                                                   │
    │   features/features-matrix_*.parquet + features-label_*.csv   │
    │   results/{clf}/{seed:05d}.csv                                │
    │   figures/forecast_*.{png,pdf}                                │
    │ forecast-results_*.csv  (top-level dump)          │
    └─────────┬─────────────────────────────────────────────────────┘
              │ ClassifierEnsemble + features + y_true (rebuilt or training-derived)
              ▼
    ┌───────────────────────────────────────────────────────────────┐
    │ evaluation/{training|prediction}/                             │
    │   classifiers/{Clf}/                                          │
    │     predictions/{y_proba,y_pred}.csv   (n_samples × n_seeds)  │
    │     figures/aggregate/{plot_name}.{png,csv}                   │
    │     figures/{plot_name}/{seed:05d}.png   (plot_per_seed=True) │
    │   labels/y_true.csv                    (prediction reuse only)│
    │   MetricsEnsemble.pkl                  (optional, via save()) │
    │   comparison/                                                 │
    │     metrics/ranking_*.csv                                     │
    │     figures/*.png                                             │
    └─────────┬─────────────────────────────────────────────────────┘
              │ ClassifierEnsemble + features
              ▼
    ┌───────────────────────────────────────────────────────────────┐
    │ explanation/{training|prediction}/                            │
    │   classifiers/{Clf}/                                          │
    │     ClassifierExplanation_{Clf}.pkl                           │
    │     shap_values/{seed:05d}.pkl                                │
    │     figures/{bar,beeswarm}/{seed:05d}.png                     │
    │   eruptions/{YYYY-MM-DD}/                                     │
    │     {Clf}_{datetime}_seed=_index=.png                         │
    └───────────────────────────────────────────────────────────────┘

           ┌────────────────────────────────────────────────────────────┐
           │  Stage-internal caches (no separate cache/ subtree):       │
           │    training/{hash}.TrainingModel.pkl       + .params.json  │  ← BaseModel.save
           │    prediction/{hash}.PredictionModel.pkl   + .params.json  │
           │    explanation/{kind}/{hash}.ExplanationModel.pkl + sidecar│
           └────────────────────────────────────────────────────────────┘
```

A cache hit on `TrainingModel` short-circuits everything in the `training/` box; 
a cache hit on `PredictionModel` short-circuits the `prediction/` box; 
a cache hit on `ExplanationModel` short-circuits the per-classifier SHAP pass. 
Evaluation is **never cached** - the on-disk matrices act as the cache and `MetricsEnsemble.compute()` is idempotent in memory once `y_probas` is populated.

---

## 6. Utility Modules

| Module               | Key functions                                                                                                  |
|----------------------|----------------------------------------------------------------------------------------------------------------|
| `utils/array.py`     | `detect_maximum_outlier`, `remove_maximum_outlier`, `remove_outliers`, `detect_anomalies_zscore`, `mask_zero_values`, `filter_nans`, `count_valid_values`, `get_completeness`, `confidence_interval`, `compute_model_probabilities`, `save_forecast_seed` |
| `utils/benchmark.py` | `benchmark_feature_selection` (side-by-side `FeatureSelector` method comparison)                               |
| `utils/window.py`    | `construct_windows`, `calculate_window_metrics`, `get_windows_information`, `chunk_daily_data`, `shannon_entropy`, `to_safe_array` |
| `utils/date_utils.py`| `to_datetime`, `normalize_dates`, `sort_dates`, `parse_label_filename`, `to_datetime_index`                    |
| `utils/ml.py`        | `random_under_sampler`, `resample`, `load_features_resampled`, `temporal_train_test_split`, `get_significant_features`, `get_classifier_models`, `grid_search_cv`, `save_model_json`, `compute_seed`, `build_y_true`, `build_classifier_ensemble_summary`, `compute_threshold_metrics`, `compute_aggregate_threshold_metrics` |
| `utils/validation.py`| `validate_random_state`, `validate_date_ranges`, `validate_window_step`, `validate_columns`, `check_sampling_consistency` |
| `utils/pathutils.py` | `resolve_output_dir`, `ensure_dir`, `save_figure`, `save_figure_as_pdf`, `save_data`, `load_json`, `load_pickle`, `setup_nslc_directories`, `generate_features_filepaths` |
| `utils/dataframe.py` | `load_label_csv`, `load_datetime_indexed`, `load_select_features`, `concat_features`, `concat_significant_features`, `find_common_features`, `plot_common_features_heatmap`, `plot_common_features_correlation`, `get_envelope_values`, `remove_anomalies`, `to_series` |
| `utils/formatting.py`| `slugify`, `slugify_class_name`, `shorten_feature_name`, `get_classifier_label`, `pdf_metadata`                |

`utils/ml.save_model_json` writes the per-classifier trained-model JSON registry (one record per seed, each with the inline top-N feature list and the path to the seed's `.pkl`). `TrainingModel.build_seed_ensemble` reads that registry via `SeedEnsemble.from_any` to package every seed into a `SeedEnsemble`, and the per-classifier `SeedEnsemble`s are then merged into a `ClassifierEnsemble` (`build_classifier_ensemble`). All three steps run at the end of `TrainingModel.fit()`.

`utils/formatting.slugify` is what turns `"Scenario 1"` into `scenario-1` for the per-scenario `output_dir` used in `scenarios.py`.
