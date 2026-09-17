# Pipeline Walkthrough

This page is the annotated tour of the two bundled entry-point scripts:

- **[Research Workflow](#research-workflow-mainpy)** - `main.py` - one-shot calculate → train → predict → evaluate → explain on a fixed date split.
- **[Scenarios Workflow](#scenarios-workflow-forecast-scenariopy)** - `forecast-scenario.py` - packaged multi-scenario orchestrator (`ForecastModelScenario`) that wraps one `ForecastModel`, captures shared per-stage kwargs, and loops `train → predict → evaluate → explain` over N `Scenario` dicts.

Both scripts share the same `ForecastModel` instance and the same `calculate()` output; they differ in how the `train → predict → evaluate → explain` legs are sequenced.

---

## Research Workflow (`main.py`)

`main.py` runs a single, linear pipeline: it computes tremor over the whole year, 
trains on the first seven months, forecasts the next four weeks, evaluates against the held-out eruptions, and finishes with per-seed SHAP explanations over the tree classifiers.

### Stage flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                       main.py - Stage Flow                           │
└──────────────────────────────────────────────────────────────────────┘

  fm = ForecastModel(network="VG", station="OJN", location="00",
                     channel="EHZ", day_to_forecast=2,
                     n_jobs=8, verbose=True)
       │
       ▼
┌─────────────────────┐
│  fm.calculate()     │  CalculateTremor → tremor_*.csv
│                     │  source = SDS  (D:\Data\OJN)
│                     │  methods = rsam, dsar, entropy
│                     │  dates: 2025-01-01 → 2025-12-31
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  fm.train()         │  TrainingModel → ClassifierEnsemble
│                     │    build_label  (window_step = 6 h, dtf = 2)
│                     │    extract_features (tsfresh, 5 tremor columns)
│                     │    fit          (4 classifiers × 25 seeds)
│                     │  cv = shuffle-stratified, scoring = recall
│                     │  resample = under,  n_jobs = 4, n_grids = 4
└──────────┬──────────┘
           │  cached → {station_dir}/training/{hash}.TrainingModel.pkl
           │
           ▼
┌─────────────────────┐
│  fm.predict()       │  PredictionModel → results
│                     │    build_label (window_step = 10 min)
│                     │    extract_features (same tremor columns)
│                     │    forecast     (4 clf × 25 seeds → consensus)
│                     │  plot_threshold = 0.7,  save_seed_result = True
└──────────┬──────────┘
           │  cached → {station_dir}/prediction/{hash}.PredictionModel.pkl
           │
           ▼
┌─────────────────────┐
│  fm.evaluate(       │  EvaluationModel(model="prediction")
│    model=           │    MetricsEnsemble.compute()
│    "prediction")    │    aggregate per-classifier metrics
│                     │    plot_per_seed = True
└──────────┬──────────┘
           │  cached → {station_dir}/evaluation/prediction/{hash}.EvaluationModel.pkl
           │
           ▼
┌─────────────────────┐
│  fm.EvaluationModel │  ClassifierComparator
│      .compare()     │    ranking CSV + comparison plots
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  fm.explain(        │  ExplanationModel(model="prediction")
│    model=           │    ExplainerEnsemble.explain()  (tree classifiers only)
│    "prediction")    │    per-seed bar/beeswarm + per-eruption waterfalls
│                     │  save_per_seed = True, plot_per_seed = False
└─────────────────────┘
           │  cached → {station_dir}/explanation/prediction/{hash}.ExplanationModel.pkl
```

### Per-stage notes

#### `fm.calculate(...)`
- Runs once across the full year so the same tremor frame is reused by every downstream stage.
- `start_date` is internally shifted back by `day_to_forecast` days (`forecast_model.py:154`) so the first label window has enough lead-in.
- `interpolate=True` fills miniSEED gaps so tsfresh receives a continuous signal.
- `plot_daily=True` emits per-day band plots under `tremor/figures/` - useful for spotting station outages before training.

#### `fm.train(...)`
- `classifiers=["lite-rf", "rf", "gb", "xgb"]` - four classifiers fit independently; their fitted `SeedEnsemble`s are bundled into a single `ClassifierEnsemble`.
- `eruption_dates` lists every known eruption - both the training-window eruptions and the held-out ones the forecast will be evaluated against.
- `scoring="recall"` instructs `GridSearchCV` to maximise positive recall - false negatives are worse than false positives for eruption forecasting.
- `seeds=25` in `main.py` is a quick run; `forecast-scenario.py` starts at `seeds=2` for a fast smoke test and is meant to be bumped for real sweeps.
- `n_jobs=4, n_grids=4` - 4 outer seed workers × 4 inner CV workers. Clamped by `BaseModel.validate()` if the box has fewer cores.

#### `fm.predict(...)`
- `window_step=10, window_step_unit="minutes"` - 144 forecasts/day. The dense grid is what makes the forecast plot smooth.
- `save_seed_result=True` writes one CSV per seed under `prediction/results/{clf-slug}/` for downstream uncertainty analysis.
- `use_cache=False` forces a fresh forecast - flip to `True` to short-circuit when nothing upstream changed.
- The cache identity threads in `_training_cache_hash`, so re-training automatically invalidates downstream predictions.

#### `fm.evaluate(model="prediction", plot_per_seed=True)`
- Reuses the in-memory `PredictionModel` - no re-extraction of features, no re-fit.
- Falls back to `train()`'s `eruption_dates` when called without an explicit list.
- Writes `(n_samples, n_seeds)` `y_proba.csv` / `y_pred.csv` matrices under `evaluation/prediction/classifiers/{Clf}/predictions/` (no per-seed JSON; per-seed metric tables live in memory on `self.metrics`).
- `plot_per_seed=True` is expensive - flip off for fast iteration.
- `use_cache=True` (default) consults `{evaluation_dir}/{hash}.EvaluationModel.pkl` before the per-classifier `predict_proba` pass; pass `use_cache=False` to force a fresh evaluation.

#### `fm.EvaluationModel.compare()`
- Reuses the cached `MetricsEnsemble` from `evaluate()` and hands it to `ClassifierComparator`.
- `comparator.get_ranking()` writes `comparison/metrics/ranking_recall.csv` (defaults to recall ranking - matches the training `scoring`).
- `comparator.plot_all()` writes ROC overlay, metric bars, seed stability violins, and a comparison grid under `evaluation/prediction/comparison/figures/`.

#### `fm.explain(model="prediction", save_per_seed=True, plot_per_seed=False, max_display=20, dpi=150)`
- Called at the end of `main.py` after `evaluate(...)`; produces per-seed SHAP bar + beeswarm plots and per-eruption waterfall plots.
- Restricted to tree classifiers (RF / `lite-rf` / GB / XGB). Non-tree classifiers in the ensemble are skipped with a warning.
- `eruption_dates` is passed explicitly in `main.py` (the same eight-eruption list used by `train(...)`); when omitted it falls back to `train()`'s dates just like `evaluate(...)`.
- `save_per_seed=True` writes per-seed `shap.Explanation` pickles alongside the bundled `ClassifierExplanation_*.pkl`; `plot_per_seed=False` skips per-seed PNG rendering (aggregate bar + beeswarm still run).
- `use_cache=True` (default) consults `{explanation_dir}/{hash}.ExplanationModel.pkl` before re-running SHAP; pass `use_cache=False` to force a fresh explanation.
- Output lands under `explanation/prediction/` - see [Explanation Workflow](Explanation-Workflow) for the full tree.

---

## Scenarios Workflow (`forecast-scenario.py`)

`forecast-scenario.py` drives the packaged multi-scenario orchestrator `ForecastModelScenario` (`src/eruption_forecast/model/forecast_model_scenario.py`). The user supplies a list of `Scenario` `TypedDict`s that vary the training and prediction windows, then calls `shared_training(...)` / `shared_prediction(...)` / `shared_evaluation(...)` / `shared_explanation(...)` once to capture the kwargs shared across every scenario. `run()` validates the setup, writes a `scenarios.json` manifest, and drives the loop — one `train → predict → evaluate → explain` pass per scenario, per-scenario artefacts namespaced under `{station_dir}/scenarios/{slug}/`.

The class wraps **one** `ForecastModel` instance. `calculate()` runs **once** before the loop; the resulting tremor frame is captured on `fm.tremor_df` and reused on every iteration. Per-scenario artefacts land under `output/{nslc}/scenarios/{slug}/` alongside a top-level `scenarios/scenarios.json`.

### Stage flow

```
┌────────────────────────────────────────────────────────────────────────────┐
│              forecast-scenario.py - ForecastModelScenario Flow             │
└────────────────────────────────────────────────────────────────────────────┘

  fms = ForecastModelScenario(
            network="VG", station="OJN", location="00", channel="EHZ",
            eruption_dates=[...], scenarios=[...],
            day_to_forecast=2, n_jobs=8, verbose=True)
       │
       ▼
┌─────────────────────┐
│  fms.calculate()    │  ForecastModel.calculate  →  tremor_*.csv
│                     │  runs ONCE, reused by every scenario below
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ fms.shared_training │  captures kwargs → self.shared_training_params
│   (window_step,     │  merged with per-scenario start/end/output_dir at run()
│    classifiers,     │
│    seeds, ...)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ fms.shared_predict- │  captures kwargs → self.shared_prediction_params
│   ion (threshold,   │  use_features_from="files" enables scenario-1 reuse
│    use_features_    │  of features matrix + labels CSV
│    from, ...)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ fms.shared_evaluat- │  required when evaluate_model=True (default)
│   ion (...)         │  run() raises RuntimeError otherwise; skip iff
│                     │  the sweep also passes evaluate_model=False
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ fms.shared_explan-  │  required when explain_model=True (default)
│   ation (...)       │  run() raises RuntimeError otherwise; skip iff
│                     │  the sweep also passes explain_model=False
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  fms.run()          │  validate() → _save_scenarios() → per-scenario loop:
│                     │
│                     │      for scenario in self.scenarios:
│                     │          fm.train(**shared_training_params,       ┐
│                     │                   output_dir=…/{slug}, start,end) │
│                     │          fm.predict(**shared_prediction_params,   │
│                     │                     **plot_kwargs, start, end)   │─┤ inner
│                     │          TelegramNotification.send_document(     │  │ loop
│                     │              forecast_plot_path, caption=name)   │  │ (per
│                     │          fm.evaluate(**shared_evaluation_params) │  │ scenario)
│                     │          fm.explain(**shared_explanation_params) ┘
│                     │
└─────────────────────┘
```

### Per-stage notes

Each `fms.*` method below is a thin wrapper over its counterpart on the wrapped `ForecastModel` (`src/eruption_forecast/model/forecast_model.py`). The wrapper signature is **the same** as the wrapped method's — same argument names, same types, same defaults — except for the per-scenario / already-known knobs the wrapper injects on your behalf, which are explicitly listed under each entry.

#### `ForecastModelScenario(...)` constructor
- Threads `network`, `station`, `location`, `channel`, `day_to_forecast`, `n_jobs`, `output_dir`, and `verbose` into a wrapped `ForecastModel`. The wrapped instance lives on `self.ForecastModel`.
- `scenarios_dir` (default `"scenarios"`) names the child directory under `{station_dir}/` that holds per-scenario artefacts and the `scenarios.json` manifest.
- `eruption_dates` is stored once on the instance and threaded into every `shared_*` call so scenario dicts do not have to repeat it.

#### `fms.calculate(...)` &nbsp;— mirrors [`ForecastModel.calculate(...)`](https://github.com/martanto/eruption-forecast/blob/master/src/eruption_forecast/model/forecast_model.py#L182)
- **Same arguments** as `ForecastModel.calculate(...)`; kwargs are forwarded verbatim.
- **Injected by the wrapper (do not pass):** `plot_eruption_dates` — auto-filled from `self.eruption_dates` so daily band plots carry markers.
- **Wrapper fallbacks:** `n_jobs=None` and `verbose=None` fall back to `self.n_jobs` / `self.verbose` before the call.
- Runs before any `shared_*` call so that `validate()` inside `run()` can check `ForecastModel.CalculateTremor is not None`.

#### `fms.shared_training(...)` &nbsp;— mirrors [`ForecastModel.train(...)`](https://github.com/martanto/eruption-forecast/blob/master/src/eruption_forecast/model/forecast_model.py#L399)
- **Same arguments** as `ForecastModel.train(...)`; captured into `self.shared_training_params` and splatted into `fm.train(**shared_training_params)` per scenario inside `run()`.
- **Injected per scenario by `run()` (do not pass):** `start_date`, `end_date`, `output_dir` — supplied from the current `Scenario` dict / `scenarios_dir/{slug}`.
- **Injected once by the wrapper (do not pass):** `eruption_dates` — filled from `self.eruption_dates`.
- **Wrapper fallbacks:** `overwrite=None` → `self.overwrite`; `verbose=None` → `self.verbose`.
- **Always required** — `validate()` raises `RuntimeError("Please run \`shared_training(...)\` first.")` if left uncalled. Every sweep must train.

#### `fms.shared_prediction(...)` &nbsp;— mirrors [`ForecastModel.predict(...)`](https://github.com/martanto/eruption-forecast/blob/master/src/eruption_forecast/model/forecast_model.py#L690)
- **Same arguments** as `ForecastModel.predict(...)`; captured into `self.shared_prediction_params` and splatted into `fm.predict(**shared_prediction_params, **plot_kwargs)` per scenario inside `run()`.
- **Injected per scenario by `run()` (do not pass):** `start_date`, `end_date`, `window_step`, `window_step_unit`, `output_dir` — supplied from the current `Scenario` dict.
- **Managed by `run()`, not by the caller:** `features_matrix_path`, `label_features_csv` — captured from scenario 1's `fm.PredictionModel` when `use_features_from="files"` and threaded into scenarios 2..N (see the coverage constraint below).
- **Wrapper fallbacks:** `overwrite=None` → `self.overwrite`; `verbose=None` → `self.verbose`.
- **Always required** — `validate()` raises `RuntimeError("Please run \`shared_prediction(...)\` first.")` if left uncalled. Every sweep must predict.

#### `fms.shared_evaluation(...)` &nbsp;— mirrors [`ForecastModel.evaluate(...)`](https://github.com/martanto/eruption-forecast/blob/master/src/eruption_forecast/model/forecast_model.py#L997)
- **Same arguments** as `ForecastModel.evaluate(...)`; captured into `self.shared_evaluation_params` and splatted into `fm.evaluate(**shared_evaluation_params)` per scenario inside `run()`.
- **Injected per scenario by `run()` (do not pass):** `output_dir`.
- **Injected once by the wrapper (do not pass):** `eruption_dates` — filled from `self.eruption_dates`.
- **Wrapper fallbacks:** `overwrite=None` → `self.overwrite`; `verbose=None` → `self.verbose`.
- **Conditionally required** — `run(evaluate_model=True)` (the default) raises `RuntimeError("Please run \`shared_evaluation(...)\` first, or pass \`evaluate_model=False\` to \`run(...)\`.")` if the params dict is empty. `run(evaluate_model=False)` waives the requirement and skips the evaluate step for that sweep.

#### `fms.shared_explanation(...)` &nbsp;— mirrors [`ForecastModel.explain(...)`](https://github.com/martanto/eruption-forecast/blob/master/src/eruption_forecast/model/forecast_model.py#L1121)
- **Same arguments** as `ForecastModel.explain(...)`; captured into `self.shared_explanation_params` and splatted into `fm.explain(**shared_explanation_params)` per scenario inside `run()`.
- **Injected per scenario by `run()` (do not pass):** `output_dir`.
- **Injected once by the wrapper (do not pass):** `eruption_dates` — filled from `self.eruption_dates`.
- **Wrapper fallbacks:** `overwrite=None` → `self.overwrite`; `verbose=None` → `self.verbose`.
- **Conditionally required** — `run(explain_model=True)` (the default) raises `RuntimeError("Please run \`shared_explanation(...)\` first, or pass \`explain_model=False\` to \`run(...)\`.")` if the params dict is empty. `run(explain_model=False)` waives the requirement and skips the explain step for that sweep.

#### `fms.run(evaluate_model=True, explain_model=True, notification_message=None)`
- Runs the `evaluate_model` / `explain_model` conditional preconditions first (raises `RuntimeError` if the corresponding `shared_*` params dict is empty when the flag is `True`), then calls `validate()` (unique slugs, `calculate()` ran, `shared_training` + `shared_prediction` populated), then writes `scenarios.json`.
- Per scenario: builds `output_dir = scenarios_dir/{slugify(name)}`, calls `fm.train`, `fm.predict`, ships the forecast PNG to Telegram via `TelegramNotification.send_document(...)`, then optionally `fm.evaluate` and `fm.explain`.
- **`use_features_from="files"` first-scenario coercion.** The loop captures the caller's original `use_features_from` value up-front, then force-sets the dict entry to `"all"` for `index == 0` so scenario 1 always extracts a fresh features matrix. When the caller *asked* for `"files"`, the loop captures scenario 1's `features_matrix_path` / `label_features_csv` on `fm.PredictionModel` and threads them into every later `fm.predict(...)` call - so scenarios 2..N never re-extract, they just read the scenario-1 parquet + label CSV.
- **Coverage constraint (important).** Because scenarios 2..N reuse scenario 1's features matrix, **scenario 1's prediction window must span every datetime any later scenario predicts over**. If a later scenario predicts over datetimes scenario 1 did not cover, the reused matrix will be missing those rows and the run will fail. Order scenarios so the widest prediction window comes first, or leave `use_features_from="all"` and let every scenario re-extract. `forecast-scenario.py` demonstrates the safe ordering: scenario 1 predicts `2025-04-01 → 2025-08-22` and scenarios 2-5 predict progressively shorter windows all contained within it.
- **Telegram caption.** `notification_message=None` (default) uses `socket.gethostname()` upper-cased as the caption prefix; passing an explicit string overrides the hostname for every scenario in the sweep.

### Resulting directory layout

```
output/
└── VG.OJN.00.EHZ/
    ├── tremor/                            # shared across all scenarios
    │   └── VG.OJN.00.EHZ_2025-01-01_2025-12-31.csv
    └── scenarios/
        ├── scenarios.json                 # ForecastModelScenario manifest (all scenario dicts)
        ├── scenario-1-2/
        │   ├── training/
        │   ├── prediction/
        │   ├── evaluation/prediction/
        │   └── explanation/prediction/
        ├── scenario-2-2/
        ...
        └── scenario-5-2/
```

See [Output Structure](Output-Structure#scenarios-layout) for the full per-scenario tree.

---

## Choosing Between the Workflows

| | Research (`main.py`) | Scenarios (`forecast-scenario.py`) |
|--|----------------------|-------------------------------------|
| Number of forecasts | 1 | N (packaged; 5 in the bundled script) |
| Tremor computation | once | once (reused) |
| Training | once | once per scenario |
| Output isolation | flat | one directory per scenario + `scenarios.json` manifest |
| Feature-matrix reuse | n/a | `use_features_from="files"` reuses scenario-1 matrix in scenarios 2..N |
| Telegram notifications | per-stage via `@notify` decorator | per-scenario forecast plot via `TelegramNotification().send_document(...)` (built into `run()`) |
| Use case | a single research run on a fixed split | leave-one-out style sweeps, ablation over training windows |

When the goal is to publish a single forecast plot on a fixed date split, use `main.py`. When you want to compare how forecast quality degrades as the training-window shifts (leave-one-out ablations, feature-matrix reuse, per-scenario artefacts, packaged Telegram hook), use `forecast-scenario.py`.
