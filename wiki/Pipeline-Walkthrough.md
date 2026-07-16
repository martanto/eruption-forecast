# Pipeline Walkthrough

This page is the annotated tour of the two bundled entry-point scripts:

- **[Research Workflow](#research-workflow-mainpy)** - `main.py` - one-shot calculate → train → predict → evaluate → explain on a fixed date split.
- **[Scenarios Workflow](#scenarios-workflow-scenariospy)** - `scenarios.py` - same pipeline executed once per scenario in a nine-scenario sweep.

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
- `seeds=25` in `main.py` is a quick run; the default in `scenarios.py` is `seeds=10` per scenario.
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

## Scenarios Workflow (`scenarios.py`)

`scenarios.py` keeps the same `ForecastModel` instance and the same tremor frame, but loops the `train → predict → evaluate` legs over nine scenarios that vary the training and prediction date splits. Each scenario's outputs land in its own directory so they can be compared side-by-side.

### Scenarios in `scenarios.py`

| # | Train window | Forecast window | Goal |
|---|--------------|-----------------|------|
| 1 | 2025-01-01 → 2025-03-31 | 2025-04-01 → 2025-04-30 | Train on 1 eruption, forecast eruption 2 |
| 2 | 2025-01-01 → 2025-03-31 | 2025-05-01 → 2025-05-31 | Train on 1 eruption, forecast eruption 3 |
| 3 | 2025-01-01 → 2025-03-31 | 2025-06-01 → 2025-06-30 | Train on 1 eruption, forecast eruption 4 |
| 5 | 2025-01-01 → 2025-04-30 | 2025-05-01 → 2025-05-31 | Train on 1+2, forecast eruption 3 |
| 6 | 2025-01-01 → 2025-05-31 | 2025-06-01 → 2025-06-30 | Train on 1+2+3, forecast eruption 4 |
| 7 | 2025-01-01 → 2025-06-30 | 2025-07-01 → 2025-07-13 | Train on 1–4, forecast eruption 5 |
| 8 | 2025-01-01 → 2025-07-26 | 2025-07-27 → 2025-08-22 | Train on 1–5, forecast eruptions 6+7 |
| 9 | 2025-01-01 → 2025-08-22 | 2025-01-01 → 2025-08-22 | Sanity check - train and predict over the full record |

(Scenario 4 is intentionally absent in `scenarios.py`.)

### Stage flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     scenarios.py - Outer-Loop Flow                       │
└──────────────────────────────────────────────────────────────────────────┘

  fm = ForecastModel(...)                  # one ForecastModel reused
       │
       ▼
┌─────────────────────┐
│  fm.calculate()     │  one full-year tremor CSV (shared across scenarios)
└──────────┬──────────┘
           │
           ▼
   for scenario in scenarios:              # nine scenarios in scenarios.py
           │
           │  output_dir = os.path.join(
           │      root_dir, "output", fm.nslc,
           │      "scenarios", slugify(name))
           │
           ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │  (scoped to scenario.output_dir)                                  │
   │                                                                   │
   │  fm.train(start, end, eruption_dates, …, output_dir=output_dir)   │
   │       │                                                           │
   │       ▼                                                           │
   │  fm.predict(start, end, …, output_dir=output_dir,                 │
   │             eruption_dates=plot_kwargs["eruption_dates"])         │
   │       │                                                           │
   │       ▼                                                           │
   │  tn = TelegramNotification(verbose=False)                        │
   │  tn.send_message(message=f"{name}: {description}") \              │
   │    .send_document(                                                │
   │      file=fm.PredictionModel.forecast_plot_path,                  │
   │      caption=f"{name}: {description}")                            │
   │       │                                                           │
   │       ▼                                                           │
   │  fm.evaluate(model="prediction", plot_per_seed=True,              │
   │              output_dir=output_dir)                               │
   │       │                                                           │
   │       ▼                                                           │
   │  fm.explain(model="prediction",                                   │
   │             eruption_dates=eruption_dates,                        │
   │             save_per_seed=True, plot_per_seed=False,              │
   │             max_display=20)                                       │
   └───────────────────────────────────────────────────────────────────┘
           │
           ▼
   next scenario  →  loop back
```

### Per-scenario notes

- `output_dir` is built once per scenario from `slugify(name)` (`"Scenario 1"` → `"scenario-1"`). Every stage call inside the loop forwards that `output_dir`, so artefacts land at `output/{nslc}/scenarios/{slug}/`.
- The outer `fm.calculate(...)` runs **once** before the loop. Its tremor frame is captured on `fm.tremor_df` and reused on every iteration - `fm.train()` reads from `fm.tremor_df`, not from disk.
- `eruption_dates` is the **full** list of eight known eruptions on every scenario. The training window simply excludes the eruptions that haven't happened yet, and the prediction window picks them up later.
- `plot_kwargs["eruption_dates"]` is forwarded into `fm.predict(...)` as `**plot_kwargs`, which routes them to `forecast_plots` so eruption-day markers are drawn on the per-scenario forecast plot.
- The Telegram hook (`TelegramNotification.send_message(...).send_document(...)`) ships the per-scenario forecast PNG to the configured chat the moment `predict()` returns. Using `send_document(...)` (rather than `send_photo(...)`) forces the file to be uploaded as a document — Telegram never re-encodes it, so the full DPI plot is preserved.
- `fm.evaluate(...)` runs **after** the notification, so by the time you see the plot in Telegram, the per-seed `y_proba` / `y_pred` matrices and aggregate metric plots are already being written in the background.
- `fm.explain(...)` runs last in each scenario - per-seed SHAP explanations are bundled into `ClassifierExplanation_*.pkl` under `explanation/prediction/classifiers/` and per-eruption waterfall plots land under `explanation/prediction/eruptions/`. See [Explanation Workflow](Explanation-Workflow) for the full output tree.

### Resulting directory layout

```
output/
└── VG.OJN.00.EHZ/
    ├── tremor/                            # shared across all scenarios
    │   └── VG.OJN.00.EHZ_2025-01-01_2025-12-31.csv
    └── scenarios/
        ├── scenario-1/
        │   ├── training/
        │   ├── prediction/
        │   ├── evaluation/prediction/
        │   ├── explanation/prediction/
        │   └── cache/
        ├── scenario-2/
        ...
        └── scenario-9/
```

See [Output Structure](Output-Structure#scenarios-layout) for the full per-scenario tree.

---

## Choosing Between the Workflows

| | Research (`main.py`) | Scenarios (`scenarios.py`) |
|--|----------------------|-----------------------------|
| Number of forecasts | 1 | N (9 in the bundled script) |
| Tremor computation | once | once (reused) |
| Training | once | once per scenario |
| Output isolation | flat | one directory per scenario |
| Telegram notifications | per-stage via `@notify` decorator | per-scenario forecast plot via `TelegramNotification().send_message(...).send_document(...)` |
| Use case | a single research run on a fixed split | leave-one-out style sweeps, ablation over training windows |

When the goal is to publish a forecast plot, use `main.py`. When the goal is to compare how forecast quality degrades as the training-window shifts, use `scenarios.py`.
