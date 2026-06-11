# Output Structure

Every pipeline run writes under a single station directory:

```
{output_dir}/{network}.{station}.{location}.{channel}/      ← station_dir
            └─ "VG.OJN.00.EHZ"                              ← nslc
```

`output_dir` defaults to `os.getcwd()` when not provided; pass `root_dir=...` to anchor relative paths under a chosen project root.

---

## Full Directory Tree

```
{station_dir}/
│
├── tremor/                                                   # CalculateTremor
│   ├── daily/                                                # Per-day CSVs (removed when cleanup_daily_dir=True)
│   ├── figures/                                              # Per-day band plots (plot_daily=True)
│   └── {nslc}_{start}_{end}.csv                              # Merged tremor CSV (DateTime index)
│
├── training/                                                 # TrainingModel
│   ├── features/{cv-slug}/
│   │   ├── features-matrix_{basename}.csv                    # Full tsfresh feature matrix
│   │   ├── features-label_{basename}.csv                     # Aligned binary labels
│   │   ├── seed/{seed:05d}.csv                               # Top-N features per seed
│   │   ├── seed/figures/{seed:05d}.png                       # Per-seed importance plots (plot_features=True)
│   │   ├── resampled/{seed:05d}.csv                          # Resampled training set per seed
│   │   ├── top_{N}_features.csv                              # Aggregated top-N across all seeds
│   │   └── top_{N}_features.png                              # Aggregated importance plot
│   │
│   └── classifiers/
│       ├── ClassifierEnsemble_{cv-slug}.pkl                  # Bundled ClassifierEnsemble (all classifiers)
│       ├── ClassifierEnsemble_{cv-slug}.json                 # Registry of per-classifier paths
│       └── {clf-slug}/{cv-slug}/
│           ├── models/{seed:05d}.pkl                         # One best_estimator_ per seed
│           ├── trained-model__{suffix}.json                  # Per-classifier trained-model registry (records with inline top-N features)
│           └── SeedEnsemble_{suffix}.pkl                     # Single-classifier SeedEnsemble
│
├── prediction/                                               # PredictionModel
│   ├── features/
│   │   ├── features-label_{basename}_step-{N}-{unit}.csv     # Forecast window grid
│   │   └── features-matrix_*.csv                             # tsfresh matrix for the grid
│   ├── results/{clf-slug}/{seed:05d}.csv                     # Per-seed probability (save_seed_result=True)
│   └── figures/forecast_{basename}.{png,pdf}                 # Forecast plot
│
├── evaluation/                                               # EvaluationModel
│   ├── training/                                             # When model.kind == "training"
│   │   ├── classifiers/{ClassifierName}/
│   │   │   ├── predictions/{y_proba,y_pred,y_true}.csv
│   │   │   ├── metrics/json/{seed:05d}.json
│   │   │   ├── metrics_summary_{start}_{end}.csv
│   │   │   ├── all_metrics_{start}_{end}.csv
│   │   │   └── figures/                                      # plot_aggregate=True
│   │   └── comparison/                                       # em.compare()
│   │       ├── metrics/ranking_*.csv
│   │       └── figures/
│   └── prediction/                                           # When model.kind == "prediction"
│       └── (identical sub-tree)
│
├── cache/                                                    # CacheModel
│   ├── TrainingModel/{hash}.pkl                              # Cached fitted TrainingModel
│   ├── TrainingModel/{hash}.params.json                      # Sidecar identity dump
│   ├── PredictionModel/{hash}.pkl                            # Cached PredictionModel
│   └── PredictionModel/{hash}.params.json
│
├── forecast.config.yaml                                      # fm.save_config()
├── training.config.yaml                                      # tm.save_config()  (standalone)
├── result_all_model_predictions_{basename}.csv               # PredictionModel.forecast() top-level dump
├── TrainingModel_{basename}.pkl                              # Optional, via fm.TrainingModel.save()
├── PredictionModel_{basename}.pkl                            # Optional, via fm.PredictionModel.save()
└── EvaluationModel_{basename}.pkl                            # Optional, via fm.EvaluationModel.save()
```

Where `basename` is typically `{start_date}_{end_date}` (training) or `{start_date}_{end_date}_ws-{window_size}` (prediction).

---

## Slug Mappings

Folder slugs come from `ClassifierModel.slug_name` and `ClassifierModel.slug_cv_name`:

| Classifier key | Folder slug |
|----------------|-------------|
| `rf` | `random-forest-classifier` |
| `lite-rf` | `lite-random-forest-classifier` |
| `gb` | `gradient-boosting-classifier` |
| `xgb` | `xgb-classifier` |
| `svm` | `svc` |
| `lr` | `logistic-regression` |
| `nn` | `mlp-classifier` |
| `dt` | `decision-tree-classifier` |
| `knn` | `k-neighbors-classifier` |
| `nb` | `gaussian-nb` |
| `voting` | `voting-classifier` |

| CV strategy | Folder slug |
|-------------|-------------|
| `shuffle` | `shuffle-split` |
| `stratified` | `stratified-k-fold` |
| `shuffle-stratified` | `stratified-shuffle-split` |
| `timeseries` *(direct `ClassifierModel` only)* | `time-series-split` |

Inside `evaluation/`, the per-classifier folder uses the **unslugified** sklearn class name (`RandomForestClassifier`) - separate from training's slug (`random-forest-classifier`).

---

## Filename Conventions

The trained-model registry JSON and ensemble pickle share a single suffix scheme:

```
trained-model__{ClassifierName}_{CVName}_seeds-{N}_features-{K}.json
SeedEnsemble_{ClassifierName}_{CVName}_seeds-{N}_features-{K}.pkl
```

Example:

```
trained-model__RandomForestClassifier_StratifiedShuffleSplit_seeds-25_features-20.json
SeedEnsemble_RandomForestClassifier_StratifiedShuffleSplit_seeds-25_features-20.pkl
```

Each `trained-model__*.json` is a list of per-seed records:

```json
[
  {"random_state": 0, "features": ["f_0", "f_1", "..."], "model_filepath": ".../models/00000.pkl"},
  {"random_state": 1, "features": ["..."], "model_filepath": ".../models/00001.pkl"}
]
```

`SeedEnsemble.from_json` (or `SeedEnsemble.from_any`, which dispatches on extension) reads this file as the single source of truth for the seed bundle. The legacy `.csv` registry remains loadable via `SeedEnsemble.from_registry` so older training outputs still work.

The `ClassifierEnsemble` is named with the CV slug only (one ensemble holds every classifier):

```
ClassifierEnsemble_stratified-shuffle-split.pkl
ClassifierEnsemble_stratified-shuffle-split.json
```

Per-seed model files inside `classifiers/{clf}/{cv}/models/` are zero-padded:

```
00000.pkl  00001.pkl  ...  00024.pkl
```

---

## Cache Layout

`CacheModel` writes content-addressed artefacts under `{station_dir}/cache/{ClassName}/`:

```
cache/
├── TrainingModel/
│   ├── 3b7a98e6...c2.pkl              # joblib-pickled fitted TrainingModel
│   └── 3b7a98e6...c2.params.json      # canonical identity dict (diff-friendly)
└── PredictionModel/
    ├── 9c12d04f...88.pkl
    └── 9c12d04f...88.params.json
```

The `.params.json` is what was hashed to produce the filename. When `use_cache=True` 
and the next run computes the same identity, the `.pkl` is loaded instead of recomputed.

`fm.train(..., use_cache=True)` (default) and `fm.predict(..., use_cache=True)` 
(default) use the cache; flip to `False` to force a clean run.

---

## Scenarios Layout

`scenarios.py` passes a per-scenario `output_dir` into each stage, so artefacts land at:

```
output/
└── {nslc}/
    ├── tremor/                                # produced ONCE outside the loop, shared
    └── scenarios/
        ├── scenario-1/
        │   ├── training/...
        │   ├── prediction/...
        │   ├── evaluation/prediction/...
        │   ├── cache/...
        │   ├── forecast.config.yaml
        │   └── result_all_model_predictions_*.csv
        ├── scenario-2/
        ...
        └── scenario-9/
```

Each scenario directory mirrors a full `{station_dir}` sub-tree, just rooted at 
`output/{nslc}/scenarios/{slug}/` instead of `output/{nslc}/`. 
Slugify is from `utils/formatting.py:slugify`: `"Scenario 1"` → `scenario-1`.

The shared `tremor/` at the top means re-running scenarios never recomputes 
tremor - only the train/predict/evaluate legs are repeated.

---

## What Lives Where - Cheat Sheet

| You want to inspect... | Look here |
|------------------------|-----------|
| The merged tremor CSV | `tremor/{nslc}_{start}_{end}.csv` |
| Per-day tremor plots | `tremor/figures/` |
| The features tsfresh extracted | `training/features/{cv}/features-matrix_*.csv` |
| Per-seed feature picks | `training/features/{cv}/seed/{seed:05d}.csv` |
| The aggregated top-N features | `training/features/{cv}/top_{N}_features.csv` |
| Individual trained models | `training/classifiers/{clf}/{cv}/models/{seed:05d}.pkl` |
| The single-classifier ensemble | `training/classifiers/{clf}/{cv}/SeedEnsemble_*.pkl` |
| The all-classifiers ensemble | `training/classifiers/ClassifierEnsemble_{cv}.pkl` |
| Forecast grid + features | `prediction/features/` |
| Per-seed forecast probabilities | `prediction/results/{clf}/{seed:05d}.csv` |
| Combined forecast CSV (consensus + per-classifier) | `result_all_model_predictions_{basename}.csv` |
| Forecast PNG/PDF | `prediction/figures/forecast_{basename}.{png,pdf}` |
| Per-seed metrics (training mode) | `evaluation/training/classifiers/{Clf}/metrics/json/{seed:05d}.json` |
| Per-seed metrics (prediction mode) | `evaluation/prediction/classifiers/{Clf}/metrics/json/{seed:05d}.json` |
| Aggregate metrics CSV | `evaluation/{kind}/classifiers/{Clf}/metrics_summary_*.csv` |
| Comparison ranking CSV | `evaluation/{kind}/comparison/metrics/ranking_*.csv` |
| Cache identity (diff-friendly) | `cache/{Stage}/{hash}.params.json` |
| Replayable pipeline config | `forecast.config.yaml` |
