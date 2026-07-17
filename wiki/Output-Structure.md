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
│   ├── training.config.yaml                                  # tm.save_config() — auto-written at end of fit()
│   ├── features/{cv-slug}/
│   │   ├── features-matrix_{basename}.parquet                # Full tsfresh feature matrix (Snappy Parquet)
│   │   ├── features-label_{basename}.csv                     # Aligned binary labels
│   │   ├── seed/{seed:05d}.csv                               # Top-N features per seed
│   │   ├── seed/figures/{seed:05d}.png                       # Per-seed importance plots (plot_features=True)
│   │   ├── resampled/{seed:05d}.csv                          # Per-seed (id + is_erupted) — features recovered via features_df.loc[ids]
│   │   ├── significant_features.csv                          # Raw per-seed rows concatenated (features + score)
│   │   ├── top_features.csv                                  # Full ranked list (all features, sorted by score desc, mean_score asc)
│   │   ├── top_{N}_features.csv                              # Top-N subset of top_features.csv
│   │   ├── top_{N}_features.png                              # Aggregated importance plot
│   │   └── sweep/{mode}/{classifier-name}/                   # ⚠ Experimental — FeatureCountSweep outputs (sweep_feature_count)
│   │       ├── cv_scores.csv                                 # Aggregated summary (N, mean, std, n_seeds)
│   │       ├── cv_scores_raw.csv                             # Full (N × seed) score matrix
│   │       ├── seed_argmax_hist.csv                          # Per-seed argmax N
│   │       ├── support.json                                  # {"n_features": N*, "seeds": {seed: [features…]}}
│   │       ├── curve.png                                     # mean±std curve + argmax histogram
│   │       └── FeatureCountSweep.pkl                         # Full sweep instance (FeatureCountSweep.load(path))
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
│   ├── prediction.config.yaml                                # pm.save_config() — auto-written at end of forecast()
│   ├── features/
│   │   ├── features-label_{basename}_step-{N}-{unit}.csv     # Forecast window grid
│   │   └── features-matrix_*.parquet                         # tsfresh matrix for the grid (Snappy Parquet)
│   ├── results/{clf-slug}/{seed:05d}.csv                     # Per-seed probability (save_seed_result=True)
│   └── figures/forecast_{basename}.{png,pdf}                 # Forecast plot
│
├── evaluation/                                               # EvaluationModel
│   ├── training/                                             # When model.kind == "training"
│   │   ├── evaluation.config.yaml                            # em.save_config() — auto-written at end of evaluate()
│   │   ├── classifiers/{ClassifierName}/
│   │   │   ├── predictions/
│   │   │   │   ├── y_proba.csv                               # (n_samples, n_seeds) matrix
│   │   │   │   └── y_pred.csv                                # (n_samples, n_seeds) matrix
│   │   │   └── figures/
│   │   │       ├── aggregate/{plot_name}.{png,csv}           # plot_aggregate=True
│   │   │       └── {plot_name}/{seed:05d}.png                # plot_per_seed=True
│   │   ├── comparison/                                       # em.compare()
│   │   │   ├── metrics/ranking_*.csv
│   │   │   └── figures/
│   │   └── MetricsEnsemble.pkl                               # Optional, via me.save()
│   └── prediction/                                           # When model.kind == "prediction"
│       ├── evaluation.config.yaml                            # em.save_config() — auto-written at end of evaluate()
│       ├── labels/y_true.csv                                 # Built by EvaluationModel.build_label()
│       └── classifiers/{ClassifierName}/…                    # Same shape as training/
│
├── explanation/                                              # ExplanationModel
│   ├── training/                                             # When upstream model.kind == "training"
│   │   ├── explanation.config.yaml                           # xm.save_config() — auto-written at end of explain()
│   │   ├── classifiers/{ClassifierName}/
│   │   │   ├── ClassifierExplanation_{ClassifierName}.pkl    # Bundled SHAP payload
│   │   │   ├── shap_values/{seed:05d}.pkl                    # Per-seed shap.Explanation (save_per_seed=True)
│   │   │   └── figures/
│   │   │       ├── bar/{seed:05d}.png                        # plot_per_seed=True
│   │   │       ├── beeswarm/{seed:05d}.png                   # plot_per_seed=True
│   │   │       └── aggregate/                                # plot_aggregate=True
│   │   │           ├── bar.{png,csv}                         # frequency-weighted importance
│   │   │           └── beeswarm.{png,csv}                    # NaN-padded union beeswarm
│   │   └── eruptions/{YYYY-MM-DD}/                           # Per-eruption waterfall sibling
│   │       └── {ClassifierName}_{datetime}_seed={i}_index={j}.png
│   └── prediction/                                           # When upstream model.kind == "prediction"
│       ├── explanation.config.yaml                           # xm.save_config() — auto-written at end of explain()
│       └── (identical sub-tree)
│
│   # Cache pickles for the three cache-using stages live next to each
│   # stage's other outputs — no separate cache/ subtree:
│   #   training/{hash}.TrainingModel.pkl                     # Cached fitted TrainingModel
│   #   training/{hash}.TrainingModel.params.json             # Sidecar identity dump
│   #   prediction/{hash}.PredictionModel.pkl                 # Cached PredictionModel
│   #   prediction/{hash}.PredictionModel.params.json
│   #   explanation/{kind}/{hash}.ExplanationModel.pkl        # Cached ExplanationModel
│   #   explanation/{kind}/{hash}.ExplanationModel.params.json
│
├── forecast.config.yaml                                      # fm.save_config()
├── forecast-results_{basename}.csv               # PredictionModel.forecast() top-level dump
├── TrainingModel_{basename}.pkl                              # Optional, via fm.TrainingModel.save()
├── PredictionModel_{basename}.pkl                            # Optional, via fm.PredictionModel.save()
├── EvaluationModel_{basename}.pkl                            # Optional, via fm.EvaluationModel.save()
└── ExplanationModel_{basename}.pkl                           # Optional, via fm.ExplanationModel.save()
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

Inside `evaluation/` and `explanation/`, the per-classifier folder uses the **unslugified** sklearn class name (`RandomForestClassifier`) - separate from training's slug (`random-forest-classifier`).

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

`BaseModel.save(identity)` writes content-addressed artefacts directly into each stage's own directory — no central `cache/` subtree:

```
training/
├── 3b7a98e6...c2.TrainingModel.pkl                 # joblib-pickled fitted TrainingModel
└── 3b7a98e6...c2.TrainingModel.params.json         # canonical identity dict (diff-friendly)

prediction/
├── 9c12d04f...88.PredictionModel.pkl
└── 9c12d04f...88.PredictionModel.params.json

explanation/{training|prediction}/
├── 4e6f2a31...77.ExplanationModel.pkl              # joblib-pickled ExplanationModel
└── 4e6f2a31...77.ExplanationModel.params.json
```

The `.params.json` is what was hashed to produce the filename. When `use_cache=True` 
and the next run computes the same identity, the `.pkl` is loaded instead of recomputed.

`fm.train(..., use_cache=True)`, `fm.predict(..., use_cache=True)`, `fm.evaluate(..., use_cache=True)`,
and `fm.explain(..., use_cache=True)` (all defaults) use the cache; flip any of them to
`False` to force a clean run of that stage. `use_cache` gates both the load and the write —
it is independent of `overwrite`, which additionally controls plot / per-classifier artefact
regeneration.

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
        │   └── forecast-results_*.csv
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
| The features tsfresh extracted | `training/features/{cv}/features-matrix_*.parquet` |
| Per-seed feature picks | `training/features/{cv}/seed/{seed:05d}.csv` |
| Raw per-seed picks concatenated | `training/features/{cv}/significant_features.csv` |
| Full ranked feature list (all features) | `training/features/{cv}/top_features.csv` |
| The aggregated top-N features | `training/features/{cv}/top_{N}_features.csv` |
| ⚠ Experimental — post-hoc sweep results | `training/features/{cv}/sweep/{mode}/{classifier-name}/` |
| Individual trained models | `training/classifiers/{clf}/{cv}/models/{seed:05d}.pkl` |
| The single-classifier ensemble | `training/classifiers/{clf}/{cv}/SeedEnsemble_*.pkl` |
| The all-classifiers ensemble | `training/classifiers/ClassifierEnsemble_{cv}.pkl` |
| Forecast grid + features | `prediction/features/` |
| Per-seed forecast probabilities | `prediction/results/{clf}/{seed:05d}.csv` |
| Combined forecast CSV (consensus + per-classifier) | `forecast-results_{basename}.csv` |
| Forecast PNG/PDF | `prediction/figures/forecast_{basename}.{png,pdf}` |
| Per-seed probability matrix | `evaluation/{kind}/classifiers/{Clf}/predictions/y_proba.csv` |
| Per-seed prediction matrix | `evaluation/{kind}/classifiers/{Clf}/predictions/y_pred.csv` |
| Aggregate metric plots + sidecar CSV | `evaluation/{kind}/classifiers/{Clf}/figures/aggregate/{plot}.{png,csv}` |
| Per-seed metric plots | `evaluation/{kind}/classifiers/{Clf}/figures/{plot}/{seed:05d}.png` |
| Comparison ranking CSV | `evaluation/{kind}/comparison/metrics/ranking_*.csv` |
| Bundled SHAP per classifier | `explanation/{kind}/classifiers/{Clf}/ClassifierExplanation_{Clf}.pkl` |
| Per-seed SHAP explanations | `explanation/{kind}/classifiers/{Clf}/shap_values/{seed:05d}.pkl` |
| Per-seed bar / beeswarm plots | `explanation/{kind}/classifiers/{Clf}/figures/{bar,beeswarm}/{seed:05d}.png` |
| Aggregate SHAP bar / beeswarm + sidecar CSV | `explanation/{kind}/classifiers/{Clf}/figures/aggregate/{bar,beeswarm}.{png,csv}` |
| Per-eruption waterfall plots | `explanation/{kind}/eruptions/{YYYY-MM-DD}/{Clf}_*.png` |
| Cache identity (diff-friendly) | `cache/{Stage}/{hash}.params.json` |
| Replayable pipeline config | `forecast.config.yaml` |
| Standalone training config | `training/training.config.yaml` |
| Standalone prediction config | `prediction/prediction.config.yaml` |
| Standalone evaluation config | `evaluation/{kind}/evaluation.config.yaml` |
| Standalone explanation config | `explanation/{kind}/explanation.config.yaml` |
