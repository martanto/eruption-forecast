# Training Workflow

The training stage turns labelled tremor windows into a `ClassifierEnsemble` - N classifiers × M random seeds - that the [Prediction Workflow](Prediction-Workflow) 
reuses for inference and the [Evaluation Workflow](Evaluation-Workflow) scores against ground truth.

Driver: `TrainingModel` (`src/eruption_forecast/model/training_model.py`). Wrapped by `ForecastModel.train(...)`.

---

## Internal Pipeline

`TrainingModel` chains three idempotent stages via method chaining:

```
                ┌──────────────────────────────────────────────┐
                │              TrainingModel                   │
                │   inherits BaseModel + CacheModel            │
                └──────────────┬───────────────────────────────┘
                               │
                  ┌────────────┼─────────────┐
                  ▼            ▼             ▼
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
        │ build_label │ →│ extract_    │ →│    fit      │
        │             │  │ features    │  │             │
        └─────────────┘  └─────────────┘  └─────────────┘
            LabelBuilder    TremorMatrix     per-seed
            or              Builder +        feature
            DynamicLabel    FeaturesBuilder  selection,
            Builder         + tsfresh        resample,
                                             GridSearchCV
                                             → SeedEnsemble
                                             → ClassifierEnsemble
```

`ForecastModel.train(...)` calls all three in one go; the standalone `TrainingModel(...)` 
lets you call them separately for fine-grained debugging.

---

## Label Building (`build_label`)

| Param | Type | Default | Purpose |
|-------|------|---------|---------|
| `window_step` | `int` | - | Stride between consecutive labelled windows |
| `window_step_unit` | `"minutes"` \| `"hours"` | - | Unit for `window_step` |
| `builder` | `"standard"` \| `"dynamic"` | `"standard"` | `LabelBuilder` vs `DynamicLabelBuilder` |
| `days_before_eruption` | `int \| None` | `None` | Required when `builder="dynamic"` |

Two builder variants live in `src/eruption_forecast/label/`:

### `LabelBuilder` - single global window

```
 start_date                                                  end_date
 │                                                              │
 ├──── window stepping at window_step ──────────────────────────┤
 │  0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1                         │
 │                        ↑                                     │
 │                  positive region                             │
 │                  starts day_to_forecast                      │
 │                  days before eruption                        │
 │                                                              │
 │  include_eruption_date=False (default)                       │
 │    → day_to_forecast + 1 positive days (dtf + eruption day)  │
 │  include_eruption_date=True                                  │
 │    → exactly day_to_forecast positive days ending on E       │
```

### `DynamicLabelBuilder` - one window per eruption, with overlap dedup

```
Phase 1: initiate
    Eruption A window         Eruption B window
    [0 0 0 0 0 0 0 0]         [0 0 0 0 0 0 0 0]

Phase 2: concat + deduplicate datetimes (sorted)
    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

Phase 3: mark positives per eruption
    [0 0 0 0 0 0 1 1 1 0 0 0 1 1 1]
                   ↑ ↑         ↑ ↑
                 Erup A      Erup B
```

`ForecastModel.train(..., label_builder="dynamic", days_before_eruption=N)` triggers the dynamic builder.

---

## Feature Pipeline (`extract_features`)

`extract_features()` chains three steps:

1. **`TremorMatrixBuilder`** slices the tremor DataFrame into label-aligned windows. Each window becomes a row block with `id`, `datetime`, and the tremor columns.
2. **`FeaturesBuilder`** runs tsfresh over the tremor matrix - one extraction per tremor column, ~700 features per column.
3. **`FeatureSelector`** is configured here (`method="tsfresh"`, `n_jobs=n_grids`) but only fires during `fit()`, per seed.

Key kwargs:

| Param | Effect |
|-------|--------|
| `select_tremor_columns=["rsam_f2", "entropy", ...]` | Restrict tsfresh input columns |
| `exclude_features=["agg_linear_trend", "length", ...]` | Drop tsfresh kinds known to be slow / collinear |
| `select_features="path/to/top_20_features.csv"` | Skip the full tsfresh sweep and only compute these features |
| `save_tremor_matrix_per_method=True` | Write one tremor-matrix CSV per column under `per_method/` |
| `minimum_completion=1.0` | Skip windows whose sample count is below this fraction of expected |

Outputs land under `{station_dir}/training/features/{cv-slug}/`:

```
training/features/{cv-slug}/
├── features-matrix_{basename}.csv      # full tsfresh feature matrix
├── features-label_{basename}.csv       # aligned labels
├── seed/{seed:05d}.csv                 # top-N features per seed
├── resampled/{seed:05d}.csv            # resampled training set per seed
└── figures/                            # per-seed importance plots (if plot_features=True)
```

---

## Classifier Fitting (`fit`)

`fit()` runs per-seed in parallel:

```
For each random_state in 0 .. seeds-1:
    1. resample              (RandomUnderSampler, RandomOverSampler, or none)
    2. FeatureSelector       (tsfresh FDR → top_n_features)
    3. for each classifier:
         grid_search_cv      (GridSearchCV over ClassifierModel.grid)
         save best_estimator to seed:05d.pkl
After the loop:
    save_model_csv  → trained-model__{Clf}_{CV}_seeds-{N}_features-{K}.csv
    build_seed_ensemble → SeedEnsemble_{suffix}.pkl
    build_classifier_ensemble → ClassifierEnsemble_{CV}.pkl
```

### Imbalance handling

| `resample_method` | Behaviour |
|-------------------|-----------|
| `"under"` | `RandomUnderSampler` always |
| `"over"` | `RandomOverSampler` always |
| `"auto"` *(default)* | `"under"` when minority share ≤ `minority_threshold` (default `0.15`), otherwise none |
| `None` | No resampling |

`sampling_strategy` (default `0.75`) is the resulting minority-to-majority ratio after resampling.

### Parallelism

```
n_jobs (outer)   = # of seed workers running in parallel
n_grids (inner)  = # of GridSearchCV / FeatureSelector workers per seed

constraint: n_jobs × n_grids ≤ total_cpu
            BaseModel.validate() clamps n_grids when violated.
            When both default to 1, n_grids is bumped to total_cpu - 2.
```

`joblib.Parallel(backend="loky")` runs the outer loop; sklearn's own `GridSearchCV(n_jobs=...)` runs the inner loop.

### Scoring

`scoring` is forwarded into `GridSearchCV` (`"balanced_accuracy"` by default, `"recall"` in the bundled `main.py`). Any [sklearn scoring key](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names) works.

---

## Classifier Catalog

Configured in `ClassifierModel` (`model/classifier_model.py`). Grid params come from `model/constants.py:DEFAULT_GRID_PARAMS`.

| Key | sklearn class | Folder slug | GPU |
|-----|---------------|-------------|-----|
| `rf` | `RandomForestClassifier` (balanced class weights) | `random-forest-classifier` | - |
| `lite-rf` | `RandomForestClassifier` with a slimmer grid | `lite-random-forest-classifier` | - |
| `gb` | `GradientBoostingClassifier` | `gradient-boosting-classifier` | - |
| `xgb` | `XGBClassifier` | `xgb-classifier` | ✓ via `use_gpu=True, gpu_id=0` |
| `svm` | `SVC` (balanced) | `svc` | - |
| `lr` | `LogisticRegression` (balanced) | `logistic-regression` | - |
| `nn` | `MLPClassifier` | `mlp-classifier` | - |
| `dt` | `DecisionTreeClassifier` (balanced) | `decision-tree-classifier` | - |
| `knn` | `KNeighborsClassifier` | `k-neighbors-classifier` | - |
| `nb` | `GaussianNB` | `gaussian-nb` | - |
| `voting` | `VotingClassifier` over `rf + xgb` | `voting-classifier` | ✓ (delegates to XGB) |

Pass `classifiers="rf"` or `classifiers=["rf", "xgb", "gb"]` to `fm.train(...)`. Each classifier produces one `SeedEnsemble`; together they form a `ClassifierEnsemble`.

---

## Cross-Validation Strategies

`ForecastModel.train(cv_strategy=...)` accepts three values:

| Value | sklearn splitter | Folder slug |
|-------|------------------|-------------|
| `"shuffle"` | `ShuffleSplit` | `shuffle-split` |
| `"stratified"` | `StratifiedKFold` | `stratified-k-fold` |
| `"shuffle-stratified"` *(default)* | `StratifiedShuffleSplit` | `stratified-shuffle-split` |

`cv_splits` (default `5`) controls the fold count. `TimeSeriesSplit` is also wired into `ClassifierModel(cv_strategy="timeseries")` 
for ad-hoc direct use, but `ForecastModel.train` does not expose it.

---

## Cache

`TrainingModel` mixes in `CacheModel`. The cache identity is built from:

- station NSLC
- tremor DataFrame fingerprint (shape + min/max + checksum)
- every train kwarg that affects output (`classifiers`, `eruption_dates`, `cv_strategy`, `cv_splits`, `scoring`, `top_n_features`, `include_eruption_date`)
- the kwargs passed into `build_label`, `extract_features`, and `fit`

Hash → `{station_dir}/cache/TrainingModel/{hash}.pkl` (+ a sidecar `{hash}.params.json` for diff-able inspection).

`fm.train(..., use_cache=True)` (default) short-circuits a re-fit when the identity matches; pass `use_cache=False` to force a fresh run.

---

## Outputs

```
{station_dir}/
├── training/
│   ├── features/{cv-slug}/...                                    (above)
│   └── classifiers/{clf-slug}/{cv-slug}/
│       ├── models/{seed:05d}.pkl                                 (per-seed best estimators)
│       ├── trained-model__{Clf}_{CV}_seeds-{N}_features-{K}.csv  (registry CSV)
│       └── SeedEnsemble_{suffix}.pkl                             (joblib-bundled SeedEnsemble)
└── training/
    └── classifiers/
        ├── ClassifierEnsemble_{CV}.pkl                           (one ClassifierEnsemble for all classifiers)
        └── ClassifierEnsemble_{CV}.json                          (registry of paths per classifier)
```

`fm.ClassifierEnsemble` is attached after `train()` returns and is what the Prediction stage consumes.

---

## Standalone Use

`TrainingModel` is a public class - call it directly when you want to skip `ForecastModel`:

```python
from eruption_forecast import TrainingModel

tm = TrainingModel(
    tremor_data="output/VG.OJN.00.EHZ/tremor/VG.OJN.00.EHZ_2025-01-01_2025-12-31.csv",
    start_date="2025-01-01", end_date="2025-07-26",
    classifiers=["rf", "xgb"],
    eruption_dates=["2025-03-20", "2025-04-22", "2025-05-18", "..."],
    window_size=2,
    cv_strategy="shuffle-stratified", cv_splits=5,
    top_n_features=20,
    n_jobs=4, n_grids=4,
)

(
    tm.build_label(window_step=6, window_step_unit="hours")
      .extract_features(select_tremor_columns=["rsam_f2", "rsam_f3", "entropy"])
      .fit(seeds=25, resample_method="auto", plot_features=True)
)

tm.save()                          # → {output_dir}/TrainingModel_{basename}.pkl
print(tm.classifier_ensemble_path) # path to the saved ClassifierEnsemble
```

### Reuse a feature matrix from a prior run

`TrainingModel.load_features(...)` skips tsfresh entirely and loads a previously written `features-matrix_*.csv` + `features-label_*.csv`. 
Pair with `select_features="path/to/top_20_features.csv"` to refit on the curated subset:

```python
tm.load_features(
    select_features="output/.../training/features/.../top_20_features.csv",
).fit(seeds=50)
```

### Persist the training config

```python
tm.save_config()   # → {output_dir}/training.config.yaml
```

Round-trips through `TrainingConfig.load(path)` for reproducibility - see [Configuration](Configuration#trainingconfig).
