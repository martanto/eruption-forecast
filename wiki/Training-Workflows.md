# Training Workflows

`ModelTrainer` supports two training modes depending on whether you need in-sample evaluation or want to train on all available data for operational forecasting.

## Overview

| | `evaluate()` | `train()` |
|---|---|---|
| In-sample metrics (accuracy, F1, AUC)? | Yes — held-out 20% test set | No |
| Uses 100% of training data? | No (80%) | Yes |
| Produces per-seed test splits? | Yes (saved to `tests/`) | No |
| Use with `ModelPredictor`? | Optional | Recommended |
| Best for? | Exploring classifiers, comparing configs | Final production model |

## Workflow 1 — evaluate()

```
      Full Dataset
           │
           ▼
      80/20 Split
      (stratified)
      ┌────┴────┐
    Train     Test
      │         │
    RandomUnder │
    Sampler     │
      │         │
    Feature     │
    Selection   │
      │         │
    GridSearchCV│
    + CV folds  │
      │         │
    Evaluate ◄──┘
      │
    Save model + metrics
```

The split happens **before** resampling and feature selection to prevent data leakage. Each seed gets its own split, its own feature selection, and its own evaluation on the held-out 20%. The results are aggregated across all seeds at the end.

```python
from eruption_forecast.model.model_trainer import ModelTrainer

trainer = ModelTrainer(
    extracted_features_csv="output/features/all_features.csv",
    label_features_csv="output/features/label_features.csv",
    output_dir="output/trainings",
    classifier="xgb",
    cv_strategy="stratified",
    number_of_significant_features=20,
    feature_selection_method="combined",
    n_jobs=4,
)

trainer.evaluate(
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)
```

### Output files

- `models/00000.pkl` — trained model per seed
- `metrics/00000.json` — per-seed evaluation metrics
- `tests/00000_X_test.csv`, `tests/00000_y_test.csv` — held-out test data per seed
- `trained_model_{suffix}.csv` — model registry
- `all_metrics_{suffix}.csv` — all per-seed metrics
- `metrics_summary_{suffix}.csv` — mean ± std summary

## Workflow 2 — train()

```
      Full Dataset
           │
           ▼
    RandomUnderSampler
      (full dataset)
           │
    Feature Selection
      (full dataset)
           │
      GridSearchCV
       + CV folds
           │
    ┌──────┴──────┐
  model.pkl   registry.csv
```

No internal split; trains on 100% of the historical period. Evaluation is deferred to `ModelPredictor` using a separate future dataset that was not seen during training.

```python
from eruption_forecast.model.model_trainer import ModelTrainer

trainer = ModelTrainer(
    extracted_features_csv="output/features/all_features.csv",
    label_features_csv="output/features/label_features.csv",
    output_dir="output/trainings",
    classifier="rf",
    n_jobs=4,
)

trainer.train(
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)
```

### Output files

- `models/00000.pkl` — trained model per seed
- `trained_model_{suffix}.csv` — model registry (used by `ModelPredictor`)
- `features/significant_features/00000.csv` — per-seed top-N features

## fit() — Unified Entry Point

`fit(with_evaluation=True/False)` dispatches to either workflow. Use it when the calling code needs a single method regardless of which workflow is active.

```python
# Equivalent to evaluate()
trainer.fit(
    with_evaluation=True,
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)

# Equivalent to train()
trainer.fit(
    with_evaluation=False,
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)
```

`ForecastModel.train()` calls `fit()` internally and exposes `with_evaluation` as a direct parameter, so you can control the workflow from the high-level API without dropping down to `ModelTrainer`.

## Multi-Seeding

Seeds run from `random_state` to `random_state + total_seed - 1`. Each seed produces an independently trained model with its own data split (for `evaluate`) or resampling (for `train`). Results are aggregated after all seeds complete.

Benefits:

- Reduces sensitivity to any single train/test split
- Provides uncertainty estimates (std across seeds)
- Enables ensemble consensus forecasting via `ModelPredictor`

## Feature Selection Methods

| Method | How it works | Speed |
|--------|-------------|-------|
| `tsfresh` | Statistical FDR (Benjamini-Hochberg) — removes features uncorrelated with the target | Fast |
| `random_forest` | Permutation importance via a quick RandomForest fit — keeps top-N by importance | Slow |
| `combined` | Stage 1: tsfresh FDR to reduce from thousands to hundreds; Stage 2: RandomForest importance to reduce to final top-N | Balanced |

Override the method:

```python
trainer = ModelTrainer(..., feature_selection_method="combined")
# or via ForecastModel
fm.set_feature_selection_method("combined").train(...)
```

## Merging Seed Models

After training completes, all 500 seed models can be bundled into a single `.pkl` file. This eliminates the per-seed file I/O at prediction time and produces a `SeedEnsemble` that is sklearn-compatible.

`SeedEnsemble` and `ClassifierEnsemble` both inherit from `BaseEnsemble`, which provides the `save()` / `load()` persistence logic. You never need to call joblib directly.

```
  500 × model.pkl  +  500 × significant_features.csv
           │
           ▼  trainer.merge_models()
           │
    SeedEnsemble_{suffix}.pkl
    (SeedEnsemble — all estimators + feature lists in one object)
           │
           ▼  trainer.merge_classifier_models({"rf": ..., "xgb": ...})
           │
    merged_classifiers_{suffix}.pkl
    (ClassifierEnsemble — one SeedEnsemble per classifier)

> **Note:** When using `ForecastModel.train()`, the `ClassifierEnsemble` is saved
> automatically to `{output_dir}/ClassifierEnsemble.pkl` after all classifiers are merged.
```

```python
# After trainer.train() completes:
merged_path = trainer.merge_models()
# → .../SeedEnsemble_RandomForestClassifier-StratifiedKFold_rs-0_ts-500_top-20.pkl

# Multi-classifier bundle
bundle_path = trainer.merge_classifier_models(
    {"rf": rf_trainer.csv, "xgb": xgb_trainer.csv}
)
# → .../trainings/merged_classifiers_{suffix}.pkl

# Load and use directly (no ModelPredictor needed):
from eruption_forecast.model.seed_ensemble import SeedEnsemble

ensemble = SeedEnsemble.load(merged_path)
mean_p, std, conf, pred = ensemble.predict_with_uncertainty(features_df)
proba = ensemble.predict_proba(features_df)  # (n_samples, 2)
```

The merged `.pkl` can also be passed directly to `ModelPredictor.predict_proba()` instead of a CSV registry path — no other changes needed.

See [Evaluation and Forecasting](Evaluation-and-Forecasting) for how `ModelPredictor` handles merged files.

## Parallelism

Two independent parallelism levels control training speed:

- **`n_jobs`** — outer loop: parallel seed workers via `joblib.Parallel(backend="loky")`. Each worker runs one full seed independently (resample → feature selection → GridSearchCV → evaluate → save). The `loky` backend is used instead of `threading` for nested-parallelism safety.
- **`grid_search_n_jobs`** — inner loop: parallel folds inside `GridSearchCV` and parallel workers inside `FeatureSelector` (tsfresh/RandomForest). These are separate scopes even though they share the same parameter.
- Enforced: `n_jobs × grid_search_n_jobs ≤ cpu_count`

## GPU Acceleration (XGBoost only)

XGBoost (`xgb`) and the voting ensemble (`voting`) support GPU training via `use_gpu=True`. Use `gpu_id` to pick a specific device on multi-GPU machines.

```python
trainer = ModelTrainer(
    extracted_features_csv="output/features/all_features.csv",
    label_features_csv="output/features/label_features.csv",
    classifier="xgb",
    use_gpu=True,    # enable CUDA
    gpu_id=1,        # use second GPU (default: 0)
)
trainer.train(random_state=0, total_seed=500)
```

When `use_gpu=True` the following restrictions are automatically enforced:

| Parallelism level | CPU (default) | GPU (`use_gpu=True`) |
|---|---|---|
| `n_jobs` (outer seed workers) | Configurable | Forced to `1` — multiple seeds sharing one GPU cause VRAM contention |
| `GridSearchCV` inner `n_jobs` | Uses `grid_search_n_jobs` | Forced to `1` — parallel CV fold workers each claim the GPU simultaneously |
| `FeatureSelector` inner `n_jobs` | Uses `grid_search_n_jobs` | **Unchanged** — feature selection is CPU-only (tsfresh / RandomForest) and safe to parallelise |

> `use_gpu=True` has no effect on non-XGBoost classifiers (`rf`, `gb`, `svm`, etc.). Passing it with those classifiers emits a warning and training continues on CPU.
