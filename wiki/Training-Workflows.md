# Training Workflows

`ModelTrainer` trains on the **entire dataset** across multiple seeds — no internal train/test split.
Evaluation is deferred to `ModelPredictor.evaluate()`, which uses the forecast period as a true
out-of-sample test set with no data leakage between training and evaluation periods.

## Overview

| Aspect | Description |
|---|---|
| Training data | 100% of historical period — no internal split |
| Evaluation | Temporal out-of-sample: forecast period as test set |
| Metrics | Computed by `ModelPredictor.evaluate()` after `predict_proba()` |
| Per-seed output | `models/`, `features/significant_features/`, `trained_model_*.csv` |

## Training Workflow — train()

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

No internal split; trains on 100% of the historical period. Evaluation is performed separately
by `ModelPredictor` on a future forecast period that was never seen during training.

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

## fit() — Thin Wrapper

`fit()` calls `train()` and returns `self` for chaining. All kwargs are forwarded verbatim to `train()`.

```python
trainer.fit(
    random_state=0,
    total_seed=500,
    sampling_strategy=0.75,
)
```

## Multi-Seeding

Seeds run from `random_state` to `random_state + total_seed - 1`. Each seed produces an independently
trained model with its own resampling and feature selection pass. Results are aggregated after all seeds complete.

Benefits:

- Reduces sensitivity to any single resampling
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
    merged_model_{suffix}.pkl
    (SeedEnsemble — all estimators + feature lists in one object)
           │
           ▼  trainer.merge_classifier_models({"rf": ..., "xgb": ...})
           │
    merged_classifiers_{suffix}.pkl
    (ClassifierEnsemble — one SeedEnsemble per classifier)
```

```python
# After trainer.train() completes:
merged_path = trainer.merge_models()
# → .../merged_model_RandomForestClassifier-StratifiedKFold_rs-0_ts-500_top-20.pkl

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

- **`n_jobs`** — outer loop: parallel seed workers via `joblib.Parallel(backend="loky")`. Each worker runs one full seed independently (resample → feature selection → GridSearchCV → save). The `loky` backend is used instead of `threading` for nested-parallelism safety.
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
