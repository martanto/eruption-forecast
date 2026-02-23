# Training Workflows

`ModelTrainer` supports two training modes depending on whether you need in-sample evaluation or want to train on all available data for operational forecasting.

## Overview

| | `train_and_evaluate()` | `train()` |
|---|---|---|
| In-sample metrics (accuracy, F1, AUC)? | Yes — held-out 20% test set | No |
| Uses 100% of training data? | No (80%) | Yes |
| Produces per-seed test splits? | Yes (saved to `tests/`) | No |
| Use with `ModelPredictor`? | Optional | Recommended |
| Best for? | Exploring classifiers, comparing configs | Final production model |

## Workflow 1 — train_and_evaluate()

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

trainer.train_and_evaluate(
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
# Equivalent to train_and_evaluate()
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

Seeds run from `random_state` to `random_state + total_seed - 1`. Each seed produces an independently trained model with its own data split (for `train_and_evaluate`) or resampling (for `train`). Results are aggregated after all seeds complete.

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

## Parallelism

- `n_jobs` — outer loop: parallel seed workers (each seed is an independent job)
- `grid_search_n_jobs` — inner loop: parallel folds inside each `GridSearchCV` call
- Enforced: `n_jobs × grid_search_n_jobs ≤ cpu_count`
- Uses `joblib.Parallel(backend="loky")` for nested-parallelism safety
