# Evaluation Workflow

The evaluation stage scores a fitted `ClassifierEnsemble` against ground truth, never re-fitting. 
It reuses the upstream `TrainingModel` or `PredictionModel` (in-memory or from a `.pkl`) and 
writes per-classifier `(n_samples, n_seeds)` `y_proba` / `y_pred` CSV matrices plus aggregate 
metric plots, with cross-classifier ranking via `ClassifierComparator`. Per-classifier per-seed
metric tables stay in memory on `self.metrics` — no per-seed JSON tree is produced.

Driver: `EvaluationModel` (`src/eruption_forecast/model/evaluation_model.py`). Wrapped by `ForecastModel.evaluate(...)`.

---

## Two Operating Modes

`EvaluationModel` dispatches on `model.kind`:

```
                       fm.evaluate(model="…")
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
       model.kind == "training"       model.kind == "prediction"
                  │                           │
   ┌──────────────┴────────────┐  ┌───────────┴─────────────────────┐
   │ Training reuse            │  │ Prediction reuse                │
   │                           │  │                                 │
   │ y_true ← TrainingModel    │  │ y_true ← fresh LabelBuilder     │
   │           .labels         │  │   over prediction window grid,  │
   │ (already ground truth on  │  │   joined to PredictionModel     │
   │  the labelled grid)       │  │   .labels by datetime           │
   │                           │  │                                 │
   │ eruption_dates: optional  │  │ eruption_dates: REQUIRED        │
   └──────────────┬────────────┘  └───────────────┬─────────────────┘
                  │                               │
                  ▼                               ▼
   output to evaluation/training/   output to evaluation/prediction/
```

Both modes share the same per-seed scoring engine (`MetricsEnsemble`) and aggregation step.

| Mode | When to use | `eruption_dates` |
|------|-------------|------------------|
| `model="training"` | In-sample / training-window diagnostics | optional - embedded in training labels |
| `model="prediction"` | Forecast-window evaluation after `predict()` | **required** to build truth on the prediction grid |

---

## What `evaluate()` Does

```
For each classifier in ClassifierEnsemble:
    MetricsEnsemble.compute()
        ├── ClassifierEnsemble.predict_proba(features_df)
        │       → (n_samples, n_seeds) probability matrix
        ├── threshold at 0.5
        │       → (n_samples, n_seeds) prediction matrix
        ├── persist predictions/{y_proba,y_pred}.csv
        ├── per-seed metric loop in joblib (compute_seed)
        │       → in-memory metrics: dict[classifier, pd.DataFrame(seed × metric)]
        └── idempotent fast-path: re-running compute() with populated
            y_probas is a no-op
```

The result of `evaluate()` is a `dict[classifier_name, pd.DataFrame]` — one DataFrame per classifier, one row per seed, one column per metric. The dict is also cached on `self.metrics` for downstream use.

Available metric columns (per seed):

```
accuracy        balanced_accuracy   precision       recall
f1_score        roc_auc             pr_auc          g_mean
true_positives  true_negatives      false_positives false_negatives
sensitivity     specificity         optimal_threshold
f1_at_optimal   recall_at_optimal   precision_at_optimal
```

### Method signature

```python
em.evaluate(
    plot_aggregate=True,         # ROC, PR, threshold, g-mean, MCC per classifier
    plot_per_seed=False,         # same dispatcher, one plot file per seed
    plot_shap=False,             # reserved no-op; SHAP moved to ExplanationModel
    compare_classifiers=True,    # also run ClassifierComparator at the end
) -> dict[str, pd.DataFrame]
```

`fm.evaluate(...)` forwards `plot_per_seed` and `plot_aggregate` from its own kwargs. `plot_shap=True` is reserved and emits a warning — SHAP plots are produced by the dedicated [Explanation Workflow](Explanation-Workflow) via `ExplanationModel.explain()`.

---

## Cross-Classifier Comparison

```python
comparator = em.compare(metrics=["recall", "roc_auc"])   # or em.compare() for defaults
comparator.get_ranking()       # → comparison/metrics/ranking_recall.csv
comparator.plot_all()          # → comparison/figures/*.png
```

`ClassifierComparator` works on the metrics already computed by `MetricsEnsemble` - repeat `em.compare()` 
calls reuse the cached `MetricsEnsemble`, so the per-classifier `predict_proba` pass is only paid once.

Outputs land under `{evaluation_dir}/comparison/`:

| Artefact | Content |
|----------|---------|
| `metrics/ranking_{metric}.csv` | Classifiers sorted by mean of `{metric}` |
| `figures/metric_bar_{metric}.png` | Bar chart per metric (mean ± std) |
| `figures/metric_bar_all.png` | All metrics in one figure |
| `figures/seed_stability_{metric}.png` | Violin + strip plot per metric across seeds |
| `figures/comparison_grid.png` | Classifier × metric grid |
| `figures/comparison_roc.png` | Overlaid mean ROC curves with ± std bands |

When invoked through `fm.EvaluationModel.compare()`, the live `(ClassifierEnsemble, features_df, y_true)` 
triple is forwarded as `ensemble_source` so the ROC overlay computes from in-memory probabilities rather than re-reading CSVs.

---

## Outputs

```
{station_dir}/evaluation/{training|prediction}/
├── classifiers/
│   └── {classifier-name}/
│       ├── predictions/
│       │   ├── y_proba.csv          # (n_samples, n_seeds)
│       │   └── y_pred.csv           # (n_samples, n_seeds)
│       └── figures/
│           ├── aggregate/{plot_name}.{png,csv}  # plot_aggregate=True
│           └── {plot_name}/{seed:05d}.png        # plot_per_seed=True
├── labels/y_true.csv                 # prediction-reuse mode only
├── MetricsEnsemble.pkl               # optional, via em.MetricsEnsemble.save()
└── comparison/                       # populated when em.compare() runs
    ├── metrics/ranking_*.csv
    └── figures/*.png
```

Per-classifier folder names use the *unslugified* sklearn class name (e.g. `RandomForestClassifier`), 
distinct from the slug used by `TrainingModel` (`random-forest-classifier`). Aggregate / per-seed
plot names come from `evaluation_plots.AGGREGATE_PLOT_DISPATCHER` and
`PER_SEED_PLOT_DISPATCHER` (`roc_curve`, `precision_recall`, `threshold_analysis`, `g_mean_curve`, `mcc_curve`, `confusion_matrix`).

---

## Cache Semantics

`EvaluationModel` does **not** override `build_identity` and so does not participate in the `BaseModel` cache layer — it has no parameter cache. Re-runs are gated by `MetricsEnsemble.compute()`'s in-memory idempotency fast-path: once `self.y_probas` is populated, repeated `compute()` calls short-circuit immediately. `overwrite` only controls plot regeneration — passing `overwrite=False` keeps existing `figures/.../{plot}.png` files in place. The on-disk `predictions/{y_proba,y_pred}.csv` matrices themselves are the persistent layer; deleting them forces the next `compute()` to refit from scratch.

---

## Standalone Use

### Reload from a saved `TrainingModel` / `PredictionModel` `.pkl`

```python
from eruption_forecast import EvaluationModel

# training-window evaluation
em = EvaluationModel.from_file(
    "output/VG.OJN.00.EHZ/TrainingModel_2025-01-01_2025-07-26.pkl",
)
metrics = em.evaluate(plot_aggregate=True)

# forecast-window evaluation - eruption_dates required
em = EvaluationModel.from_file(
    "output/VG.OJN.00.EHZ/PredictionModel_2025-07-27_2025-08-22.pkl",
    eruption_dates=["2025-08-02", "2025-08-18"],
)
metrics = em.evaluate(plot_aggregate=True)
comparator = em.compare()
print(comparator.get_ranking())
comparator.plot_all()
```

### Inspect the per-seed metrics

```python
rf_metrics = metrics["RandomForestClassifier"]
print(rf_metrics["recall"].describe())
#       mean  std    min    max
#       0.81  0.06   0.69   0.92
```

### Drive ranking by a custom metric

```python
ranking = em.compare(metrics="balanced_accuracy").get_ranking(metric="balanced_accuracy")
```

### Persist the evaluation config

```python
em.save_config()   # → {evaluation_dir}/evaluation.config.yaml
```

`evaluate()` already auto-calls `save_config()` once `self.metrics` is set, so 
a standalone evaluation always leaves a YAML snapshot at 
`{output_dir}/evaluation/{training|prediction}/evaluation.config.yaml`. The 
path is already mode-namespaced, so a training-reuse and a prediction-reuse 
run sharing the same `output_dir` never collide. The upstream `model` 
parameter is intentionally omitted from the config (live model instances are 
not serializable); the captured fields are `eruption_dates`, `overwrite`, 
`output_dir`, `root_dir`, `n_jobs`, and `verbose`. 
See [Configuration](Configuration#per-stage-configs-standalone).

---

## ASCII Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│             EvaluationModel  (no cache participation)           │
│                                                                 │
│   ┌──────────────────────────────────────────┐                  │
│   │ MetricsEnsemble.compute()                │                  │
│   │   per-classifier predict_proba           │                  │
│   │   (n_samples × n_seeds) y_proba / y_pred │                  │
│   │   per-seed metrics → self.metrics (mem)  │                  │
│   │   idempotent once y_probas is populated  │                  │
│   └────────────────────┬─────────────────────┘                  │
│                        │ cached on self.MetricsEnsemble         │
│                        ▼                                        │
│                em.compare() → ClassifierComparator              │
│                    ranking CSV + comparison plots               │
└─────────────────────────────────────────────────────────────────┘
```
