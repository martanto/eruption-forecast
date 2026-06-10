# Evaluation Workflow

The evaluation stage scores a fitted `ClassifierEnsemble` against ground truth, never re-fitting. It reuses the upstream `TrainingModel` or `PredictionModel` (in-memory or from a `.pkl`) and writes per-seed JSON + aggregate CSV + plots per classifier, plus cross-classifier ranking via `ClassifierComparator`.

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
   ┌──────────────┴────────────┐  ┌───────────┴────────────────────┐
   │ Training reuse            │  │ Prediction reuse                │
   │                           │  │                                 │
   │ y_true ← TrainingModel    │  │ y_true ← fresh LabelBuilder     │
   │           .labels         │  │   over prediction window grid,  │
   │ (already ground truth on  │  │   joined to PredictionModel     │
   │  the labelled grid)       │  │   .labels by datetime           │
   │                           │  │                                 │
   │ eruption_dates: optional  │  │ eruption_dates: REQUIRED         │
   └──────────────┬────────────┘  └───────────────┬─────────────────┘
                  │                               │
                  ▼                               ▼
       output to evaluation/training/    output to evaluation/prediction/
```

Both modes share the same per-seed scoring engine (`MetricsEnsemble`) and aggregation step.

| Mode | When to use | `eruption_dates` |
|------|-------------|------------------|
| `model="training"` | In-sample / training-window diagnostics | optional — embedded in training labels |
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
        ├── persist: predictions/{y_proba,y_pred,y_true}.csv
        ├── per-seed MetricsComputer.compute_all_metrics()
        │       → metrics/json/{seed:05d}.json
        └── aggregate mean ± std across seeds
                → metrics_summary_{start}_{end}.csv
                → all_metrics_{start}_{end}.csv
                → returned to caller as pd.DataFrame
```

The result of `evaluate()` is a `dict[classifier_name, pd.DataFrame]` — one DataFrame per classifier, one row per seed, one column per metric.

Available metrics (from `MetricsComputer`):

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
    plot_aggregate=True,         # mean ± std plots per classifier
    plot_per_seed=False,         # one plot set per seed (expensive)
    plot_shap=False,             # reserved — currently a warning
    compare_classifiers=True,    # also run ClassifierComparator at the end
) -> dict[str, pd.DataFrame]
```

`fm.evaluate(...)` forwards `plot_per_seed` and `plot_aggregate` from its own kwargs.

---

## Cross-Classifier Comparison

```python
comparator = em.compare(metrics=["recall", "roc_auc"])   # or em.compare() for defaults
comparator.get_ranking()       # → comparison/metrics/ranking_recall.csv
comparator.plot_all()          # → comparison/figures/*.png
```

`ClassifierComparator` works on the metrics already computed by `MetricsEnsemble` — repeat `em.compare()` calls reuse the cached `MetricsEnsemble`, so the per-classifier `predict_proba` pass is only paid once.

Outputs land under `{evaluation_dir}/comparison/`:

| Artefact | Content |
|----------|---------|
| `metrics/ranking_{metric}.csv` | Classifiers sorted by mean of `{metric}` |
| `figures/metric_bar_{metric}.png` | Bar chart per metric (mean ± std) |
| `figures/metric_bar_all.png` | All metrics in one figure |
| `figures/seed_stability_{metric}.png` | Violin + strip plot per metric across seeds |
| `figures/comparison_grid.png` | Classifier × metric grid |
| `figures/comparison_roc.png` | Overlaid mean ROC curves with ± std bands |

When invoked through `fm.EvaluationModel.compare()`, the live `(ClassifierEnsemble, features_df, y_true)` triple is forwarded as `ensemble_source` so the ROC overlay computes from in-memory probabilities rather than re-reading CSVs.

---

## Outputs

```
{station_dir}/evaluation/{training|prediction}/
├── classifiers/
│   └── {classifier-name}/
│       ├── predictions/
│       │   ├── y_proba.csv          # (n_samples, n_seeds)
│       │   ├── y_pred.csv           # (n_samples, n_seeds)
│       │   └── y_true.csv           # (n_samples,)
│       ├── metrics/
│       │   ├── json/{seed:05d}.json # per-seed metrics
│       │   ├── metrics_summary_{start}_{end}.csv
│       │   └── all_metrics_{start}_{end}.csv
│       └── figures/                  # aggregate plots when plot_aggregate=True
└── comparison/                       # populated when em.compare() runs
    ├── metrics/ranking_*.csv
    └── figures/*.png
```

Per-classifier folder names use the *unslugified* sklearn class name (e.g. `RandomForestClassifier`), distinct from the slug used by `TrainingModel` (`random-forest-classifier`).

---

## Cache Semantics

`EvaluationModel` does **not** mix in `CacheModel` — it has no parameter-cache. What it does instead is **reuse on-disk JSON metrics**: per-seed metrics files are only re-computed when missing or when `overwrite=True`. Re-running `evaluate()` on the same instance is therefore very fast once the JSON tree exists.

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

# forecast-window evaluation — eruption_dates required
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

---

## ASCII Quick Reference

```
┌─────────────────────────────────────────────────────────────────┐
│             EvaluationModel  (no CacheModel mix-in)              │
│                                                                  │
│   ┌─────────────────────────────────────────┐                    │
│   │ MetricsEnsemble.compute()                │                    │
│   │   per-classifier predict_proba           │                    │
│   │   per-seed JSON + (y_proba,y_pred,y_true)│                    │
│   │   aggregate mean ± std → CSV             │                    │
│   └────────────────────┬────────────────────┘                    │
│                        │ cached on self.MetricsEnsemble          │
│                        ▼                                          │
│                em.compare() → ClassifierComparator               │
│                    ranking CSV + comparison plots                │
└─────────────────────────────────────────────────────────────────┘
```
