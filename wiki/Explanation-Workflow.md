# Explanation Workflow

The explanation stage produces per-seed SHAP explanations for a fitted
`ClassifierEnsemble`, never re-fitting it. It reuses the upstream
`TrainingModel` or `PredictionModel` (in-memory or from a `.pkl`) and
writes per-classifier `ClassifierExplanation.pkl` artefacts, per-seed
`shap.Explanation` pickles, per-seed bar / beeswarm plots, and
per-eruption waterfall plots.

Driver: `ExplanationModel` (`src/eruption_forecast/model/explanation_model.py`),
delegating SHAP work to `ExplainerEnsemble`
(`src/eruption_forecast/ensemble/explainer_ensemble.py`). Wrapped by
`ForecastModel.explain(...)`.

---

## TreeExplainer constraint

`ExplainerEnsemble` only supports `shap.TreeExplainer`, which restricts
the stage to tree-based classifiers. From the 11 supported by
`TrainingModel`:

| Supported (tree) | Skipped (non-tree, warning logged) |
|------------------|-------------------------------------|
| `rf`, `lite-rf`, `gb`, `xgb` | `svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting` |

Non-tree classifiers are skipped at the `ExplainerEnsemble.explain()`
loop with a warning so a mixed-classifier ensemble still produces SHAP
output for whichever classifiers qualify.

---

## Two operating modes

`ExplanationModel` dispatches on `model.kind`:

```
                       fm.explain(model="…")
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
       model.kind == "training"       model.kind == "prediction"
                  │                           │
   ┌──────────────┴────────────┐  ┌───────────┴─────────────────────┐
   │ Training reuse            │  │ Prediction reuse                │
   │                           │  │                                 │
   │ features_df ← Training    │  │ features_df ← PredictionModel   │
   │   Model.features_df       │  │   .features_df                  │
   │ labels ← TrainingModel    │  │ labels ← PredictionModel.labels │
   │   .labels                 │  │                                 │
   │                           │  │ eruption_dates: required for    │
   │ eruption_dates: optional  │  │   waterfall plot rendering      │
   └──────────────┬────────────┘  └───────────────┬─────────────────┘
                  │                               │
                  ▼                               ▼
   output to explanation/training/ output to explanation/prediction/
```

Both modes share the same per-seed SHAP engine
(`ExplainerEnsemble.explain_seed`) and per-classifier
`ClassifierExplanation` payload.

| Mode | When to use | `eruption_dates` |
|------|-------------|------------------|
| `model="training"` | In-sample feature attribution diagnostics | optional — waterfalls skipped if missing |
| `model="prediction"` | Forecast-window attribution after `predict()` | required for waterfall plots |

---

## What `explain()` does

```
For each SeedEnsemble in ClassifierEnsemble:
    skip non-tree classifier (warn)
    For each seed in SeedEnsemble.seeds:
        shap.TreeExplainer(model, features_df[seed.feature_names])
            → shap.Explanation
        normalise_shap_values()       # pick positive-class slice
        shorten_feature_name()        # readable tsfresh labels
        persist seed pickle → shap_values/{seed:05d}.pkl   # save_per_seed=True
    bundle into ClassifierExplanation
    persist → ClassifierExplanation_{classifier_name}.pkl
```

Result on the instance: `em.explanations: list[ClassifierExplanation]`.

### `ForecastModel.explain()` signature

```python
fm.explain(
    model="prediction",                       # "training" | "prediction"
    eruption_dates=None,                      # falls back to train() dates
    save_per_seed=True,
    plot_per_seed=True,
    plot_aggregate=True,                      # aggregate bar + beeswarm per classifier
    figsize=None,                             # auto-sized from max_display
    max_display=20,
    group_remaining_features=False,
    dpi=150,
    check_additivity=False,
    overwrite_classifier_explanation=False,
    output_dir=None,
    overwrite=None,
    n_jobs=None,
    verbose=None,
) -> Self
```

Internally calls `ExplanationModel.explain(...)` then `.plot(...)`.

### Standalone `ExplanationModel.explain()` signature

```python
em.explain(
    save_per_seed=True,
    check_additivity=False,
    overwrite_classifier_explanation=False,
) -> Self
```

### Standalone `ExplanationModel.plot()` signature

```python
em.plot(
    figsize=None,
    max_display=20,
    group_remaining_features=False,
    dpi=150,
    plot_per_seed=True,
    plot_aggregate=True,
)
```

`plot()` always renders per-eruption waterfalls when `eruption_dates`
is available; per-seed bar + beeswarm rendering is gated on
`plot_per_seed`; per-classifier aggregate bar + beeswarm rendering
(stacks every seed into the NaN-padded union feature space) is gated
on `plot_aggregate`.

---

## Plot inventory

| Plot | Producer | Output stem |
|------|----------|-------------|
| Per-seed beeswarm | `ExplainerEnsemble.plot_seed()` | `classifiers/{ClfName}/figures/beeswarm/{seed:05d}.png` |
| Per-seed bar | `ExplainerEnsemble.plot_seed()` | `classifiers/{ClfName}/figures/bar/{seed:05d}.png` |
| Aggregate bar (frequency-weighted mean \|SHAP\| across seeds) | `ExplainerEnsemble.plot_aggregate()` → `plot_aggregate_shap_bar()` | `classifiers/{ClfName}/figures/aggregate/bar.{png,csv}` |
| Aggregate beeswarm (NaN-padded union feature space) | `ExplainerEnsemble.plot_aggregate()` → `plot_aggregate_shap_beeswarm()` | `classifiers/{ClfName}/figures/aggregate/beeswarm.{png,csv}` |
| Per-eruption waterfall (highest-probability window per eruption) | `ExplainerEnsemble.plot_waterfall()` → `plot_classifier_waterfall()` | `eruptions/{eruption_date}/{ClfName}_{datetime}_seed=_index=.png` |

Standalone plot helpers in
`src/eruption_forecast/plots/explanation_plots.py`:

| Helper | Use case |
|--------|----------|
| `plot_shap_waterfall(explanation, ...)` | One waterfall for one observation |
| `plot_shap_beeswarm(explanation, ...)` | One beeswarm for one seed |
| `plot_shap_bar(explanation, ...)` | One bar plot for one seed |
| `plot_aggregate_shap_bar(classifier_explanation, ...)` | Frequency-weighted aggregate bar across seeds (builds importance table internally) |
| `plot_aggregate_shap_beeswarm(classifier_explanation, ...)` | Stacked-seeds aggregate beeswarm (builds NaN-padded union explanation internally) |
| `plot_classifier_waterfall(classifier_explanation, ...)` | Per-eruption highest-probability waterfall (the `plot_waterfall` workhorse) |

All renderers route through
`plots/styles.py::shap_figure` and `save_figure`, which closes the
matplotlib figure after saving.

---

## Outputs

```
{station_dir}/explanation/{training|prediction}/
├── classifiers/
│   └── {ClassifierName}/                                  # e.g. RandomForestClassifier
│       ├── ClassifierExplanation_{ClassifierName}.pkl     # bundled explanations
│       ├── shap_values/
│       │   └── {seed:05d}.pkl                             # per-seed shap.Explanation
│       └── figures/
│           ├── beeswarm/{seed:05d}.png                    # plot_per_seed=True
│           ├── bar/{seed:05d}.png                         # plot_per_seed=True
│           └── aggregate/                                 # plot_aggregate=True
│               ├── bar.png
│               ├── bar.csv                                # frequency-weighted importance table
│               ├── beeswarm.png
│               └── beeswarm.csv                           # tidy non-NaN cells for offline redraw
└── eruptions/                                             # sibling of classifiers/
    └── {YYYY-MM-DD}/
        └── {ClassifierName}_{YYYY-MM-DD_HH-MM-SS}_seed={i}_index={j}.png
```

Per-classifier folder names use the *unslugified* sklearn class name
(`RandomForestClassifier`), matching `EvaluationModel`'s convention.

---

## Cache semantics

`ExplanationModel` mixes in `CacheModel`. The cache identity is
content-addressable:

```
ExplanationModel cache identity = {
    class:          "ExplanationModel",
    upstream_hash:  hash(model_kind, classifier_names, features shape+columns,
                         date range),
    explain_params: {save_per_seed: bool},
}
```

A change to the upstream `ClassifierEnsemble` or the feature matrix
invalidates the cache automatically. Cache files land under
`cache/ExplanationModel/{hash}.pkl + {hash}.params.json`, alongside the
existing `TrainingModel` and `PredictionModel` caches.

A cache hit restores `self.explanations` and skips the SHAP pass. The
per-seed `shap_values/{seed:05d}.pkl` files and per-classifier
`ClassifierExplanation_*.pkl` artefacts on disk are independent of the
cache pickle — they survive cache deletion and allow `explain()` to
short-circuit at the per-classifier level even if the top-level cache
pickle is missing.

---

## Standalone use

### Reload from a saved `PredictionModel` `.pkl`

```python
from eruption_forecast import ExplanationModel

em = ExplanationModel.from_file(
    "output/VG.OJN.00.EHZ/PredictionModel_2025-07-27_2025-08-22.pkl",
    eruption_dates=["2025-08-02", "2025-08-18"],
    n_jobs=4,
)
em.explain(save_per_seed=True)
em.plot(max_display=20, plot_per_seed=True)

print(em.explanations[0].classifier_name)   # "RandomForestClassifier"
print(em.explanations[0].seeds[0].random_state)
print(em.explanations[0].seeds[0].shap_values.shape)
```

### Reload from a saved `TrainingModel` `.pkl`

```python
em = ExplanationModel.from_file(
    "output/VG.OJN.00.EHZ/TrainingModel_2025-01-01_2025-07-26.pkl",
)
em.explain().plot(plot_per_seed=False)
```

### Drive the waterfall path directly

```python
from eruption_forecast.plots.explanation_plots import plot_classifier_waterfall

for classifier_explanation in em.explanations:
    plot_classifier_waterfall(
        classifier_explanation=classifier_explanation,
        classifier_ensemble=em.ClassifierEnsemble,
        labels=em.model.labels,
        eruption_dates=["2025-08-02", "2025-08-18"],
        output_dir=em.explanation_dir + "/eruptions",
        max_display=20,
    )
```

### Persist the explanation config

```python
em.save_config()   # → {explanation_dir}/explanation.config.yaml
```

`explain()` already auto-calls `save_config()` after the SHAP pass + 
`self.save()`, so a standalone explanation always leaves a YAML snapshot at 
`{output_dir}/explanation/{training|prediction}/explanation.config.yaml`. The 
path is already namespaced by upstream stage. The upstream `model` parameter 
is intentionally omitted from the config (live model instances are not 
serializable); the captured fields are `eruption_dates`, `overwrite`, 
`output_dir`, `root_dir`, `n_jobs`, and `verbose`. 
See [Configuration](Configuration#per-stage-configs-standalone).

---

## ASCII quick reference

```
┌─────────────────────────────────────────────────────────────────┐
│             ExplanationModel  (BaseModel + CacheModel)          │
│                                                                 │
│   ┌──────────────────────────────────────────┐                  │
│   │ ExplainerEnsemble.explain()              │                  │
│   │   per-classifier TreeExplainer pass      │                  │
│   │   per-seed shap.Explanation              │                  │
│   │   bundle → ClassifierExplanation.pkl     │                  │
│   └────────────────────┬─────────────────────┘                  │
│                        │ cached on self.explanations            │
│                        ▼                                        │
│            em.plot() → ExplainerEnsemble.plot_seed()            │
│                      → ExplainerEnsemble.plot_waterfall()       │
│                                                                 │
│   Output: explanation/{training|prediction}/                    │
│           classifiers/{ClfName}/ + eruptions/{date}/            │
└─────────────────────────────────────────────────────────────────┘
```
