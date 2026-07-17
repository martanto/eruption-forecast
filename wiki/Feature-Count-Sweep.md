# Feature Count Sweep

> **⚠️ Experimental — not production-ready.** The feature-count sweep is a
> post-hoc diagnostic added on the `ft/feature-count-sweep` branch. Its
> public surface (`FeatureCountSweep`, `sweep_feature_count`), defaults
> (`mode="forecast"`, `estimator_mode="default"`), on-disk layout
> (`sweep/{mode}/{classifier}/`), and CLI wrappers may change without a
> deprecation cycle. Do **not** rely on the results as a substitute for a
> retrained model — treat the recommended `N*` as a hypothesis to verify
> with a fresh `TrainingModel.fit(top_n_features=N*)` run before adopting it.

Post-hoc "which `top_n_features` should I have picked?" utility. Reuses
per-seed FDR rankings, resampled ids, and tuned hyperparameters that
`TrainingModel.fit()` already wrote to disk, then re-scores each seed at
several candidate `N` values.

Two entry points, both re-exported from `eruption_forecast.features`:

```python
from eruption_forecast.features import FeatureCountSweep, sweep_feature_count
```

- **`sweep_feature_count(source, ...)`** — cold-flow wrapper. Discovers the
  per-seed inputs from a completed `training/` directory (or a live
  `TrainingModel` instance) and returns a fitted `FeatureCountSweep` (or a
  `{classifier: sweep}` dict).
- **`FeatureCountSweep(...)`** — the underlying engine, callable directly
  when you already have `(X, y, per-seed rankings)` in memory.

---

## Quick Start

Distilled from
[`example_feature_count_sweep.py`](https://github.com/martanto/eruption-forecast/blob/ft/feature-count-sweep/example_feature_count_sweep.py).
Prerequisite: the scenario has already run `ForecastModel.predict(...)`
(or `PredictionModel.forecast(...)` with `save_model=True`), so a
`{hash}.PredictionModel.pkl` cache pickle exists.

```python
import glob, os

from eruption_forecast import EvaluationModel
from eruption_forecast.features import sweep_feature_count

SCENARIO_DIR = "output/VG.OJN.00.EHZ/scenarios/scenario-1"
TRAINING_DIR = os.path.join(SCENARIO_DIR, "training")
ERUPTION_DATES = ["2025-04-10", "2025-04-22"]

# 1) Prepare an EvaluationModel in prediction-reuse mode. build_label()
#    populates em.y_true cheaply (no per-classifier predict_proba pass).
prediction_pkl = glob.glob(os.path.join(SCENARIO_DIR, "prediction", "*.PredictionModel.pkl"))[0]
em = EvaluationModel.from_file(prediction_pkl, eruption_dates=ERUPTION_DATES)
em.build_label(window_step=em.window_step, window_step_unit=em.window_step_unit)

# 2) Sweep every classifier in the training directory.
results = sweep_feature_count(
    source=TRAINING_DIR,
    mode="forecast",              # score against the prediction window (default)
    evaluation_source=em,
    n_candidates=list(range(1, 26)),
    scoring="roc_auc",
    save=True,                    # writes cv_scores.csv, curve.png, ...
)

# 3) Recommended N* per classifier.
for name, sweep in results.items():
    print(f"{name}: N* = {sweep.n_features_}")

# Reload later without re-running the sweep:
#   from eruption_forecast.features import FeatureCountSweep
#   sweep = FeatureCountSweep.load(
#       f"{TRAINING_DIR}/features/{cv_slug}/sweep/forecast/RandomForestClassifier/FeatureCountSweep.pkl"
#   )
```

Pass `classifier_name="RandomForestClassifier"` to focus on a single
classifier (returns a `FeatureCountSweep` instead of a dict).

Fast within-training proxy — no `evaluation_source` needed:

```python
results = sweep_feature_count(
    source=TRAINING_DIR,
    mode="cv",                    # RFECV-analog inside the resampled subset
    n_candidates=list(range(1, 26)),
    cv=5,
    scoring="roc_auc",
    n_jobs=4,                     # parallel over CV folds (cv mode only)
    save=True,
)
```

---

## Two Scoring Modes

| Mode | Scoring surface | Cost | When to use |
|------|-----------------|------|-------------|
| `"forecast"` *(default)* | Held-out **prediction-window** features + ground-truth `y_true` from an `EvaluationModel` in prediction-reuse mode | One `fit()` per `(seed, N)` | Match the project's real forecasting evaluation |
| `"cv"` | `RFECV`-analog — cross-validation *inside* the trainer's resampled subset | `cv_splits` fits per `(seed, N)` | Fast proxy, but does **not** measure forecast-window generalisation |

Motivation for `forecast` being the default: the CV proxy runs inside the
same tiny under-sampled subset the trainer used, and on realistic
eruption datasets it saturates at AP ≈ 1.0 across every `N`, producing
no signal about temporal generalisation. Forecast mode answers "which
`N` gives the best score on the held-out forecast window?" directly.

### Fail-fast on missing features

In forecast mode, if any per-seed top-`N` list contains a feature that
does not appear in the prediction feature matrix, the sweep raises
`KeyError` immediately — it is **never** silently intersected or
skipped. This surfaces the mismatch caused by
`PredictionModel.extract_features(select_tremor_columns=..., exclude_features=...)`
narrowing the extraction set. Fix by widening the prediction extraction
(or by re-training with a narrower `top_n_features` pool).

---

## `estimator_mode` — Default vs Tuned

Controls how each seed's `best_model.pkl` is prepared before every refit:

| Value | Behaviour | Bias |
|-------|-----------|------|
| `"default"` *(recommended)* | Reset the tuned estimator via `estimator.__class__()` — every candidate `N` is compared with the same untuned base learner | None (matches sklearn `RFECV` / yellowbrick convention) |
| `"tuned"` | Keep the trainer's `GridSearchCV`-picked hyperparameters | Biases the curve toward the trained `top_n_features` because those params were selected at that specific `N` |

A large gap between the two curves is a signal that the trainer's
hyperparameters are strongly `N`-specific — a smell worth investigating
before adopting the recommended `N*`.

---

## Picking `N*`

The recommendation is the `N` with the highest mean score across seeds,
with an optional tie-break toward the smallest `N`:

- `parsimony=False` — strict argmax on the mean.
- `parsimony=True, parsimony_tolerance=None` *(default)* — **1-SE rule**
  (adaptive): pick the smallest `N` whose mean is within one standard
  error of the peak.
- `parsimony=True, parsimony_tolerance=t` — fractional tolerance: pick
  the smallest `N` whose mean is ≥ `peak × (1 − t)`.

For single-seed sweeps the standard error is undefined; the tolerance
band collapses to zero and the peak `N` is returned.

The per-seed argmax `N` is exposed on `sweep.seed_argmax_` — a
diagnostic that reveals whether a shared `N*` across seeds is a
reasonable assumption in the first place.

---

## Forecast-Mode Wiring

`sweep_feature_count(mode="forecast", ...)` requires an
`EvaluationModel` in **prediction-reuse mode** whose `y_true` has
already been populated. The cheapest way to prepare one:

```python
from eruption_forecast import EvaluationModel

em = EvaluationModel.from_file(
    "output/VG.OJN.00.EHZ/prediction/{hash}.PredictionModel.pkl",
    eruption_dates=["2025-04-10", "2025-04-22", "..."],
)
em.build_label(
    window_step=em.window_step,
    window_step_unit=em.window_step_unit,
)
```

`build_label(...)` populates `em.y_true` and writes `y_true.csv` under
`evaluation/prediction/labels/` without running the full per-classifier
`predict_proba` pass that `em.evaluate()` would trigger. The sweep
needs only `em.features_df` and `em.y_true`.

The `_resolve_forecast_inputs` guard validates the source and raises
`ValueError` when:

- `evaluation_source is None`
- `evaluation_source.model_kind != "prediction"` (training-reuse labels are
  the training labels themselves and do not form a valid held-out set)
- `evaluation_source.y_true` is empty (need to call `build_label` /
  `evaluate` first)
- `evaluation_source.features_df` is empty (upstream `PredictionModel`
  never ran `extract_features()`)

---

## Complete Example

```python
import glob
import os

from eruption_forecast import EvaluationModel
from eruption_forecast.features import sweep_feature_count

SCENARIO_DIR = "output/VG.OJN.00.EHZ/scenarios-new/scenario-11"
TRAINING_DIR = os.path.join(SCENARIO_DIR, "training")
PREDICTION_DIR = os.path.join(SCENARIO_DIR, "prediction")

ERUPTION_DATES = ["2025-04-10", "2025-04-22", "2025-05-18"]

# Auto-discover the persisted PredictionModel cache pickle.
prediction_pickle = sorted(
    glob.glob(os.path.join(PREDICTION_DIR, "*.PredictionModel.pkl"))
)[0]

em = EvaluationModel.from_file(prediction_pickle, eruption_dates=ERUPTION_DATES)
em.build_label(window_step=em.window_step, window_step_unit=em.window_step_unit)

results = sweep_feature_count(
    source=TRAINING_DIR,
    mode="forecast",                        # default
    evaluation_source=em,
    classifier_name=None,                   # sweep every classifier
    n_candidates=list(range(1, 26, 1)),
    scoring="roc_auc",
    parsimony=True,
    parsimony_tolerance=None,               # → 1-SE rule
    estimator_mode="default",               # RFECV convention
    random_state=42,
    save=True,
    verbose=True,
)

for classifier_name, sweep in results.items():
    print(f"[{classifier_name}] N* = {sweep.n_features_}")
    print(sweep.cv_scores_[["mean", "std"]].round(4).to_string())
```

The working script lives at
[`example_feature_count_sweep.py`](https://github.com/martanto/eruption-forecast/blob/ft/feature-count-sweep/example_feature_count_sweep.py)
on the branch. It ships `run_sweep_forecast()` (recommended default),
`run_sweep_cv()` (fast proxy), and single-classifier variants.

---

## Outputs

`sweep_feature_count(save=True)` writes under
`{training_dir}/features/{cv-slug}/sweep/{mode}/{classifier}/` — mode is
part of the path so `forecast/` and `cv/` runs never collide:

```
training/features/{cv-slug}/sweep/{mode}/{classifier-name}/
├── cv_scores.csv          # aggregated summary: N, mean, std, n_seeds
├── cv_scores_raw.csv      # full (N × seed) score matrix
├── seed_argmax_hist.csv   # per-seed argmax N (diagnostic)
├── support.json           # {"n_features": N*, "seeds": {seed: [features...]}}
├── curve.png              # two-panel plot: mean±std curve + argmax histogram
└── FeatureCountSweep.pkl  # full instance, reloadable via FeatureCountSweep.load(...)
```

`FeatureCountSweep.load(path)` restores the full instance so you can
re-plot or re-inspect without re-running the sweep.

---

## Attributes After `fit()`

| Attribute | Type | Notes |
|-----------|------|-------|
| `sweep.cv_scores_` | `pd.DataFrame` | Indexed by `N`; columns `mean`, `std`, `n_seeds` |
| `sweep.cv_scores_raw_` | `pd.DataFrame` | Full `(N × seed)` matrix — the source data for aggregation |
| `sweep.n_features_` | `int` | Recommended `N*` |
| `sweep.seed_argmax_` | `pd.Series` | Per-seed argmax `N` |
| `sweep.support_` | `dict[int, list[str]]` | For each seed, the top-`N*` feature list |
| `sweep.mode` | `str` | `"cv"` or `"forecast"` (echoed back for downstream code) |
| `sweep._classifier_name` | `str \| None` | Informational, set by the wrapper |

---

## Constructor / Function Reference

### `FeatureCountSweep`

```python
FeatureCountSweep(
    estimator: BaseEstimator | None = None,
    *,
    strategy: Literal["shared", "per-seed"] = "per-seed",
    n_candidates: list[int] | None = None,
    cv: BaseCrossValidator | int = 5,
    scoring: str = "average_precision",
    parsimony: bool = True,
    parsimony_tolerance: float | None = None,
    resample_method: Literal["under", "over", "auto"] | None = None,
    minority_threshold: float = 0.15,
    estimator_mode: Literal["default", "tuned"] = "default",
    mode: Literal["cv", "forecast"] = "forecast",
    n_jobs: int = 1,
    random_state: int = 42,
    verbose: bool = False,
)
```

```python
sweep.fit(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    per_seed_inputs: dict[int, dict[str, Any]] | None = None,
    shared_ranking: pd.Series | pd.Index | list[str] | None = None,
    per_seed_rankings: dict[int, ...] | None = None,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> Self
```

`per_seed_inputs` / `shared_ranking` / `per_seed_rankings` are mutually
exclusive. `X_test` / `y_test` are required in `mode="forecast"`.

### `sweep_feature_count`

```python
sweep_feature_count(
    source: str | os.PathLike | TrainingModel,
    *,
    mode: Literal["cv", "forecast"] = "forecast",
    evaluation_source: EvaluationModel | None = None,
    classifier_name: str | None = None,
    n_candidates: list[int] | None = None,
    cv: BaseCrossValidator | int = 5,
    scoring: str = "average_precision",
    parsimony: bool = True,
    parsimony_tolerance: float | None = None,
    resample_method: Literal["under", "over", "auto"] | None = None,
    minority_threshold: float = 0.15,
    estimator_mode: Literal["default", "tuned"] = "default",
    n_jobs: int = 1,
    random_state: int = 42,
    output_dir: str | None = None,
    save: bool = True,
    verbose: bool = False,
) -> FeatureCountSweep | dict[str, FeatureCountSweep]
```

Returns a single `FeatureCountSweep` when `classifier_name` is given,
otherwise a dict keyed by discovered classifier names.

The trained `top_n_features` (peeked from the first per-seed CSV) is
automatically merged into `n_candidates` so the curve always includes
the baseline the operator actually trained at.

---

## Cross-References

- [Training Workflow → Feature Pipeline](Training-Workflow#feature-pipeline-extract_features) — where the per-seed rankings and resampled ids originate.
- [Evaluation Workflow](Evaluation-Workflow) — how `EvaluationModel` builds `y_true` for prediction-reuse mode.
- [Prediction Workflow → Feature Scoping](Prediction-Workflow#feature-scoping-via-use_features_from) — governs which features exist in the held-out matrix and therefore what the sweep can score against.
- [Output Structure → Sweep outputs](Output-Structure#feature-count-sweep-experimental) — full on-disk layout.
