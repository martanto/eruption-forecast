# Usage

This page walks from a 20-line Quick Start to a fully annotated example mirroring the bundled `main.py`. 
For a per-stage internal tour, see [Pipeline Walkthrough](Pipeline-Walkthrough); for parameter-level detail, see [API Reference](API-Reference).

---

## Quick Start

```python
from eruption_forecast import ForecastModel

fm = ForecastModel(
    station="OJN", channel="EHZ", network="VG", location="00",
    day_to_forecast=2,           # also threaded as window_size into Training/Prediction
    n_jobs=4,
    verbose=True,
)

(
    fm.calculate(
        start_date="2025-01-01", end_date="2025-12-31",
        source="sds", sds_dir=r"D:\Data\OJN",
        methods=["rsam", "dsar", "entropy"],
    )
    .train(
        start_date="2025-01-01", end_date="2025-07-26",
        eruption_dates=["2025-03-20", "2025-04-22", "2025-05-18"],
        window_step=6, window_step_unit="hours",
        classifiers=["rf", "xgb"],
        seeds=25,
    )
    .predict(
        start_date="2025-07-27", end_date="2025-08-22",
        window_step=10, window_step_unit="minutes",
        plot_threshold=0.7,
    )
    .evaluate(model="prediction")
)
```

That's the whole pipeline. The four stages - `calculate → train → predict → evaluate` - each return `self`, so they chain.

---

## Stage Cheat Sheet

| Stage | What it does | Key output (under `{station_dir}/`) |
|-------|--------------|-------------------------------------|
| `calculate()` | Reads SDS/FDSN waveforms, computes RSAM / DSAR / entropy per band | `tremor/{nslc}_{start}_{end}.csv` |
| `train()` | Builds labels → extracts tsfresh features → fits one `SeedEnsemble` per classifier into a `ClassifierEnsemble` | `training/...` + `cache/TrainingModel/` |
| `predict()` | Re-extracts features over the forecast grid → runs ensemble inference | `prediction/...` + `cache/PredictionModel/` |
| `evaluate()` | Re-uses the in-session `TrainingModel` or `PredictionModel` for per-seed metrics + aggregate plots | `evaluation/{training\|prediction}/...` |

All paths root at `{output_dir}/{nslc}/`, where `nslc = "{network}.{station}.{location}.{channel}"` - see [Output Structure](Output-Structure) for the full tree.

---

## Annotated Example (`main.py`)

This is the bundled Research Workflow. Every kwarg below is the actual value in the repo's `main.py`.

```python
from eruption_forecast import ForecastModel
from eruption_forecast.decorators import timer, notify


@timer("Run Forecasting")
@notify("Run Forecasting")           # Telegram push when this function finishes
def main(sds_dir: str, n_jobs: int = 2):
    fm = ForecastModel(
        network="VG",
        station="OJN",
        location="00",
        channel="EHZ",
        day_to_forecast=2,           # window_size used by Training + Prediction
        n_jobs=n_jobs,
        verbose=True,
    )

    # 1. Tremor metrics - RSAM + DSAR + Shannon Entropy
    fm.calculate(
        start_date="2025-01-01",
        end_date="2025-12-31",
        source="sds",
        sds_dir=sds_dir,
        methods=["rsam", "dsar", "entropy"],
        remove_tremor_anomalies=False,
        interpolate=True,            # fill miniSEED gaps linearly
        plot_daily=True,             # save per-day tremor PNG
        save_plot=True,
        plot_overwrite=True,
        overwrite=False,             # skip days already on disk
        n_jobs=n_jobs,
    )

    # 2. Train four classifiers across 25 seeds
    fm.train(
        start_date="2025-01-01",
        end_date="2025-07-26",
        classifiers=["lite-rf", "rf", "gb", "xgb"],
        eruption_dates=[
            "2025-03-20", "2025-04-10", "2025-04-22", "2025-05-18",
            "2025-06-17", "2025-07-07", "2025-08-02", "2025-08-18",
        ],
        window_step=6, window_step_unit="hours",
        label_builder="standard",
        cv_strategy="shuffle-stratified",
        scoring="recall",                  # maximise eruption recall during GridSearchCV
        select_tremor_columns=[            # restrict tsfresh input columns
            "rsam_f2", "rsam_f3", "rsam_f4",
            "dsar_f3-f4", "entropy",
        ],
        exclude_features=[                 # drop slow / collinear tsfresh kinds
            "agg_linear_trend", "linear_trend_timewise", "length",
            "has_duplicate_max", "has_duplicate_min", "has_duplicate",
        ],
        seeds=25,
        resample_method="under",           # RandomUnderSampler for class imbalance
        plot_features=True,                # feature-importance PNG per seed
        n_jobs=4, n_grids=4,               # 4 seed workers × 4 GridSearchCV workers
    )

    # 3. Forecast the next 4 weeks at 10-minute resolution
    fm.predict(
        start_date="2025-07-27",
        end_date="2025-08-22",
        window_step=10, window_step_unit="minutes",
        save_seed_result=True,             # write per-seed probability CSV
        plot_threshold=0.7,                # forecast plot threshold
        use_cache=False,                   # ignore any earlier cached prediction
        verbose=True,
    )

    # 4. Evaluate the forecast against ground truth
    fm.evaluate(model="prediction", plot_per_seed=True)

    # 5. Cross-classifier ranking
    if fm.EvaluationModel:
        comparator = fm.EvaluationModel.compare()
        comparator.get_ranking()
        comparator.plot_all()


if __name__ == "__main__":
    main(sds_dir=r"D:\Data\OJN", n_jobs=8)
```

### Why these parameters?

| Parameter | Why |
|-----------|-----|
| `day_to_forecast=2` | Mark the 2-day window before each eruption as positive; doubles as the tsfresh `window_size` |
| `window_step=6, "hours"` *(train)* | Coarse stride keeps the labelled set manageable while preserving multiple windows per eruption |
| `window_step=10, "minutes"` *(predict)* | Dense stride during forecasting - 144 forecasts/day |
| `scoring="recall"` | False negatives are far more costly than false positives in eruption forecasting |
| `select_tremor_columns=[...]` | High-frequency RSAM bands and entropy carry most of the precursor signal at OJN |
| `resample_method="under"` | Eruption-positive windows are < 5 % of the training set → undersample the majority |
| `n_jobs=4, n_grids=4` | 4 outer seed workers × 4 inner CV workers ≈ 16 cores busy on a 16-core box |
| `plot_threshold=0.7` | A 0.7 probability cut-off mirrors the operational alert threshold at the observatory |

---

## Accessing Results

After the pipeline runs, every artefact is reachable from the `fm` object:

```python
fm.tremor_df                  # pd.DataFrame - merged tremor CSV
fm.TrainingModel              # TrainingModel - labels, features, fit state
fm.ClassifierEnsemble         # ClassifierEnsemble - fitted across classifiers + seeds
fm.PredictionModel            # PredictionModel - forecast grid + caching state
fm.results                    # pd.DataFrame - per-window forecast probabilities
fm.EvaluationModel            # EvaluationModel - matrix CSVs + in-memory metrics
fm.evaluation_results         # dict[classifier_name, pd.DataFrame] - per-seed metrics
fm.ExplanationModel           # ExplanationModel (after .explain()) - SHAP payloads
fm.ExplanationModel.explanations  # list[ClassifierExplanation] - per-classifier SHAP
```

Probability columns in `fm.results`:

```
{clf}_eruption_probability     # mean across seeds
{clf}_uncertainty              # std across seeds
{clf}_confidence               # 1 - normalised uncertainty
{clf}_prediction               # binary at plot_threshold
consensus_eruption_probability # mean across classifiers
consensus_*                    # same suite at the consensus level
```

---

## Common Variants

### FDSN instead of SDS

```python
fm.calculate(
    start_date="2025-01-01", end_date="2025-12-31",
    source="fdsn",
    client_url="https://service.iris.edu",   # any FDSN endpoint
    methods=["rsam", "dsar", "entropy"],
)
```

Downloads are cached locally as SDS - see [Data Sources](Data-Sources#fdsn--web-service-with-local-cache).

### Skip recomputation by replaying a saved config

```python
fm.save_config()                                # {station_dir}/forecast.config.yaml
# ...later...
fm2 = ForecastModel.from_config("output/VG.OJN.00.EHZ/forecast.config.yaml")
fm2.run()                                       # idempotently replays every captured stage
```

See [Configuration](Configuration) for the YAML schema.

### Persist intermediate stage objects

```python
fm.TrainingModel.save()        # → {station_dir}/TrainingModel_{basename}.pkl
fm.PredictionModel.save()      # → {station_dir}/PredictionModel_{basename}.pkl
fm.EvaluationModel.save()      # → {station_dir}/EvaluationModel_{basename}.pkl
```

### Per-stage config snapshots

Each stage model also auto-saves its own `*.config.yaml` at the end of its main 
run method, so a standalone run leaves a YAML snapshot next to its artefacts 
without any extra wiring:

```
{station_dir}/training/training.config.yaml         # auto at end of fit()
{station_dir}/prediction/prediction.config.yaml     # auto at end of forecast()
{station_dir}/evaluation/{kind}/evaluation.config.yaml  # auto at end of evaluate()
{station_dir}/explanation/{kind}/explanation.config.yaml # auto at end of explain()
```

Call `tm.save_config(path=..., fmt="json")` etc. manually for a custom path or 
JSON output. See [Configuration](Configuration#per-stage-configs-standalone).

### Standalone evaluation from a saved `.pkl`

```python
from eruption_forecast import EvaluationModel

em = EvaluationModel.from_file(
    "output/VG.OJN.00.EHZ/PredictionModel_2025-07-27_2025-08-22.pkl",
    eruption_dates=["2025-08-02", "2025-08-18"],
)
em.evaluate(plot_aggregate=True)
em.compare().plot_all()
```

### Loop over multiple training/prediction splits

Use `scenarios.py` - see [Pipeline Walkthrough → Scenarios Workflow](Pipeline-Walkthrough#scenarios-workflow-scenariospy).
