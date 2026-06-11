# Prediction Workflow

The prediction stage runs forecast inference on a future (or held-out) time window using 
a previously trained `ClassifierEnsemble`. No labels are required at this stage - predictions are produced over an evenly-spaced window grid.

Driver: `PredictionModel` (`src/eruption_forecast/model/prediction_model.py`). Wrapped by `ForecastModel.predict(...)`.

---

## Internal Pipeline

```
                ┌──────────────────────────────────────────────┐
                │             PredictionModel                  │
                │   inherits BaseModel + CacheModel            │
                │   self.ClassifierEnsemble loaded eagerly     │
                └──────────────┬───────────────────────────────┘
                               │
                  ┌────────────┼─────────────┐
                  ▼            ▼             ▼
        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
        │ build_label │→ │ extract_    │→ │  forecast   │
        │             │  │ features    │  │             │
        └─────────────┘  └─────────────┘  └─────────────┘
            unlabelled       prediction-      ClassifierEnsemble
            window grid      mode tsfresh     .predict_with_uncertainty
            (id + datetime)  (no relevance    → per-classifier +
                              filtering)        consensus probabilities
```

`ForecastModel.predict(...)` chains all three internally.

---

## Trained Model Sources

`PredictionModel(model=...)` accepts five forms via `ClassifierEnsemble.from_any()`:

| Source | Example |
|--------|---------|
| `ClassifierEnsemble` object | `fm.TrainingModel.ClassifierEnsemble` |
| `SeedEnsemble` object | a single-classifier ensemble (auto-wrapped) |
| Path to `ClassifierEnsemble*.pkl` | `output/.../training/classifiers/ClassifierEnsemble_StratifiedShuffleSplit.pkl` |
| Path to `ClassifierEnsemble*.json` | the JSON registry written next to the `.pkl` |
| Path to a `SeedEnsemble_*.pkl` | bundle for a single classifier |
| Path to a trained-model registry CSV | `trained-model__RandomForestClassifier_...csv` |

When called via `fm.predict(...)`, the in-memory `fm.ClassifierEnsemble` is passed directly - no disk round-trip.

---

## Build Forecast Grid (`build_label`)

Despite the name, prediction labels are placeholders - every `is_erupted` value is `0`. 
The role of `build_label()` here is to lay out the *window grid* the model will score against.

| Param | Type | Notes |
|-------|------|-------|
| `window_step` | `int` | Stride between consecutive forecast windows |
| `window_step_unit` | `"minutes"` \| `"hours"` | A 10-minute step produces 144 forecasts/day |

The grid is cached as `{prediction_dir}/features/features-label_{basename}_step-{N}-{unit}.csv`. 
On a re-run the cached grid is loaded unless `overwrite=True`.

---

## Extract Features (`extract_features`)

Runs the same `TremorMatrixBuilder → FeaturesBuilder` chain as training, 
but in **prediction mode** - no tsfresh relevance filtering, because there are no labels to test relevance against.

`select_tremor_columns`, `save_tremor_matrix_per_method`, and `exclude_features` 
are mirrored from the training call so that the prediction feature columns line up with the 
trained model's input schema. `ForecastModel.predict(...)` forwards 
`self.select_tremor_columns`, `self.save_tremor_matrix_per_method`, and `self.exclude_features` from the upstream `train()` call automatically.

---

## Forecast (`forecast`)

```python
results = ClassifierEnsemble.predict_with_uncertainty(
    X=features_df,
    save=save_seed_result,             # write per-seed CSV under result_dir/{clf}/
    output_dir=result_dir,
    overwrite=overwrite,
)
```

The DataFrame returned by `forecast()` (also stored on `self.results` and `fm.results`) has one row per forecast window and the following columns:

| Column suffix | Meaning |
|---------------|---------|
| `{clf}_eruption_probability` | Mean P(eruption) across the classifier's seeds |
| `{clf}_uncertainty` | Std-dev across the classifier's seeds |
| `{clf}_confidence` | `1 - normalised_uncertainty` |
| `{clf}_prediction` | Binary prediction at the seed-averaged threshold |
| `consensus_eruption_probability` | Mean of `{clf}_eruption_probability` across classifiers |
| `consensus_uncertainty` | Pooled std across classifiers + seeds |
| `consensus_confidence` | `1 - normalised consensus uncertainty` |
| `consensus_prediction` | Binary prediction on the consensus mean |

The result CSV is written at `{station_dir}/result_all_model_predictions_{basename}.csv`.

### Plotting

```
prediction/figures/forecast_{basename}.png       # always
prediction/figures/forecast_{basename}.pdf       # when plot_pdf=True (default)
```

| `forecast(...)` param | Default | Effect |
|-----------------------|---------|--------|
| `save_seed_result` | `True` | Per-seed CSVs under `prediction/results/{clf}/` |
| `plot_threshold` | `0.5` | Horizontal threshold line on the forecast plot |
| `plot_title` | `None` | Optional title |
| `plot_pdf` | `True` | Also save a vector PDF |
| `**plot_kwargs` | - | Forwarded to `eruption_forecast.plots.plot_forecast` - e.g. `eruption_dates=[...]` to render eruption markers |

---

## Cache

`PredictionModel` mixes in `CacheModel`. The cache identity includes:

- NSLC
- tremor DataFrame fingerprint
- **`training_hash`** - the cache hash of the upstream `TrainingModel`
- `start_date`, `end_date`, `window_size`
- `build_label` kwargs (`window_step`, `window_step_unit`)
- `extract_features` kwargs (`select_tremor_columns`, `save_tremor_matrix_per_method`, `exclude_features`)

Threading `training_hash` means re-training automatically invalidates the prediction cache. Hashes land at `{station_dir}/cache/PredictionModel/{hash}.pkl` + `{hash}.params.json`.

---

## Outputs

```
{station_dir}/
├── prediction/
│   ├── features/
│   │   ├── features-label_{basename}_step-{N}-{unit}.csv  # forecast grid
│   │   └── features-matrix_*.csv                           # tsfresh matrix
│   ├── results/{clf-slug}/{seed:05d}.csv                   # per-seed probability (save_seed_result=True)
│   └── figures/forecast_{basename}.{png,pdf}               # forecast plot
├── result_all_model_predictions_{basename}.csv             # top-level results dump
└── cache/PredictionModel/{hash}.pkl                        # CacheModel artefact
```

`fm.PredictionModel.forecast_plot_path` exposes the path to the rendered plot - used by `scenarios.py` to attach the figure to a Telegram notification.

---

## Standalone Use

```python
from eruption_forecast import PredictionModel

pm = (
    PredictionModel(
        model="output/VG.OJN.00.EHZ/training/classifiers/ClassifierEnsemble_StratifiedShuffleSplit.pkl",
        tremor_data="output/VG.OJN.00.EHZ/tremor/VG.OJN.00.EHZ_2025-01-01_2025-12-31.csv",
        start_date="2025-07-27",
        end_date="2025-08-22",
        window_size=2,                 # must match the trained model's window_size
        output_dir="output/VG.OJN.00.EHZ",
        n_jobs=4,
    )
    .build_label(window_step=10, window_step_unit="minutes")
    .extract_features(
        select_tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4", "entropy"],
    )
)

df_forecast = pm.forecast(
    plot_threshold=0.7,
    plot_pdf=True,
    eruption_dates=["2025-08-02", "2025-08-18"],   # → forecast_plots plot kwargs
)

pm.save()                       # → {output_dir}/PredictionModel_{basename}.pkl
print(pm.forecast_plot_path)    # path to the saved plot
print(df_forecast.head())       # 10-minute resolution forecast
```

### Reload a saved `.pkl`

```python
pm = PredictionModel.load("output/VG.OJN.00.EHZ/PredictionModel_2025-07-27_2025-08-22.pkl")
df = pm.results
```

The reloaded `pm.results` is the same DataFrame returned by `forecast()` - no re-inference needed for downstream analysis.
