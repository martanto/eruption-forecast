# Quick Start

This page shows a complete end-to-end example — from raw seismic data to trained models and eruption forecasts — using the high-level `ForecastModel` API.

For per-stage detail, see the [Pipeline Walkthrough](Pipeline-Walkthrough) page.

---

## Prerequisites

- Package installed and dependencies synced (`uv sync`)
- A seismic data archive in [SDS format](Data-Sources) **or** access to an FDSN web service
- Known eruption dates for your volcano

---

## Complete Example

```python
from eruption_forecast import ForecastModel

# 1. Initialise with station metadata and time range
fm = ForecastModel(
    root_dir="output",               # All outputs anchored here
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",         # Training period start
    end_date="2025-12-31",           # Training period end
    window_size=2,                   # 2-day tremor windows for feature extraction
    volcano_id="Lewotobi Laki-laki",
    n_jobs=4,
    verbose=True,
)

# 2. Calculate tremor metrics from raw seismic data
fm.calculate(
    source="sds",
    sds_dir="/path/to/sds/data",
    methods=["rsam", "dsar", "entropy"],
    plot_daily=True,
    save_plot=True,
    remove_outlier_method="maximum",

# 3. Build binary labels from known eruption dates
).build_label(
    start_date="2025-01-01",
    end_date="2025-07-24",           # Label window ends before last eruption
    day_to_forecast=2,               # Mark windows 2 days before eruption as positive
    window_step=6,
    window_step_unit="hours",
    eruption_dates=[
        "2025-03-20",
        "2025-04-22",
        "2025-05-18",
        "2025-06-17",
        "2025-07-07",
    ],

# 4. Extract 700+ tsfresh features from tremor windows
).extract_features(
    select_tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4", "entropy"],
    save_tremor_matrix_per_method=True,
    exclude_features=["agg_linear_trend", "linear_trend_timewise", "length"],
    use_relevant_features=True,

# 5. Train Random Forest ensemble across 500 random seeds
).train(
    classifier="rf",
    cv_strategy="stratified",
    random_state=0,
    total_seed=500,
    number_of_significant_features=20,
    sampling_strategy=0.75,
    save_all_features=True,
    plot_significant_features=True,

# 6. Forecast eruption probability for a future period
).forecast(
    start_date="2025-07-28",
    end_date="2025-08-04",
    window_step=10,
    window_step_unit="minutes",
)
```

---

## What Happens at Each Step

| Step | What it does | Key output |
|------|-------------|------------|
| `calculate()` | Reads SDS/FDSN waveforms, computes RSAM, DSAR, and Shannon Entropy per frequency band | `tremor/tremor_*.csv` |
| `build_label()` | Slides windows over the date range, labels windows near eruptions as positive | `label_*.csv` |
| `extract_features()` | Aligns tremor to label windows, runs tsfresh, selects top features | `features/all_extracted_features_*.csv` |
| `train()` | Resamples for class balance, runs GridSearchCV per seed, saves each model | `trainings/predictions/.../` |
| `forecast()` | Extracts features from future tremor, loads all 500 models, outputs probability per window | `forecast/predictions.csv` |

---

## Accessing Results

```python
print(fm.trained_model_csv)    # Path to the model registry CSV
print(fm.predictions_csv)      # Path to the forecast predictions CSV
print(fm.forecast_plot_path)   # Path to the eruption probability plot
```

---

## Common Variants

### Use FDSN instead of SDS

```python
fm.calculate(
    source="fdsn",
    client_url="https://service.iris.edu",
)
```

Downloaded miniSEED files are cached locally as SDS so subsequent runs skip the network.

### Skip calculation if tremor data already exists

```python
fm.load_tremor_data(
    tremor_csv="output/VG.OJN.00.EHZ/tremor/tremor_2025-01-01_2025-12-31.csv"
).build_label(...).extract_features(...).train(...).forecast(...)
```

### Train multiple classifiers and run consensus forecasting

```python
fm.train(
    classifier=["rf", "xgb", "gb"],
    cv_strategy="stratified",
    total_seed=500,
).forecast(
    start_date="2025-07-28",
    end_date="2025-08-04",
    window_step=10,
    window_step_unit="minutes",
)
```

### Save and replay the pipeline

```python
fm.save_config()   # Writes output/VG.OJN.00.EHZ/config.yaml

# Later — replay every stage from the saved config
fm2 = ForecastModel.from_config("output/VG.OJN.00.EHZ/config.yaml")
fm2.run()
```

See [Configuration](Configuration) for full save/replay details.
