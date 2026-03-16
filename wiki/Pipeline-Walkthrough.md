# Pipeline Walkthrough

The pipeline transforms raw seismic waveforms into probabilistic eruption forecasts through six processing stages followed by a forecast stage. `ForecastModel` orchestrates all stages via method chaining — you invoke them in sequence on a single object. You can also instantiate and call each stage class independently when you need finer control over intermediate outputs or want to slot the package into a larger workflow.

---

## Stage 1 — Calculate Tremor Metrics

Raw seismic waveforms are too high-dimensional and noisy to use directly as model input. `CalculateTremor` compresses each day of continuous waveform data into compact, interpretable metrics sampled at 10-minute intervals. Three metrics are calculated across multiple frequency bands in parallel:

- **RSAM** (Real Seismic Amplitude Measurement): The mean absolute amplitude within each frequency band. Rising RSAM in mid-to-high frequency bands is a classical precursor to volcanic unrest.
- **DSAR** (Displacement Seismic Amplitude Ratio): The ratio of RSAM between consecutive frequency bands. DSAR captures shifts in the spectral shape of tremor, which correlate with changes in the volcanic fluid system.
- **Shannon Entropy**: A single broadband measure of signal complexity. Low entropy indicates a repetitive, coherent signal; high entropy indicates random broadband noise.

### Frequency Band Naming

The default frequency bands and their aliases are:

| Alias | Band (Hz) |
|-------|-----------|
| `f0`  | 0.01 – 0.1 |
| `f1`  | 0.1 – 2    |
| `f2`  | 2 – 5      |
| `f3`  | 4.5 – 8    |
| `f4`  | 8 – 16     |

RSAM columns are named `rsam_f0` through `rsam_f4`. DSAR columns carry the names of both participating bands: `dsar_f0-f1`, `dsar_f1-f2`, `dsar_f2-f3`, `dsar_f3-f4`. Shannon Entropy produces a single column named `entropy`.

### Reading from a Local SDS Archive

SDS (SeisComP Data Structure) is the primary on-disk waveform format. Pass the root of the archive to `from_sds()`:

```python
from eruption_forecast import CalculateTremor

tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
    n_jobs=4,
).from_sds(sds_dir="/data/sds").run()

# Access the resulting DataFrame
print(tremor.df.head())

# Inspect where the merged CSV was saved
print(f"Saved to: {tremor.csv}")
```

You can restrict which metrics are calculated:

```python
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
    methods=["rsam", "dsar", "entropy"],  # default: all three
).from_sds(sds_dir="/data/sds").run()
```

### Reading from an FDSN Web Service

When a local archive is unavailable, `from_fdsn()` downloads waveforms from any FDSN-compatible service and caches them locally as SDS miniSEED so subsequent runs skip the network:

```python
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
).from_fdsn(client_url="https://service.iris.edu").run()
```

### Custom Frequency Bands

Call `change_freq_bands()` before `from_sds()` or `from_fdsn()` to override the defaults:

```python
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
).change_freq_bands([
    (0.1, 1.0),   # Low frequency
    (1.0, 5.0),   # Mid frequency
    (5.0, 10.0),  # High frequency
]).from_sds(sds_dir="/data/sds").run()
```

### Output Format

- DateTime index at 10-minute intervals
- RSAM columns: `rsam_f0`, `rsam_f1`, `rsam_f2`, `rsam_f3`, `rsam_f4`
- DSAR columns: `dsar_f0-f1`, `dsar_f1-f2`, `dsar_f2-f3`, `dsar_f3-f4`
- Shannon Entropy column: `entropy`

---

## Stage 2 — Build Training Labels

Before a model can learn, every time window of tremor data must be assigned a binary label: `1` if the window precedes an eruption, `0` otherwise. `LabelBuilder` generates this labeling by sliding a window across the full date range and assigning labels based on known eruption dates.

### The Labeling Zone Logic

A window gets `is_erupted = 1` when its **end time** falls within the interval `[eruption_date − day_to_forecast, eruption_date]`. All other windows get `is_erupted = 0`. This design captures the pre-eruptive signal window you want the model to recognise — for example, setting `day_to_forecast=2` tells the labeler to mark the 48 hours leading into each eruption as positive.

### Code Example

```python
from eruption_forecast import LabelBuilder

labels = LabelBuilder(
    start_date="2020-01-01",
    end_date="2020-12-31",
    window_step=12,          # Slide by 12 hours
    window_step_unit="hours",
    day_to_forecast=2,       # Label 2 days before eruption as positive
    eruption_dates=[
        "2020-03-15",
        "2020-06-20",
        "2020-09-10",
    ],
    volcano_id="VOLCANO_001",
).build()

# Access the DataFrame
print(labels.df.head())
# Columns: id, is_erupted

# Check label distribution
print(f"Positive labels: {(labels.df['is_erupted'] == 1).sum()}")
print(f"Negative labels: {(labels.df['is_erupted'] == 0).sum()}")
```

### Labeling Strategy Visualization

Example with `window_step=12h`, `day_to_forecast=2d`, eruption on January 15.

```
Timeline (each tick = 12 hours):

── Jan10 ───┬─── Jan11 ───┬─── Jan12 ───┬─── Jan13 ───┬─── Jan14 ───┬─── Jan15 ───┐ ☄
 00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │ 
     │      │      │      │      │      │      │      │      │      │      │      │
      ← window_step: 12h →              │      │      │      │      │      │      │
                                        │◄──────── day_to_forecast=2d ───────────►│
                                        │             label = 1 zone              │
```

```
 ID  Window data span                     End time (index)    Label
 ──  ──────────────────────────────────── ──────────────────  ──────
  1  Jan09·12:00 ══════════ Jan10·12:00   Jan10 12:00           0
  2  Jan10·00:00 ══════════ Jan11·00:00   Jan11 00:00           0
  3  Jan10·12:00 ══════════ Jan11·12:00   Jan11 12:00           0
  4  Jan11·00:00 ══════════ Jan12·00:00   Jan12 00:00           0
  5  Jan11·12:00 ══════════ Jan12·12:00   Jan12 12:00           0
  6  Jan12·00:00 ══════════ Jan13·00:00   Jan13 00:00           1  ← label zone starts
  7  Jan12·12:00 ══════════ Jan13·12:00   Jan13 12:00           1
  8  Jan13·00:00 ══════════ Jan14·00:00   Jan14 00:00           1
  9  Jan13·12:00 ══════════ Jan14·12:00   Jan14 12:00           1
 10  Jan14·00:00 ══════════ Jan15·00:00   Jan15 00:00           1  ← eruption day
```

> The window's datetime index is its end time. A window gets `label=1` when its end time falls within `[eruption_date − day_to_forecast, eruption_date]`.

### Key Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `window_step` | `int` | How far to shift the window between consecutive labels | `12` |
| `window_step_unit` | `"minutes"` \| `"hours"` | Unit for `window_step` | `"hours"` |
| `day_to_forecast` | `int` (days) | How many days before the eruption to start labeling as positive (`is_erupted=1`) | `2` |
| `eruption_dates` | `list[str]` | Known eruption dates in `YYYY-MM-DD` format | `["2025-03-20"]` |
| `start_date` / `end_date` | `str` | Date range for generating all label windows | `"2025-01-01"` |
| `volcano_id` | `str` | Identifier used in output filenames | `"LEWOTOBI"` |

---

## Stage 3 — Build Tremor Matrix

Labels define which time windows are positive and which are negative, but the model needs the actual tremor values inside each window. `TremorMatrixBuilder` slices the tremor time series into windows that align exactly with the label DataFrame, then concatenates them into a single unified matrix.

Every row in the output matrix corresponds to one 10-minute sample inside one window. The matrix carries an `id` column (window identifier matching the label), a `datetime` column, and the selected tremor columns. This format is what tsfresh expects: all windows stacked vertically, identified by `id`.

### Code Example

```python
from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder

builder = TremorMatrixBuilder(
    tremor_df=tremor.df,
    label_df=labels.df,
    output_dir="output/features",
    window_size=1,
).build(
    select_tremor_columns=["rsam_f0", "rsam_f1", "dsar_f0-f1"],
    save_tremor_matrix_per_method=True,
)

print(builder.df.shape)  # (n_windows × n_samples_per_window, n_columns)
```

### Output Shape

The shape is `(n_windows × n_samples_per_window, n_columns)`. For a one-day window at 10-minute sampling, each window contains 144 samples. With 500 windows and three tremor columns selected, `builder.df` will have shape `(72 000, 5)` — the five columns being `id`, `datetime`, and the three tremor columns.

Use `select_tremor_columns` to limit which tremor columns are carried into feature extraction. Including only the columns relevant to your classification task reduces computation time in Stage 4 significantly.

---

## Stage 4 — Extract Features

Raw tremor amplitude values at each time step are unlikely to be the most discriminating input for a classifier. `FeaturesBuilder` applies tsfresh to extract over 700 statistical and temporal features per tremor column per window. These features describe properties of the tremor signal that a classifier can act on directly.

### What tsfresh Computes

Feature categories extracted per column per window:

- **Statistical**: mean, median, standard deviation, variance, min, max, quantiles, skewness, kurtosis
- **Time-domain**: autocorrelation, partial autocorrelation, number of peaks, peak positions, linear trend coefficients
- **Frequency-domain**: FFT coefficients, spectral entropy, spectral centroid
- **Complexity**: approximate entropy, sample entropy, Lempel-Ziv complexity
- **Peaks**: number of peaks above mean, number of peaks above a threshold

### Training Mode vs Prediction Mode

`FeaturesBuilder` operates in two modes depending on whether labels are provided:

- **Training mode** (labels provided): Filters windows to match the label DataFrame, saves an aligned label CSV alongside the features CSV, and optionally applies tsfresh's relevance filter to discard statistically insignificant features.
- **Prediction mode** (no labels): Extracts features for all windows without filtering. Used when running inference on future data that has no ground-truth labels.

### Code Example

```python
from eruption_forecast import FeaturesBuilder

features_builder = FeaturesBuilder(
    tremor_matrix_df=builder.df,
    label_df=labels.df,
    output_dir="output/features",
    n_jobs=4,
)

# Extract features; optionally exclude specific feature calculators
features = features_builder.extract_features(
    select_tremor_columns=["rsam_f0", "rsam_f1"],
    exclude_features=["length", "has_duplicate"],
)

print(f"Features shape: {features.shape}")
# Example: (5000 windows, 1500 features)
```

---

## Stage 5 — Select Features

With 700+ features per tremor column, the raw feature matrix is wide. Many features will be noise or redundant. `FeatureSelector` narrows the matrix down to the most informative features before they are passed to the classifier.

Three selection methods are available, each with different trade-offs:

| Method | Reduces | Captures Interactions | Speed |
|--------|---------|-----------------------|-------|
| `tsfresh` | 1000s → 100s | No | Fast |
| `random_forest` | Direct → N | Yes | Slow |
| `combined` | 1000s → 100s → N | Yes | Fast |

- **`tsfresh`**: Applies a statistical hypothesis test (Benjamini-Hochberg FDR correction) to retain only features with a significant relationship to the label. Fast and principled, but treats each feature independently.
- **`random_forest`**: Trains a Random Forest and ranks features by permutation importance. Captures feature interactions but is slower at wide feature matrices.
- **`combined`** (recommended): Runs `tsfresh` first to coarsely prune the feature set, then applies `random_forest` on the reduced matrix. Achieves the interaction-capturing benefit of Random Forest at reasonable speed.

### Code Examples

```python
from eruption_forecast.features import FeatureSelector

# Two-stage combined selection (recommended)
selector = FeatureSelector(method="combined", n_jobs=4, verbose=True)
X_selected = selector.fit_transform(
    X_train, y_train,
    fdr_level=0.05,  # Stage 1: tsfresh FDR threshold
    top_n=30,        # Stage 2: final feature count
)
print(f"Reduced: {X_train.shape[1]} → {X_selected.shape[1]} features")

# tsfresh statistical selection only
selector = FeatureSelector(method="tsfresh", n_jobs=4)
X_selected = selector.fit_transform(X_train, y_train, fdr_level=0.05)

# RandomForest permutation importance only
selector = FeatureSelector(method="random_forest", n_jobs=4)
X_selected = selector.fit_transform(X_train, y_train, top_n=30)

# Retrieve feature scores after fitting
scores = selector.get_feature_scores()
print(scores.head(10))
```

Feature selection runs inside `ModelTrainer` automatically — you do not need to call `FeatureSelector` directly unless you are working outside the standard training pipeline.

---

## Stage 6 — Train Models

`ModelTrainer` trains a classifier across multiple random seeds to produce a robust ensemble of models. Because volcanic datasets are heavily imbalanced (eruption windows are rare), each seed independently resamples the training data with `RandomUnderSampler` before fitting.

### Two Training Workflows

Two workflows are available depending on your evaluation strategy:

```
   evaluate()                  train()
  ─────────────────────            ────────────────────
      Full Dataset                      Full Dataset
           │                                │
           ▼                                ▼
      80/20 Split                    RandomUnderSampler
      (stratified)                     (full dataset)
      ┌────┴────┐                           │
    Train     Test                   Feature Selection
      │         │                      (full dataset)
    RandomUnder │                           │
    Sampler     │                      GridSearchCV
      │         │                       + CV folds
    Feature     │                           │
    Selection   │                    ┌──────┴──────┐
      │         │                model.pkl   registry.csv
    GridSearchCV│
    + CV folds  │
      │         │
    Evaluate ◄──┘
      │
    Save model + metrics
```

- **`evaluate()`** — splits the full dataset 80/20 internally. Trains on the 80% split and evaluates each seed on the held-out 20%. Use this when you want per-seed metrics and an in-sample accuracy estimate to compare classifiers or tune hyperparameters.
- **`train()`** — treats the entire dataset as the training set with no internal split. Evaluation is deferred to `ModelPredictor` using a separate future dataset. Use this for final production models where no data should be withheld.

For full detail on both workflows, see the [Training Workflows](Training-Workflows) wiki page.

### `evaluate()` — with held-out test set

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

### `train()` — full dataset training

```python
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

`fit()` is a unified entry point that dispatches to either workflow based on the `with_evaluation` flag:

```python
# Equivalent to evaluate()
trainer.fit(with_evaluation=True, random_state=0, total_seed=500, sampling_strategy=0.75)

# Equivalent to train()
trainer.fit(with_evaluation=False, random_state=0, total_seed=500, sampling_strategy=0.75)
```

`ForecastModel.train()` calls `fit()` internally and exposes `with_evaluation` as a direct parameter, so you can control the workflow from the high-level API without dropping down to `ModelTrainer`.

---

## Stage 7 — Forecast

`ModelPredictor` applies trained models to new data and produces per-window eruption probability estimates. It supports two modes:

- **Evaluation mode** (`predict()` / `predict_best()`): Requires ground-truth labels. Evaluates each seed model against known eruption labels and aggregates metrics across seeds. Used when you have a labelled future dataset and want to measure out-of-sample performance.
- **Forecast mode** (`predict_proba()`): No labels required. Produces a time series of eruption probabilities. Used for operational forecasting when ground truth is unavailable.

Both modes support single-model and multi-model consensus inference.

### Evaluation Mode (labelled data)

```python
from eruption_forecast.model.model_predictor import ModelPredictor

predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models=trainer.csv,  # trained_model_*.csv from train()
    output_dir="output/predictions",
)

# Metrics for every (classifier, seed) combination
df_metrics = predictor.predict(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
)
print(df_metrics[["balanced_accuracy", "f1_score"]].describe())

# Best single seed by a chosen criterion
evaluator = predictor.predict_best(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
    criterion="balanced_accuracy",
)
print(evaluator.summary())
evaluator.plot_all()
```

`predict_best()` accepts any metric column as `criterion`: `"accuracy"`, `"balanced_accuracy"`, `"f1_score"`, `"precision"`, `"recall"`, `"roc_auc"`, `"pr_auc"`.

### Forecast Mode — Single Model (unlabelled data)

```python
predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models=trainer.csv,
    output_dir="output/predictions",
)

df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",  # or pd.DataFrame
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

### Forecast Mode — Multi-Model Consensus

Pass a dict of model registry paths to aggregate across classifiers. `predict_proba()` first aggregates within each classifier across seeds, then averages across classifiers to produce a consensus probability:

```python
predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models={
        "rf":  "output/VG.OJN.00.EHZ/trainings/predictions/random-forest-classifier/stratified-shuffle-split/trained_model_RandomForestClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv",
        "xgb": "output/VG.OJN.00.EHZ/trainings/predictions/xgb-classifier/stratified-shuffle-split/trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv",
    },
    output_dir="output/predictions",
)

df_forecast = predictor.predict_proba(
    tremor_data="path/to/tremor.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

**Output columns (multi-model):**

| Column | Description |
|--------|-------------|
| `{name}_eruption_probability` | Mean P(eruption) across seeds of that classifier |
| `{name}_uncertainty` | Std across seeds of that classifier |
| `{name}_confidence` | Seed-level agreement fraction (0.5–1.0) |
| `{name}_prediction` | Hard label for that classifier |
| `consensus_eruption_probability` | Mean P(eruption) averaged across all classifiers |
| `consensus_uncertainty` | Std of per-classifier means (inter-model disagreement) |
| `consensus_confidence` | Fraction of classifiers voting with consensus majority |
| `consensus_prediction` | Hard label — `1` if `consensus_eruption_probability >= 0.5` |

Results are saved to `predictions.csv`. The plot shows each classifier as a dashed line and the consensus as a solid black line with a shaded uncertainty band (`eruption_forecast.png` in `figures/`).

---

## Running the Full Pipeline with ForecastModel

`ForecastModel` wires all stages together with a fluent method-chaining API. The `root_dir` parameter anchors all output paths.

```python
from eruption_forecast import ForecastModel

# Initialize with station metadata and global parameters
fm = ForecastModel(
    root_dir="output",
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=2,
    volcano_id="Lewotobi Laki-laki",
    n_jobs=4,
    verbose=True,
)

# Run the complete pipeline with method chaining
fm.calculate(
    source="sds",
    sds_dir="/path/to/sds/data",
    methods=["rsam", "dsar", "entropy"],
    plot_daily=True,
    save_plot=True,
    remove_outlier_method="maximum",
).build_label(
    start_date="2025-01-01",
    end_date="2025-07-24",
    day_to_forecast=2,
    window_step=6,
    window_step_unit="hours",
    eruption_dates=[
        "2025-03-20",
        "2025-04-22",
        "2025-05-18",
        "2025-06-17",
        "2025-07-07",
    ],
).extract_features(
    select_tremor_columns=["rsam_f2", "rsam_f3", "rsam_f4", "dsar_f3-f4", "entropy"],
    save_tremor_matrix_per_method=True,
    exclude_features=["agg_linear_trend", "linear_trend_timewise", "length"],
    use_relevant_features=True,
).train(
    classifier="rf",
    cv_strategy="stratified",
    random_state=0,
    total_seed=500,
    with_evaluation=False,
    number_of_significant_features=20,
    sampling_strategy=0.75,
    save_all_features=True,
    plot_significant_features=True,
).forecast(
    start_date="2025-07-28",
    end_date="2025-08-04",
    window_step=10,
    window_step_unit="minutes",
)
```

**What this pipeline does:**

1. **Calculate tremor** — computes RSAM, DSAR, and Shannon Entropy from raw SDS waveforms with maximum-outlier removal and daily plots saved
2. **Build labels** — creates binary labels for January 1 through July 24, marking 2-day windows before each known eruption as positive
3. **Extract features** — builds the tremor matrix for selected columns, runs tsfresh, and retains statistically relevant features
4. **Train models** — trains a Random Forest across 500 random seeds on the full dataset (`with_evaluation=False`), saving models and significant feature lists
5. **Forecast** — runs the trained ensemble on July 28 through August 4 and writes a probability time series with a consensus plot

If pre-computed tremor data already exists, skip the `calculate()` call:

```python
fm.load_tremor_data(
    tremor_csv="output/VG.OJN.00.EHZ/tremor/tremor_2025-01-01_2025-12-31.csv"
).build_label(...).extract_features(...).train(...).forecast(...)
```

See `main.py` in the repository root for a complete working example.

---

## Optional Stage — Merge Seed Models

After training 500 seeds, each estimator and its feature list live in separate files on disk. Calling `merge_models()` collapses them all into a single `SeedEnsemble` pkl, removing the per-seed I/O overhead at prediction time.

```
  500 × models/00000.pkl
  500 × features/significant_features/00000.csv
           │
           ▼  trainer.merge_models()
           │
  merged_model_RandomForestClassifier-StratifiedKFold_rs-0_ts-500_top-20.pkl
  (SeedEnsemble — one object, one file)
```

```python
from eruption_forecast.model.seed_ensemble import SeedEnsemble

# Merge right after training
merged_path = trainer.merge_models()

# Load and predict
ensemble = SeedEnsemble.load(merged_path)

mean_p, std, confidence, prediction = ensemble.predict_with_uncertainty(
    features_df,
    threshold=0.7,
)
print(f"Mean P(eruption):    {mean_p.mean():.4f}")
print(f"Mean confidence:     {confidence.mean():.4f}")
print(f"Eruption windows:    {prediction.sum()}")

# sklearn-compatible interface
proba = ensemble.predict_proba(features_df)   # (n_windows, 2)
```

**Multiple classifiers** can be bundled into one file:

```python
bundle_path = trainer.merge_classifier_models(
    {"rf": rf_trainer.csv, "xgb": xgb_trainer.csv}
)
```

Pass a merged pkl directly to `ModelPredictor` — `.pkl` vs `.csv` is detected automatically:

```python
from eruption_forecast.model.model_predictor import ModelPredictor

predictor = ModelPredictor(
    start_date="2025-07-28",
    end_date="2025-08-04",
    trained_models=merged_path,   # or bundle_path for multi-classifier
)
df_forecast = predictor.predict_proba(
    tremor_data="output/VG.OJN.00.EHZ/tremor/tremor_*.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
```

For full details see [Training Workflows](Training-Workflows) and [Evaluation and Forecasting](Evaluation-and-Forecasting).
