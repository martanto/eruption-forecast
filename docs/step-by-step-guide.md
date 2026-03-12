# Step-by-Step Usage Guide

Detailed usage examples for each pipeline stage.

> Back to [README](../README.md)

### 1. Calculate Tremor Metrics

Process raw seismic data to calculate RSAM (amplitude), DSAR (ratios), and Shannon Entropy across frequency bands.

```python
from eruption_forecast import CalculateTremor

# From a local SDS archive (all three metrics enabled by default)
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
    n_jobs=4,
).from_sds(sds_dir="/data/sds").run()

# Select specific metrics
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
    methods=["rsam", "dsar", "entropy"],  # default: all three
).from_sds(sds_dir="/data/sds").run()

# From an FDSN web service (downloads and caches locally as SDS)
tremor = CalculateTremor(
    start_date="2025-01-01",
    end_date="2025-01-31",
    station="OJN",
    channel="EHZ",
).from_fdsn(client_url="https://service.iris.edu").run()

# Custom frequency bands (works with both SDS and FDSN)
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

# Access the DataFrame
print(tremor.df.head())
# Output columns: rsam_f0, rsam_f1, rsam_f2, dsar_f0-f1, dsar_f1-f2, entropy

# Get file path
print(f"Saved to: {tremor.csv}")
```

**Output format:**
- DateTime index with 10-minute intervals
- RSAM columns: `rsam_f0`, `rsam_f1`, `rsam_f2`, `rsam_f3`, `rsam_f4`
- DSAR columns: `dsar_f0-f1`, `dsar_f1-f2`, `dsar_f2-f3`, `dsar_f3-f4`
- Shannon Entropy: `entropy` (signal complexity, single broadband column)
- Default bands: (0.01–0.1), (0.1–2), (2–5), (4.5–8), (8–16) Hz

### 2. Build Training Labels

Create binary labels (erupted/not erupted) based on known eruption dates.

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

**Label logic:**
- Windows whose **end time** falls within `[eruption_date − day_to_forecast, eruption_date]` are labeled `is_erupted = 1`
- All other windows: `is_erupted = 0`
- Each window's datetime index is its **end time** (tremor data for the preceding `window_size` days)

#### Labeling Strategy Visualization

Example with `window_step=12h`, `day_to_forecast=2d`, `eruption=Jan 15`.

```
Timeline (each tick = 12 hours):

──── Jan10 ──────── Jan11 ──────── Jan12 ──────── Jan13 ──────── Jan14 ────── Jan15  ☄
 00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │  00  │  12  │  00
     │      │      │      │      │      │      │      │      │      │      │      │
     ← window_step: 12h →        │      │      │      │      │      │      │      │
                                 │      ◄───────── day_to_forecast=2d ───────────►│
                                 │                    label = 1 zone              │
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

> **Key:** The window's datetime index = its **end time**. A window gets `label=1` when its
> end time falls within `[eruption_date − day_to_forecast, eruption_date]`.

**Parameter reference:**

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `window_step` | `int` | How far to shift the window between consecutive labels | `12` |
| `window_step_unit` | `"minutes"` \| `"hours"` | Unit for `window_step` | `"hours"` |
| `day_to_forecast` | `int` (days) | How many days before the eruption to start labeling as positive (`is_erupted=1`) | `2` |
| `eruption_dates` | `list[str]` | Known eruption dates in `YYYY-MM-DD` format | `["2025-03-20"]` |
| `start_date` / `end_date` | `str` | Date range for generating all label windows | `"2025-01-01"` |
| `volcano_id` | `str` | Identifier used in output filenames | `"LEWOTOBI"` |

### 3. Build Tremor Matrix

Align tremor time-series with label windows to create a unified feature matrix.

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

### 4. Extract Time-Series Features

Extract 700+ statistical features from tremor data using tsfresh.

```python
from eruption_forecast import FeaturesBuilder

features_builder = FeaturesBuilder(
    tremor_matrix_df=builder.df,
    label_df=labels.df,
    output_dir="output/features",
    n_jobs=4,
)

# Extract all features
features = features_builder.extract_features(
    select_tremor_columns=["rsam_f0", "rsam_f1"],
    exclude_features=["length", "has_duplicate"],
)

print(f"Features shape: {features.shape}")
# Example: (5000 windows, 1500 features)
```

**Feature types extracted:**
- Statistical: mean, median, std, variance, min, max, quantiles
- Time-domain: autocorrelation, partial autocorrelation
- Frequency-domain: FFT coefficients, spectral entropy
- Complexity: approximate entropy, sample entropy
- Peaks: number of peaks, peak positions

### 5. Feature Selection

Select the most informative features using one of three available methods.

```python
from eruption_forecast.features import FeatureSelector

# Two-stage combined selection (recommended)
selector = FeatureSelector(method="combined", n_jobs=4, verbose=True)
X_selected = selector.fit_transform(
    X_train, y_train,
    fdr_level=0.05,  # Stage 1: tsfresh FDR level
    top_n=30,        # Stage 2: final feature count
)
print(f"Reduced: {X_train.shape[1]} → {X_selected.shape[1]} features")

# tsfresh statistical selection only
selector = FeatureSelector(method="tsfresh", n_jobs=4)
X_selected = selector.fit_transform(X_train, y_train, fdr_level=0.05)

# RandomForest permutation importance only
selector = FeatureSelector(method="random_forest", n_jobs=4)
X_selected = selector.fit_transform(X_train, y_train, top_n=30)

# Get comprehensive feature scores
scores = selector.get_feature_scores()
print(scores.head(10))
```

**Selection methods:**

| Method | Reduces | Captures Interactions | Speed |
|--------|---------|-----------------------|-------|
| `tsfresh` | 1000s → 100s | No | Fast |
| `random_forest` | Direct → N | Yes | Slow |
| `combined` | 1000s → 100s → N | Yes | Fast |

### 6. Train Models with Multiple Seeds

Two training workflows are available depending on your evaluation strategy.

```
        train_and_evaluate()                          train()
   ──────────────────────────────           ──────────────────────────
         Full Dataset                             Full Dataset
               │                                       │
               ▼                                       ▼
     80/20 Stratified Split                   RandomUnderSampler
        ┌──────┴───────┐                        (full dataset)
      Train           Test                            │
   (imbalanced)   (imbalanced,                 Feature Selection
        │          never touched)               (resampled data)
        ▼               │                            │
  RandomUnderSampler    │                            ▼
  (training set only)   │                       GridSearchCV
        │               │                   (CV folds on resampled
        ▼               │                       balanced data)
  Feature Selection     │                            │
  (resampled data)      │                     ┌──────┴──────┐
        │               │                 model.pkl   registry.csv
        ▼               │
   GridSearchCV         │
  (CV folds on          │
   resampled data)      │
        │               │
        └───► Evaluate ◄┘
              on Test set
                  │
       ┌──────────┴──────────┐
   model.pkl  metrics.json  registry.csv
```

#### Which workflow should I use?

- **`train_and_evaluate()`** — uses the **all calculated tremor dataset**, splits it 80/20 internally, trains on the 80% and evaluates on the held-out 20%. Both train and test come from the same date range.
- **`train()`** — treats data as two separate time periods: a **current/present dataset** used for training (passed via `extracted_features_csv`) and a **future dataset** evaluated separately via `ModelPredictor`. No internal split; the model is trained on 100% of the current data.

| Question | `train_and_evaluate()` | `train()` |
|---|---|---|
| Do I want to measure in-sample performance? | Yes — evaluates each seed on held-out 20% | No metrics computed |
| Do I have a separate future dataset to evaluate on? | — | Use with `ModelPredictor` |
| Am I exploring classifiers and hyperparameters? | Quick feedback per run | Not suitable |
| Am I training the final production model? | Wastes 20% of data | Uses 100% of data |

#### `train_and_evaluate()` — with held-out test set (80/20 split)

Splits data **before** resampling and feature selection to prevent data leakage.
Evaluates each seed on the held-out 20% and aggregates metrics across seeds.

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

#### `train()` — full dataset training (no split)

Trains on the **entire current/present dataset** across multiple seeds — no internal 80/20 split.
The dataset passed here represents your known historical period. Evaluation is deferred to
`ModelPredictor` using a **separate future dataset** that was not seen during training.

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

#### `fit()` — Unified entry point

`fit()` dispatches to `train_and_evaluate()` or `train()` based on the
`with_evaluation` flag. Use it when the calling code needs a single method
regardless of which workflow is active.

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

`ForecastModel.train()` calls `fit()` internally and exposes `with_evaluation`
as a direct parameter, so you can control the workflow from the high-level API
without dropping down to `ModelTrainer`.

#### ModelTrainer constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extracted_features_csv` | `str` | — | Path to extracted features CSV (output of `FeaturesBuilder`) |
| `label_features_csv` | `str` | — | Path to aligned labels CSV (output of `FeaturesBuilder`) |
| `output_dir` | `str \| None` | `None` | Output directory; resolved against `root_dir` when relative. Defaults to `root_dir/output/trainings` |
| `root_dir` | `str \| None` | `None` | Anchor for resolving relative `output_dir`. Defaults to `os.getcwd()` |
| `prefix_filename` | `str \| None` | `None` | Optional prefix prepended to every output filename |
| `classifier` | `str` | `"rf"` | Classifier type — see [Supported Classifiers](#7-supported-classifiers). `ForecastModel.train()` also accepts a `list[str]` or comma-separated string to train multiple classifiers in sequence |
| `cv_strategy` | `str` | `"shuffle-stratified"` | Cross-validation strategy — `"shuffle"`, `"stratified"`, `"shuffle-stratified"`, or `"timeseries"` |
| `cv_splits` | `int` | `5` | Number of CV folds |
| `number_of_significant_features` | `int` | `20` | Top-N features retained per seed and aggregated across seeds |
| `feature_selection_method` | `str` | `"tsfresh"` | Feature selection algorithm — `"tsfresh"`, `"random_forest"`, or `"combined"` |
| `overwrite` | `bool` | `False` | Re-run even if output files already exist |
| `n_jobs` | `int` | `1` | Parallel seed workers (outer loop). Pass `-1` to use all available cores. Enforced: `n_jobs × grid_search_n_jobs ≤ cpu_count` |
| `grid_search_n_jobs` | `int` | `1` | Parallel jobs inside each `GridSearchCV` call and `FeatureSelector` (inner loop). Uses `loky` backend — safe for Intel's scikit-learn extension. When `use_gpu=True`, `GridSearchCV` is forced to `n_jobs=1` but `FeatureSelector` keeps the configured value |
| `use_gpu` | `bool` | `False` | Enable GPU acceleration for XGBoost via `device="cuda:<gpu_id>"`. Forces outer `n_jobs=1` and `GridSearchCV` inner `n_jobs=1` to prevent VRAM contention. Has no effect for non-XGBoost classifiers — emits a warning if set with `rf`, `gb`, `svm`, etc. |
| `gpu_id` | `int` | `0` | GPU device index when `use_gpu=True`. Use `0` for the first GPU, `1` for the second |
| `verbose` | `bool` | `False` | Print progress messages |
| `debug` | `bool` | `False` | Enable debug-level logging |

#### `train_and_evaluate()` method parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `0` | Starting random seed; seeds are `random_state, random_state+1, …, random_state+total_seed−1` |
| `total_seed` | `int` | `500` | Number of seeds (independent train/test splits) to run |
| `sampling_strategy` | `str \| float` | `0.75` | Under-sampling ratio for `RandomUnderSampler` on training data only |
| `save_all_features` | `bool` | `False` | Save all ranked features per seed (can produce many files) |
| `plot_significant_features` | `bool` | `False` | Save a feature-importance plot per seed |

#### `train()` method parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_state` | `int` | `0` | Starting random seed |
| `total_seed` | `int` | `500` | Number of seeds to run |
| `sampling_strategy` | `str \| float` | `0.75` | Under-sampling ratio for `RandomUnderSampler` on full dataset |
| `save_all_features` | `bool` | `False` | Save all ranked features per seed |
| `plot_significant_features` | `bool` | `False` | Save a feature-importance plot per seed |

#### `fit()` method parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `with_evaluation` | `bool` | `True` | `True` → `train_and_evaluate()` (80/20 split + metrics); `False` → `train()` (full dataset, no metrics) |
| `**kwargs` | — | — | Forwarded to `train_and_evaluate()` or `train()` (same parameters as those methods) |

### 7. Supported Classifiers

| Classifier | Description | Imbalance Handling |
|------------|-------------|---------------------|
| `rf` | Random Forest (balanced, robust, default) | `class_weight="balanced"` |
| `gb` | Gradient Boosting (handles imbalance natively) | None (natural) |
| `xgb` | XGBoost (excellent for imbalanced data, GPU-capable) | `scale_pos_weight` grid search |
| `svm` | Support Vector Machine (high-dimensional) | `class_weight="balanced"` |
| `lr` | Logistic Regression (interpretable, fast) | `class_weight="balanced"` |
| `nn` | Neural Network MLP (complex patterns) | None |
| `dt` | Decision Tree (interpretable baseline) | `class_weight="balanced"` |
| `knn` | K-Nearest Neighbors (simple baseline) | None |
| `nb` | Gaussian Naive Bayes (fast baseline) | None |
| `voting` | Soft VotingClassifier (RF + XGBoost ensemble, GPU-capable) | Combined |
| `lite-rf` | Random Forest with a smaller grid for faster training | `class_weight="balanced"` |

### 8. Hyperparameter Grids

Each classifier comes with a built-in hyperparameter grid for `GridSearchCV`. You can override the grid via `ClassifierModel`:

```python
from eruption_forecast.model.classifier_model import ClassifierModel

clf = ClassifierModel("xgb", random_state=42)

# Override default grid
clf.grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "scale_pos_weight": [1, 5, 10, 20],
}
```

**Default grids by classifier:**

<details>
<summary>Random Forest (<code>rf</code>)</summary>

```python
{
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
```
</details>

<details>
<summary>Gradient Boosting (<code>gb</code>)</summary>

```python
{
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
```
</details>

<details>
<summary>XGBoost (<code>xgb</code>)</summary>

```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3],
    "scale_pos_weight": [1, 5, 10, 15],  # Tunes positive-class weighting
}
```

> `scale_pos_weight` controls how much extra weight positive (eruption) samples receive. Higher values increase sensitivity at the cost of more false positives.
</details>

<details>
<summary>Voting Ensemble (<code>voting</code>)</summary>

```python
{
    "rf__n_estimators": [100, 200],
    "rf__max_depth": [10, None],
    "xgb__n_estimators": [50, 100],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__max_depth": [5, 7],
}
```

> Combines Random Forest and XGBoost with soft voting (probability averaging).
</details>

### 9. Cross-Validation Strategies

| Strategy | Class | Best For |
|----------|-------|----------|
| `shuffle` | `ShuffleSplit` | Random splits without stratification |
| `stratified` | `StratifiedKFold` | Preserves class distribution across folds |
| `shuffle-stratified` | `StratifiedShuffleSplit` | Randomized stratified folds — **default** |
| `timeseries` | `TimeSeriesSplit` | Temporal data, strict no-future-leakage |

### 10. Predict on Future Data with ModelPredictor

`ModelPredictor` supports two modes after `train()`:

#### Single model — evaluation mode (labelled data)

Evaluates each seed model against known eruption labels and aggregates metrics across seeds.

```python
from eruption_forecast.model.model_predictor import ModelPredictor

predictor = ModelPredictor(
    start_date="2025-03-16",
    end_date="2025-03-22",
    trained_models=trainer.csv,  # trained_model_*.csv from train()
    output_dir="output/predictions",
)

# One row per (classifier, seed)
df_metrics = predictor.predict(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
)
print(df_metrics[["balanced_accuracy", "f1_score"]].describe())

# Best (classifier, seed) overall
evaluator = predictor.predict_best(
    future_features_csv="output/features/future_all_features.csv",
    future_labels_csv="output/features/future_label_features.csv",
    criterion="balanced_accuracy",
)
print(evaluator.summary())
evaluator.plot_all()
```

`predict_best()` accepts any metric column as `criterion`:
`"accuracy"`, `"balanced_accuracy"`, `"f1_score"`, `"precision"`, `"recall"`, `"roc_auc"`, `"pr_auc"`.

#### ModelPredictor constructor parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start_date` | `str \| datetime` | — | Start date for prediction period (format: YYYY-MM-DD) |
| `end_date` | `str \| datetime` | — | End date for prediction period (format: YYYY-MM-DD) |
| `trained_models` | `str \| dict[str, str]` | — | Single `trained_model_*.csv` path (from `train()`) or a `{name: path}` dict for multi-model consensus |
| `overwrite` | `bool` | `False` | Overwrite existing output files |
| `n_jobs` | `int` | `1` | Number of parallel jobs for feature extraction |
| `output_dir` | `str \| None` | `None` | Output directory; defaults to `<root_dir>/output/predictions` |
| `root_dir` | `str \| None` | `None` | Root directory for resolving output paths |
| `verbose` | `bool` | `False` | Enable verbose logging |

#### Single model — forecast mode (unlabelled data)

When no ground-truth labels are available, use `predict_proba()`.

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

#### Multi-model consensus

Pass a dict of classifier registries.  `predict_proba()` aggregates within
each classifier (across seeds) and then across classifiers (consensus).

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
| `consensus_prediction` | Hard label — `1` if `consensus_eruption_probability ≥ 0.5` |

Results are saved to `predictions.csv`.  The plot shows each classifier as a
dashed line and the consensus as a solid black line with a shaded uncertainty
band (`eruption_forecast.png` in `figures/`).

### 11. Model Evaluation

```python
from eruption_forecast.model.model_evaluator import ModelEvaluator

# From in-memory objects
evaluator = ModelEvaluator(
    model=trained_model,
    X_test=X_test,
    y_test=y_test,
    model_name="xgb_42",
    output_dir="output/eval",
)

# Or load directly from files
evaluator = ModelEvaluator.from_files(
    model_path="output/trainings/classifier/XGBClassifier/stratified/models/00042.pkl",
    X_test="output/features/all_features.csv",
    y_test="output/features/label_features.csv",
    selected_features=["feat_a", "feat_b"],  # optional
    model_name="xgb_42",
    output_dir="output/eval",
)

# Print a formatted summary
print(evaluator.summary())

# Get metrics as a dict
metrics = evaluator.get_metrics()
# Keys: accuracy, balanced_accuracy, precision, recall, f1_score, roc_auc, pr_auc,
#       true_positives, true_negatives, false_positives, false_negatives,
#       sensitivity, specificity, optimal_threshold, f1_at_optimal,
#       recall_at_optimal, precision_at_optimal

# Generate all plots (saved to output_dir)
evaluator.plot_all()
# Produces: confusion_matrix, roc_curve, pr_curve, threshold_analysis,
#            feature_importance, calibration, prediction_distribution

# Find optimal decision threshold
threshold, threshold_metrics = evaluator.optimize_threshold(criterion="f1")
print(f"Optimal threshold: {threshold:.3f}")
print(f"F1 at threshold:   {threshold_metrics['f1']:.4f}")
```

#### Save Single-Seed Metrics to JSON

After evaluating a single seed, call `save_metrics()` to persist the metrics dict as JSON:

```python
# Metrics are serialized with np.nan → null
path = evaluator.save_metrics()
# Saved to: {output_dir}/{model_name}_metrics.json

# Or specify an explicit path
path = evaluator.save_metrics("results/xgb_42_metrics.json")
print(f"Saved: {path}")
```

#### Aggregate Evaluation (All Seeds) with MultiModelEvaluator

When training with `with_evaluation=True`, each seed's held-out test split is saved to a `tests/`
directory alongside per-seed `metrics/*.json` files. Use `MultiModelEvaluator` to aggregate across all seeds:

```python
from eruption_forecast import MultiModelEvaluator

base = "output/trainings/evaluations/xgb-classifier/stratified-shuffle-split"
trained_model_csv = f"{base}/trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv"

# --- From a model registry CSV (enables plots) ---
evaluator = MultiModelEvaluator(trained_model_csv=trained_model_csv)

# Generate all 7 aggregate plots at once (saved to <registry_dir>/figures/)
figs = evaluator.plot_all(dpi=150, show_individual=True)
# Keys: roc_curve, pr_curve, calibration, prediction_distribution,
#       confusion_matrix, threshold_analysis, feature_importance

# Or generate individual aggregate plots
evaluator.plot_roc(show_individual=True)
evaluator.plot_precision_recall(show_individual=True)
evaluator.plot_calibration(n_bins=10)
evaluator.plot_prediction_distribution()
evaluator.plot_confusion_matrix(normalize=None)
evaluator.plot_threshold_analysis(show_individual=True)
evaluator.plot_feature_importance(top_n=20)
```

Each plot method accepts `save=True` (default), `filename=None`, `dpi=150`, and `title=None`. Figures and their CSV data are saved to `{output_dir}/figures/` (defaults to `<registry_dir>/figures/`).

#### Aggregate Metrics from JSON Files

Use `get_aggregate_metrics()` to summarize per-seed JSON metrics written by `save_metrics()`:

```python
# --- From per-seed metrics JSON files ---
metrics_dir = f"{base}/metrics"
evaluator = MultiModelEvaluator(metrics_dir=metrics_dir)

# Returns a DataFrame: index = metric name, columns = mean/std/min/max
summary = evaluator.get_aggregate_metrics()
print(summary.loc["f1_score"])
# mean     0.7842
# std      0.0321
# min      0.6900
# max      0.8500

# Save to CSV
path = evaluator.save_aggregate_metrics()
# Saved to: {metrics_dir}/figures/aggregate_metrics.csv

path = evaluator.save_aggregate_metrics("my_summary.csv")
```

You can also provide an explicit list of JSON files or a custom output directory:

```python
import glob

json_files = sorted(glob.glob(f"{base}/metrics/*.json"))
evaluator = MultiModelEvaluator(
    metrics_files=json_files,
    output_dir="output/eval/aggregate",
)
summary = evaluator.get_aggregate_metrics()
evaluator.save_aggregate_metrics()
```

#### Combining Metrics and Plots

Pass both `metrics_dir` and `trained_model_csv` to get everything in one object:

```python
evaluator = MultiModelEvaluator(
    metrics_dir=f"{base}/metrics",
    trained_model_csv=trained_model_csv,
    output_dir="output/eval/aggregate",
)

# JSON-based aggregate stats
summary = evaluator.get_aggregate_metrics()
evaluator.save_aggregate_metrics()

# Registry-based aggregate plots
figs = evaluator.plot_all()
```

### 12. Analyze Training Results

```python
import pandas as pd

# Suffix format: {ClassifierName}-{CVName}_rs-{random_state}_ts-{total_seed}_top-{n}
# e.g., XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20
base = "output/trainings/evaluations/classifiers/xgb-classifier/stratified-shuffle-split"
suffix = "XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20"

# All per-seed metrics
metrics = pd.read_csv(f"{base}/all_metrics_{suffix}.csv")

# Summary statistics (mean ± std)
summary = pd.read_csv(f"{base}/metrics_summary_{suffix}.csv", index_col=0)
print(summary[["balanced_accuracy", "f1_score", "precision", "recall"]])

# Best seed
best_seed = metrics.loc[metrics["balanced_accuracy"].idxmax()]
print(f"Best seed:          {best_seed['random_state']}")
print(f"Balanced Accuracy:  {best_seed['balanced_accuracy']:.4f}")
print(f"F1 Score:           {best_seed['f1_score']:.4f}")

# Aggregated significant features
sig_features = pd.read_csv("output/trainings/evaluations/features/stratified-shuffle-split/significant_features.csv")
print(sig_features.head(10))
```

### 13. Merge Seed Models into One File

After training 500 seeds, each seed's estimator and its significant-feature list live in separate files. Merging them into a single `SeedEnsemble` pkl eliminates all per-seed I/O at prediction time and gives you a standard sklearn estimator you can use directly.

#### Option A — via ModelTrainer (recommended after training)

```python
# trainer is the ModelTrainer returned by ForecastModel.train() or constructed directly
merged_path = trainer.merge_models()
# Default output: alongside the registry CSV
# → output/VG.OJN.00.EHZ/trainings/predictions/
#     random-forest-classifier/stratified-k-fold/
#     merged_model_RandomForestClassifier-StratifiedKFold_rs-0_ts-500_top-20.pkl

print(f"Merged ensemble saved to: {merged_path}")
```

Custom output path:

```python
merged_path = trainer.merge_models(
    output_path="output/deploy/rf_ensemble.pkl"
)
```

#### Option B — standalone function

```python
from eruption_forecast.utils.ml import merge_seed_models

merged_path = merge_seed_models(
    registry_csv="output/.../trained_model_RandomForestClassifier-StratifiedKFold_rs-0_ts-500_top-20.csv"
)
```

#### Option C — multi-classifier bundle

Bundle all classifiers into one file for one-shot loading with `ModelPredictor`:

```python
from eruption_forecast.utils.ml import merge_all_classifiers

bundle_path = merge_all_classifiers(
    trained_models={
        "rf":  "output/.../predictions/random-forest-classifier/stratified-k-fold/trained_model_*.csv",
        "xgb": "output/.../predictions/xgb-classifier/stratified-k-fold/trained_model_*.csv",
    }
)
# Default output: output/.../trainings/merged_classifiers_*.pkl
```

Or via `ModelTrainer`:

```python
bundle_path = trainer.merge_classifier_models(
    trained_models={"rf": rf_trainer.csv, "xgb": xgb_trainer.csv}
)
```

#### Using SeedEnsemble directly

```python
from eruption_forecast.model.seed_ensemble import SeedEnsemble

ensemble = SeedEnsemble.load(merged_path)
print(ensemble)
# SeedEnsemble(classifier_name='RandomForestClassifier',
#              cv_strategy='unknown', n_seeds=500)

# Full uncertainty output — mirrors compute_model_probabilities()
mean_p, std, confidence, prediction = ensemble.predict_with_uncertainty(
    features_df,
    threshold=0.7,   # default: ERUPTION_PROBABILITY_THRESHOLD
)
print(f"Mean P(eruption): {mean_p.mean():.4f}")
print(f"Eruption windows: {prediction.sum()}")

# sklearn-compatible output — useful for cross_val_score, Pipeline, etc.
proba = ensemble.predict_proba(features_df)   # shape (n_windows, 2)
labels = ensemble.predict(features_df)        # shape (n_windows,)
```

#### Passing a merged pkl to ModelPredictor

`ModelPredictor` detects `.pkl` vs `.csv` automatically — no API change needed:

```python
from eruption_forecast.model.model_predictor import ModelPredictor

# Single merged classifier
predictor = ModelPredictor(
    start_date="2025-07-28",
    end_date="2025-08-04",
    trained_models=merged_path,   # path ending in .pkl
)

# Multi-classifier bundle
predictor = ModelPredictor(
    start_date="2025-07-28",
    end_date="2025-08-04",
    trained_models=bundle_path,   # dict[str, SeedEnsemble] inside
)

df_forecast = predictor.predict_proba(
    tremor_data="output/VG.OJN.00.EHZ/tremor/tremor_2025-01-01_2025-12-31.csv",
    window_size=2,
    window_step=12,
    window_step_unit="hours",
    plot=True,
)
print(df_forecast[["model_eruption_probability", "model_uncertainty"]].head())
```
