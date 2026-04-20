# Output Structure

All outputs are written under:

```
{output_dir}/{network}.{station}.{location}.{channel}/
```

For example, with `network="VG"`, `station="OJN"`, `location="00"`, `channel="EHZ"`:

```
output/VG.OJN.00.EHZ/
```

---

## Full Directory Tree

```
output/
└── VG.OJN.00.EHZ/
    │
    ├── tremor/
    │   ├── daily/                            # Per-day CSV files (removed if cleanup_daily_dir=True)
    │   ├── figures/                          # Daily tremor plots (created if plot_daily=True)
    │   └── tremor_*.csv                      # Final merged tremor data
    │
    ├── features/
    │   ├── tremor_matrix_*.csv               # Aligned tremor matrix (all columns)
    │   ├── tremor_matrix_per_method/         # Per-column tremor matrices (optional)
    │   ├── all_extracted_features_*.csv      # tsfresh output per tremor column
    │   └── label_features_*.csv             # Labels aligned with features
    │
    ├── trainings/
    │   │
    │   └── predictions/                       # Output of train()
    │       ├── predictions_trained_models.json   # {ClassifierName: trained_model_*.csv} registry
    │       ├── predictions_config.yaml           # Pipeline config snapshot (written by save_model)
    │       ├── predictions_forecast_model.pkl    # Serialised ForecastModel (written by save_model)
    │       ├── features/                         # Shared across all classifiers
    │       │   └── {cv-slug}/
    │       │       ├── significant_features/
    │       │       │   ├── 00000.csv
    │       │       │   └── ...
    │       │       ├── all_features/ (optional)
    │       │       ├── figures/significant/ (optional)
    │       │       ├── significant_features.csv
    │       │       └── top_{n}_significant_features.csv
    │       └── classifiers/                      # Per-classifier outputs
    │           └── {classifier-slug}/
    │               └── {cv-slug}/
    │                   ├── models/
    │                   │   ├── 00000.pkl
    │                   │   └── ...
    │                   ├── trained_model_{suffix}.csv    # Registry used by ModelPredictor
    │                   └── merged_model_{suffix}.pkl     # SeedEnsemble (optional — call merge_models())
    │
    ├── trainings/merged_classifiers_{suffix}.pkl   # Multi-classifier bundle (optional — call merge_classifier_models())
    │
    ├── forecast/
    │   ├── predictions.csv                   # Forecast output (predict_proba)
    │   └── figures/
    │       └── eruption_forecast.png
    │
    ├── config_forecast.yaml                  # Pipeline config snapshot (written by forecast())
    └── forecast_model.pkl                    # Serialised ForecastModel (default path for save_model())
```

---

## File Name Suffixes

Training output files follow this naming pattern:

```
{ClassifierName}-{CVName}_rs-{random_state}_ts-{total_seed}_top-{n}
```

Example:
```
XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20
```

So the model registry is:
```
trained_model_XGBClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv
```

---

## Classifier and CV Slugs

| Classifier key | Folder slug |
|----------------|-------------|
| `rf` | `random-forest-classifier` |
| `xgb` | `xgb-classifier` |
| `gb` | `gradient-boosting-classifier` |
| `svm` | `svm-classifier` |
| `lr` | `logistic-regression-classifier` |
| `nn` | `mlp-classifier` |
| `dt` | `decision-tree-classifier` |
| `knn` | `knn-classifier` |
| `nb` | `gaussian-nb-classifier` |
| `voting` | `voting-classifier` |

| CV strategy key | Folder slug |
|-----------------|-------------|
| `shuffle` | `shuffle-split` |
| `stratified` | `stratified-k-fold` |
| `shuffle-stratified` | `stratified-shuffle-split` |
| `timeseries` | `time-series-split` |

---

## ModelPredictor Output

### Forecast mode (`predict_proba()`)

```
{output_dir}/
├── predictions.csv             # eruption_probability, uncertainty, confidence, prediction
└── figures/
    └── eruption_forecast.png
```
