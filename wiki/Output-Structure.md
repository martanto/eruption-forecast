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
    │   ├── evaluations/            # Output of train(with_evaluation=True)
    │   │   ├── evaluations_trained_models.json   # {ClassifierName: trained_model_*.csv} registry
    │   │   ├── evaluations_config.yaml           # Pipeline config snapshot (written by save_model)
    │   │   ├── evaluations_forecast_model.pkl    # Serialised ForecastModel (written by save_model)
    │   │   └── {classifier-slug}/                # e.g., random-forest-classifier
    │   │       └── {cv-slug}/                    # e.g., stratified-shuffle-split
    │   │           ├── features/
    │   │           │   ├── significant_features/     # Per-seed top-N features
    │   │           │   │   ├── 00000.csv
    │   │           │   │   └── ...
    │   │           │   ├── all_features/             # All ranked features (optional)
    │   │           │   │   ├── 00000.csv
    │   │           │   │   └── ...
    │   │           │   ├── figures/significant/      # Feature importance plots (optional)
    │   │           │   │   └── 00000.jpg
    │   │           │   ├── tests/                    # Per-seed held-out test splits
    │   │           │   │   ├── 00000_X_test.csv
    │   │           │   │   ├── 00000_y_test.csv
    │   │           │   │   └── ...
    │   │           │   ├── significant_features.csv  # Aggregated top features (all seeds)
    │   │           │   └── top_{n}_significant_features.csv
    │   │           ├── models/
    │   │           │   ├── 00000.pkl                 # Trained model — seed 0
    │   │           │   ├── 00001.pkl
    │   │           │   └── ...
    │   │           ├── metrics/
    │   │           │   ├── 00000.json                # Per-seed metrics JSON
    │   │           │   └── ...
    │   │           ├── figures/                      # Aggregate evaluation plots
    │   │           │   ├── aggregate_roc_curve.png / .csv
    │   │           │   ├── aggregate_pr_curve.png / .csv
    │   │           │   ├── aggregate_calibration.png / .csv
    │   │           │   ├── aggregate_prediction_distribution.png / .csv
    │   │           │   ├── aggregate_confusion_matrix.png / .csv
    │   │           │   ├── aggregate_threshold_analysis.png / .csv
    │   │           │   ├── aggregate_feature_importance.png / .csv
    │   │           │   ├── aggregate_shap_summary.png
    │   │           │   ├── aggregate_shap_summary.pkl   # shap.Explanation (joblib)
    │   │           │   └── aggregate_metrics.csv
    │   │           ├── trained_model_{suffix}.csv    # Registry of all trained models
    │   │           ├── merged_model_{suffix}.pkl     # SeedEnsemble (all seeds merged — optional)
    │   │           ├── all_metrics_{suffix}.csv      # All per-seed metrics
    │   │           └── metrics_summary_{suffix}.csv  # Mean ± std summary
    │   │
    │   └── predictions/                       # Output of train(with_evaluation=False)
    │       ├── predictions_trained_models.json   # {ClassifierName: trained_model_*.csv} registry
    │       ├── predictions_config.yaml           # Pipeline config snapshot (written by save_model)
    │       ├── predictions_forecast_model.pkl    # Serialised ForecastModel (written by save_model)
    │       └── {classifier-slug}/
    │           └── {cv-slug}/
    │               ├── features/
    │               │   ├── significant_features/
    │               │   │   ├── 00000.csv
    │               │   │   └── ...
    │               │   ├── all_features/ (optional)
    │               │   ├── figures/significant/ (optional)
    │               │   ├── significant_features.csv
    │               │   └── top_{n}_significant_features.csv
    │               ├── models/
    │               │   ├── 00000.pkl
    │               │   └── ...
    │               ├── trained_model_{suffix}.csv    # Registry used by ModelPredictor
    │               └── merged_model_{suffix}.pkl     # SeedEnsemble (optional — call merge_models())
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

### Evaluation mode (`predict()` / `predict_best()`)

```
{output_dir}/
├── metrics/
│   ├── all_metrics.csv
│   └── metrics_summary.csv
└── seed_00000/                 # Only created when plot=True
    ├── seed_00000_confusion_matrix.png
    ├── seed_00000_roc_curve.png
    ├── seed_00000_pr_curve.png
    ├── seed_00000_threshold_analysis.png
    ├── seed_00000_feature_importance.png
    ├── seed_00000_calibration.png
    └── seed_00000_prediction_distribution.png
```

### Forecast mode (`predict_proba()`)

```
{output_dir}/
├── predictions.csv             # eruption_probability, uncertainty, confidence, prediction
└── figures/
    └── eruption_forecast.png
```
