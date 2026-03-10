# Output Directory Structure

> Back to [README](../README.md)

## Output Directory Structure

All outputs are organized under `{output_dir}/{network}.{station}.{location}.{channel}/`
(e.g., `output/VG.OJN.00.EHZ/`).

```
output/
└── VG.OJN.00.EHZ/
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
    └── trainings/
        ├── evaluations/        # Output of train_and_evaluate()
        │   ├── features/                             # Shared across all classifiers
        │   │   └── {cv-slug}/                        # e.g., stratified-shuffle-split
        │   │       ├── significant_features/         # Per-seed top-N features
        │   │       │   ├── 00000.csv
        │   │       │   └── ...
        │   │       ├── all_features/                 # All ranked features (optional)
        │   │       │   ├── 00000.csv
        │   │       │   └── ...
        │   │       ├── figures/
        │   │       │   └── significant/              # Feature importance plots (optional)
        │   │       │       └── 00000.jpg
        │   │       ├── tests/                        # Per-seed held-out test splits
        │   │       │   ├── 00000_X_test.csv
        │   │       │   ├── 00000_y_test.csv
        │   │       │   └── ...
        │   │       ├── significant_features.csv      # Aggregated features (all seeds)
        │   │       └── top_{n}_significant_features.csv
        │   └── classifiers/                          # Per-classifier outputs
        │       └── {classifier-slug}/                # e.g., random-forest-classifier
        │           └── {cv-slug}/                    # e.g., stratified-shuffle-split
        │               ├── models/
        │               │   ├── 00000.pkl     # Trained model (seed 0)
        │               │   ├── 00001.pkl
        │               │   └── ...
        │               ├── metrics/
        │               │   ├── 00000.json    # Per-seed metrics
        │               │   └── ...
        │               ├── figures/              # Aggregate evaluation plots (MultiModelEvaluator)
        │               │   ├── aggregate_roc_curve.png
        │               │   ├── aggregate_roc_curve.csv
        │               │   ├── aggregate_pr_curve.png
        │               │   ├── aggregate_pr_curve.csv
        │               │   ├── aggregate_calibration.png
        │               │   ├── aggregate_calibration.csv
        │               │   ├── aggregate_prediction_distribution.png
        │               │   ├── aggregate_prediction_distribution.csv
        │               │   ├── aggregate_confusion_matrix.png
        │               │   ├── aggregate_confusion_matrix.csv
        │               │   ├── aggregate_threshold_analysis.png
        │               │   ├── aggregate_threshold_analysis.csv
        │               │   ├── aggregate_feature_importance.png
        │               │   ├── aggregate_feature_importance.csv
        │               │   ├── aggregate_shap_summary.png
        │               │   ├── aggregate_shap_summary.pkl   # shap.Explanation (joblib)
        │               │   └── aggregate_metrics.csv        # from save_aggregate_metrics()
        │               ├── trained_model_{suffix}.csv    # Registry of all trained models
        │               ├── merged_model_{suffix}.pkl     # SeedEnsemble (optional — call merge_models())
        │               ├── all_metrics_{suffix}.csv      # All seed metrics
        │               └── metrics_summary_{suffix}.csv  # Mean ± std summary
        │
        └── predictions/                   # Output of train()
            ├── features/                             # Shared across all classifiers
            │   └── {cv-slug}/
            │       ├── significant_features/         # Per-seed top-N features
            │       │   ├── 00000.csv
            │       │   └── ...
            │       ├── all_features/                 # All ranked features (optional)
            │       │   ├── 00000.csv
            │       │   └── ...
            │       ├── figures/
            │       │   └── significant/              # Feature importance plots (optional)
            │       │       └── 00000.jpg
            │       ├── significant_features.csv      # Aggregated features (all seeds)
            │       └── top_{n}_significant_features.csv
            └── classifiers/                          # Per-classifier outputs
                └── {classifier-slug}/
                    └── {cv-slug}/
                        ├── models/
                        │   ├── 00000.pkl
                        │   └── ...
                        ├── trained_model_{suffix}.csv    # Registry used by ModelPredictor
                        └── merged_model_{suffix}.pkl     # SeedEnsemble (optional — call merge_models())
```

**ModelPredictor output** (`output_dir/predictions/`):

*Evaluation mode (`predict()` / `predict_best()`):*

```
predictions/
├── metrics/
│   ├── all_metrics.csv
│   └── metrics_summary.csv
└── seed_00000/                    # Only created when plot=True
    ├── seed_00000_confusion_matrix.png
    ├── seed_00000_roc_curve.png
    ├── seed_00000_pr_curve.png
    ├── seed_00000_threshold_analysis.png
    ├── seed_00000_feature_importance.png
    ├── seed_00000_calibration.png
    └── seed_00000_prediction_distribution.png
```

*Forecast mode (`predict_proba()`):*

```
predictions/
├── predictions.csv                # eruption_probability, uncertainty, confidence, prediction
└── figures/
    └── eruption_forecast.png      # Probability + confidence time-series plot
```
