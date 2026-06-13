Plotting layer. matplotlib-backed, `Agg`-friendly renderers shared across stages.

Files:
- styles.py — global rcParams and palette helpers.
- tremor_plots.py — `plot_tremor` daily + merged tremor figures.
- feature_plots.py — `plot_significant_features` per-seed selected-feature bar charts.
- evaluation_plots.py — aggregate + per-seed dispatchers (ROC, PR, threshold, g-mean, MCC) called by `MetricsEnsemble.plot_aggregate()` / `plot_seed()`.
- forecast_plots.py — `plot_forecast` per-classifier + consensus probability time-series.
- explanation_plots.py — SHAP beeswarm / waterfall / bar wrappers used by `ExplanationModel`.
