"""Publication-quality scientific plotting for volcanic eruption forecasting.

This package provides a unified set of plotting utilities styled after Nature/Science
journal standards. All figures use the Okabe-Ito colorblind-safe palette by default,
clean spines, and font sizes matched to typical journal column widths.

Key sub-modules and their public exports:

- ``styles``: Color palettes (``NATURE_COLORS``, ``OKABE_ITO``), rcParams helpers
  (``apply_nature_style``, ``setup_nature_style``), and utilities
  (``get_color``, ``configure_spine``, ``get_figure_size``).
- ``tremor_plots``: ``plot_tremor`` — multi-panel RSAM/DSAR/entropy time-series.
- ``feature_plots``: ``plot_significant_features``, ``replot_significant_features``,
  ``plot_frequency_band_contribution`` — tsfresh feature importance visualisation.
- ``evaluation_plots``: full suite of classifier evaluation charts including ROC,
  precision-recall, calibration, confusion matrix, learning curves, threshold
  analysis, feature importance, prediction distribution, seed stability, and
  classifier comparison.
- ``shap_plots``: ``plot_shap_summary``, ``plot_aggregate_shap_summary``,
  ``plot_shap_waterfall`` — SHAP beeswarm and waterfall explainability plots
  (XGBoost >= 3.x compatible via Independent masker).
- ``forecast_plots``: ``plot_forecast``, ``plot_forecast_from_file`` — eruption
  probability time-series with per-classifier panels and consensus band.
"""

# Style configuration
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    NATURE_COLORS,
    get_color,
    configure_spine,
    get_figure_size,
    apply_nature_style,
    setup_nature_style,
)

# SHAP plots
from eruption_forecast.plots.shap_plots import (
    plot_shap_summary,
    plot_shap_waterfall,
    plot_aggregate_shap_summary,
    plot_aggregate_shap_waterfall,
)

# Tremor plotting
from eruption_forecast.plots.tremor_plots import plot_tremor

# Feature plotting
from eruption_forecast.plots.feature_plots import (
    plot_significant_features,
    replot_significant_features,
    plot_frequency_band_contribution,
)

# Forecast plotting
from eruption_forecast.plots.forecast_plots import (
    plot_forecast,
    plot_forecast_from_file,
)

# Evaluation plotting
from eruption_forecast.plots.evaluation_plots import (
    plot_roc_curve,
    plot_calibration,
    plot_learning_curve,
    plot_seed_stability,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_threshold_analysis,
    plot_classifier_comparison,
    plot_precision_recall_curve,
    plot_prediction_distribution,
    plot_aggregate_learning_curve,
)


__all__ = [
    # Style configuration
    "NATURE_COLORS",
    "OKABE_ITO",
    "apply_nature_style",
    "configure_spine",
    "get_color",
    "get_figure_size",
    "setup_nature_style",
    # Tremor plotting
    "plot_tremor",
    # Feature plotting
    "plot_significant_features",
    "replot_significant_features",
    "plot_frequency_band_contribution",
    # Evaluation plotting
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_threshold_analysis",
    "plot_feature_importance",
    "plot_calibration",
    "plot_prediction_distribution",
    "plot_classifier_comparison",
    "plot_seed_stability",
    "plot_learning_curve",
    "plot_aggregate_learning_curve",
    # SHAP plots
    "plot_shap_summary",
    "plot_shap_waterfall",
    "plot_aggregate_shap_summary",
    "plot_aggregate_shap_waterfall",
    # Forecast plotting
    "plot_forecast",
    "plot_forecast_from_file",
]
