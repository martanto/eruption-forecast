"""Publication-quality scientific plotting for volcanic eruption forecasting.

This module provides Nature/Science journal-standard visualizations for:
- Tremor time-series analysis
- Feature importance and selection
- Model evaluation (ROC, PR, confusion matrix, calibration)
- Eruption forecast probability plots

All plots use consistent styling, colorblind-safe palettes, and publication-ready
typography.
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

# Tremor plotting
from eruption_forecast.plots.tremor_plots import plot_tremor

# Feature plotting
from eruption_forecast.plots.feature_plots import plot_significant_features

# Forecast plotting
from eruption_forecast.plots.forecast_plots import (
    plot_forecast,
    plot_forecast_with_events,
)

# Evaluation plotting
from eruption_forecast.plots.evaluation_plots import (
    plot_roc_curve,
    plot_calibration,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_threshold_analysis,
    plot_precision_recall_curve,
    plot_prediction_distribution,
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
    # Evaluation plotting
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_threshold_analysis",
    "plot_feature_importance",
    "plot_calibration",
    "plot_prediction_distribution",
    # Forecast plotting
    "plot_forecast",
    "plot_forecast_with_events",
]
