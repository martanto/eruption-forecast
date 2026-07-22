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

# Label plotting — imported last to avoid triggering model/utils circular chains
from eruption_forecast.label.label_plots import plot_label_distribution

# Tremor plotting
from eruption_forecast.plots.tremor_plots import (
    plot_tremor,
    plot_tremor_from_file,
)

# Feature plotting
from eruption_forecast.plots.feature_plots import (
    plot_significant_features,
    replot_significant_features,
    plot_common_features_heatmap,
    plot_common_features_correlation,
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
    plot_confusion_matrix,
    plot_feature_importance,
    plot_threshold_analysis,
    plot_precision_recall_curve,
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
    "plot_tremor_from_file",
    # Feature plotting
    "plot_significant_features",
    "replot_significant_features",
    "plot_frequency_band_contribution",
    "plot_common_features_heatmap",
    "plot_common_features_correlation",
    # Evaluation plotting
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
    "plot_threshold_analysis",
    "plot_feature_importance",
    # Forecast plotting
    "plot_forecast",
    "plot_forecast_from_file",
    # Label plotting
    "plot_label_distribution",
]
