"""Tests to ensure no circular import errors exist in the package."""

import importlib
import sys


MODULES = [
    "eruption_forecast",
    "eruption_forecast.config.constants",
    "eruption_forecast.config.pipeline_config",
    "eruption_forecast.decorators.decorator_class",
    "eruption_forecast.features.constants",
    "eruption_forecast.features.features_builder",
    "eruption_forecast.features.feature_selector",
    "eruption_forecast.features.tremor_matrix_builder",
    "eruption_forecast.label.constants",
    "eruption_forecast.label.label_builder",
    "eruption_forecast.label.label_data",
    "eruption_forecast.logger",
    "eruption_forecast.model.classifier_model",
    "eruption_forecast.model.forecast_model",
    "eruption_forecast.model.metrics_computer",
    "eruption_forecast.model.model_evaluator",
    "eruption_forecast.model.model_predictor",
    "eruption_forecast.model.model_trainer",
    "eruption_forecast.model.multi_model_evaluator",
    "eruption_forecast.plots.evaluation_plots",
    "eruption_forecast.plots.feature_plots",
    "eruption_forecast.plots.forecast_plots",
    "eruption_forecast.plots.styles",
    "eruption_forecast.plots.tremor_plots",
    "eruption_forecast.sources.fdsn",
    "eruption_forecast.sources.sds",
    "eruption_forecast.tremor.calculate_tremor",
    "eruption_forecast.tremor.tremor_data",
    "eruption_forecast.utils.array",
    "eruption_forecast.utils.dataframe",
    "eruption_forecast.utils.date_utils",
    "eruption_forecast.utils.formatting",
    "eruption_forecast.utils.ml",
    "eruption_forecast.utils.pathutils",
    "eruption_forecast.utils.window",
]


def test_no_circular_imports():
    """Verify that importing each module does not raise a circular import error."""
    for module_name in MODULES:
        # Remove from sys.modules to force a fresh import each time.
        for key in list(sys.modules.keys()):
            if key.startswith("eruption_forecast"):
                del sys.modules[key]

        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise AssertionError(
                f"Circular or missing import detected in '{module_name}': {exc}"
            ) from exc


def test_top_level_public_api():
    """Verify that the top-level public API is importable without errors."""
    from eruption_forecast import (  # noqa: F401
        CalculateTremor,
        FeaturesBuilder,
        ForecastModel,
        LabelBuilder,
        LabelData,
        ModelEvaluator,
        ModelTrainer,
        MultiModelEvaluator,
        PipelineConfig,
        TremorData,
        TremorMatrixBuilder,
    )
