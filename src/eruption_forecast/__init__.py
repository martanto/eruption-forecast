#!/usr/bin/env python
"""
Volcanic eruption forecasting using seismic tremor analysis and machine learning.

This package implements a complete pipeline for processing seismic data, extracting features,
and training predictive models for volcanic eruption forecasting. It includes modules for:

- Tremor calculation (RSAM, DSAR metrics)
- Label generation for supervised learning
- Feature extraction using tsfresh
- Model training and evaluation
- Eruption forecasting

The pipeline can be orchestrated through the `ForecastModel` class or used as individual components.

Examples:
    >>> from eruption_forecast import ForecastModel
    >>> fm = ForecastModel(root_dir="output", station="OJN", channel="EHZ",
    ...                    start_date="2025-01-01", end_date="2025-12-31",
    ...                    window_size=2, volcano_id="VOLCANO_001")
    >>> fm.calculate(source="sds", sds_dir="/data/sds").build_label(
    ...     eruption_dates=["2025-06-15"], day_to_forecast=2
    ... ).extract_features().train(classifier="xgb")
"""

from importlib.metadata import version

from eruption_forecast.label.label_data import LabelData
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.model.model_trainer import ModelTrainer
from eruption_forecast.model.forecast_model import ForecastModel
from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.config.pipeline_config import PipelineConfig
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator
from eruption_forecast.features.tremor_matrix_builder import TremorMatrixBuilder


__version__ = version("eruption-forecast")
__author__ = "Martanto"
__author_email__ = "martanto@live.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025, Martanto"
__url__ = "https://github.com/martanto/eruption-forecast"

__all__ = [
    "__version__",
    "__author__",
    "__author_email__",
    "__license__",
    "__copyright__",
    "CalculateTremor",
    "LabelBuilder",
    "FeaturesBuilder",
    "TremorMatrixBuilder",
    "ForecastModel",
    "ModelTrainer",
    "ModelEvaluator",
    "MultiModelEvaluator",
    "LabelData",
    "TremorData",
    "PipelineConfig",
]
