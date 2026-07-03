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

The pipeline can be orchestrated through the ``ForecastModel`` class or used as individual components.

Examples:
    >>> from eruption_forecast import ForecastModel
    >>> fm = ForecastModel(
    ...     network="VG", station="OJN", location="00", channel="EHZ",
    ...     day_to_forecast=2, root_dir="output", n_jobs=4,
    ... )
    >>> (
    ...     fm.calculate(
    ...         start_date="2025-01-01", end_date="2025-12-31",
    ...         source="sds", sds_dir="/data/sds",
    ...     )
    ...     .train(
    ...         start_date="2025-01-01", end_date="2025-12-31",
    ...         eruption_dates=["2025-06-15"],
    ...         classifiers=["rf", "xgb"],
    ...     )
    ...     .predict(start_date="2025-07-01", end_date="2025-07-31")
    ...     .evaluate(model="prediction")
    ...     .explain(model="prediction")
    ... )
"""

from importlib.metadata import version

from eruption_forecast.logger import enable_logging, disable_logging
from eruption_forecast.decorators.timer import timer
from eruption_forecast.label.label_data import LabelData
from eruption_forecast.decorators.notify import notify
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.model.forecast_model import ForecastModel
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.notification.telegram import TelegramNotification
from eruption_forecast.model.evaluation_model import EvaluationModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.model.explanation_model import ExplanationModel
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.label.dynamic_label_builder import DynamicLabelBuilder
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
    "__url__",
    "LabelData",
    "TremorData",
    "LabelBuilder",
    "ForecastModel",
    "TrainingModel",
    "EvaluationModel",
    "ExplanationModel",
    "PredictionModel",
    "CalculateTremor",
    "FeaturesBuilder",
    "DynamicLabelBuilder",
    "TremorMatrixBuilder",
    "disable_logging",
    "enable_logging",
    "notify",
    "timer",
    "TelegramNotification",
]
