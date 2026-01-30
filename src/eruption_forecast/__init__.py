#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard library imports
from importlib.metadata import version

# Project imports
from eruption_forecast.features.features_builder import FeaturesBuilder
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.label.label_data import LabelData
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.model.forecast_model import ForecastModel

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
    "LabelData",
    "TremorData",
    "FeaturesBuilder",
    "ForecastModel",
]
