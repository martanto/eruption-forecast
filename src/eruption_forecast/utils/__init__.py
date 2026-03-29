"""Utility modules for eruption forecasting.

This package contains focused sub-modules that provide shared infrastructure
across the eruption-forecast pipeline. The public surface re-exported here
covers the most commonly needed helpers; the full set of functions is
available by importing from the individual sub-modules directly.

Sub-modules
-----------
- ``array`` — array manipulation, outlier detection, and seed-probability aggregation
- ``date_utils`` — date conversion, normalisation, sorting, and label-filename parsing
- ``dataframe`` — DataFrame validation, anomaly removal, and feature concatenation
- ``formatting`` — slug generation for class names and filesystem-safe strings
- ``ml`` — resampling, feature selection, metrics, and model-merging utilities
- ``pathutils`` — output-directory resolution, figure saving, and JSON loading
- ``validation`` — centralised guard functions for dates, columns, and window steps
- ``window`` — sliding-window construction and per-window metric computation
"""

from eruption_forecast.utils.date_utils import parse_label_filename
from eruption_forecast.utils.validation import (
    validate_columns,
    validate_date_ranges,
    validate_window_step,
    validate_random_state,
    check_sampling_consistency,
)


__all__ = [
    "parse_label_filename",
    "validate_random_state",
    "validate_date_ranges",
    "validate_window_step",
    "check_sampling_consistency",
    "validate_columns",
]
