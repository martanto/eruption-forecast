"""Constants for the label building module.

This module defines constants used throughout the label building process,
including filename format prefixes, validation thresholds, and default parameter
values for window configuration and labeling logic.

The label filename format is:
    label_{start_date}_{end_date}_step-{window_step}-{unit}_dtf-{day_to_forecast}.csv

Where:
    - start_date, end_date: YYYY-MM-DD format
    - window_step: Integer step size
    - unit: "hours" or "minutes"
    - day_to_forecast: Integer days before eruption

Examples:
    >>> # Valid label filename
    >>> filename = f"{LABEL_PREFIX}2020-01-01_2020-12-31_step-12-hours_dtf-2{LABEL_EXTENSION}"
    >>> print(filename)
    'label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv'

    >>> # Validation constants
    >>> print(f"Minimum date range: {MIN_DATE_RANGE_DAYS} days")
    Minimum date range: 7 days

    >>> # Check valid units
    >>> "hours" in VALID_WINDOW_STEP_UNITS
    True
"""

# Label filename format constants
LABEL_PREFIX = "label_"
LABEL_EXTENSION = ".csv"
WINDOW_STEP_PREFIX = "step-"
DAY_TO_FORECAST_PREFIX = "dtf-"

# Validation constants
MIN_DATE_RANGE_DAYS = 7
"""Minimum number of days required between start_date and end_date."""

# Default parameter values
DEFAULT_WINDOW_STEP = 12
"""Default window step size."""

DEFAULT_WINDOW_STEP_UNIT = "hours"
"""Default unit for window step ('hours' or 'minutes')."""

DEFAULT_DAY_TO_FORECAST = 2
"""Default number of days before eruption to start labeling as positive."""

# Valid window step units
VALID_WINDOW_STEP_UNITS = ["minutes", "hours"]
"""Valid units for window_step_unit parameter."""

# Required DataFrame columns
REQUIRED_LABEL_COLUMNS = ["id", "is_erupted"]
"""Required columns in label DataFrames."""

# Date format
DATE_FORMAT = "%Y-%m-%d"
"""Standard date format string (YYYY-MM-DD)."""

# Example filename for error messages
EXAMPLE_LABEL_FILENAME = "label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv"
"""Example of a correctly formatted label filename."""
