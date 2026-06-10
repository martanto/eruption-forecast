# Label filename format constants
WINDOW_STEP_PREFIX = "step-"
DAY_TO_FORECAST_PREFIX = "dtf-"

# Validation constants
MIN_DATE_RANGE_DAYS = 7
"""Minimum number of days required between start_date and end_date."""

# Valid window step units
VALID_WINDOW_STEP_UNITS = ["minutes", "hours"]
"""Valid units for window_step_unit parameter."""

# Date format
DATE_FORMAT = "%Y-%m-%d"
"""Standard date format string (YYYY-MM-DD)."""

# Example filename for error messages
EXAMPLE_LABEL_FILENAME = "label_2020-01-01_2020-12-31_step-6-hours_dtf-2_ie-1.csv"
"""Example of a correctly formatted label filename."""
