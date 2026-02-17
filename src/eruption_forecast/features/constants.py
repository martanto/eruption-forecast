"""Constants for the feature extraction module.

This module defines constants used throughout feature extraction, matrix building,
and feature selection. Constants include required DataFrame column names,
time conversion values, output filenames, and tsfresh feature exclusions.

Constants:
    ID_COLUMN (str): Column name for window identifiers ("id").
    DATETIME_COLUMN (str): Column name for datetime values ("datetime").
    ERUPTED_COLUMN (str): Column name for eruption labels ("is_erupted").
    FEATURES_COLUMN (str): Column name for feature names in significant features output.
    SECONDS_PER_DAY (int): Number of seconds in one day (86400).
    SIGNIFICANT_FEATURES_FILENAME (str): Default filename for concatenated significant features.
    DEFAULT_EXCLUDE_FEATURES (set[str]): tsfresh calculator names excluded by default.

Examples:
    >>> from eruption_forecast.features.constants import ID_COLUMN, DATETIME_COLUMN
    >>> required_columns = [ID_COLUMN, DATETIME_COLUMN]
    >>> print(required_columns)
    ['id', 'datetime']
    >>>
    >>> # Check sampling period
    >>> from eruption_forecast.features.constants import SECONDS_PER_DAY
    >>> samples_per_day = SECONDS_PER_DAY // 600  # 600s = 10-minute intervals
    >>> print(samples_per_day)
    144
"""

# Required DataFrame columns
ID_COLUMN = "id"
"""Column name for window identifiers."""

DATETIME_COLUMN = "datetime"
"""Column name for datetime values."""

ERUPTED_COLUMN = "is_erupted"
"""Column name for eruption labels."""

FEATURES_COLUMN = "features"
"""Column name for feature names in significant features output."""

# Time conversion
SECONDS_PER_DAY = 86400
"""Number of seconds in one day (24 * 60 * 60)."""

# Output filenames
SIGNIFICANT_FEATURES_FILENAME = "significant_features"
"""Default filename for concatenated significant features output."""

# tsfresh feature exclusions
DEFAULT_EXCLUDE_FEATURES: set[str] = {
    "agg_linear_trend",
    "has_duplicate_max",
    "has_duplicate_min",
    "has_duplicate",
    "linear_trend_timewise",
    "length",
    "sum_of_reoccurring_data_points",
    "sum_of_reoccurring_values",
    "value_count",
}
"""tsfresh calculator names excluded from feature extraction by default."""
