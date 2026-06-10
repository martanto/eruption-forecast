# Required DataFrame columns
ID_COLUMN = "id"
"""Column name for window identifiers."""

DATETIME_COLUMN = "datetime"
"""Column name for datetime values."""

ERUPTED_COLUMN = "is_erupted"
"""Column name for eruption labels."""

# Time conversion
SECONDS_PER_DAY = 86400
"""Number of seconds in one day (24 * 60 * 60)."""

# tsfresh feature exclusions
DEFAULT_EXCLUDE_FEATURES: set[str] = {
    "agg_linear_trend",
    "has_duplicate_max",
    "has_duplicate_min",
    "has_duplicate",
    "linear_trend_timewise",
    "length",
    "quantile",
    "sum_of_reoccurring_data_points",
    "sum_of_reoccurring_values",
    "value_count",
}
"""tsfresh calculator names excluded from feature extraction by default."""
