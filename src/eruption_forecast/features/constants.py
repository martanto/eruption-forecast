"""Constants for the feature extraction module.

This module defines constants used throughout the feature extraction process,
including required column names and time conversion values.
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
