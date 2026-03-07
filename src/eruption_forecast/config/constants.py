"""
Centralized configuration constants for the eruption-forecast package.

This module contains all hardcoded values used throughout the package to ensure
consistency and ease of maintenance.
"""

# Calculating methods
CALCULATE_METHODS = ["rsam", "dsar", "entropy"]
"""Default tremor calculation methods"""

# Model Training Constants
TRAIN_TEST_SPLIT: float = 0.2
"""Train/test split ratio for model evaluation (default: 80/20 split)."""

DEFAULT_CV_SPLITS: int = 5
"""Default number of cross-validation splits."""

DEFAULT_N_SIGNIFICANT_FEATURES: int = 20
"""Default number of top features to select."""

DEFAULT_SAMPLING_STRATEGY: float = 0.75
"""Default RandomUnderSampler strategy (ratio of minority to majority class)."""

# Prediction Constants
ERUPTION_PROBABILITY_THRESHOLD: float = 0.5
"""Threshold for classifying eruption probability as positive."""

THRESHOLD_RESOLUTION: int = 101
"""Number of threshold points to evaluate (for ROC/PR curves)."""

LEARNING_CURVE_SCORINGS: list[str] = ["balanced_accuracy", "f1_weighted"]
"""Scoring metrics computed for every learning curve during training."""

# Tremor Calculation Constants
DEFAULT_WINDOW_DURATION_MINUTES: int = 10
"""Duration of each tremor calculation window in minutes."""

DEFAULT_SAMPLING_FREQUENCY: str = "10min"
"""Expected pandas frequency string for tremor time-series resampling."""

DEFAULT_MINIMUM_COMPLETION_RATIO: float = 0.3
"""Minimum fraction of valid data points required in a window before it is accepted."""

BANDPASS_FILTER_CORNERS: int = 4
"""Number of corners (poles) for the ObsPy bandpass filter."""

DEFAULT_FREQUENCY_BANDS: list[tuple[float, float]] = [
    (0.01, 0.1),
    (0.1, 2),
    (2, 5),
    (4.5, 8),
    (8, 16),
]
"""Default seismic frequency bands (Hz) used for RSAM and DSAR calculation."""

# Plotting Constants
MATPLOTLIB_BACKEND: str = "Agg"
"""Non-interactive Matplotlib backend, safe for worker threads."""

CLASS_LABELS: list[str] = ["Not Erupted", "Erupted"]
"""Display labels for the binary classification classes (index 0 = not erupted, 1 = erupted)."""

PLOT_DPI: int = 300
"""Default DPI for saving plot figures."""

PLOT_SEPARATOR_LENGTH: int = 50
"""Character length for console output separators."""
