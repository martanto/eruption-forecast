"""
Centralized configuration constants for the eruption-forecast package.

This module contains all hardcoded values used throughout the package to ensure
consistency and ease of maintenance.
"""

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

# Plotting Constants
PLOT_DPI: int = 300
"""Default DPI for saving plot figures."""

PLOT_SEPARATOR_LENGTH: int = 50
"""Character length for console output separators."""
