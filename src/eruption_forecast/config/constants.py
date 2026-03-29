"""Centralised numeric and string constants for the eruption-forecast package.

Default values that are referenced across multiple modules live here so they
can be changed in one place. Every constant is annotated with its type and
accompanied by a module-level docstring that explains its purpose and
default value.

Contents:
    - **Tremor calculation**: ``CALCULATE_METHODS``, ``DEFAULT_FREQUENCY_BANDS``,
      ``BANDPASS_FILTER_CORNERS``, ``DEFAULT_WINDOW_DURATION_MINUTES``,
      ``DEFAULT_SAMPLING_FREQUENCY``, ``DEFAULT_MINIMUM_COMPLETION_RATIO``.
    - **Model training**: ``TRAIN_TEST_SPLIT``, ``DEFAULT_CV_SPLITS``,
      ``DEFAULT_N_SIGNIFICANT_FEATURES``, ``DEFAULT_SAMPLING_STRATEGY``,
      ``LEARNING_CURVE_SCORINGS``.
    - **Inference / evaluation**: ``ERUPTION_PROBABILITY_THRESHOLD``,
      ``THRESHOLD_RESOLUTION``, ``CLASS_LABELS``.
    - **Plotting / output**: ``PLOT_DPI``, ``PLOT_SEPARATOR_LENGTH``,
      ``MATPLOTLIB_BACKEND``.

Import these constants directly from ``eruption_forecast.config`` (which
re-exports them) or from this module.
"""

CALCULATE_METHODS = ["rsam", "dsar", "entropy"]
"""Default tremor calculation methods."""

TRAIN_TEST_SPLIT: float = 0.2
"""Train/test split ratio for model evaluation (80/20 split)."""

DEFAULT_CV_SPLITS: int = 5
"""Default number of cross-validation splits."""

DEFAULT_N_SIGNIFICANT_FEATURES: int = 20
"""Default number of top features to select."""

DEFAULT_SAMPLING_STRATEGY: float = 0.75
"""Default RandomUnderSampler strategy (ratio of minority to majority class)."""

ERUPTION_PROBABILITY_THRESHOLD: float = 0.5
"""Threshold for classifying eruption probability as positive."""

# TODO: Consider to reduce the value. Affecting in smoothing PR curves
THRESHOLD_RESOLUTION: int = 101
"""Number of threshold points to evaluate (for ROC/PR curves)."""

LEARNING_CURVE_SCORINGS: list[str] = ["balanced_accuracy", "f1_weighted"]
"""Scoring metrics computed for every learning curve during training."""

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

MATPLOTLIB_BACKEND: str = "Agg"
"""Non-interactive Matplotlib backend, safe for worker threads."""

CLASS_LABELS: list[str] = ["Not Erupted", "Erupted"]
"""Display labels for the binary classification classes (index 0 = not erupted, 1 = erupted)."""

PLOT_DPI: int = 300
"""Default DPI for saving plot figures."""

PLOT_SEPARATOR_LENGTH: int = 50
"""Character length for console output separators."""
