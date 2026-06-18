CALCULATE_METHODS = ["rsam", "dsar", "entropy"]
"""Default tremor calculation methods."""

DEFAULT_SAMPLING_STRATEGY: float = 0.75
"""Default RandomUnderSampler strategy (ratio of minority to majority class)."""

ERUPTION_PROBABILITY_THRESHOLD: float = 0.5
"""Threshold for classifying eruption probability as positive."""

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
