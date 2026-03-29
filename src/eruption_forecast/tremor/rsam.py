"""Real Seismic Amplitude Measurement (RSAM) calculator.

This module provides the ``RSAM`` class, which computes mean absolute seismic
amplitude over fixed-duration time windows for each frequency band. RSAM is a
widely used metric for monitoring volcanic tremor intensity and is the primary
amplitude measure produced by ``CalculateTremor``.

Key class:
    - ``RSAM``: Accepts an ObsPy ``Trace`` or ``Stream``, applies a bandpass filter
      via ``filter()``, and computes windowed mean absolute amplitude via
      ``calculate()``. Returns a ``pd.Series`` indexed by window start times.

Usage::

    rsam = RSAM(stream=stream).filter(freqmin=2.0, freqmax=5.0).calculate()
"""

from typing import Self, Literal
from datetime import datetime
from collections.abc import Callable

import numpy as np
import pandas as pd
from obspy import Trace, Stream
from loguru import logger

from eruption_forecast.utils.window import calculate_window_metrics
from eruption_forecast.config.constants import (
    BANDPASS_FILTER_CORNERS,
    DEFAULT_WINDOW_DURATION_MINUTES,
    DEFAULT_MINIMUM_COMPLETION_RATIO,
)


class RSAM:
    """Real Seismic Amplitude Measurement (RSAM) calculator.

    Calculates mean absolute amplitude over time windows for seismic signals.
    RSAM is a widely-used metric for monitoring volcanic tremor intensity.

    Attributes:
        trace (Trace): ObsPy Trace object containing seismic waveform data.
        start_datetime (datetime): Start datetime of the trace.
        start_datetime_str (str): Start date in "YYYY-MM-DD" format.
        verbose (bool): If True, enables verbose logging.
        debug (bool): If True, enables debug-level logging.
        is_filtered (bool): True if bandpass filter has been applied.
        series (pd.Series): Calculated RSAM time-series (set after calculate()).

    Args:
        stream (Stream): ObsPy Stream object containing seismic trace.
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug-level logging. Defaults to False.

    Examples:
        >>> from obspy import read
        >>> stream = read("seismic_data.mseed")
        >>> rsam = RSAM(stream, verbose=True)
        >>> series = rsam.apply_filter(0.1, 2.0).calculate()
        >>> print(series.head())
    """

    def __init__(self, stream: Stream, verbose: bool = False, debug: bool = False):
        """Initialize the RSAM calculator with a seismic stream and logging settings.

        Extracts the first trace from the stream, records the trace start time,
        and initialises result attributes to their empty defaults. No filtering
        or metric computation occurs until apply_filter() and calculate() are called.

        Args:
            stream (Stream): ObsPy Stream object containing the seismic waveform.
                The first trace (stream[0]) is used for all subsequent operations.
            verbose (bool, optional): Emit progress log messages. Defaults to False.
            debug (bool, optional): Emit debug log messages. Defaults to False.
        """
        trace: Trace = stream[0]
        start_datetime: datetime = trace.stats.starttime.datetime

        self.trace = trace
        self.start_datetime = start_datetime
        self.verbose = verbose
        self.debug = debug
        self.start_datetime_str = start_datetime.strftime("%Y-%m-%d")

        # Will be set after apply_filter() is called.
        self.is_filtered = False

        # Will be set after calculate() is called.
        self.series: pd.Series = pd.Series(dtype=float)

    def apply_filter(self, freq_min: float, freq_max: float) -> Self:
        """Apply bandpass filter to the seismic trace.

        Filters the trace using a 4th-order Butterworth bandpass filter.
        This method modifies the trace in-place and sets is_filtered to True.

        Args:
            freq_min (float): Minimum frequency in Hz (lower corner).
            freq_max (float): Maximum frequency in Hz (upper corner).

        Returns:
            Self: The RSAM instance for method chaining.

        Examples:
            >>> rsam = RSAM(stream)
            >>> rsam.apply_filter(0.1, 2.0)
            >>> print(rsam.is_filtered)  # True
        """
        self.trace = self.trace.filter(
            "bandpass", freqmin=freq_min, freqmax=freq_max, corners=BANDPASS_FILTER_CORNERS
        )

        # Set is_filtered to True
        self.is_filtered = True

        if self.debug:
            logger.debug(
                f"{self.start_datetime_str} :: RSAM Filtered using ({freq_min},{freq_max})"
            )

        return self

    def calculate(
        self,
        window_duration_minutes: int = DEFAULT_WINDOW_DURATION_MINUTES,
        metric_function: Callable[[np.ndarray], float] = np.nanmean,
        value_multiplier: float = 1.0,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        minimum_completion_ratio: float = DEFAULT_MINIMUM_COMPLETION_RATIO,
        interpolate: bool = True,
    ) -> pd.Series:
        """Calculate RSAM metrics over sliding time windows.

        Divides the trace into non-overlapping windows and computes a metric
        (typically mean absolute amplitude) for each window. Supports outlier
        removal and linear interpolation of missing values.

        Args:
            window_duration_minutes (int): Duration of each window in minutes.
                Defaults to 10.
            metric_function (Callable[[np.ndarray], float]): Function to calculate
                the metric over each window (e.g., np.mean, np.max, np.median).
                Defaults to np.mean.
            value_multiplier (float): Scaling factor applied to all metric values.
                Defaults to 1.0 (no scaling).
            remove_outlier_method (Literal["all", "maximum"]): Outlier removal strategy.
                "maximum" removes only the single maximum outlier per window;
                "all" removes all detected outliers. Defaults to "maximum".
            minimum_completion_ratio (float): Minimum fraction of data points required
                in a window to compute the metric (0.0-1.0). Windows with fewer points
                are set to NaN. Defaults to 0.3.
            interpolate (bool): If True, applies linear interpolation to fill NaN values.
                Defaults to True.

        Returns:
            pd.Series: Time-series of RSAM values with DatetimeIndex.

        Examples:
            >>> rsam = RSAM(stream)
            >>> series = rsam.apply_filter(0.1, 2.0).calculate(window_duration_minutes=10)
            >>> print(series.describe())
        """
        trace = self.trace

        series = calculate_window_metrics(
            trace=trace,
            window_duration_minutes=window_duration_minutes,
            metric_function=metric_function,
            remove_outlier_method=remove_outlier_method,
            value_multiplier=value_multiplier,
            minimum_completion_ratio=minimum_completion_ratio,
            absolute_value=True,
        )

        # Note: value_multiplier is already applied in calculate_window_metrics
        # No need to apply it again here

        if interpolate:
            series = series.interpolate(method="time")

        self.series = series

        return series
