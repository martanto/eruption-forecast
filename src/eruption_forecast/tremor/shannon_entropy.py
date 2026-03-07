from typing import Self, Literal

import pandas as pd
from obspy import Trace, Stream

from eruption_forecast.utils.window import shannon_entropy, calculate_window_metrics
from eruption_forecast.config.constants import (
    BANDPASS_FILTER_CORNERS,
    DEFAULT_WINDOW_DURATION_MINUTES,
    DEFAULT_MINIMUM_COMPLETION_RATIO,
)


class ShanonEntropy:
    """Calculate Shannon entropy from a seismic stream.

    Computes Shannon entropy from seismic waveform data by applying a bandpass
    filter and computing the entropy metric over fixed-duration time windows.
    Shannon entropy quantifies the complexity or disorder of the seismic signal
    and is useful for distinguishing volcanic tremor from background noise.

    Attributes:
        trace (Trace): ObsPy Trace object extracted from the input stream.
        verbose (bool): Whether to enable verbose logging.
        debug (bool): Whether to enable debug-level logging.
        freqmin (float): Lower corner frequency for the bandpass filter in Hz.
        freqmax (float): Upper corner frequency for the bandpass filter in Hz.

    Args:
        stream (Stream): ObsPy Stream object containing the seismic waveform.
        verbose (bool, optional): If True, enables verbose logging. Defaults to False.
        debug (bool, optional): If True, enables debug-level logging. Defaults to False.

    Examples:
        >>> from obspy import read
        >>> stream = read("seismic_data.mseed")
        >>> entropy = ShanonEntropy(stream).filter(1.0, 16.0).calculate()
    """

    def __init__(self, stream: Stream, verbose: bool = False, debug: bool = False):
        """Initialize ShanonEntropy from an ObsPy Stream.

        Extracts the first trace from the stream and sets default filter bounds
        of 1–16 Hz.

        Args:
            stream (Stream): ObsPy Stream object containing seismic waveform data.
                Only the first trace is used.
            verbose (bool, optional): If True, enables verbose logging. Defaults to False.
            debug (bool, optional): If True, enables debug-level logging. Defaults to False.
        """
        trace: Trace = stream[0]

        self.trace: Trace = trace
        self.verbose = verbose
        self.debug = debug

        self.freqmin: float = 1.0
        self.freqmax: float = 16.0

    def filter(self, freqmin: float, freqmax: float) -> Self:
        """Set the bandpass filter frequency bounds.

        Updates the lower and upper corner frequencies that will be applied
        to the seismic trace before entropy calculation.

        Args:
            freqmin (float): Minimum frequency of data to be filtered.
            freqmax (float): Maximum frequency of data to be filtered.

        Returns:
            Self: ShanonEntropy instance for method chaining.

        Raises:
            ValueError: If freqmin is greater than or equal to freqmax.
        """
        if freqmin > freqmax:
            raise ValueError(
                f"Frequency minimum ({freqmin}) must be less than frequency maximum ({freqmax})."
            )

        self.freqmin = float(freqmin)
        self.freqmax = float(freqmax)

        return self

    def calculate(
        self,
        window_duration_minutes: int = DEFAULT_WINDOW_DURATION_MINUTES,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        minimum_completion_ratio: float = DEFAULT_MINIMUM_COMPLETION_RATIO,
        window_overlap: float | None = None,
        interpolate: bool = True,
    ) -> pd.Series:
        """Calculate the Shannon entropy metric over windowed seismic data.

        Applies the configured bandpass filter to the trace and computes Shannon
        entropy for each time window using ``calculate_window_metrics()``. Windows
        with insufficient data (below ``minimum_completion_ratio``) are set to NaN,
        and optionally gap-filled with linear interpolation.

        Args:
            window_duration_minutes (int, optional): Duration of each window in minutes.
                Defaults to 10.
            remove_outlier_method (Literal["all", "maximum"], optional): Outlier removal
                strategy. "maximum" removes only the single maximum outlier per window;
                "all" removes all detected outliers. Defaults to "maximum".
            minimum_completion_ratio (float, optional): Minimum fraction of valid data
                points required per window (0.0–1.0). Windows below this threshold
                are set to NaN. Defaults to 0.3.
            window_overlap (float | None, optional): Fractional window overlap as a
                percentage (0–100). If None, windows are non-overlapping. Defaults to None.
            interpolate (bool, optional): If True, applies linear interpolation to fill
                NaN values after windowing. Defaults to True.

        Returns:
            pd.Series: Time-series of Shannon entropy values with DatetimeIndex at
                ``window_duration_minutes`` intervals.
        """
        trace: Trace = self.trace.copy()
        trace.filter("bandpass", freqmin=self.freqmin, freqmax=self.freqmax, corners=BANDPASS_FILTER_CORNERS)

        series = calculate_window_metrics(
            trace=trace,
            window_duration_minutes=window_duration_minutes,
            metric_function=shannon_entropy,
            remove_outlier_method=remove_outlier_method,
            mask_zero_value=True,
            minimum_completion_ratio=minimum_completion_ratio,
            absolute_value=True,
            window_overlap=window_overlap,
        )

        if interpolate:
            series = series.interpolate(method="time")

        return series
