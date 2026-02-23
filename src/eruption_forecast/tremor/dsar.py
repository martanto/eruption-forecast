from typing import Literal

import numpy as np
import pandas as pd
from obspy import Trace, Stream

from eruption_forecast.utils.window import calculate_window_metrics


class DSAR:
    """Displacement Seismic Amplitude Ratio (DSAR) calculator.

    Computes the ratio of mean absolute amplitudes between two frequency bands.
    DSAR helps identify changes in the spectral content of volcanic tremor.

    Attributes:
        remove_outlier_method (Literal["maximum", "all"]): Outlier removal strategy.
        verbose (bool): If True, enables verbose logging.
        debug (bool): If True, enables debug-level logging.
        first_dsar (pd.Series | None): Amplitude series from the first stream (set after calculate()).
        second_dsar (pd.Series | None): Amplitude series from the second stream (set after calculate()).
        series (pd.Series | None): Calculated DSAR ratio series (set after calculate()).

    Args:
        remove_outlier_method (Literal["maximum", "all"]): Outlier removal strategy.
            "maximum" removes only the single maximum outlier; "all" removes all outliers.
            Defaults to "maximum".
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug-level logging. Defaults to False.

    Examples:
        >>> from obspy import read
        >>> stream_lf = read("low_freq.mseed")
        >>> stream_hf = read("high_freq.mseed")
        >>> dsar = DSAR(verbose=True)
        >>> ratio = dsar.calculate(stream_lf, stream_hf)
        >>> print(ratio.head())
    """

    def __init__(
        self,
        remove_outlier_method: Literal["maximum", "all"] = "maximum",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the DSAR calculator with outlier removal and logging settings.

        Sets processing flags and initialises result attributes to None. No
        computation occurs until calculate() is called.

        Args:
            remove_outlier_method (Literal["maximum", "all"], optional): Outlier
                removal strategy applied to each amplitude series before computing
                the ratio. "maximum" removes only the global maximum outlier;
                "all" removes all Z-score outliers. Defaults to "maximum".
            verbose (bool, optional): Emit progress log messages. Defaults to False.
            debug (bool, optional): Emit debug log messages. Defaults to False.
        """
        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.remove_outlier_method = remove_outlier_method
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Will be set after calculate() method called
        # ------------------------------------------------------------------
        self.first_dsar: pd.Series | None = None
        self.second_dsar: pd.Series | None = None
        self.series: pd.Series | None = None

    def calculate(
        self,
        first_stream: Stream | pd.Series,
        second_stream: Stream | pd.Series,
        window_duration_minutes: int = 10,
        value_multiplier: float = 1.0,
        minimum_completion_ratio: float = 0.3,
        interpolate: bool = True,
    ) -> pd.Series:
        """Calculate Displacement Seismic Amplitude Ratio (DSAR).

        Computes the ratio of mean absolute amplitudes between two streams:
        DSAR = Mean(|Stream1|) / Mean(|Stream2|). If a Stream object is provided,
        it is first converted to a windowed amplitude series.

        Args:
            first_stream (Stream | pd.Series): First stream (typically low frequency band).
                If Stream, amplitudes are computed over windows. If Series, used directly.
            second_stream (Stream | pd.Series): Second stream (typically high frequency band).
                If Stream, amplitudes are computed over windows. If Series, used directly.
            window_duration_minutes (int): Duration of each window in minutes.
                Defaults to 10.
            value_multiplier (float): Scaling factor applied to the DSAR ratio.
                Defaults to 1.0 (no scaling).
            minimum_completion_ratio (float): Minimum fraction of data points required
                in a window to compute metrics (0.0-1.0). Defaults to 0.3.
            interpolate (bool): If True, applies linear interpolation to fill NaN values.
                Defaults to True.

        Returns:
            pd.Series: DSAR ratio time-series with DatetimeIndex.

        Examples:
            >>> from obspy import read
            >>> lf = read("low_freq.mseed")
            >>> hf = read("high_freq.mseed")
            >>> dsar = DSAR()
            >>> ratio = dsar.calculate(lf, hf, window_duration_minutes=10)
            >>> print(ratio.mean())
        """
        # Calculate mean of absolute values for both streams
        # Note: We use absolute_value=True inside calculate_window_metrics
        if isinstance(first_stream, Stream):
            trace: Trace = first_stream[0]
            first_stream = calculate_window_metrics(
                trace=trace,
                window_duration_minutes=window_duration_minutes,
                metric_function=np.mean,
                remove_outlier_method=self.remove_outlier_method,
                minimum_completion_ratio=minimum_completion_ratio,
                absolute_value=True,
            )
        if isinstance(second_stream, Stream):
            trace: Trace = second_stream[0]
            second_stream = calculate_window_metrics(
                trace=trace,
                window_duration_minutes=window_duration_minutes,
                metric_function=np.mean,
                remove_outlier_method=self.remove_outlier_method,
                minimum_completion_ratio=minimum_completion_ratio,
                absolute_value=True,
            )

        first_dsar: pd.Series = first_stream
        second_dsar: pd.Series = second_stream

        self.first_dsar = first_dsar
        self.second_dsar: pd.Series[float] = pd.Series(second_dsar)

        # Calculate DSAR ratio
        # Pandas handles division of Series with same index automatically
        series: pd.Series[float] = first_dsar / second_dsar

        if value_multiplier != 1.0:
            series: pd.Series[float] = series.apply(lambda x: x * value_multiplier)

        if interpolate:
            series: pd.Series[float] = series.interpolate(method="linear")

        self.series = series

        return series
