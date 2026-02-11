# Standard library imports
from typing import Literal

# Third party imports
import numpy as np
import pandas as pd
from obspy import Stream, Trace

# Project imports
from eruption_forecast.utils import calculate_window_metrics


class DSAR:
    """Calculate Displacement Seismic Amplitude Ratio (DSAR).

    Args:
        remove_outlier_method (Literal["maximum", "all"], optional): Remove outlier method. Defaults to "maximum".
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
        debug (bool, optional): Whether to enable debug logging. Defaults to False.
    """

    def __init__(
        self,
        remove_outlier_method: Literal["maximum", "all"] = "maximum",
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # =========================
        # Set DEFAULT properties
        # =========================
        self.remove_outlier_method = remove_outlier_method
        self.verbose = verbose
        self.debug = debug

        # =========================
        # Will be set after calculate() method called
        # =========================
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

        DSAR = Mean(Abs(Stream1)) / Mean(Abs(Stream2))

        Args:
            first_stream (Stream | pd.Series): First stream (Low Frequency).
            second_stream (Stream | pd.Series): Second stream (High Frequency).
            window_duration_minutes (int, optional): Window duration in minutes. Defaults to 10.
            value_multiplier (float, optional): Value multiplier. Defaults to 1.0.
            minimum_completion_ratio (float, optional): Minimum completion ratio. Defaults to 0.3.
            interpolate (bool, optional): Interpolate data. Defaults to True.

        Returns:
            pd.Series: DSAR series.
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

        if value_multiplier > 1:
            series: pd.Series[float] = series.apply(lambda x: x * value_multiplier)

        if interpolate:
            series: pd.Series[float] = series.interpolate(method="linear")

        self.series = series

        return series
