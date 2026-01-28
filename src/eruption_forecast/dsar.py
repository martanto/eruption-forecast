# Standard library imports
from typing import Optional, Union

# Third party imports
import numpy as np
import pandas as pd
from obspy import Stream

from .utils import calculate_window_metrics


class DSAR:
    """Calculate Displacement Seismic Amplitude Ratio (DSAR).

    Args:
        remove_outliers (bool, optional): Whether to remove outliers. Defaults to True.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
        debug (bool, optional): Whether to enable debug logging. Defaults to False.
    """

    def __init__(
        self, remove_outliers: bool = True, verbose: bool = False, debug: bool = False
    ) -> None:
        self.remove_outliers = remove_outliers

        self.first_dsar: Optional[pd.Series] = None
        self.second_dsar: Optional[pd.Series] = None
        self.series: Optional[pd.Series] = None

        self.verbose = verbose
        self.debug = debug

    def calculate(
        self,
        first_stream: Union[Stream, pd.Series],
        second_stream: Union[Stream, pd.Series],
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
        first_dsar = first_stream
        if isinstance(first_stream, Stream):
            first_dsar = calculate_window_metrics(
                trace=first_stream[0],
                window_duration_minutes=window_duration_minutes,
                metric_function=np.mean,
                remove_outliers=self.remove_outliers,
                minimum_completion_ratio=minimum_completion_ratio,
                absolute_value=True,
            )

        second_dsar = second_stream
        if isinstance(second_stream, Stream):
            second_dsar = calculate_window_metrics(
                trace=second_stream[0],
                window_duration_minutes=window_duration_minutes,
                metric_function=np.mean,
                remove_outliers=self.remove_outliers,
                minimum_completion_ratio=minimum_completion_ratio,
                absolute_value=True,
            )

        self.first_dsar = first_dsar
        self.second_dsar = second_dsar

        # Calculate DSAR ratio
        # Pandas handles division of Series with same index automatically
        series = first_dsar / second_dsar

        if value_multiplier > 1:
            series = series.apply(lambda x: x * value_multiplier)

        if interpolate:
            series = series.interpolate(method="linear")

        self.series = series

        return series
