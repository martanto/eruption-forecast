# Standard library imports
from datetime import datetime
from typing import Optional

# Third party imports
import numpy as np
import pandas as pd
from loguru import logger
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
        self.verbose = verbose
        self.debug = debug

    def calculate(
        self,
        first_stream: Stream,
        second_stream: Stream,
        window_duration_minutes: int = 10,
        value_multiplier: float = 1.0,
        minimum_completion_ratio: float = 0.3,
    ) -> pd.Series:
        """Calculate Displacement Seismic Amplitude Ratio (DSAR).

        DSAR = Mean(Abs(Stream1)) / Mean(Abs(Stream2))

        Args:
            first_stream (Stream): First stream (Low Frequency).
            second_stream (Stream): Second stream (High Frequency).
            window_duration_minutes (int, optional): Window duration in minutes. Defaults to 10.
            value_multiplier (float, optional): Value multiplier. Defaults to 1.0.
            minimum_completion_ratio (float, optional): Minimum completion ratio. Defaults to 0.3.

        Returns:
            pd.Series: DSAR series.
        """
        # Calculate mean of absolute values for both streams
        # Note: We use absolute_value=True inside calculate_window_metrics
        series1 = calculate_window_metrics(
            trace=first_stream[0],
            window_duration_minutes=window_duration_minutes,
            metric_function=np.mean,
            remove_outliers=self.remove_outliers,
            minimum_completion_ratio=minimum_completion_ratio,
            absolute_value=True,
        )

        series2 = calculate_window_metrics(
            trace=second_stream[0],
            window_duration_minutes=window_duration_minutes,
            metric_function=np.mean,
            remove_outliers=self.remove_outliers,
            minimum_completion_ratio=minimum_completion_ratio,
            absolute_value=True,
        )

        # Calculate DSAR ratio
        # Pandas handles division of Series with same index automatically
        dsar_series = series1 / series2

        if value_multiplier > 1:
            dsar_series = dsar_series.apply(lambda x: x * value_multiplier)

        return dsar_series
