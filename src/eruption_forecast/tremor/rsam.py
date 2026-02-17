from typing import Self, Literal
from datetime import datetime
from collections.abc import Callable

import numpy as np
import pandas as pd
from obspy import Trace, Stream
from loguru import logger

from eruption_forecast.utils import calculate_window_metrics


class RSAM:
    """Real Seismic Amplitude Measurement (RSAM) class.

    Args:
        stream (Stream): Obspy stream object.
        verbose (bool, optional): Verbose mode. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.
    """

    def __init__(self, stream: Stream, verbose: bool = False, debug: bool = False):
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        trace: Trace = stream[0]
        start_datetime: datetime = trace.stats.starttime.datetime

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.trace = trace
        self.start_datetime = start_datetime
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self.start_datetime_str = start_datetime.strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # Will be set after apply_filter() method called
        # ------------------------------------------------------------------
        self.is_filtered = False

        # ------------------------------------------------------------------
        # Will be set after calculate() method called
        # ------------------------------------------------------------------
        self.series: pd.Series = pd.Series(dtype=float)

    def apply_filter(self, freq_min: float, freq_max: float) -> Self:
        """Apply filter to the trace.

        Args:
            freq_min (float): Minimum frequency.
            freq_max (float): Maximum frequency.

        Returns:
            Self: RSAM object
        """
        self.trace = self.trace.filter(
            "bandpass", freqmin=freq_min, freqmax=freq_max, corners=4
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
        window_duration_minutes: int = 10,
        metric_function: Callable[[np.ndarray], float] = np.mean,
        value_multiplier: float = 1.0,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        minimum_completion_ratio: float = 0.3,
        interpolate: bool = True,
    ) -> pd.Series:
        """Calculate metrics for defined time windows of a Trace.

        Args:
            window_duration_minutes (int, optional): Duration of each window in minutes. Defaults to 10.
            metric_function (callable, optional): Function to calculate metric (e.g., np.mean, np.max). Defaults to np.mean.
            value_multiplier (float, optional): Value multiplier. Defaults to 1.0.
            remove_outlier_method (Literal["maximum", "all"], optional): Remove outlier method. Defaults to "maximum".
            minimum_completion_ratio (float, optional): Minimum ratio of data points required to calculate metric. Defaults to 0.3.
            interpolate (bool, optional): Interpolate data. Defaults to True.

        Returns:
            pd.Series: Series containing calculated RSAM metrics with datetime index.
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
            series = series.interpolate(method="linear")

        self.series = series

        return series
