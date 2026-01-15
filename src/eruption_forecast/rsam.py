# Standard library imports
from datetime import datetime, timedelta
from typing import Optional, Self

# Third party imports
import numpy as np
import pandas as pd
from loguru import logger
from obspy import Stream, Trace

from .utils import calculate_window_metrics


class RSAM:
    """Real Seismic Amplitude Measurement (RSAM) class.

    Args:
        stream (Stream): Obspy stream object.
        verbose (bool, optional): Verbose mode. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.
    """

    def __init__(self, stream: Stream, verbose: bool = False, debug: bool = False):
        trace: Trace = stream[0].copy()

        self.stream = stream.copy()
        self.verbose = verbose
        self.debug = debug

        start_datetime: datetime = trace.stats.starttime.datetime

        self.trace = trace
        self.start_datetime = start_datetime
        self.start_datetime_str = start_datetime.strftime("%Y-%m-%d")
        self.is_filtered = False
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
        metric_function: callable = np.mean,
        value_multiplier: float = 1.0, 
        remove_outlier: bool = True
        minimum_completion_ratio: float = 0.3,
    ) -> Self:
        """Calculate metrics for defined time windows of a Trace.

        Args:
            window_duration_minutes (int, optional): Duration of each window in minutes. Defaults to 10.
            metric_function (callable, optional): Function to calculate metric (e.g., np.mean, np.max). Defaults to np.mean.
            value_multiplier (float, optional): Value multiplier. Defaults to 1.0.
            remove_outlier (bool, optional): Remove outlier. Defaults to True.
            minimum_completion_ratio (float, optional): Minimum ratio of data points required to calculate metric. Defaults to 0.3.

        Returns:
            Self: RSAM object
        """
        trace = self.trace

        series = calculate_window_metrics(
            trace,
            window_duration_minutes,
            metric_function,
            value_multiplier,
            remove_outlier,
            minimum_completion_ratio,
            absolute_value=True,
        )

        if value_multiplier > 1:
            series = series.apply(lambda values: values * value_multiplier)

        self.series = series

        return self
