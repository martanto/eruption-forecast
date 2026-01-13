# Standard library imports
from datetime import datetime, timedelta
from typing import Optional, Self

# Third party imports
import numpy as np
import pandas as pd
from loguru import logger
from obspy import Stream, Trace

from .utils import delete_outliers, get_windows_information


class RSAM:
    def __init__(self, stream: Stream, verbose: bool = False, debug: bool = False):
        trace: Trace = stream[0].copy()

        self.stream = stream.copy()
        self.verbose = verbose
        self.debug = debug

        start_datetime: datetime = trace.stats.starttime.datetime
        start_datetime = start_datetime.replace(hour=0, minute=0, second=0)

        self.trace = trace
        self.start_datetime = start_datetime
        self.start_datetime_str = start_datetime.strftime("%Y-%m-%d")
        self.is_filtered = False
        self.series: pd.Series = pd.Series()
        self.windows_information = get_windows_information(trace)

    def apply_filter(self, freq_min: float, freq_max: float) -> Self:
        self.trace = self.trace.filter(
            "bandpass", freqmin=freq_min, freqmax=freq_max, corners=4
        )
        self.is_filtered = True

        if self.debug:
            logger.debug(
                f"{self.start_datetime_str} :: RSAM Filtered using ({freq_min},{freq_max})"
            )

        return self

    def calculate(
        self, value_multiplier: float = 1.0, remove_outlier: bool = True
    ) -> Self:
        trace = self.trace
        start_datetime = self.start_datetime

        # building 10 minutes series data
        trace_data = np.abs(trace.data)

        windows = self.windows_information
        total_window = windows["total_window"]
        sample_per_ten_minute = windows["samples_per_ten_minute"]

        indices = []
        data = []
        for index_window in range(total_window):
            next_index = index_window + 1
            first_index = int(index_window * sample_per_ten_minute)
            last_index = int(next_index * sample_per_ten_minute)
            ten_minutes_data = trace_data[first_index:last_index]

            length_ten_minutes_data = len(ten_minutes_data)
            minimum_samples = int(np.ceil(0.3 * length_ten_minutes_data))

            if self.debug and (length_ten_minutes_data == 0):
                logger.debug(f"{self.start_datetime_str} :: 10 minutes data empty for index_window = {index_window} or ten_minutes_data[{first_index}:{last_index}]")

            # Removing outlier
            if remove_outlier and (length_ten_minutes_data > minimum_samples):
                ten_minutes_data = delete_outliers(ten_minutes_data)

            index = start_datetime + timedelta(minutes=index_window * 10)

            mean_data = np.nan
            if length_ten_minutes_data > 0:
                mean_data = np.mean(np.array(ten_minutes_data))

            indices.append(index)
            data.append(mean_data)

        series = pd.Series(data=data, index=indices, name="datetime")

        if value_multiplier > 1:
            series = series.apply(lambda values: values * value_multiplier)

        self.series = series

        return self
