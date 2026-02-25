"""Time window operations for seismic data processing.

This module provides functions for creating time windows, extracting window
information from seismic traces, and calculating window-based metrics for
tremor analysis.
"""

from typing import Literal
from datetime import datetime, timedelta
from collections.abc import Callable

import numpy as np
import pandas as pd
from obspy import Trace
from scipy.stats import norm

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import (
    remove_outliers,
    chunk_daily_data,
    remove_maximum_outlier,
)
from eruption_forecast.utils.validation import (
    validate_date_ranges,
    validate_window_step,
)


def get_windows_information(
    trace: Trace,
    window_duration_minutes: int = 10,
) -> dict[str, int | float]:
    """Get window and sample information from an ObsPy Trace.

    Calculates the number of samples, windows, and sampling statistics from a seismic
    trace. This is useful for understanding the temporal resolution and data completeness.

    Args:
        trace (Trace): ObsPy Trace object containing seismic waveform data.
        window_duration_minutes (int, optional): Duration of each window in minutes.
            Defaults to 10.

    Returns:
        dict[str, int | float]: Dictionary containing window and sample information:
            - number_of_samples (int): Total number of samples in the trace.
            - samples_per_day (float): Expected samples per 24-hour period.
            - sample_per_window (float): Expected samples per window.
            - total_windows (int): Total number of windows per day.
            - sample_window (int): Actual number of windows in the trace.

    Raises:
        TypeError: If input is not an ObsPy Trace object.

    Examples:
        >>> trace = obspy.read()[0]  # Load example trace
        >>> info = get_windows_information(trace, window_duration_minutes=10)
        >>> print(info["total_windows"])
        144
    """
    if not isinstance(trace, Trace):
        raise TypeError("Input must be an ObsPy Trace object")

    sampling_rate = trace.stats.sampling_rate
    number_of_samples = trace.stats.npts

    samples_per_day = sampling_rate * 60 * 60 * 24
    sample_per_window = sampling_rate * 60 * window_duration_minutes
    total_windows = int(np.ceil(samples_per_day / sample_per_window))
    sample_window = int(np.ceil(number_of_samples / sample_per_window))

    if sample_window != total_windows:
        logger.warning(
            f"sample_window ({sample_window}) is not the same as total_windows ({total_windows})"
        )

    return {
        "number_of_samples": number_of_samples,
        "samples_per_day": samples_per_day,
        "sample_per_window": sample_per_window,
        "total_windows": total_windows,
        "sample_window": sample_window,
    }


def shannon_entropy(data: np.ndarray) -> float:
    """Calculate the Shannon entropy of a seismic data window.

    Models the amplitude distribution as a Gaussian and computes the differential
    entropy as ``H = -Σ p(x) · log2(p(x))``. Returns NaN when signal energy is
    below 1.0, which guards against computing entropy on near-zero noise.

    Args:
        data (np.ndarray): 1-D array of seismic amplitude values for one window.

    Returns:
        float: Shannon entropy value, or NaN if energy is below threshold.
    """
    energy = np.sum(np.square(data))

    if energy < 1.0:
        return np.nan

    y = norm.pdf(data, loc=np.mean(data), scale=np.std(data))

    # Zero PDF values would produce -inf in log; mask them before summing
    y_masked = np.ma.MaskedArray(y, (y == 0))
    y = y_masked.filled(np.nan)

    entropy = -1 * np.sum(y * np.log2(y))

    return entropy


def calculate_window_metrics(
    trace: Trace,
    window_duration_minutes: int = 10,
    metric_function: Callable[[np.ndarray], float] = np.mean,
    remove_outlier_method: Literal["maximum", "all"] | None = None,
    mask_zero_value: bool = False,
    minimum_completion_ratio: float = 0.3,
    absolute_value: bool = False,
    window_overlap: float | None = None,
    value_multiplier: float = 1.0,
) -> pd.Series:
    """Calculate metrics for defined time windows of an ObsPy Trace.

    Divides a seismic trace into fixed-duration time windows and computes a statistical
    metric for each window. Supports outlier removal and data quality filtering. This
    is the core function for computing RSAM and other tremor metrics.

    Args:
        trace (Trace): ObsPy Trace object containing seismic waveform data.
        window_duration_minutes (int, optional): Duration of each window in minutes.
            Defaults to 10.
        metric_function (Callable[[np.ndarray], float], optional): Function to calculate
            metric (e.g., np.mean, np.max, np.median). Defaults to np.mean.
        remove_outlier_method (Literal["maximum", "all"] | None, optional): Outlier removal
            strategy. "maximum" removes single outlier, "all" removes all outliers, None
            disables removal. Defaults to None.
        mask_zero_value (bool, optional): If True, mask zero values before processing.
            Defaults to False.
        minimum_completion_ratio (float, optional): Minimum ratio of data points required
            to calculate the metric (0.0-1.0). Windows with fewer samples return NaN.
            Defaults to 0.3.
        absolute_value (bool, optional): If True, use absolute values of trace data.
            Defaults to False.
        window_overlap (float, optional): Overlap windows percentage. Defaults to None.
        value_multiplier (float, optional): Multiplier applied to the final metric value.
            Defaults to 1.0.

    Returns:
        pd.Series: Series containing the calculated metrics with datetime index and float dtype.

    Raises:
        TypeError: If input is not an ObsPy Trace object.

    Examples:
        >>> trace = obspy.read()[0]
        >>> rsam = calculate_window_metrics(
        ...     trace, window_duration_minutes=10, metric_function=np.mean
        ... )
        >>> print(rsam.head())
    """
    start_datetime = trace.stats.starttime.datetime
    sampling_rate = trace.stats.sampling_rate

    data = trace.data
    if absolute_value:
        data = np.abs(data)

    chunking_data = chunk_daily_data(
        data=data,
        sampling_rate=sampling_rate,
        window_min=window_duration_minutes,
        window_overlap=window_overlap,
        mask_zero_value=mask_zero_value,
    )

    indices = []
    data_points = []
    for index, window_data in enumerate(chunking_data):
        # Initialize metric_value to np.nan
        metric_value = np.nan

        has_sufficient_samples = True
        if isinstance(window_data, np.ma.MaskedArray):
            valid_samples = window_data.count()
            valid_samples_ratio = valid_samples / len(window_data)
            has_sufficient_samples = valid_samples_ratio >= minimum_completion_ratio

        if not has_sufficient_samples:
            metric_value = np.nan

        elif len(window_data) == 1:
            metric_value = window_data[0]

        elif remove_outlier_method is None:
            metric_value = metric_function(window_data)

        elif remove_outlier_method:
            window_data = (
                remove_maximum_outlier(window_data)
                if remove_outlier_method == "maximum"
                else remove_outliers(window_data)
            )

            # Re-check length after outlier removal just in case,
            # though remove_maximum_outlier mostly removes one
            if len(window_data) > 0:
                metric_value = metric_function(window_data)
                if value_multiplier != 1.0 and not np.isnan(metric_value):
                    metric_value *= value_multiplier

        indices.append(
            start_datetime + timedelta(minutes=index * window_duration_minutes)
        )
        data_points.append(metric_value)

    return pd.Series(data=data_points, index=indices, name="datetime", dtype=float)


def construct_windows(
    start_date: str | datetime,
    end_date: str | datetime,
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
) -> pd.DataFrame:
    """Construct time windows for label and tremor data.

    Generates a sliding window time series from start_date to end_date with specified
    step size. This is used by LabelBuilder to create time windows for labeling.

    Args:
        start_date (str | datetime): Start date in YYYY-MM-DD format or datetime object.
        end_date (str | datetime): End date in YYYY-MM-DD format or datetime object.
        window_step (int): Step size between consecutive windows.
        window_step_unit (Literal["minutes", "hours"]): Unit of window step.

    Returns:
        pd.DataFrame: DataFrame with datetime index representing time windows. The index
            is named "datetime" and spans from start_date (00:00:00) to end_date (23:59:59).

    Raises:
        ValueError: If window_step exceeds the date range duration.

    Examples:
        >>> windows = construct_windows("2025-01-01", "2025-01-02", 6, "hours")
        >>> print(len(windows))
        9
    """
    window_step, window_step_unit = validate_window_step(window_step, window_step_unit)
    start_date, end_date, n_days = validate_date_ranges(start_date, end_date)

    maximum_window_step = (
        n_days * 60 * 24 if window_step_unit == "minutes" else n_days * 24
    )

    if window_step > maximum_window_step:
        raise ValueError(
            f"window_step must be less than or equal to {maximum_window_step} "
            f"{window_step_unit}.\n"
            f"window_step: {window_step}, maximum_window_step: {maximum_window_step}"
        )

    start_date = start_date.replace(hour=0, minute=0, second=0)
    end_date = end_date.replace(hour=23, minute=59, second=59)

    freq = (
        timedelta(minutes=window_step)
        if window_step_unit == "minutes"
        else timedelta(hours=window_step)
    )

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq=freq,
        inclusive="both",
    )

    df = pd.DataFrame(index=dates)
    df.index.name = "datetime"

    return df
