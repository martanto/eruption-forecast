# Standard library imports
from datetime import datetime, timedelta
from typing import Optional, Callable

# Third party imports
import numpy as np
import pandas as pd
from loguru import logger
from obspy import Trace


def detect_outliers(
    data: np.ndarray, outlier_threshold: float = 0.5
) -> tuple[bool, int | float, float]:
    """Detect outliers in an array and return an array with outliers
    using z-score ((X - μ) / σ)

    Args:
        data (np.ndarray): Array of data from trace
        outlier_threshold (float, optional): Degree of outliers. Defaults to 0.5.

    Returns:
        tuple[bool, int | float, float]:
            outlier (bool) : true if outlier is detected, false otherwise.
            outlier_index : Index of the outlier
            outlier_value : Value of the outlier
    """
    if isinstance(data, pd.Series):
        data = data.values

    outlier_index = np.argmax(data)
    outlier_value = data[outlier_index]

    if np.std(data) == 0:
        return True, int(outlier_index), float(outlier_value)

    # Z-score = (X - μ) / σ
    z_score = (outlier_value - np.mean(data)) / np.std(data)

    if z_score > 10**outlier_threshold:
        return True, int(outlier_index), float(outlier_value)

    return False, np.nan, np.nan


def delete_outliers(data: np.ndarray) -> np.ndarray:
    """Delete outliers from an array

    Args:
        data (np.ndarray): Array of data

    Returns:
        np.ndarray: Array without outliers
    """
    outlier, outlier_index, _ = detect_outliers(data)

    if outlier:
        data = np.delete(data, int(outlier_index))
        data = delete_outliers(data)

    return data


def get_windows_information(
    trace: Trace,
    window_duration_minutes: int = 10,
) -> dict[str, int | float]:
    """Get windows and samples information from Trace

    Args:
        trace (Trace): Trace
        window_duration_minutes (int, optional): Duration of each window in minutes. Defaults to 10.

    Returns:
        dict[str, int | float]: Windows and samples information

    Example:
        Returns examples:
            {
                "number_of_samples": number_of_samples,
                "samples_per_day": samples_per_day,
                "sample_per_window": sample_per_window,
                "total_windows": total_windows,
                "sample_window": sample_window,
            }
    """
    if not isinstance(trace, Trace):
        raise TypeError("Input must be an Obspy Trace object")

    sampling_rate = trace.stats.sampling_rate
    number_of_samples = trace.stats.npts

    samples_per_day = sampling_rate * 60 * 60 * 24
    sample_per_window = sampling_rate * 60 * window_duration_minutes
    total_windows = int(np.ceil(samples_per_day / sample_per_window))
    sample_window = int(np.ceil(number_of_samples / sample_per_window))

    if sample_window != total_windows:
        logger.warning(
            f"sample_window ({sample_window}) not the same as total_windows ({total_windows})"
        )

    return {
        "number_of_samples": number_of_samples,
        "samples_per_day": samples_per_day,
        "sample_per_window": sample_per_window,
        "total_windows": total_windows,
        "sample_window": sample_window,
    }


def calculate_window_metrics(
    trace: Trace,
    window_duration_minutes: int = 10,
    metric_function: Callable[[np.ndarray], float] = np.mean,
    remove_outliers: bool = True,
    minimum_completion_ratio: float = 0.3,
    absolute_value: bool = False,
    value_multiplier: float = 1.0,
) -> pd.Series:
    """Calculate metrics for defined time windows of a Trace.

    Args:
        trace (Trace): Obspy Trace object.
        window_duration_minutes (int, optional): Duration of each window in minutes. Defaults to 10.
        metric_function (callable, optional): Function to calculate metric (e.g., np.mean, np.max). Defaults to np.mean.
        remove_outliers (bool, optional): Whether to remove outliers before calculation. Defaults to True.
        minimum_completion_ratio (float, optional): Minimum ratio of data points required to calculate metric. Defaults to 0.3.
        absolute_value (bool, optional): Whether to calculate absolute value. Defaults to False.
        value_multiplier (float, optional): Multiplier for metric value. Defaults to 1.0.

    Returns:
        pd.Series: Series containing the calculated metrics with datetime index.
    """
    if not isinstance(trace, Trace):
        raise TypeError("Input must be an Obspy Trace object")

    start_datetime = trace.stats.starttime.datetime
    start_datetime = start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)

    trace_data = abs(trace.data) if absolute_value else trace.data
    sampling_rate = trace.stats.sampling_rate

    samples_per_window = int(sampling_rate * 60 * window_duration_minutes)
    samples_per_day = int(sampling_rate * 60 * 60 * 24)
    total_windows = int(np.ceil(samples_per_day / samples_per_window))

    indices: list[datetime] = []
    data_points: list[float] = []

    for index_window in range(total_windows):
        first_index = int(index_window * samples_per_window)
        last_index = int((index_window + 1) * samples_per_window)

        window_data = trace_data[first_index:last_index]
        length_window_data = len(window_data)
        minimum_samples = int(np.ceil(minimum_completion_ratio * length_window_data))

        # Initialize metric_value to np.nan
        metric_value = np.nan

        if length_window_data > 0:
            if remove_outliers and (length_window_data > minimum_samples):
                window_data = delete_outliers(window_data)

            # Re-check length after outlier removal just in case,
            # though delete_outliers mostly removes one
            if len(window_data) > 0:
                metric_value = metric_function(window_data)

                if value_multiplier != 1.0 and not np.isnan(metric_value):
                    metric_value *= value_multiplier

        # Calculate timestamp for the window
        window_time = start_datetime + timedelta(
            minutes=index_window * window_duration_minutes
        )

        indices.append(window_time)
        data_points.append(metric_value)

    return pd.Series(data=data_points, index=indices, name="datetime", dtype=float)
