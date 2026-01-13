# Standard library imports
from typing import Optional

# Third party imports
import numpy as np
import pandas as pd
from obspy import Trace


def detect_outliers(
    data: np.ndarray, outlier_threshold: Optional[float] = 0.5
) -> tuple[bool, int, float]:
    """Detect outliers in an array and return an array with outliers
    using z-score ((X - μ) / σ)

    Args:
        data (np.ndarray): Array of data from trace
        outlier_threshold (float, optional): Degree of outliers. Defaults to 0.5.

    Returns:
        tuple[bool, int, float]:
            outlier (bool) : true if outlier is detected, false otherwise.
            outlier_index : Index of the outlier
            outlier_value : Value of the outlier
    """
    if isinstance(data, pd.Series):
        data = data.values

    outlier_index = np.argmax(data)
    outlier_value = data[outlier_index]

    if np.std(data) == 0:
        return True, outlier_index, float(outlier_value)

    # Z-score = (X - μ) / σ
    z_score = (outlier_value - np.mean(data)) / np.std(data)

    if z_score > 10**outlier_threshold:
        return True, outlier_index, float(outlier_value)

    return False, np.nan, np.nan


def delete_outliers(data: np.ndarray) -> np.ndarray:
    """Delete outlier based on z-score

    Args:
        data (np.ndarray): Array of data from trace

    Returns:
        np.ndarray: Array of data from trace after z-score
    """
    outlier, outlier_index, _ = detect_outliers(data)
    if outlier:
        data = np.delete(data, outlier_index)

    return data


def get_windows_information(trace: Trace) -> dict[str, int | float]:
    """Get windows and samples information from Trace

    Args:
        trace (Trace): Trace

    Returns:
        dict[str, int | float]: Windows and samples information

    Example:
        Returns examples:
            {
                "number_of_samples": number_of_samples,
                "samples_per_day": samples_per_day,
                "samples_per_ten_minute": samples_per_ten_minute,
                "total_window": total_window,
                "sample_window": sample_window,
            }
    """
    sampling_rate = trace.stats.sampling_rate
    number_of_samples = trace.stats.npts

    samples_per_day = sampling_rate * 60 * 60 * 24
    samples_per_ten_minute = sampling_rate * 60 * 10
    total_window = int(np.ceil(samples_per_day / samples_per_ten_minute))

    sample_window = int(np.ceil(number_of_samples / samples_per_ten_minute))

    if sample_window == total_window:
        return {
            "number_of_samples": number_of_samples,
            "samples_per_day": samples_per_day,
            "samples_per_ten_minute": samples_per_ten_minute,
            "total_window": total_window,
            "sample_window": sample_window,
        }

    raise ValueError(
        f"sample_window ({sample_window}) not the same as total_window {total_window}"
    )
