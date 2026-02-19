"""Array operations and outlier detection utilities.

This module provides functions for array manipulation and outlier detection
using z-score methods. Supports removing zero values, detecting maximum outliers,
and filtering all outliers from numpy arrays.
"""

import numpy as np

from eruption_forecast.logger import logger


def mask_zero_values(data: np.ndarray) -> np.ndarray:
    """Remove zero values from an array.

    Filters out all zero values (0.0) from the input numpy array, returning only
    non-zero elements. This function is commonly used to clean data before outlier
    detection or statistical calculations.

    Args:
        data (np.ndarray): Input array of numerical data.

    Returns:
        np.ndarray: Array with zero values removed, preserving original order.

    Raises:
        TypeError: If input is not a numpy array.

    Examples:
        >>> mask_zero_values(np.array([1, 0, 2, 0, 3]))
        array([1, 2, 3])
        >>> mask_zero_values(np.array([0.0, 1.5, 0.0, 2.5]))
        array([1.5, 2.5])
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    non_zero_mask = data != 0.0
    return data[non_zero_mask]


def detect_maximum_outlier(
    data: np.ndarray, outlier_threshold: float = 3.0
) -> tuple[bool, int | float, float]:
    """Detect if maximum value in array is an outlier using z-score method.

    Uses z-score ((X - μ) / σ) to determine if the maximum value (by absolute value)
    in the array is statistically an outlier. A value is considered an outlier if its
    z-score exceeds the threshold (default 3.0, equivalent to 3 standard deviations).
    NaN values are automatically filtered before detection.

    Args:
        data (np.ndarray): Array of numerical data.
        outlier_threshold (float, optional): Z-score threshold for outlier detection.
            Defaults to 3.0 (3 standard deviations).

    Returns:
        tuple[bool, int | float, float]:
            - is_outlier (bool): True if maximum value is an outlier.
            - outlier_index (int | float): Index of the maximum value, or np.nan if no outlier.
            - outlier_value (float): Maximum value, or np.nan if no outlier.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If array is empty or outlier_threshold is not positive.

    Examples:
        >>> detect_maximum_outlier(np.array([1, 2, 3, 100]))  # 100 is outlier
        (True, 3, 100.0)
        >>> detect_maximum_outlier(np.array([1, 2, 3, 4]))  # No outlier
        (False, nan, nan)
        >>> detect_maximum_outlier(np.array([5, 5, 5, 5]))  # Identical values
        (False, nan, nan)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if len(data) == 0:
        raise ValueError("Cannot detect outliers in empty array")

    if outlier_threshold <= 0:
        raise ValueError("Outlier threshold must be positive")

    # Handle NaN values
    if np.any(np.isnan(data)):
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return False, np.nan, np.nan

    outlier_index = np.argmax(np.abs(data))
    outlier_value = data[outlier_index]

    # If all values are identical, no outlier exists
    std = np.std(data)
    if std == 0:
        return False, np.nan, np.nan

    # Calculate z-score: Z = (X - μ) / σ
    mean = np.mean(data)
    z_score = abs((outlier_value - mean) / std)

    # Check if z-score exceeds threshold
    if z_score > outlier_threshold:
        return True, int(outlier_index), float(outlier_value)

    return False, np.nan, np.nan


def remove_maximum_outlier(
    data: np.ndarray, mask_zero_value: bool = True, outlier_threshold: float = 3.0
) -> np.ndarray:
    """Remove single maximum outlier from array using z-score method.

    Detects if the maximum value (by absolute value) is an outlier and removes it.
    This function removes at most one outlier per call. To remove all outliers,
    use remove_outliers() instead. Optionally masks zero values before detection.

    Args:
        data (np.ndarray): Input array of numerical data.
        mask_zero_value (bool, optional): If True, remove zero values before processing.
            Defaults to True.
        outlier_threshold (float, optional): Z-score threshold for outlier detection.
            Defaults to 3.0.

    Returns:
        np.ndarray: Array with maximum outlier removed (if detected), or original array
            if no outlier found.

    Raises:
        TypeError: If input is not a numpy array.

    Examples:
        >>> remove_maximum_outlier(np.array([1, 2, 3, 100]))
        array([1, 2, 3])
        >>> remove_maximum_outlier(np.array([0, 1, 2, 3]), mask_zero_value=False)
        array([0, 1, 2, 3])
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    # Make a copy to avoid modifying the original
    data = data.copy()

    # Optionally mask zero values
    if mask_zero_value:
        data = mask_zero_values(data)

    # Return empty array if no data left
    if len(data) == 0:
        return np.array([])

    # Detect and remove maximum outlier
    try:
        is_outlier, outlier_index, _ = detect_maximum_outlier(
            data, outlier_threshold=outlier_threshold
        )

        if is_outlier and not np.isnan(outlier_index):
            data = np.delete(data, int(outlier_index))
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not detect outlier: {e}")
        return data

    return data


def remove_outliers(
    data: np.ndarray,
    outlier_threshold: float = 3.0,
    mask_zero_value: bool = True,
    return_outliers: bool = False,
) -> np.ndarray:
    """Remove all outliers from array based on z-score threshold.

    Removes all values whose z-score exceeds the threshold in a single pass.
    Unlike remove_maximum_outlier which removes only one value, this function
    removes all outliers simultaneously. NaN values are automatically filtered.

    Args:
        data (np.ndarray): Input array of numerical data.
        outlier_threshold (float, optional): Z-score threshold in standard deviations.
            Defaults to 3.0 (3σ).
        mask_zero_value (bool, optional): If True, remove zero values before processing.
            Defaults to True.
        return_outliers (bool, optional): If True, return outliers instead of filtered data.
            Defaults to False.

    Returns:
        np.ndarray: Array with outliers removed, or array of outliers if return_outliers=True.
            Returns empty array if all values are identical (std=0).

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If outlier_threshold is not positive.

    Examples:
        >>> remove_outliers(np.array([1, 2, 3, 100, 200]))
        array([1, 2, 3])
        >>> remove_outliers(np.array([1, 2, 3, 100, 200]), return_outliers=True)
        array([100, 200])
        >>> remove_outliers(np.array([1, 2, 3, 4]), outlier_threshold=2.0)
        array([1, 2, 3, 4])
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    if outlier_threshold <= 0:
        raise ValueError("Outlier threshold must be positive")

    # Make a copy to avoid modifying the original
    data = data.copy()

    # Optionally mask zero values
    if mask_zero_value:
        data = mask_zero_values(data)

    # Return empty array if no data left
    if len(data) == 0:
        return np.array([])

    # Handle NaN values
    if np.any(np.isnan(data)):
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return np.array([])

    # Calculate mean and standard deviation
    mean = np.mean(data)
    std = np.std(data)

    # If all values are identical, no outliers exist
    if std == 0:
        return np.array([]) if return_outliers else data

    # Calculate z-scores: Z = |X - μ| / σ
    z_scores = np.abs((data - mean) / std)

    # Create mask for non-outliers (z-score <= threshold)
    non_outlier_mask = z_scores <= outlier_threshold

    # Get filtered data and outliers
    filtered_data = data[non_outlier_mask]
    outliers = data[~non_outlier_mask]

    if return_outliers:
        return outliers

    return filtered_data


def chunk_daily_data(
    data: np.ndarray,
    sampling_rate: float,
    window_min: int = 10,
    window_overlap: float | None = None,
    mask_zero_value: bool = True,
    debug: bool = False,
) -> np.ndarray | np.ma.MaskedArray:
    """Chunk a daily time series into fixed-size windows.

    Splits a 1-D seismic data array into consecutive windows of equal length.
    Incomplete trailing windows are zero-padded with NaN before slicing.
    When ``mask_zero_value=True``, zero samples (indicating dead channels or
    missing data) and NaN values are masked so downstream metrics ignore them.

    Args:
        data (np.ndarray): 1-D array of daily seismic amplitude data.
        sampling_rate (float): Sampling rate in Hz.
        window_min (int, optional): Window duration in minutes. Defaults to 10.
        window_overlap (float, optional): Overlap between consecutive windows as a
            percentage in [0, 100). If None, non-overlapping windows are used.
            Defaults to None.
        mask_zero_value (bool, optional): If True, mask zero and NaN samples.
            Zeros in seismic data typically indicate dead or missing samples.
            Defaults to True.
        debug (bool, optional): Debug mode. Defaults to False.

    Returns:
        np.ndarray | np.ma.MaskedArray: Array of shape (n_windows, samples_per_window).
            Returns a masked array when ``mask_zero_value=True``, otherwise a plain
            float64 ndarray.

    Raises:
        ValueError: If ``window_overlap`` is not in range [0, 100).
    """
    window_samples = int(sampling_rate * 60 * window_min)
    samples_per_day = 24 * 60 * 60 * sampling_rate

    if window_overlap is None:
        step_samples = window_samples
    else:
        if not (0.0 <= window_overlap < 100.0):
            raise ValueError("window_overlap must be in range [0, 100)")
        step_samples = int(window_samples * (1 - window_overlap / 100.0))

    n_windows = (int(samples_per_day) - window_samples) // step_samples + 1

    # Pad data so the last incomplete window is filled with NaN
    difference_samples = int(samples_per_day) - len(data)
    if difference_samples != 0:
        data = np.concatenate([data, np.full(difference_samples, np.nan)])

    if debug:
        if len(data) != 144:
            raise ValueError(f"n_windows should be 144, but got {len(data)}")

    chunks = np.array(
        [
            data[i * step_samples : i * step_samples + window_samples]
            for i in range(n_windows)
        ],
        dtype=float,
    )

    # Mask zeros (dead samples) and NaN (gaps + padding)
    if mask_zero_value:
        mask = (chunks == 0) | np.isnan(chunks)
        return np.ma.MaskedArray(chunks, mask=mask)

    return chunks
