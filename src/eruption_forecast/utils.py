from __future__ import annotations

import os
import json
from typing import TYPE_CHECKING, Any, Literal, cast
from datetime import datetime, timedelta
from collections.abc import Callable

import numpy as np
import joblib
import pandas as pd
from obspy import Trace
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from tsfresh.transformers import FeatureSelector
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import (
    GridSearchCV,
)

from eruption_forecast.logger import logger


if TYPE_CHECKING:
    from eruption_forecast.model.classifier_model import ClassifierModel


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


def calculate_window_metrics(
    trace: Trace,
    window_duration_minutes: int = 10,
    metric_function: Callable[[np.ndarray], float] = np.mean,
    remove_outlier_method: Literal["maximum", "all"] | None = None,
    mask_zero_value: bool = False,
    minimum_completion_ratio: float = 0.3,
    absolute_value: bool = False,
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
    if not isinstance(trace, Trace):
        raise TypeError("Input must be an ObsPy Trace object")

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
        metric_value = window_data[0] if length_window_data == 1 else np.nan

        if remove_outlier_method and (length_window_data > minimum_samples):
            window_data = (
                remove_maximum_outlier(window_data, mask_zero_value=mask_zero_value)
                if remove_outlier_method == "maximum"
                else remove_outliers(window_data, mask_zero_value=mask_zero_value)
            )

            # Re-check length after outlier removal just in case,
            # though remove_maximum_outlier mostly removes one
            if len(window_data) > 0:
                # Update metric value
                metric_value = metric_function(window_data)

                if value_multiplier != 1.0 and not np.isnan(metric_value):
                    metric_value *= value_multiplier

        # Calculate timestamp for the window
        window_time = start_datetime + timedelta(
            minutes=index_window * window_duration_minutes
        )

        indices.append(window_time)
        data_points.append(float(metric_value))

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

    maximum_window_step = n_days * 24
    if window_step_unit == "minutes":
        maximum_window_step = n_days * 60 * 24

    if window_step > maximum_window_step:
        raise ValueError(
            f"window_step must be less than or equal to {maximum_window_step} "
            f"{window_step_unit}.\n"
            f"window_step: {window_step}, maximum_window_step: {maximum_window_step}"
        )

    start_date = start_date.replace(hour=0, minute=0, second=0)
    end_date = end_date.replace(hour=23, minute=59, second=59)

    freq = timedelta(hours=window_step)
    if window_step_unit == "minutes":
        freq = timedelta(minutes=window_step)

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq=freq,
        inclusive="both",
    )

    df = pd.DataFrame(index=dates)
    df.index.name = "datetime"

    return df


def to_series(
    df: pd.DataFrame, column_value: str, column_index: str = "id"
) -> pd.Series:
    """Convert a DataFrame column into a Series with a custom index.

    Extracts a column from a DataFrame and uses another column as the index.
    Commonly used to convert label DataFrames into Series for tsfresh processing.

    Args:
        df (pd.DataFrame): Input DataFrame containing both value and index columns.
        column_value (str): Column name whose values become the Series values.
        column_index (str, optional): Column name whose values become the Series index.
            Defaults to "id".

    Returns:
        pd.Series: Series with values from column_value and index from column_index.

    Raises:
        ValueError: If column_value or column_index is not in DataFrame columns.

    Examples:
        >>> df = pd.DataFrame({"id": [1, 2, 3], "is_erupted": [0, 1, 0]})
        >>> series = to_series(df, column_value="is_erupted", column_index="id")
        >>> print(series)
        1    0
        2    1
        3    0
    """
    if column_value not in df.columns:
        raise ValueError(
            f"Param column_value ({column_value}) not in columns in DataFrame."
        )

    if column_index not in df.columns:
        raise ValueError(
            f"Param column_index ({column_index}) not in columns in DataFrame."
        )

    series = pd.Series(df[column_value])
    series.index = df[column_index]
    return series


def to_datetime(date: str | datetime, variable_name: str | None = None) -> datetime:
    """Ensure date object is a datetime object.

    Converts date strings in YYYY-MM-DD format to datetime objects. If already a
    datetime object, returns it unchanged. Used for standardizing date inputs.

    Args:
        date (str | datetime): Date string in YYYY-MM-DD format or datetime object.
        variable_name (str | None, optional): Variable name for error messages.
            Defaults to None.

    Returns:
        datetime: Datetime object.

    Raises:
        ValueError: If date string is not in YYYY-MM-DD format.

    Examples:
        >>> to_datetime("2025-03-20")
        datetime.datetime(2025, 3, 20, 0, 0)
        >>> to_datetime(datetime(2025, 3, 20))
        datetime.datetime(2025, 3, 20, 0, 0)
    """
    if isinstance(date, datetime):
        return date

    variable_name = f"{variable_name}" if variable_name else "Date"

    try:
        return datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(  # noqa: B904
            f"{variable_name} value {date} is not in valid YYYY-MM-DD format."
        )


def validate_date_ranges(
    start_date: str | datetime, end_date: str | datetime
) -> tuple[datetime, datetime, int]:
    """Validate date range and ensure start_date is before end_date.

    Converts date strings to datetime objects and validates that start_date
    comes before end_date. Returns the validated dates and duration in days.

    Args:
        start_date (str | datetime): Start date in YYYY-MM-DD format or datetime object.
        end_date (str | datetime): End date in YYYY-MM-DD format or datetime object.

    Returns:
        tuple[datetime, datetime, int]: Tuple containing:
            - start_date (datetime): Validated start date.
            - end_date (datetime): Validated end date.
            - n_days (int): Total number of days between start and end (end - start).

    Raises:
        ValueError: If start_date >= end_date.

    Examples:
        >>> start, end, days = validate_date_ranges("2025-01-01", "2025-01-10")
        >>> print(days)
        9
    """
    if isinstance(start_date, str):
        start_date = to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = to_datetime(end_date)

    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    if start_date >= end_date:
        raise ValueError(
            f"Start date ({start_date_str}) should be less than end date ({end_date_str})"
        )

    n_days: int = int((end_date - start_date).days)

    return start_date, end_date, n_days


def validate_window_step(
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
) -> tuple[int, Literal["minutes", "hours"]]:
    """Validate window step and step unit.

    Ensures window_step is an integer and window_step_unit is either "minutes"
    or "hours". Used to validate sliding window parameters before construction.

    Args:
        window_step (int): Step size between consecutive windows (must be positive integer).
        window_step_unit (Literal["minutes", "hours"]): Unit of window step.

    Returns:
        tuple[int, Literal["minutes", "hours"]]: Validated window step and unit.

    Raises:
        TypeError: If window_step is not an integer or window_step_unit is not a string.
        ValueError: If window_step_unit is not "minutes" or "hours".

    Examples:
        >>> validate_window_step(6, "hours")
        (6, 'hours')
        >>> validate_window_step(30, "minutes")
        (30, 'minutes')
    """
    if not isinstance(window_step, int):
        raise TypeError(f"window_step must be an integer. Your value is {window_step}")
    if not isinstance(window_step_unit, str):
        raise TypeError(
            f"window_step_unit must be a string. Your value is {window_step_unit}"
        )
    if window_step_unit not in ["minutes", "hours"]:
        raise ValueError(
            f"window_step_unit must be 'minutes' or 'hours'. Your value is {window_step_unit}"
        )

    return window_step, window_step_unit


def sort_dates(dates: list[str]) -> list[str]:
    """Sort a list of date strings chronologically.

    Converts date strings to datetime objects, sorts them chronologically,
    and returns them as formatted strings. Used for sorting eruption dates.

    Args:
        dates (list[str]): List of date strings in YYYY-MM-DD format.

    Returns:
        list[str]: Sorted list of date strings in YYYY-MM-DD format.

    Examples:
        >>> sort_dates(["2025-03-20", "2025-01-15", "2025-02-10"])
        ['2025-01-15', '2025-02-10', '2025-03-20']
    """
    date_series = pd.Series(dates)
    date_series = date_series.apply(pd.to_datetime, format="%Y-%m-%d").sort_values()
    date_list: list[str] = list(date_series.dt.strftime("%Y-%m-%d"))

    return date_list


def check_sampling_consistency(
    df: pd.DataFrame,
    expected_freq: str = "10min",
    tolerance: str = "1min",
    verbose: bool = False,
) -> tuple[bool, pd.DataFrame, pd.DataFrame, int | None]:
    """Check sampling rate consistency and identify inconsistencies.

    Validates that a DataFrame has consistent time intervals between consecutive rows.
    Identifies and separates rows with inconsistent sampling rates based on tolerance.
    This is crucial for ensuring data quality in tremor time series.

    Args:
        df (pd.DataFrame): DataFrame with pd.DatetimeIndex.
        expected_freq (str, optional): Expected sampling frequency (e.g., "10min", "1H").
            Defaults to "10min".
        tolerance (str, optional): Tolerance for considering sampling periods as equal
            (e.g., "1min", "30s"). Defaults to "1min".
        verbose (bool, optional): If True, print detailed information about inconsistencies.
            Defaults to False.

    Returns:
        tuple[bool, pd.DataFrame, pd.DataFrame, int | None]: Tuple containing:
            - is_consistent (bool): True if all samples are consistent, False otherwise.
            - consistent_data (pd.DataFrame): DataFrame with consistent samples only.
            - inconsistent_data (pd.DataFrame): DataFrame with inconsistent samples.
            - sampling_rate (int | None): Sampling rate in seconds if consistent, None otherwise.

    Raises:
        ValueError: If DataFrame has fewer than 2 rows.
        TypeError: If DataFrame index is not DatetimeIndex.

    Examples:
        >>> df = pd.DataFrame({"value": [1, 2, 3]},
        ...                   index=pd.date_range("2025-01-01", periods=3, freq="10min"))
        >>> is_consistent, consistent, inconsistent, rate = check_sampling_consistency(df)
        >>> print(is_consistent)
        True
    """
    if len(df) <= 2:
        raise ValueError(
            "DataFrame must have at least 2 rows to check sampling consistency"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")

    df = df.sort_index()
    sampling_rate = None

    # Calculate time differences between consecutive timestamps
    time_diffs = df.index.to_series().diff()

    # Expected time difference
    expected_diff = pd.Timedelta(expected_freq)
    tolerance_diff = pd.Timedelta(tolerance)

    # Find inconsistent sampling rates (outside tolerance range)
    lower_bound = expected_diff - tolerance_diff
    upper_bound = expected_diff + tolerance_diff

    # First row will be NaT (no previous timestamp), so we skip it
    inconsistent_mask: pd.Series = ~(
        (time_diffs >= lower_bound) & (time_diffs <= upper_bound)
    )
    inconsistent_mask.iloc[0] = False

    # Get inconsistent data
    inconsistent_data = df[inconsistent_mask]

    # Get consistent data (remove inconsistencies)
    consistent_data = df[~inconsistent_mask]

    is_consistent = True if inconsistent_data.empty else False

    # Get sampling rate if consistent
    if is_consistent:
        sampling_rate = (df.index[1] - df.index[0]).seconds

    if verbose:
        print(f"Total rows: {len(df)}")
        print(f"Inconsistent rows found: {len(inconsistent_data)}")
        print(f"Consistent rows: {len(consistent_data)}")

        if len(inconsistent_data) > 0:
            print("\nInconsistent time differences:")
            print(time_diffs[inconsistent_mask].describe())

    return is_consistent, consistent_data, inconsistent_data, sampling_rate


def validate_columns(
    df: pd.DataFrame, columns: list[str], exclude_columns: list[str] | None = None
) -> None:
    """Validate that specified columns exist in DataFrame.

    Checks that all specified columns exist in the DataFrame, except those in
    the exclude list. Raises ValueError with detailed message if any column is missing.

    Args:
        df (pd.DataFrame): DataFrame to validate.
        columns (list[str]): List of column names to validate.
        exclude_columns (list[str] | None, optional): List of column names to skip
            validation. Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If any column in columns (except exclude_columns) does not exist
            in the DataFrame.

    Examples:
        >>> df = pd.DataFrame({"rsam_f0": [1, 2], "rsam_f1": [3, 4]})
        >>> validate_columns(df, ["rsam_f0", "rsam_f1"])  # No error
        >>> validate_columns(df, ["rsam_f2"])  # Raises ValueError
    """
    if exclude_columns is None:
        exclude_columns = []

    for column in columns:
        if column in exclude_columns:
            continue
        if column not in df.columns.tolist():
            raise ValueError(
                f"Column {column} does not exist in dataframe. "
                f"Columns available are: {df.columns}. "
                f"{df.head(5)}"
            )
    return None


def concat_features(csv_list: list[str], filepath: str) -> tuple[str, pd.DataFrame]:
    """Concatenate feature CSVs into one DataFrame and save.

    Reads multiple feature CSV files, concatenates them column-wise (axis=1),
    and saves the combined DataFrame to the specified filepath. This is used
    to merge per-column tsfresh feature extractions.

    Args:
        csv_list (list[str]): List of CSV file paths to concatenate.
        filepath (str): Output filepath to save the concatenated CSV.

    Returns:
        tuple[str, pd.DataFrame]: Tuple containing:
            - filepath (str): Path where the CSV was saved.
            - df (pd.DataFrame): Concatenated DataFrame.

    Raises:
        ValueError: If csv_list has fewer than 2 files or if all CSVs are empty.

    Examples:
        >>> csv_files = ["features_f0.csv", "features_f1.csv"]
        >>> path, df = concat_features(csv_files, "all_features.csv")
        >>> print(df.shape)
    """
    if len(csv_list) <= 1:
        raise ValueError(
            f"Requires at least 2 CSV files. Total your CSV file is {len(csv_list)}"
        )

    df = pd.concat([pd.read_csv(file, index_col=0) for file in csv_list], axis=1)

    if df.empty:
        raise ValueError("There is no data in the csv files.")

    df.to_csv(filepath, index=True)

    return filepath, df


def random_under_sampler(
    features: pd.DataFrame,
    labels: pd.Series,
    sampling_strategy: str | float = "auto",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply random under-sampling to balance classes.

    Handles imbalanced eruption/non-eruption datasets by randomly removing
    samples from the majority class (non-eruption) to match the minority
    class (eruption) based on the sampling strategy. This improves classifier
    performance on imbalanced data.

    Args:
        features (pd.DataFrame): Features DataFrame with training samples.
        labels (pd.Series): Binary labels Series (0=non-eruption, 1=eruption).
        sampling_strategy (str | float, optional): Sampling ratio or strategy.
            If "auto", balances to 50/50. If float, represents desired ratio
            of minority/majority class. Defaults to "auto".
        random_state (int, optional): Random seed for reproducibility.
            Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tuple containing:
            - features (pd.DataFrame): Balanced features DataFrame.
            - labels (pd.Series): Balanced labels Series.

    Examples:
        >>> balanced_X, balanced_y = random_under_sampler(
        ...     features, labels, sampling_strategy=0.75, random_state=42
        ... )
        >>> print(balanced_y.value_counts())
    """
    sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy, random_state=random_state
    )

    features, labels = sampler.fit_resample(features, labels)

    return features, labels


def get_significant_features(
    features: pd.DataFrame,
    labels: pd.Series | pd.DataFrame,
    fdr_level: float = 0.05,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Get significant features ranked by p-value using tsfresh FeatureSelector.

    Uses tsfresh's FeatureSelector with Benjamini-Hochberg FDR correction to identify
    features with statistically significant correlation to the target labels. This is
    the first stage of feature selection in the pipeline.

    Args:
        features (pd.DataFrame): Extracted features DataFrame from tsfresh.
        labels (pd.Series | pd.DataFrame): Binary eruption labels. If DataFrame,
            will extract "is_erupted" column.
        fdr_level (float, optional): False discovery rate threshold (0.0-1.0).
            Lower values are more conservative. Defaults to 0.05.
        n_jobs (int, optional): Number of parallel jobs for computation. Defaults to 1.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tuple containing:
            - features_filtered (pd.DataFrame): Filtered features DataFrame with only
              significant features.
            - significant_features (pd.Series): Features sorted by p-value (ascending),
              with feature names as index and p-values as values. Index name is "features",
              series name is "p_values".

    Examples:
        >>> filtered_features, sig_features = get_significant_features(
        ...     features_df, labels_series, fdr_level=0.05, n_jobs=4
        ... )
        >>> top_10_features = sig_features.head(10).index.tolist()
        >>> print(f"Selected {len(filtered_features.columns)} significant features")
    """
    if isinstance(labels, pd.DataFrame):
        labels = to_series(labels, column_value="is_erupted")

    selector = FeatureSelector(
        n_jobs=n_jobs, fdr_level=fdr_level, ml_task="classification"
    )

    # Extracted features with potentially reduced column
    features_filtered: pd.DataFrame = selector.fit_transform(X=features, y=labels)

    _significant_features = pd.Series(selector.p_values, index=selector.features)
    _significant_features = _significant_features.sort_values()
    _significant_features.name = "p_values"
    _significant_features.index.name = "features"

    return features_filtered, _significant_features


def resolve_output_dir(
    output_dir: str | None,
    root_dir: str | None,
    default_subpath: str,
) -> str:
    """Resolve an output directory path against an anchor directory.

    Provides a consistent way to resolve output paths relative to a stable
    root directory instead of relying on the current working directory. This
    is critical for the pipeline's output directory structure.

    Resolution rules:
    1. Absolute ``output_dir`` → used as-is (``root_dir`` is ignored).
    2. Relative ``output_dir`` → joined with ``root_dir`` (or ``os.getcwd()`` if None).
    3. ``None`` ``output_dir`` → ``root_dir / default_subpath``.

    Args:
        output_dir (str | None): Caller-supplied output directory (absolute, relative, or None).
        root_dir (str | None): Anchor directory for resolving relative paths.
            If None, falls back to ``os.getcwd()``.
        default_subpath (str): Sub-path appended to the anchor when ``output_dir`` is None.

    Returns:
        str: Resolved absolute or anchored output directory path.

    Examples:
        >>> resolve_output_dir(None, "/data/project", "output")
        '/data/project/output'
        >>> resolve_output_dir("custom", "/data/project", "output")
        '/data/project/custom'
        >>> resolve_output_dir("/abs/path", "/data/project", "output")
        '/abs/path'
    """
    anchor = root_dir if root_dir is not None else os.getcwd()
    if output_dir is None:
        return os.path.join(anchor, default_subpath)
    if os.path.isabs(output_dir):
        return output_dir
    return os.path.join(anchor, output_dir)


def normalize_dates(
    start_date: str | datetime,
    end_date: str | datetime,
) -> tuple[datetime, datetime, str, str]:
    """Normalize start and end dates to standard format.

    Converts date strings to datetime objects and formats them consistently.
    Start date is set to 00:00:00, end date is set to 23:59:59. Returns both
    datetime objects and formatted strings for convenience.

    Args:
        start_date (str | datetime): Start date in YYYY-MM-DD format or datetime object.
        end_date (str | datetime): End date in YYYY-MM-DD format or datetime object.

    Returns:
        tuple[datetime, datetime, str, str]: Tuple containing:
            - start_date (datetime): Start date at 00:00:00.
            - end_date (datetime): End date at 23:59:59.
            - start_date_str (str): Start date formatted as YYYY-MM-DD.
            - end_date_str (str): End date formatted as YYYY-MM-DD.

    Examples:
        >>> start, end, start_str, end_str = normalize_dates("2025-01-01", "2025-01-31")
        >>> print(start_str, end_str)
        2025-01-01 2025-01-31
    """
    start_date = to_datetime(start_date).replace(hour=0, minute=0, second=0)
    end_date = to_datetime(end_date).replace(hour=23, minute=59, second=59)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    return start_date, end_date, start_date_str, end_date_str


def get_metrics(
    classifier_model: ClassifierModel,
    labels_test,
    labels_pred,
    labels_train: pd.Series,
    top_n: int,
    grid_search: GridSearchCV,
    random_state: int,
    metrics_filepath: str | None = None,
) -> dict[str | Any, int | str | Any]:
    """Compute classification metrics from model predictions.

    Calculates confusion matrix components, accuracy, balanced accuracy,
    F1 score, precision, recall, and best cross-validation parameters.
    Optionally saves the metrics to a JSON file. This is the core metrics
    function used by ModelTrainer.

    Args:
        classifier_model (ClassifierModel): Fitted classifier model with name,
            cv_strategy, and n_splits attributes.
        labels_test (array-like): True test labels.
        labels_pred (array-like): Predicted labels from the model.
        labels_train (pd.Series): Training labels used for training.
        top_n (int): Number of features used in training.
        grid_search (GridSearchCV): Fitted GridSearchCV object with
            best_params_ and best_score_ attributes.
        random_state (int): Random state seed used for reproducibility.
        metrics_filepath (str | None, optional): Path to save metrics as
            a JSON file. If None, does not save. Defaults to None.

    Returns:
        dict[str | Any, int | str | Any]: Metrics dictionary with keys:
            - ``random_state`` (int): Random seed used.
            - ``classifier`` (str): Classifier name.
            - ``cv_strategy`` (str): Cross-validation strategy.
            - ``cv_splits`` (int): Number of CV splits.
            - ``n_train`` (int): Number of training samples.
            - ``n_test`` (int): Number of test samples.
            - ``n_features`` (int): Number of features used.
            - ``true_negatives`` (int): TN count.
            - ``false_positives`` (int): FP count.
            - ``false_negatives`` (int): FN count.
            - ``true_positives`` (int): TP count.
            - ``accuracy`` (float): Accuracy score.
            - ``balanced_accuracy`` (float): Balanced accuracy score.
            - ``f1_score`` (float): F1 score.
            - ``precision`` (float): Precision score.
            - ``recall`` (float): Recall score.
            - ``best_params`` (dict): Best hyperparameters from GridSearchCV.
            - ``best_cv_score`` (float): Best CV score from GridSearchCV.

    Examples:
        >>> metrics = get_metrics(
        ...     classifier_model=clf_model,
        ...     labels_test=y_test,
        ...     labels_pred=y_pred,
        ...     labels_train=y_train,
        ...     top_n=20,
        ...     grid_search=gs,
        ...     random_state=42,
        ...     metrics_filepath="output/metrics.json",
        ... )
        >>> print(f"F1: {metrics['f1_score']:.3f}")
        F1: 0.850
    """

    # Confusion matrix for Binary Classification
    # Read more: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#binary-classification
    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(labels_test, labels_pred).ravel().tolist()
    )

    metrics = {
        "random_state": random_state,
        "classifier": classifier_model.name,
        "cv_strategy": classifier_model.cv_strategy,
        "cv_splits": classifier_model.n_splits,
        "n_train": len(labels_train),
        "n_test": len(labels_test),
        "n_features": top_n,
        "true_negatives": true_negative,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_positives": true_positive,
        "accuracy": accuracy_score(labels_test, labels_pred),
        "balanced_accuracy": balanced_accuracy_score(labels_test, labels_pred),
        "f1_score": f1_score(labels_test, labels_pred),
        "precision": precision_score(labels_test, labels_pred),
        "recall": recall_score(labels_test, labels_pred),
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
    }

    # Save metrics
    if metrics_filepath is not None:
        with open(metrics_filepath, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


def slugify_class_name(class_name: str) -> str:
    """Convert a class name to a slug for use in filenames.

    Converts CamelCase class names to lowercase hyphen-separated slugs.
    Handles consecutive uppercase letters (e.g., HTTP, XML) correctly.
    Used for creating classifier-specific directory names.

    Args:
        class_name (str): Class name in CamelCase format.

    Returns:
        str: Slugified class name in lowercase with hyphens.

    Examples:
        >>> slugify_class_name("MyClassName")
        'my-class-name'
        >>> slugify_class_name("HTTPSConnection")
        'https-connection'
        >>> slugify_class_name("XMLParser")
        'xml-parser'
        >>> slugify_class_name("XGBClassifier")
        'xgb-classifier'
    """
    import re

    # Insert hyphens before uppercase letters (except at start)
    s = re.sub("([a-z0-9])([A-Z])", r"\1-\2", class_name)
    # Handle consecutive uppercase letters (e.g., HTTP)
    s = re.sub("([A-Z]+)([A-Z][a-z])", r"\1-\2", s)

    return s.lower()


def compute_seed_eruption_probability(
    random_state: int,
    features_df: pd.DataFrame,
    significant_features_csv: str,
    model_filepath: str,
    output_dir: str | None = None,
    save: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eruption probability for a single random seed model.

    Loads a trained model and computes eruption probabilities for the given features.
    Supports both predict_proba (probabilistic) and decision_function (SVM) methods.
    Can cache results to disk for faster repeated predictions.

    Args:
        random_state (int): Random seed identifying the model.
        features_df (pd.DataFrame): Extracted features DataFrame for prediction.
        significant_features_csv (str): Path to CSV containing significant feature names.
        model_filepath (str): Path to the saved model (.pkl file).
        output_dir (str | None, optional): Directory to save predictions. If None,
            uses "output/predictions/seeds". Defaults to None.
        save (bool, optional): If True, save probabilities to CSV. Defaults to False.
        overwrite (bool, optional): If True, overwrite existing cached predictions.
            Defaults to False.
        verbose (bool, optional): If True, log save operations. Defaults to False.
        debug (bool, optional): If True, log detailed debug information. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing:
            - probabilities_eruption (np.ndarray): 1D array of eruption probabilities (P(class=1)).
            - probabilities_scores (np.ndarray): 2D array of shape (n_windows, 2) with
              columns [P(non-eruption), P(eruption)].

    Raises:
        ValueError: If model output is 1-dimensional.
        RuntimeError: If model supports neither predict_proba nor decision_function.

    Examples:
        >>> proba_1d, proba_2d = compute_seed_eruption_probability(
        ...     random_state=42,
        ...     features_df=features,
        ...     significant_features_csv="sig_features.csv",
        ...     model_filepath="model_00042.pkl",
        ...     save=True
        ... )
        >>> print(proba_1d.mean())
        0.35
    """
    output_dir = output_dir or os.path.join(os.getcwd(), "output", "predictions")
    output_dir = os.path.join(output_dir, "seeds")

    filename = f"p_{random_state:05d}.csv"
    filepath = os.path.join(output_dir, f"{filename}")

    if os.path.exists(filepath) and not overwrite:
        seed_df = pd.read_csv(filepath, index_col=0)
        eruption_probabilities = seed_df["p_eruption"]
        return eruption_probabilities.to_numpy(), seed_df.to_numpy()

    df_sig = pd.read_csv(significant_features_csv, index_col=0)
    feature_names: list[str] = df_sig.index.tolist()

    # Load trained model
    model = joblib.load(model_filepath)

    # Select features dataframe with top-n significant features
    X = features_df[feature_names]

    if hasattr(model, "predict_proba"):
        # probabilities_scores has shape (n_rows, n_windows). Where n_window will reflect
        # the number of the classifiications. In this case n_windows is 2 which indicates
        # 0 as nnn-eruption, and 1 as an eruption.
        probabilities_scores: np.ndarray = model.predict_proba(X)

        if probabilities_scores.ndim == 1:
            raise ValueError(
                f"Your probability for seed {random_state} scores only has 1 dimensions. "
                f"It should have 2 dimensions which consists of `0` (non-eruption) and `1` (eruption)."
            )

        # Select column 1 as eruption probabilities representative. Column 0 is non-eruption.
        probabilities_eruption: np.ndarray = probabilities_scores[:, 1]

        if debug:
            logger.debug(f"{random_state:05d} :: predict_probe was used")
            logger.debug(
                f"{random_state:05d} :: probabilities_eruption values: {probabilities_eruption}"
            )

    elif hasattr(model, "decision_function"):
        probabilities_scores: np.ndarray = model.decision_function(X)
        probabilities_eruption: np.ndarray = 1.0 / (1.0 + np.exp(-probabilities_scores))

        if debug:
            logger.debug(f"{random_state:05d} :: decision_function was used")
            logger.debug(
                f"{random_state:05d} :: probabilities_eruption values: {probabilities_eruption}"
            )

    else:
        raise RuntimeError(
            f"Model at {model_filepath} supports neither "
            "predict_proba nor decision_function."
        )

    if save and not overwrite:
        os.makedirs(output_dir, exist_ok=True)
        probabilities_df = pd.DataFrame(
            probabilities_scores, columns=["p_non_eruption", "p_eruption"]
        )

        probabilities_df.index.name = "label_id"
        probabilities_df.to_csv(filepath, index=True)

        if verbose:
            logger.info(f"Saved seed {random_state:05d} probability to: {filepath}")

    return probabilities_eruption, probabilities_scores


def compute_model_probabilities(
    df_models: pd.DataFrame,
    features_df: pd.DataFrame,
    features_csv_column: str = "significant_features_csv",
    trained_model_filepath_column: str = "trained_model_filepath",
    classifier_name: str = "model",
    threshold: float = 0.7,
    number_of_seeds: int | None = None,
    output_dir: str | None = None,
    save_predictions: bool = False,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate eruption probabilities across all seeds of a single classifier.

    Computes consensus predictions by averaging probabilities from multiple models
    trained with different random seeds. This reduces variance and improves prediction
    reliability. Calculates mean probability, uncertainty (std), confidence, and
    binary predictions.

    Args:
        df_models (pd.DataFrame): Model registry DataFrame with random_state as index.
            Must contain columns specified by features_csv_column and
            trained_model_filepath_column.
        features_df (pd.DataFrame): Extracted feature matrix for the prediction windows.
        features_csv_column (str, optional): Column name containing paths to significant
            features CSVs. Defaults to "significant_features_csv".
        trained_model_filepath_column (str, optional): Column name containing paths to
            trained model files. Defaults to "trained_model_filepath".
        classifier_name (str, optional): Classifier name for logging. Defaults to "model".
        threshold (float, optional): Minimum mean probability threshold to classify as
            eruption (0.0-1.0). Defaults to 0.7.
        number_of_seeds (int | None, optional): Maximum number of seeds to use. If None,
            uses all seeds. Defaults to None.
        output_dir (str | None, optional): Directory to save per-seed predictions.
            Defaults to None.
        save_predictions (bool, optional): If True, save per-seed predictions to CSV.
            Defaults to False.
        overwrite (bool, optional): If True, overwrite existing cached predictions.
            Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing arrays
            of shape (n_windows,):
            - mean_probability (np.ndarray): Mean eruption probability across seeds.
            - std_proba (np.ndarray): Standard deviation of probabilities (uncertainty).
            - confidence (np.ndarray): Voting agreement fraction (0.5-1.0).
            - prediction (np.ndarray): Binary predictions (0 or 1) based on threshold.

    Examples:
        >>> mean_p, std_p, conf, pred = compute_model_probabilities(
        ...     df_models=model_registry,
        ...     features_df=features,
        ...     threshold=0.6,
        ...     number_of_seeds=100
        ... )
        >>> print(f"Mean eruption probability: {mean_p.mean():.3f}")
        Mean eruption probability: 0.450
    """
    seed_eruption_probabilities: list[np.ndarray] = []

    for random_state, row in df_models.iterrows():
        random_state_int = int(cast(int, random_state))
        significant_features_csv: str = row[features_csv_column]
        model_filepath: str = row[trained_model_filepath_column]

        probabilities_eruption, _ = compute_seed_eruption_probability(
            random_state=random_state_int,
            significant_features_csv=significant_features_csv,
            model_filepath=model_filepath,
            features_df=features_df,
            output_dir=output_dir,
            overwrite=overwrite,
            save=save_predictions,
        )

        seed_eruption_probabilities.append(probabilities_eruption)
        logger.debug(
            f"[{classifier_name}] Seed {random_state:05d} — "
            f"mean P(eruption): {probabilities_eruption.mean():.4f}"
        )

        if number_of_seeds is not None and random_state_int > number_of_seeds:
            break

    probabilities_eruption_matrix = np.stack(
        seed_eruption_probabilities, axis=1
    )  # (n_seeds, n_windows)

    mean_probability: np.ndarray = probabilities_eruption_matrix.mean(axis=1)
    std_proba: np.ndarray = probabilities_eruption_matrix.std(axis=1)
    prediction: np.ndarray = (mean_probability >= threshold).astype(int)

    votes_for_eruption: np.ndarray = (probabilities_eruption_matrix >= 0.5).sum(axis=1)
    n_seeds = probabilities_eruption_matrix.shape[0]
    majority_votes = np.where(
        prediction == 1, votes_for_eruption, n_seeds - votes_for_eruption
    )
    confidence: np.ndarray = majority_votes / n_seeds

    return mean_probability, std_proba, confidence, prediction


def label_id_to_datetime(
    label_df: pd.DataFrame | pd.Series, target_df: pd.DataFrame
) -> pd.DataFrame:
    """Add datetime column to target DataFrame by merging with label DataFrame.

    Merges the label DataFrame (containing id and datetime) with the target DataFrame
    to add datetime information. This is useful for adding temporal context to
    extracted features, tremor matrices, or prediction results.

    Args:
        label_df (pd.DataFrame | pd.Series): Label DataFrame or Series with datetime
            index containing 'id' and 'datetime' columns. If Series, will be converted
            to DataFrame.
        target_df (pd.DataFrame): Target DataFrame to which datetime will be added.
            Can be extracted features, eruption probabilities, or tremor matrix.
            Must have matching index with label_df.

    Returns:
        pd.DataFrame: Target DataFrame with datetime column merged from label_df.

    Examples:
        >>> label_df = pd.DataFrame({"id": [1, 2], "datetime": ["2025-01-01", "2025-01-02"]})
        >>> target_df = pd.DataFrame({"feature_1": [0.5, 0.8]}, index=[1, 2])
        >>> result = label_id_to_datetime(label_df, target_df)
        >>> print(result.columns)
        Index(['feature_1', 'id', 'datetime'], dtype='object')
    """
    if isinstance(label_df, pd.Series):
        label_df = pd.DataFrame(label_df)

    return target_df.merge(label_df, left_index=True, right_index=True)
