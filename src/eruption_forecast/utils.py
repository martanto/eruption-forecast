# Standard library imports
from datetime import datetime, timedelta
from typing import Callable, Literal, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from obspy import Trace
from tsfresh.transformers import FeatureSelector

# Project imports
from eruption_forecast.logger import logger


def mask_zero_values(data: np.ndarray) -> np.ndarray:
    """Remove zero values from an array.

    Args:
        data (np.ndarray): Input array of numerical data.

    Returns:
        np.ndarray: Array with zero values removed.

    Examples:
        >>> mask_zero_values(np.array([1, 0, 2, 0, 3]))
        array([1, 2, 3])
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("Input must be a numpy array")

    non_zero_mask = data != 0.0
    return data[non_zero_mask]


def detect_maximum_outlier(
    data: np.ndarray, outlier_threshold: float = 3.0
) -> Tuple[bool, Union[int, float], float]:
    """Detect if maximum value in array is an outlier using z-score method.

    Uses z-score ((X - μ) / σ) to determine if the maximum value in the array
    is statistically an outlier. A value is considered an outlier if its z-score
    exceeds the threshold (default 3.0, equivalent to 3 standard deviations).

    Args:
        data (np.ndarray): Array of numerical data.
        outlier_threshold (float, optional): Z-score threshold for outlier detection.
            Defaults to 3.0 (3 standard deviations).

    Returns:
        Tuple[bool, Union[int, float], float]:
            - is_outlier (bool): True if maximum value is an outlier
            - outlier_index (int | float): Index of the maximum value, or np.nan if no outlier
            - outlier_value (float): Maximum value, or np.nan if no outlier

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If array is empty or outlier_threshold is not positive

    Examples:
        >>> detect_maximum_outlier(np.array([1, 2, 3, 100]))  # 100 is outlier
        (True, 3, 100.0)
        >>> detect_maximum_outlier(np.array([1, 2, 3, 4]))  # No outlier
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
    Optionally masks zero values before outlier detection.

    Args:
        data (np.ndarray): Input array of numerical data.
        mask_zero_value (bool, optional): If True, remove zero values before processing.
            Defaults to True.
        outlier_threshold (float, optional): Z-score threshold for outlier detection.
            Defaults to 3.0.

    Returns:
        np.ndarray: Array with maximum outlier removed (if detected).

    Raises:
        TypeError: If input is not a numpy array

    Examples:
        >>> remove_maximum_outlier(np.array([1, 2, 3, 100]))
        array([1, 2, 3])
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

    Iteratively removes all values whose z-score exceeds the threshold.
    Unlike remove_maximum_outlier which removes only one value, this function
    removes all outliers in a single pass.

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

    Raises:
        TypeError: If input is not a numpy array
        ValueError: If outlier_threshold is not positive

    Examples:
        >>> remove_outliers(np.array([1, 2, 3, 100, 200]))
        array([1, 2, 3])
        >>> remove_outliers(np.array([1, 2, 3, 100, 200]), return_outliers=True)
        array([100, 200])
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
) -> dict[str, Union[int, float]]:
    """Get window and sample information from an ObsPy Trace.

    Args:
        trace (Trace): ObsPy Trace object.
        window_duration_minutes (int, optional): Duration of each window in minutes. Defaults to 10.

    Returns:
        dict[str, Union[int, float]]: Window and sample information.

    Example:
        Example return value:
            {
                "number_of_samples": number_of_samples,
                "samples_per_day": samples_per_day,
                "sample_per_window": sample_per_window,
                "total_windows": total_windows,
                "sample_window": sample_window,
            }
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
    remove_outlier_method: Optional[Literal["maximum", "all"]] = None,
    mask_zero_value: bool = False,
    minimum_completion_ratio: float = 0.3,
    absolute_value: bool = False,
    value_multiplier: float = 1.0,
) -> pd.Series:
    """Calculate metrics for defined time windows of an ObsPy Trace.

    Args:
        trace (Trace): ObsPy Trace object.
        window_duration_minutes (int, optional): Duration of each window in minutes. Defaults to 10.
        metric_function (callable, optional): Function to calculate metric (e.g., np.mean, np.max). Defaults to np.mean.
        mask_zero_value (bool, optional): Mask zero values. Defaults to False.
        remove_outlier_method (Literal["maximum", "all"], optional): Remove outlier method. Defaults to "maximum".
        minimum_completion_ratio (float, optional): Minimum ratio of data points required to calculate the metric. Defaults to 0.3.
        absolute_value (bool, optional): Whether to use absolute values. Defaults to False.
        value_multiplier (float, optional): Multiplier for the metric value. Defaults to 1.0.

    Returns:
        pd.Series: Series containing the calculated metrics with datetime index.
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
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
) -> pd.DataFrame:
    """Construct time windows for label and tremor data.

    Args:
        start_date (Union[str, datetime]): Start date in YYYY-MM-DD format or datetime object.
        end_date (Union[str, datetime]): End date in YYYY-MM-DD format or datetime object.
        window_step (int): Step size between windows.
        window_step_unit (Literal["minutes", "hours"]): Unit of window step.

    Returns:
        pd.DataFrame: DataFrame with datetime index representing time windows.
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


def to_datetime(
    date: Union[str, datetime], variable_name: Optional[str] = None
) -> datetime:
    """Ensure date object is a datetime object.

    Args:
        date (Union[str, datetime]): Date string in YYYY-MM-DD format or datetime object.
        variable_name (str, optional): Variable name for error messages. Defaults to None.

    Returns:
        datetime: Datetime object.
    """
    if isinstance(date, datetime):
        return date

    variable_name = f"{variable_name}" if variable_name else "Date"

    try:
        return datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"{variable_name} value {date} is not in valid YYYY-MM-DD format."
        )


def validate_date_ranges(
    start_date: Union[str, datetime], end_date: Union[str, datetime]
) -> Tuple[datetime, datetime, int]:
    """Validate date range.

    Args:
        start_date (Union[str, datetime]): Start date in YYYY-MM-DD format or datetime object.
        end_date (Union[str, datetime]): End date in YYYY-MM-DD format or datetime object.

    Raises:
        ValueError: If date range is not valid.

    Returns:
        Tuple[datetime, datetime, int]: Start date, end date, and total number of days.
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
) -> Tuple[int, Literal["minutes", "hours"]]:
    """Validate window step and step unit.

    Args:
        window_step (int): Step size between windows.
        window_step_unit (Literal["minutes", "hours"]): Unit of window step.

    Raises:
        ValueError: If window step or unit is invalid.

    Returns:
        Tuple[int, Literal["minutes", "hours"]]: Window step and unit (minutes or hours).
    """
    if not isinstance(window_step, int):
        raise TypeError(
            f"window_step must be an integer. Your value is {window_step}"
        )
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
    """Convert the list of dates into a pandas Series.

    Args:
        dates (list[str]): List of dates.

    Returns:
        list[str]: List of dates.
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
) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
    """
    Check 10-minute sampling rate consistency, identify inconsistencies, and remove them.

    Args:
        df (pd.DataFrame): DataFrame with pd.DatetimeIndex.
        expected_freq (optional, str): Expected sampling frequency. Defaults to "10min".
        tolerance (optional, str): Tolerance in seconds for considering sampling periods as equal (default: "1min").
        verbose (optional, bool): Print detailed information. Defaults to False.

    Returns:
        bool: True if consistent. False otherwise.
        pd.DataFrame: Consistency DataFrame with pd.DatetimeIndex.
        pd.DataFrame: Inconsistency DataFrame with pd.DatetimeIndex.
    """
    if len(df) <= 2:
        raise ValueError(
            "DataFrame must have at least 2 rows to check sampling consistency"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")

    df = df.sort_index()

    # Calculate time differences between consecutive timestamps
    time_diffs = df.index.to_series().diff()

    # Expected time difference
    expected_diff = pd.Timedelta(expected_freq)
    tolerance_diff = pd.Timedelta(tolerance)

    # Find inconsistent sampling rates (outside tolerance range)
    lower_bound = expected_diff - tolerance_diff
    upper_bound = expected_diff + tolerance_diff

    # First row will be NaT (no previous timestamp), so we skip it
    inconsistent_mask = ~((time_diffs >= lower_bound) & (time_diffs <= upper_bound))
    inconsistent_mask.iloc[0] = False

    # Get inconsistent data
    inconsistent_data = df[inconsistent_mask]

    # Get consistent data (remove inconsistencies)
    consistent_data = df[~inconsistent_mask]

    is_consistent = True if inconsistent_data.empty else False

    if verbose:
        print(f"Total rows: {len(df)}")
        print(f"Inconsistent rows found: {len(inconsistent_data)}")
        print(f"Consistent rows: {len(consistent_data)}")

        if len(inconsistent_data) > 0:
            print(f"\nInconsistent time differences:")
            print(time_diffs[inconsistent_mask].describe())

    return is_consistent, consistent_data, inconsistent_data


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Validate columns in dataframe.

    Args:
        df (pd.DataFrame): DataFrame with pd.DatetimeIndex.
        columns (list[str]): List of columns to validate.

    Raises:
        ValueError: If columns are invalid.

    Returns:
        None
    """
    for column in columns:
        if column in ["id", "datetime"]:
            continue
        if column not in df.columns.tolist():
            raise ValueError(
                f"Column {column} does not exist in dataframe. "
                f"Columns available are: {df.columns}"
            )
    return None


def concat_features(
    csv_list: list[str], filepath: str, return_as_filepath: bool = False
) -> Union[str, Tuple[str, pd.DataFrame]]:
    """Concatenate features from csv_list into one dataframe.

    Args:
        csv_list (list[str]): List of csv files.
        filepath (str): Filepath to save csv file.
        return_as_filepath (bool, optional): Return as CSV filepath. Defaults to False.

    Returns:
        str: Filepath of csv file.
        (str, pd.DataFrame): Filepath and DataFrame
    """
    if len(csv_list) <= 1:
        raise ValueError(
            f"Requires at least 2 CSV files. Total your CSV file is {len(csv_list)}"
        )

    df = pd.concat([pd.read_csv(file, index_col=0) for file in csv_list], axis=1)

    if df.empty:
        raise ValueError("There is no data in the csv files.")

    df.to_csv(filepath, index=True)

    if return_as_filepath:
        return filepath

    return filepath, df


def random_under_sampler(
    features: pd.DataFrame,
    labels: pd.Series,
    sampling_strategy: Union[str, float] = "auto",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Randomly under sampling features.

    Handling imbalance dataset of eruption and non-eruption.

    Args:
        features (pd.DataFrame): Features dataframe.
        labels (pd.Series): Labels dataframe.
        sampling_strategy (str, optional): Sampling strategy. Defaults to "auto".
        random_state (int, optional): Random state. Defaults to 42.

    Returns:
        pd.DataFrame: Randomly under sampling features.
        pd.Series: Randomly under sampling labels.
    """
    sampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy, random_state=random_state
    )

    features, labels = sampler.fit_resample(features, labels)

    return features, labels


def get_significant_features(
    features: pd.DataFrame,
    labels: pd.Series,
    n_jobs: int = 1,
) -> pd.Series:
    """Get significant features.

    Args:
        features (pd.DataFrame): Features dataframe.
        labels (pd.Series): Labels dataframe.
        number_of_significant_features (int, optional): Number of significant features. Defaults to 20.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.

    Returns:
        pd.Series: Significant features.
    """
    selector = FeatureSelector(n_jobs=n_jobs, ml_task="classification")
    selector.fit_transform(X=features, y=labels)  # type: ignore

    _significant_features = pd.Series(selector.p_values, index=features.columns)
    _significant_features = _significant_features.sort_values()
    _significant_features.name = "values"
    _significant_features.index.name = "features"

    return _significant_features


def normalize_dates(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> tuple[datetime, datetime, str, str]:
    """Normalize start and end dates to standard format.

    Converts date strings to datetime objects and formats them consistently.
    Start date is set to 00:00:00, end date is set to 23:59:59.

    Args:
        start_date: Start date in YYYY-MM-DD format or datetime object.
        end_date: End date in YYYY-MM-DD format or datetime object.

    Returns:
        Tuple of (start_date, end_date, start_date_str, end_date_str)
    """
    start_date = to_datetime(start_date).replace(hour=0, minute=0, second=0)
    end_date = to_datetime(end_date).replace(hour=23, minute=59, second=59)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    return start_date, end_date, start_date_str, end_date_str
