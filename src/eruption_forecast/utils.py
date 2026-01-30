# Standard library imports
from datetime import datetime, timedelta
from typing import Callable, Literal, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
from obspy import Trace

# Project imports
from eruption_forecast.logger import logger


def detect_outliers(
    data: np.ndarray, outlier_threshold: float = 0.5
) -> tuple[bool, Union[int, float], float]:
    """Detect outliers in an array using z-score ((X - μ) / σ)

    Args:
        data (np.ndarray): Array of data from a trace.
        outlier_threshold (float, optional): Outlier threshold degree. Defaults to 0.5.

    Returns:
        tuple[bool, Union[int, float], float]:
            outlier (bool): True if outlier is detected, False otherwise.
            outlier_index: Index of the outlier.
            outlier_value: Value of the outlier.
    """
    if isinstance(data, pd.Series):
        data = np.array(data.values)

    outlier_index = np.argmax(data)
    outlier_value = data[outlier_index]

    if np.std(data) == 0:
        return True, int(outlier_index), float(outlier_value)

    # Z-score = (X - μ) / σ
    z_score = (outlier_value - np.mean(data)) / np.std(data)

    # If z_score is greater than 10^outlier_threshold or 3σ, it is an outlier
    if z_score > 10**outlier_threshold:
        return True, int(outlier_index), float(outlier_value)

    return False, np.nan, np.nan


def delete_outliers(data: np.ndarray) -> np.ndarray:
    """Remove outliers from an array.

    Args:
        data (np.ndarray): Array of data.

    Returns:
        np.ndarray: Array without outliers.
    """
    if np.sum(data) == 0:
        return np.array([])

    outlier, outlier_index, _ = detect_outliers(data)

    if outlier:
        data = np.delete(data, int(outlier_index))

    return data


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
    remove_outliers: bool = True,
    minimum_completion_ratio: float = 0.3,
    absolute_value: bool = False,
    value_multiplier: float = 1.0,
) -> pd.Series:
    """Calculate metrics for defined time windows of an ObsPy Trace.

    Args:
        trace (Trace): ObsPy Trace object.
        window_duration_minutes (int, optional): Duration of each window in minutes. Defaults to 10.
        metric_function (callable, optional): Function to calculate metric (e.g., np.mean, np.max). Defaults to np.mean.
        remove_outliers (bool, optional): Whether to remove outliers before calculation. Defaults to True.
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

        if length_window_data > 1:
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
        data_points.append(float(metric_value))

    return pd.Series(data=data_points, index=indices, name="datetime", dtype=float)


def construct_windows(
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> pd.DataFrame:
    """Construct time windows for label and tremor data.

    Args:
        window_step (int): Step size between windows.
        window_step_unit (Literal["minutes", "hours"]): Unit of window step.
        start_date (Union[str, datetime]): Start date in YYYY-MM-DD format or datetime object.
        end_date (Union[str, datetime]): End date in YYYY-MM-DD format or datetime object.

    Returns:
        pd.DataFrame: DataFrame with datetime index representing time windows.
    """
    window_step, window_step_unit = validate_window_step(window_step, window_step_unit)
    start_date, end_date, n_days = validate_date_ranges(start_date, end_date)

    maximum_window_step = n_days * 24
    if window_step_unit == "minutes":
        maximum_window_step = n_days * 60 * 24

    assert window_step <= maximum_window_step, (
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

    assert start_date < end_date, ValueError(
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
    assert isinstance(window_step, int), ValueError(
        f"window_step must be an integer. Your value is {window_step}"
    )
    assert isinstance(window_step_unit, str), ValueError(
        f"window_step_unit must be a string. Your value is {window_step_unit}"
    )
    assert window_step_unit in [
        "minutes",
        "hours",
    ], ValueError(
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
        pd.DataFrame: Conisntency DataFrame with pd.DatetimeIndex.
        pd.DataFrame: Inconisntency DataFrame with pd.DatetimeIndex.
    """
    assert len(df) > 2, ValueError(
        "DataFrame must have at least 2 rows to check sampling consistency"
    )
    assert isinstance(df.index, pd.DatetimeIndex), ValueError(
        "DataFrame index must be DatetimeIndex"
    )

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
