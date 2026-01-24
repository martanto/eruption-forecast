# Standard library imports
from datetime import datetime, timedelta
from typing import Callable, Literal, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
from loguru import logger
from obspy import Trace


def detect_outliers(
    data: np.ndarray, outlier_threshold: float = 0.5
) -> tuple[bool, Union[int, float], float]:
    """Detect outliers in an array using z-score ((X - μ) / σ)

    Args:
        data (np.ndarray): Array of data from trace
        outlier_threshold (float, optional): Degree of outliers. Defaults to 0.5.

    Returns:
        tuple[bool, Union[int, float], float]:
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

    # If z_score is greater than 10^outlier_threshold or 3σ, it is an outlier
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

    return data


def get_windows_information(
    trace: Trace,
    window_duration_minutes: int = 10,
) -> dict[str, Union[int, float]]:
    """Get window and sample information from a Trace

    Args:
        trace (Trace): Trace
        window_duration_minutes (int, optional): Duration of each window in minutes. Defaults to 10.

    Returns:
        dict[str, Union[int, float]]: Window and sample information

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


def construct_windows(
    window_step: int,
    window_step_unit: Literal["minutes", "hours"],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
) -> pd.DataFrame:
    """Construct windows for label and tremor data

    Args:
        window_step (int): Step size
        window_step_unit (Literal["minutes", "hours"]): Unit of window step.
        start_date (Union[str, datetime]): Start date
        end_date (Union[str, datetime]): End date

    Returns:
        pd.DataFrame
    """
    window_step, window_step_unit = validate_window_step(window_step, window_step_unit)
    start_date, end_date, n_days = validate_date_ranges(start_date, end_date)

    maximum_window_step = n_days * 24
    if window_step_unit == "minutes":
        maximum_window_step = n_days * 60 * 24

    assert window_step <= maximum_window_step, (
        f"window_step must be less than or equal to {maximum_window_step} "
        f"{window_step_unit}. \\n"
        f"window_step: {window_step}, maximum_window_step: {maximum_window_step}"
    )

    start_date = start_date.replace(hour=0, minute=0, second=0)
    end_date = end_date.replace(hour=23, minute=59, second=59)

    freq_in_hours = timedelta(hours=window_step)
    if window_step_unit == "minutes":
        freq_in_hours = timedelta(minutes=window_step)

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq=freq_in_hours,
        inclusive="both",
    )

    df = pd.DataFrame(index=dates)
    df.index.name = "datetime"

    return df


def to_datetime(
    date: Union[str, datetime], variable_name: Optional[str] = None
) -> datetime:
    """Ensure date object is a datetime object

    Args:
        date (str): Date string
        variable_name (str, optional): Variable name. Defaults to None.

    Returns:
        datetime: Datetime
    """
    if isinstance(date, datetime):
        return date

    variable_name = f"{variable_name}" if variable_name else "Date"

    try:
        return datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise ValueError(
            f"{variable_name} value {date} " f"is not valid YYYY-MM-DD format."
        )


def validate_date_ranges(
    start_date: Union[str, datetime], end_date: Union[str, datetime]
) -> Tuple[datetime, datetime, int]:
    """Validate date ranges

    Args:
        start_date (Union[str, datetime]): Start date in YYYY-MM-DD format.
        end_date (Union[str, datetime]): End date in YYYY-MM-DD format.

    Raise:
        ValueError: Date ranges are not valid.

    Returns:
        Tuple[datetime, datetime, int]: start date, end date, and total number of days
    """
    if isinstance(start_date, str):
        start_date: datetime = to_datetime(start_date)
    if isinstance(end_date, str):
        end_date: datetime = to_datetime(end_date)

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
    """Validate window step and step unit

    Args:
        window_step (int): Step size
        window_step_unit (Literal["minutes", "hours"]): Unit of window step.

    Raise:
        ValueError: Window step or unit is invalid.

    Returns:
        Tuple[int, Literal["minutes", "hours"]]: Window step and unit (minutes, hours)
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
