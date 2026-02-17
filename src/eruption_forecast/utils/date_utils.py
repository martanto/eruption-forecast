"""Date and time validation and conversion utilities.

This module provides functions for date validation, conversion, normalization,
and window step validation. Ensures consistent date handling across the package.
"""

from typing import Literal
from datetime import datetime

import pandas as pd


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
