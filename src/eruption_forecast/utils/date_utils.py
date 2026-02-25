"""Date and time conversion and normalization utilities.

This module provides functions for date conversion, normalization, sorting,
and filename parsing. Validation of date ranges and window steps has been
moved to :mod:`eruption_forecast.utils.validation`.
"""

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


def parse_label_filename(basename: str) -> dict:
    """Parse label parameters from a label file basename (without extension).

    Extracts all structured parameters encoded in the standardised label filename
    format: ``label_{start}_{end}_step-{X}-{unit}_dtf-{X}``.

    Args:
        basename (str): Filename without the ``.csv`` extension, e.g.
            ``"label_2020-01-01_2020-12-31_step-12-hours_dtf-2"``.

    Returns:
        dict: Parsed parameters with keys:
            - ``start_date`` (datetime): Parsed start date.
            - ``end_date`` (datetime): Parsed end date.
            - ``start_date_str`` (str): Start date string in ``"YYYY-MM-DD"`` format.
            - ``end_date_str`` (str): End date string in ``"YYYY-MM-DD"`` format.
            - ``window_step`` (int): Window step numeric value.
            - ``window_step_unit`` (str): Window step unit (``"hours"`` or ``"minutes"``).
            - ``day_to_forecast`` (int): Days before eruption for positive label.

    Raises:
        ValueError: If ``basename`` does not match the expected
            ``label_{start}_{end}_step-{X}-{unit}_dtf-{X}`` format, including
            wrong number of underscore-separated parts, missing ``step-`` or
            ``dtf-`` prefixes, non-integer step or day values, or an unrecognised
            window step unit.

    Examples:
        >>> params = parse_label_filename("label_2020-01-01_2020-12-31_step-12-hours_dtf-2")
        >>> params["window_step"]
        12
        >>> params["window_step_unit"]
        'hours'
        >>> params["day_to_forecast"]
        2
    """
    _EXPECTED_FORMAT = "label_{start}_{end}_step-{X}-{unit}_dtf-{X}"
    _VALID_UNITS = ("minutes", "hours")

    parts = basename.split("_")
    if len(parts) != 5:
        raise ValueError(
            f"Label filename has {len(parts)} underscore-separated part(s); "
            f"expected 5. Got: '{basename}'. Expected format: {_EXPECTED_FORMAT}"
        )

    _, start_date_str, end_date_str, window_step_and_unit, day_to_forecast_part = parts

    if not window_step_and_unit.startswith("step-"):
        raise ValueError(
            f"Window step segment must start with 'step-'. "
            f"Got: '{window_step_and_unit}'. Expected format: {_EXPECTED_FORMAT}"
        )

    if not day_to_forecast_part.startswith("dtf-"):
        raise ValueError(
            f"Day-to-forecast segment must start with 'dtf-'. "
            f"Got: '{day_to_forecast_part}'. Expected format: {_EXPECTED_FORMAT}"
        )

    window_step_parts = window_step_and_unit.split("-")
    if len(window_step_parts) != 3:
        raise ValueError(
            f"Window step segment must have format 'step-{{int}}-{{unit}}'. "
            f"Got: '{window_step_and_unit}'. Expected format: {_EXPECTED_FORMAT}"
        )

    try:
        window_step = int(window_step_parts[1])
    except ValueError:
        raise ValueError(
            f"Window step value must be an integer. "
            f"Got: '{window_step_parts[1]}'. Expected format: {_EXPECTED_FORMAT}"
        )

    window_step_unit = window_step_parts[2]
    if window_step_unit not in _VALID_UNITS:
        raise ValueError(
            f"Window step unit must be one of {_VALID_UNITS}. "
            f"Got: '{window_step_unit}'. Expected format: {_EXPECTED_FORMAT}"
        )

    dtf_parts = day_to_forecast_part.split("-")
    if len(dtf_parts) != 2:
        raise ValueError(
            f"Day-to-forecast segment must have format 'dtf-{{int}}'. "
            f"Got: '{day_to_forecast_part}'. Expected format: {_EXPECTED_FORMAT}"
        )

    try:
        day_to_forecast = int(dtf_parts[1])
    except ValueError:
        raise ValueError(
            f"Day-to-forecast value must be an integer. "
            f"Got: '{dtf_parts[1]}'. Expected format: {_EXPECTED_FORMAT}"
        )

    start_date = to_datetime(start_date_str)
    end_date = to_datetime(end_date_str)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "start_date_str": start_date_str,
        "end_date_str": end_date_str,
        "window_step": window_step,
        "window_step_unit": window_step_unit,
        "day_to_forecast": day_to_forecast,
    }


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
