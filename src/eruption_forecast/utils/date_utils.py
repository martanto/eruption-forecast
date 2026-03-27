"""Date and time conversion, normalisation, and label-filename parsing utilities.

This module standardises all date handling across the pipeline.  Every public
function accepts either a ``datetime`` object or a ``"YYYY-MM-DD"`` string and
returns a well-typed result.  Validation of date ranges and window step
constraints is intentionally delegated to
:mod:`eruption_forecast.utils.validation` to avoid circular imports.

Key functions
-------------
- ``to_datetime`` — coerce a string or ``datetime`` to a ``datetime``; raises a
  descriptive ``ValueError`` for malformed inputs
- ``normalize_dates`` — anchor start to 00:00:00 and end to 23:59:59; returns both
  ``datetime`` objects and ``"YYYY-MM-DD"`` strings
- ``sort_dates`` — sort a list of date strings chronologically; optionally return
  ``datetime`` objects
- ``parse_label_filename`` — extract all structured parameters (start/end date,
  window step, unit, day-to-forecast) from a label CSV basename
- ``set_datetime_index`` — attach a ``DatetimeIndex`` to a DataFrame by joining with
  an ``id → datetime`` mapping; used for features and forecast outputs
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


def sort_dates(dates: list[str], as_datetime: bool = False) -> list[str] | list[datetime]:
    """Sort a list of date strings chronologically.

    Converts date strings to datetime objects, sorts them chronologically,
    and returns them either as formatted strings or as datetime objects.

    Args:
        dates (list[str]): List of date strings in YYYY-MM-DD format.
        as_datetime (bool, optional): If True, returns a list of datetime objects
            instead of formatted strings. Defaults to False.

    Returns:
        list[str] | list[datetime]: Sorted dates as YYYY-MM-DD strings when
        ``as_datetime`` is False, or as datetime objects when True.

    Examples:
        >>> sort_dates(["2025-03-20", "2025-01-15", "2025-02-10"])
        ['2025-01-15', '2025-02-10', '2025-03-20']
    """
    date_series = pd.Series(dates)
    date_series = date_series.apply(pd.to_datetime, format="%Y-%m-%d").sort_values()

    if as_datetime:
        return date_series.tolist()

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


def set_datetime_index(
    datetime_map: pd.DataFrame | pd.Series,
    df: pd.DataFrame,
    on: str | None = None,
) -> pd.DataFrame:
    """Set a DatetimeIndex on a DataFrame by merging with a datetime mapping.

    Merges ``datetime_map`` (containing ``id`` and ``datetime``) with ``df``,
    then replaces the index with a ``DatetimeIndex``. This is useful for adding
    temporal context to extracted features, eruption probabilities, or tremor
    matrices.

    If ``df`` already has a ``DatetimeIndex``, it is returned unchanged.

    Args:
        datetime_map (pd.DataFrame | pd.Series): Source of the ``id → datetime``
            mapping. Accepted forms:
            - DataFrame with an integer ``id`` index and a ``"datetime"`` column.
            - Series named ``"datetime"`` with an integer ``id`` index.
            - DatetimeIndex-based DataFrame with an ``"id"`` column used as the
              merge key.
        df (pd.DataFrame): DataFrame whose index will be replaced with a
            ``DatetimeIndex``. Can hold features, eruption probabilities, or tremor
            data.
        on (str | None, optional): Column in ``df`` to use as the join key against
            the index of ``datetime_map``. When ``None`` (default), merges on the
            index of both DataFrames.

    Returns:
        pd.DataFrame: Copy of ``df`` with a ``DatetimeIndex`` derived from
        ``datetime_map``'s ``"datetime"`` column.

    Raises:
        ValueError: If ``datetime_map`` is a Series not named ``"datetime"``.
        ValueError: If a DatetimeIndex-based ``datetime_map`` has no ``"id"`` column.
        ValueError: If ``datetime_map`` does not contain a ``"datetime"`` column
            after normalisation.
        ValueError: If ``on`` is specified but the column does not exist in ``df``.

    Examples:
        >>> datetime_map = pd.DataFrame(
        ...     {"datetime": ["2025-01-01", "2025-01-02"]}, index=[1, 2]
        ... )
        >>> df = pd.DataFrame({"feature_1": [0.5, 0.8]}, index=[1, 2])
        >>> result = set_datetime_index(datetime_map, df)
        >>> print(result.columns.tolist())
        ['feature_1']
        >>> print(type(result.index))
        <class 'pandas.core.indexes.datetimes.DatetimeIndex'>
        >>> # Merge on a column instead of the index
        >>> df_with_id = pd.DataFrame({"id": [1, 2], "feature_1": [0.5, 0.8]})
        >>> result = set_datetime_index(datetime_map, df_with_id, on="id")
        >>> print(result.columns.tolist())
        ['feature_1']
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    if isinstance(datetime_map, pd.Series):
        if datetime_map.name != "datetime":
            raise ValueError(
                f"Series passed as datetime_map must be named 'datetime'. Got: '{datetime_map.name}'"
            )
        datetime_map = datetime_map.to_frame()

    if isinstance(datetime_map.index, pd.DatetimeIndex):
        if "id" not in datetime_map.columns:
            raise ValueError(
                "DatetimeIndex-based datetime_map must have an 'id' column to use as merge key. "
                f"Got columns: {datetime_map.columns.tolist()}"
            )
        datetime_map = datetime_map.reset_index(drop=False)
        datetime_map = datetime_map.set_index("id")

    if "datetime" not in datetime_map.columns:
        raise ValueError(
            f"datetime_map must have a 'datetime' column. Got: {datetime_map.columns.tolist()}"
        )

    if on is not None and on not in df.columns:
        raise ValueError(
            f"Column '{on}' specified in 'on' does not exist in df. "
            f"Got columns: {df.columns.tolist()}"
        )

    datetime_map: pd.DataFrame = datetime_map[["datetime"]]

    if on is not None:
        result = df.merge(datetime_map, left_on=on, right_index=True).drop(columns=[on])
    else:
        result = df.merge(datetime_map, left_index=True, right_index=True)

    result.index = pd.to_datetime(result["datetime"])
    result = result.drop(columns=["datetime"])

    return result
