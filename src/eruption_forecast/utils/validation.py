"""Input validation utilities for eruption forecasting.

This module centralises all guard/validation functions used across the package:
date range validation, window step validation, random state validation,
DataFrame column validation, and sampling consistency checks.
"""

from typing import Literal
from datetime import datetime

import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.date_utils import to_datetime


def validate_random_state(random_state: int) -> None:
    """Validate that a random_state value is non-negative.

    Args:
        random_state (int): The random seed to validate.

    Raises:
        ValueError: If random_state is negative.

    Examples:
        >>> validate_random_state(42)   # OK
        >>> validate_random_state(-1)   # raises ValueError
    """
    if random_state < 0:
        raise ValueError(f"random_state must be >= 0. Got {random_state}")


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

    time_diffs = df.index.to_series().diff()
    expected_diff = pd.Timedelta(expected_freq)
    tolerance_diff = pd.Timedelta(tolerance)
    lower_bound = expected_diff - tolerance_diff
    upper_bound = expected_diff + tolerance_diff

    inconsistent_mask: pd.Series = ~(
        (time_diffs >= lower_bound) & (time_diffs <= upper_bound)
    )
    inconsistent_mask.iloc[0] = False

    inconsistent_data = df[inconsistent_mask]
    consistent_data = df[~inconsistent_mask]
    is_consistent = inconsistent_data.empty

    if is_consistent:
        sampling_rate = (df.index[1] - df.index[0]).seconds

    if verbose:
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Inconsistent rows found: {len(inconsistent_data)}")
        logger.info(f"Consistent rows: {len(consistent_data)}")
        if len(inconsistent_data) > 0:
            logger.warning("\nInconsistent time differences:")
            logger.warning(time_diffs[inconsistent_mask].describe())

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
