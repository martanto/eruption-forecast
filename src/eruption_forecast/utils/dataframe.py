"""DataFrame validation and manipulation utilities.

This module provides functions for DataFrame operations including column
validation, sampling consistency checks, series conversion, and feature
concatenation for tsfresh processing.
"""

from __future__ import annotations

import pandas as pd


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
