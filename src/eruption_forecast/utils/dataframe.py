"""DataFrame validation and manipulation utilities.

This module provides functions for DataFrame operations including column
validation, sampling consistency checks, series conversion, and feature
concatenation for tsfresh processing.
"""

import numpy as np
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import detect_anomalies_zscore


def remove_anomalies(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    interpolate: bool = False,
    threshold: float = 3.5,
    inplace: bool = False,
    debug: bool = False,
) -> pd.DataFrame:
    """Remove anomalies from a DataFrame.

    Applies Z-score based anomaly detection column-wise, replacing flagged values
    with NaN. Optionally interpolates the cleaned series using time-based interpolation.
    Operates in-place or on a copy depending on the ``inplace`` flag.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        columns (list[str] | None, optional): List of column names to check for
            anomalies. If None, all columns are checked. Defaults to None.
        interpolate (bool, optional): If True, interpolate the DataFrame after
            anomaly removal using time-based interpolation. Defaults to False.
        threshold (float, optional): Z-score threshold for anomaly detection.
            Values with |z-score| > threshold are flagged. Defaults to 3.5.
        inplace (bool, optional): If True, modify the input DataFrame in place.
            Defaults to False.
        debug (bool, optional): If True, log the number of anomalies removed per
            column. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with anomalous values replaced by NaN.

    Raises:
        TypeError: If ``df.index`` is not a ``pd.DatetimeIndex``.
        ValueError: If ``threshold`` is not a positive number.
        ValueError: If any column in ``columns`` does not exist in ``df``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Dataframe index should be a DatetimeIndex")

    if threshold <= 0:
        raise ValueError("Threshold should be a positive number")

    _df = df if inplace else df.copy()
    total_data = _df.shape[0]

    df_columns = _df.columns.tolist()
    columns = columns or df_columns

    for column in columns:
        if column not in df_columns:
            raise ValueError(
                f"Column `{column}` not in dataframe columns: {df_columns}"
            )

    for column in columns:
        anomalies = detect_anomalies_zscore(_df[column].values, threshold=threshold)
        anomalies_removed = anomalies.sum()
        percentage_removed = anomalies_removed / total_data * 100

        # Replace anomalies with NaN
        _df.loc[anomalies, column] = np.nan

        if debug:
            logger.info(
                f"Column {column}: Removed {anomalies_removed} ({percentage_removed:.2f}%) anomalie(s)"
            )

        # Interpolate
        if interpolate:
            _df[column] = _df[column].interpolate(method="time")
    return _df


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


def load_label_csv(label_features_csv: str) -> pd.Series:
    """Load a label CSV and return a Series indexed by window ID.

    Reads the aligned label CSV produced by ``FeaturesBuilder``, sets the
    ``id`` column as the index, drops the ``datetime`` column if present,
    and returns the ``is_erupted`` column as a Series.

    Args:
        label_features_csv (str): Path to the label CSV file. Must contain
            an ``id`` column and an ``is_erupted`` column.

    Returns:
        pd.Series: Binary eruption labels indexed by window ID.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        >>> labels = load_label_csv("output/features/label_features.csv")
        >>> print(labels.value_counts())
        0    450
        1     50
        Name: is_erupted, dtype: int64
    """
    df = pd.read_csv(label_features_csv)
    if "id" in df.columns:
        df = df.set_index("id")
    if "datetime" in df.columns:
        df = df.drop("datetime", axis=1)
    return df["is_erupted"]


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
