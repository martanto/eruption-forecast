import os

import numpy as np
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import detect_anomalies_zscore
from eruption_forecast.utils.date_utils import to_datetime_index


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

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame(
        ...     {"rsam_f0": [1.0, 1e9, 1.1, 0.9]},
        ...     index=pd.date_range("2025-01-01", periods=4, freq="10min"),
        ... )
        >>> cleaned = remove_anomalies(df, threshold=3.5, interpolate=True)
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
        anomalies = detect_anomalies_zscore(_df[column].to_numpy(), threshold=threshold)
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


def load_datetime_indexed(label_csv: str, features_path: str) -> pd.DataFrame:
    """Load a label CSV and an ``id``-indexed data file, then re-index by datetime.

    Thin file-path wrapper around :func:`to_datetime_index`. Loads the label
    CSV (DatetimeIndex + ``id`` column) and the features/probability file
    (``id``-indexed), then delegates the merge and ``DatetimeIndex``
    replacement to :func:`to_datetime_index`. The features file format is
    dispatched by extension: ``.parquet`` via ``pd.read_parquet`` and ``.csv``
    via ``pd.read_csv`` with the first column as index.

    Args:
        label_csv (str): Path to the aligned label CSV produced by
            ``FeaturesBuilder`` (e.g. ``features-label_{start}_{end}.csv``).
            Must contain a datetime index and an ``id`` column.
        features_path (str): Path to the ``id``-indexed features matrix or
            probability matrix. Supported suffixes: ``.parquet``, ``.csv``.

    Returns:
        pd.DataFrame: Copy of the features/probability frame with a
        ``DatetimeIndex`` derived from ``label_csv``. The ``id`` and
        ``datetime`` columns are absent from the result.

    Raises:
        ValueError: If ``features_path`` has a suffix other than ``.parquet``
            or ``.csv``.
        ValueError: Propagated from :func:`to_datetime_index` when the loaded
            frames cannot be aligned (length mismatch, missing ``id`` column,
            missing ``datetime`` column).

    Examples:
        >>> # Parquet features matrix
        >>> df = load_datetime_indexed(
        ...     label_csv="output/.../features-label_2020-01-01_2020-12-31.csv",
        ...     features_path="output/.../features-matrix_2020-01-01_2020-12-31.parquet",
        ... )
        >>> isinstance(df.index, pd.DatetimeIndex)
        True
        >>> # CSV probability matrix
        >>> df = load_datetime_indexed(
        ...     label_csv="output/.../prediction/labels/label-features_2020-07.csv",
        ...     features_path="output/.../predictions/y_proba.csv",
        ... )
    """
    labels = pd.read_csv(label_csv, index_col=0, parse_dates=True)

    suffix = os.path.splitext(features_path)[1].lower()
    if suffix == ".parquet":
        features = pd.read_parquet(features_path)
    elif suffix == ".csv":
        features = pd.read_csv(features_path, index_col=0)
    else:
        raise ValueError(
            f"Unsupported features_path suffix '{suffix}'. "
            f"Expected '.parquet' or '.csv'. Got: {features_path}"
        )

    return to_datetime_index(labels, features)


def load_features_matrix(label_csv: str, features_path: str) -> pd.DataFrame:
    """Load a features matrix and return it with a ``DatetimeIndex``.

    Domain-named alias for :func:`load_datetime_indexed`. Forwards both
    arguments unchanged so the parquet / csv dispatch and datetime
    attachment from a sibling ``features-label_*.csv`` happen exactly
    once. Prefer this name at call sites that specifically load a
    features matrix; use :func:`load_datetime_indexed` when the payload
    is a probability matrix or any other ``id``-indexed frame.

    Args:
        label_csv (str): Path to the sibling ``features-label_*.csv``
            (``DatetimeIndex`` + ``id`` column) used to attach datetimes
            to the ``id``-indexed features matrix.
        features_path (str): Path to the ``id``-indexed features matrix.
            Supported suffixes: ``.parquet``, ``.csv``.

    Returns:
        pd.DataFrame: Features frame with a ``DatetimeIndex`` derived from
        ``label_csv``. The ``id`` and ``datetime`` columns are absent from
        the result.

    Raises:
        ValueError: Propagated from :func:`load_datetime_indexed` on an
            unsupported path suffix or from
            :func:`~eruption_forecast.utils.date_utils.to_datetime_index`
            when the frames cannot be aligned.

    Examples:
        >>> df = load_features_matrix(
        ...     label_csv="output/.../training/features/stratified-shuffle-split/features-label_2025-01-03_2025-03-31.csv",
        ...     features_path="output/.../training/features/stratified-shuffle-split/features-matrix_2025-01-03_2025-03-31.parquet",
        ... )
        >>> isinstance(df.index, pd.DatetimeIndex)
        True
    """
    return load_datetime_indexed(label_csv=label_csv, features_path=features_path)


def get_envelope_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling min/max envelopes for per-classifier probability and prediction columns.

    Adds eight new columns to ``df`` in place (no copy is made):

    - ``consensus_probability_max`` / ``consensus_probability_min``: row-wise
      max/min across all ``*_probability`` columns (excluding ``consensus_*``).
    - ``consensus_probability_max_envelope`` / ``consensus_probability_min_envelope``:
      centered rolling max/min (window=6) of the above.
    - ``consensus_prediction_max`` / ``consensus_prediction_min``: row-wise
      max/min across all ``*_prediction`` columns (excluding ``consensus_*``).
    - ``consensus_prediction_max_envelope`` / ``consensus_prediction_min_envelope``:
      centered rolling max/min (window=6) of the above.

    Args:
        df (pd.DataFrame): Consensus forecast DataFrame containing per-classifier
            columns ending in ``_probability`` and ``_prediction``. Modified in place.

    Returns:
        pd.DataFrame: The same DataFrame with the eight envelope columns added.

    Raises:
        ValueError: If no columns ending with ``_probability`` (excluding
            ``consensus_*``) are found in ``df``.
        ValueError: If no columns ending with ``_prediction`` (excluding
            ``consensus_*``) are found in ``df``.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "rf_probability": [0.2, 0.8, 0.5],
        ...     "rf_prediction": [0, 1, 0],
        ... })
        >>> result = get_envelope_values(df)
        >>> list(result.columns)  # doctest: +ELLIPSIS
        ['rf_probability', 'rf_prediction', ..., 'consensus_prediction_max_envelope']
    """
    prob_cols = [
        col
        for col in df.columns
        if (col.endswith("_probability") and not col.startswith("consensus"))
    ]

    pred_cols = [
        col
        for col in df.columns
        if (col.endswith("_prediction") and not col.startswith("consensus"))
    ]

    # Ensure model ``_probability`` and ``_prediction`` column exists
    if not prob_cols:
        raise ValueError(
            "No probability columns found. Expected columns ending with '_probability' "
            "(excluding 'consensus_*')."
        )

    if not pred_cols:
        raise ValueError(
            "No prediction columns found. Expected columns ending with '_prediction' "
            "(excluding 'consensus_*')."
        )

    df["consensus_probability_max"] = df[prob_cols].max(axis=1)
    df["consensus_probability_min"] = df[prob_cols].min(axis=1)
    df["consensus_probability_min_envelope"] = (
        df["consensus_probability_min"]
        .rolling(window=6, center=True, min_periods=1)
        .min()
    )
    df["consensus_probability_max_envelope"] = (
        df["consensus_probability_max"]
        .rolling(window=6, center=True, min_periods=1)
        .max()
    )

    df["consensus_prediction_max"] = df[pred_cols].max(axis=1)
    df["consensus_prediction_min"] = df[pred_cols].min(axis=1)
    df["consensus_prediction_min_envelope"] = (
        df["consensus_prediction_min"]
        .rolling(window=6, center=True, min_periods=1)
        .min()
    )
    df["consensus_prediction_max_envelope"] = (
        df["consensus_prediction_max"]
        .rolling(window=6, center=True, min_periods=1)
        .max()
    )

    return df
