import os
import warnings

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
    """Load a label CSV and return the ``is_erupted`` Series with its native axis.

    Reads the aligned label CSV produced by ``FeaturesBuilder``. Post the
    features-matrix DatetimeIndex migration, callers align labels with
    features via a shared ``DatetimeIndex`` — so this helper now preserves
    the CSV's datetime axis when present and only falls back to the legacy
    ``id``-indexed shape when the CSV has no parseable datetime column
    (e.g. an older artefact hand-authored before the migration).

    Args:
        label_features_csv (str): Path to the label CSV file. Must contain
            an ``is_erupted`` column. The datetime column (typically the
            first column, unnamed on read via ``index_col=0``) is parsed
            as the ``DatetimeIndex`` when present.

    Returns:
        pd.Series: Binary eruption labels. Indexed by ``pd.DatetimeIndex``
        when the CSV carries a parseable datetime column; otherwise
        indexed by window ``id`` for backwards compatibility.

    Raises:
        FileNotFoundError: If the file does not exist.

    Examples:
        >>> labels = load_label_csv("output/features/features-label_...csv")
        >>> isinstance(labels.index, pd.DatetimeIndex)
        True
        >>> print(labels.value_counts())
        0    450
        1     50
        Name: is_erupted, dtype: int64
    """
    df = pd.read_csv(label_features_csv, index_col=0, parse_dates=True)

    # New shape: DatetimeIndex + ``id`` column + ``is_erupted`` column.
    # Preserve the DatetimeIndex so callers can align with the
    # DatetimeIndex-first features matrix on the shared temporal axis.
    if isinstance(df.index, pd.DatetimeIndex):
        return df["is_erupted"]

    # Legacy fallback: no parseable datetime column at index_col=0. Reload
    # unindexed and reproduce the historical ``id``-indexed behaviour.
    df = pd.read_csv(label_features_csv)
    if "id" in df.columns:
        df = df.set_index("id")
    if "datetime" in df.columns:
        df = df.drop("datetime", axis=1)
    return df["is_erupted"]


def load_datetime_indexed(label_csv: str, features_path: str) -> pd.DataFrame:
    """Load a features / probability frame and ensure a ``DatetimeIndex``.

    .. deprecated::
        The features-matrix DatetimeIndex migration made the sibling
        ``features-label_*.csv`` join redundant — features and probability
        parquets produced by :class:`FeaturesBuilder` / :class:`SeedEnsemble`
        now carry a ``DatetimeIndex`` directly. This helper stays for one
        release as a compatibility shim: on-disk artefacts that already
        carry a ``DatetimeIndex`` are returned as-is (``label_csv``
        unused); legacy integer-``id``-indexed artefacts still fall back
        to the CSV-join path via
        :func:`~eruption_forecast.utils.date_utils.to_datetime_index`.

    Args:
        label_csv (str): Path to the sibling ``features-label_*.csv``
            (used only when ``features_path`` is still integer-``id``-indexed).
        features_path (str): Path to the features / probability frame.
            Supported suffixes: ``.parquet``, ``.csv``.

    Returns:
        pd.DataFrame: Frame with a ``DatetimeIndex``. For DatetimeIndex-first
        parquets this is the loaded frame itself; for legacy integer-``id``-
        indexed frames it is the merge output.

    Raises:
        ValueError: If ``features_path`` has a suffix other than ``.parquet``
            or ``.csv``.
        ValueError: Propagated from :func:`to_datetime_index` when a legacy
            frame cannot be aligned.

    Examples:
        >>> # New-format parquet — ``label_csv`` unused:
        >>> df = load_datetime_indexed(
        ...     label_csv="unused-post-migration.csv",
        ...     features_path="output/.../features-matrix-dt_2025-01-03_2025-03-31.parquet",
        ... )
        >>> isinstance(df.index, pd.DatetimeIndex)
        True
    """
    warnings.warn(
        "load_datetime_indexed is deprecated: FeaturesBuilder and SeedEnsemble "
        "now write DatetimeIndex-first parquets directly, so the CSV-join is "
        "no longer required. Read the parquet with pd.read_parquet(...) instead. "
        "This shim will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )

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

    # DatetimeIndex-first artefact — the ``label_csv`` join is redundant.
    if isinstance(features.index, pd.DatetimeIndex):
        return features

    labels = pd.read_csv(label_csv, index_col=0, parse_dates=True)
    return to_datetime_index(labels, features)


def load_features_matrix(label_csv: str, features_path: str) -> pd.DataFrame:
    """Load a features matrix and ensure a ``DatetimeIndex``.

    .. deprecated::
        Domain-named alias for :func:`load_datetime_indexed`. Same
        semantics — see that function's deprecation notice. Prefer
        ``pd.read_parquet(features_path)`` for DatetimeIndex-first
        parquets produced by :class:`FeaturesBuilder` after the
        features-matrix DatetimeIndex migration.

    Args:
        label_csv (str): Path to the sibling ``features-label_*.csv``
            (used only when the features matrix is still integer-``id``-indexed).
        features_path (str): Path to the features matrix. Supported
            suffixes: ``.parquet``, ``.csv``.

    Returns:
        pd.DataFrame: Features frame with a ``DatetimeIndex``.

    Raises:
        ValueError: Propagated from :func:`load_datetime_indexed`.

    Examples:
        >>> df = load_features_matrix(
        ...     label_csv="unused-post-migration.csv",
        ...     features_path="output/.../training/features/stratified-shuffle-split/features-matrix-dt_2025-01-03_2025-03-31.parquet",
        ... )
        >>> isinstance(df.index, pd.DatetimeIndex)
        True
    """
    warnings.warn(
        "load_features_matrix is deprecated: FeaturesBuilder now writes "
        "DatetimeIndex-first parquets directly. Read the parquet with "
        "pd.read_parquet(...) instead. This shim will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Delegate to the shim to keep the fallback behaviour in one place — but
    # suppress the inner warning so callers only see one message per call.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
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
