import os
import glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import detect_anomalies_zscore
from eruption_forecast.utils.date_utils import to_datetime_index
from eruption_forecast.utils.formatting import shorten_feature_name


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


def load_select_features(
    value: str | list[str], number_of_features: int = 20
) -> list[str]:
    """Resolve a ``select_features`` value into a list of tsfresh feature names.

    Accepts either a CSV path written by :func:`concat_significant_features`
    (``top_{N}_features.csv``, ``top_features.csv``, or
    ``significant_features.csv``), in which case the ``features`` index
    column is read, or an explicit list of tsfresh feature names. Empty
    lists and blank entries raise.

    Args:
        value (str | list[str]): A CSV path or a list of fully-qualified
            tsfresh feature names (e.g. ``"rsam_f2__mean"``,
            ``"rsam_f1__autocorrelation__lag_1"``).
        number_of_features (int, optional): Cap the returned list to the top
            ``number_of_features`` entries. The CSV produced by
            :func:`concat_significant_features` is already sorted descending
            by frequency, so truncation preserves the highest-ranked names;
            an in-memory list is assumed to be in priority order. Pass ``0``
            or a negative value to disable truncation. Defaults to ``20``.

    Returns:
        list[str]: Cleaned list of feature names, capped to the top
            ``number_of_features`` entries.

    Raises:
        FileNotFoundError: If ``value`` is a path that does not exist.
        ValueError: If ``value`` resolves to an empty list, contains blank
            entries, or is of an unsupported type.

    Examples:
        >>> load_select_features("output/.../top_20_features.csv")
        ['rsam_f2__mean', 'rsam_f1__autocorrelation__lag_1', ...]
        >>> load_select_features(["rsam_f2__mean", "entropy__variance"])
        ['rsam_f2__mean', 'entropy__variance']
        >>> load_select_features("output/.../top_50_features.csv", number_of_features=10)
        # returns at most 10 names from the top of the ranked CSV
    """
    if isinstance(value, str):
        if not os.path.isfile(value):
            raise FileNotFoundError(f"select_features CSV not found: {value}")
        names = pd.read_csv(value, index_col=0).index.astype(str).tolist()
    else:
        names = [str(name) for name in value]

    cleaned = [name.strip() for name in names if name and name.strip()]
    if not cleaned:
        raise ValueError("select_features resolved to an empty list.")
    if number_of_features > 0:
        cleaned = cleaned[:number_of_features]
    return cleaned


def concat_features(
    paths: list[str],
    filepath: str,
    frames: list[pd.DataFrame] | None = None,
) -> tuple[str, pd.DataFrame]:
    """Concatenate per-column feature Parquets (and optional in-memory frames) and save.

    Reads each path in ``paths`` as Parquet, optionally combines them with
    already-loaded DataFrames in ``frames``, concatenates everything column-wise
    (``axis=1``), and saves the merged DataFrame to ``filepath`` as
    Snappy-compressed Parquet. Used to merge per-column tsfresh feature
    extractions; ``frames`` lets callers skip a disk round-trip for columns
    whose DataFrames are already in memory.

    Args:
        paths (list[str]): Paths to per-column feature Parquet files to read
            and concatenate.
        filepath (str): Output filepath for the merged Parquet (should end in
            ``.parquet``).
        frames (list[pd.DataFrame] | None, optional): Pre-loaded feature
            DataFrames to include in the concatenation alongside the Parquet
            paths. Defaults to ``None`` (Parquet-only behaviour).

    Returns:
        tuple[str, pd.DataFrame]: Tuple containing:
            - filepath (str): Path where the merged Parquet was saved.
            - df (pd.DataFrame): Concatenated DataFrame.

    Raises:
        ValueError: If the combined input count (``len(paths) + len(frames)``)
            is fewer than 2, or if all inputs concatenate to an empty frame.

    Examples:
        >>> parquet_files = ["features_f0.parquet", "features_f1.parquet"]
        >>> path, df = concat_features(parquet_files, "all_features.parquet")
        >>> print(df.shape)
        >>> # mix in already-loaded frames
        >>> path, df = concat_features(
        ...     ["features_f0.parquet"],
        ...     "all_features.parquet",
        ...     frames=[fresh_df_f1, fresh_df_f2],
        ... )
    """
    frames = frames or []
    total_inputs = len(paths) + len(frames)
    if total_inputs <= 1:
        raise ValueError(
            f"Requires at least 2 inputs (paths + frames). Got {total_inputs}."
        )

    pieces: list[pd.DataFrame] = [pd.read_parquet(path) for path in paths]
    pieces.extend(frames)

    df = pd.concat(pieces, axis=1)

    if df.empty:
        raise ValueError("There is no data in the input files.")

    df.to_parquet(filepath, engine="pyarrow", compression="snappy", index=True)

    return filepath, df


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


def concat_significant_features(
    features_csvs: list[str],
    features_dir: str,
    number_of_features: int | None = None,
) -> pd.DataFrame:
    """Concatenate per-seed significant-feature CSVs and save a ranked summary.

    Reads all CSVs in ``features_csvs``, concatenates them row-wise, and
    writes the raw combined data to ``{features_dir}/significant_features.csv``.
    When ``number_of_features`` is provided, also aggregates by feature name,
    counts occurrences across seeds (``score``), computes mean score
    (``mean_score``), sorts descending by frequency and ascending by mean
    p-value, and writes two additional CSVs:

    - ``{features_dir}/top_features.csv`` — the full ranked list (all
      features, sorted).
    - ``{features_dir}/top_{number_of_features}_features.csv`` — the top-N
      subset of the same ranking, used downstream by
      :func:`load_select_features`.

    Args:
        features_csvs (list[str]): Paths to per-seed significant-feature CSV
            files, each expected to contain a ``features`` column and a
            ``score`` column.
        features_dir (str): Directory where output CSVs are written.
        number_of_features (int | None, optional): If set, produce the
            ``top_features.csv`` full ranking plus a
            ``top_{number_of_features}_features.csv`` capped to the top-N
            features by occurrence count. If ``None`` or ``<= 0``, only the
            raw ``significant_features.csv`` is written. Defaults to ``None``.

    Returns:
        pd.DataFrame: The combined DataFrame (all seeds concatenated), or the
            full ranked DataFrame when ``number_of_features`` is specified.

    Raises:
        ValueError: If ``combined_features_df`` is empty after concatenation.

    Examples:
        >>> csvs = ["seed_0_features.csv", "seed_1_features.csv"]
        >>> df = concat_significant_features(csvs, "output/features", number_of_features=20)
        >>> df.index.name
        'features'
    """
    combined_features_df = pd.concat(
        [pd.read_csv(file) for file in features_csvs],
        ignore_index=True,
    )

    if combined_features_df.empty:
        raise ValueError("No data found inside csv files.")

    combined_features_df.to_csv(
        os.path.join(features_dir, "significant_features.csv"), index=False
    )

    if number_of_features is not None and number_of_features > 0:
        combined_features_df = (
            combined_features_df.groupby(by="features")
            .agg(score=("score", "count"), mean_score=("score", "mean"))
            .sort_values(
                by=["score", "mean_score"],
                ascending=[False, True],
            )
        )
        combined_features_df.index.name = "features"
        combined_features_df.to_csv(
            os.path.join(features_dir, "top_features.csv"),
            index=True,
        )

        top_n_features_df = combined_features_df.head(number_of_features)
        top_n_features_df.to_csv(
            os.path.join(features_dir, f"top_{number_of_features}_features.csv"),
            index=True,
        )

    return combined_features_df


def find_common_features(
    top_features_csv: list[str], output_dir: str | None = None
) -> pd.DataFrame:
    """Return features that appear in every ``top_N_features.csv``.

    Loads each CSV produced by the scenario training stage and intersects
    their feature indices, so the result contains only features that every
    scenario agreed on. The ``score`` and ``mean_score`` columns are summed
    across the input CSVs so the output keeps the same shape as a source
    ``top_N_features.csv`` and can be sorted the same way (descending by
    ``score``, ascending by ``mean_score``).

    Args:
        top_features_csv (list[str]): Paths to ``top_{N}_features.csv`` files
            (one per scenario). Each file must have a ``features`` index
            column and ``score`` / ``mean_score`` columns.
        output_dir (str | None, optional): Directory where the resulting
            ``common_top_features.csv`` is written. When ``None``, falls back
            to the current working directory (``os.getcwd()``). Defaults to
            ``None``.

    Returns:
        pd.DataFrame: DataFrame indexed by the common feature names with
        ``score`` and ``mean_score`` columns summed across the input CSVs.
        Also written to ``{output_dir}/common_top_features.csv``.

    Raises:
        ValueError: If ``top_features_csv`` is empty or the intersection is
            empty.
        FileNotFoundError: If any of the paths does not exist (raised by
            :func:`load_select_features`).

    Examples:
        >>> csvs = [
        ...     "output/.../scenarios/scenario-1/training/features/stratified-shuffle-split/top_20_features.csv",
        ...     "output/.../scenarios/scenario-2/training/features/stratified-shuffle-split/top_20_features.csv",
        ... ]
        >>> df = find_common_features(csvs)
        >>> df.index.tolist()[:3]
        ['rsam_f2__mean', 'entropy__variance', 'dsar_f3-f4__median']
        >>> df = find_common_features(csvs, output_dir="output/.../scenarios")
    """
    if not top_features_csv:
        raise ValueError("top_features_csv is empty.")

    frames: list[pd.DataFrame] = []
    for path in top_features_csv:
        load_select_features(path, number_of_features=0)
        frames.append(pd.read_csv(path, index_col=0))

    common = sorted(set.intersection(*(set(df.index) for df in frames)))
    if not common:
        raise ValueError("No features are common to all CSVs.")

    aligned = [
        df.loc[df.index.intersection(common), ["score", "mean_score"]] for df in frames
    ]
    combined = pd.concat(aligned).groupby(level=0).sum()
    combined = combined.sort_values(by=["score", "mean_score"], ascending=[False, True])

    out_path = os.path.join(output_dir or os.getcwd(), "common_top_features.csv")
    combined.to_csv(out_path, index=True)
    return combined


def plot_common_features_heatmap(
    top_features_csv: dict[str, str],
    output_path: str | None = None,
    cmap: str = "viridis",
) -> plt.Axes:
    """Heatmap of per-scenario ``score`` for the common-feature subset.

    Computes the cross-scenario intersection via :func:`find_common_features`,
    re-reads each input CSV to recover its per-scenario ``score`` column, and
    renders a heatmap with common features on the rows and scenarios on the
    columns. Rows are ordered by the ranking produced by
    :func:`find_common_features` (most universally strong on top).

    Args:
        top_features_csv (dict[str, str]): Mapping of column label → path to
            a ``top_{N}_features.csv`` file (one entry per scenario). The
            dict keys are used directly as the heatmap's x-axis labels, so
            insertion order controls left-to-right column order.
        output_path (str | None, optional): Where to save the PNG. When
            ``None``, writes to ``{cwd}/common_top_features_heatmap.png``.
            Defaults to ``None``.
        cmap (str, optional): Matplotlib/seaborn colormap name. Defaults to
            ``"viridis"``.

    Returns:
        plt.Axes: The heatmap axes, so the caller can further annotate it.
    """
    common_df = find_common_features(list(top_features_csv.values()))
    common_features = list(common_df.index)
    labels = list(top_features_csv.keys())

    matrix = pd.DataFrame(index=common_features, columns=labels, dtype=float)
    for label, path in top_features_csv.items():
        per_scenario = pd.read_csv(path, index_col=0)
        matrix[label] = per_scenario["score"].reindex(common_features)

    matrix.index = [shorten_feature_name(name) for name in matrix.index]

    fig, ax = plt.subplots(
        figsize=(
            max(6.0, 1.6 * len(labels) + 3),
            max(4.0, 0.6 * len(common_features) + 2),
        )
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".0f",
        cmap=cmap,
        cbar_kws={"label": "frequency", "shrink": 0.8, "pad": 0.02},
        linewidths=0.5,
        square=True,
        ax=ax,
    )

    ax.tick_params(axis="x", rotation=90)
    ax.set_title(f"{len(common_features)} features × {len(labels)} scenarios")
    fig.tight_layout()

    out = output_path or os.path.join(os.getcwd(), "common_top_features_heatmap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return ax


def plot_common_features_correlation(
    top_features_csv: dict[str, str],
    output_path: str | None = None,
    cmap: str = "RdBu_r",
) -> plt.Axes:
    """Pearson correlation heatmap across scenarios for the common-feature subset.

    For each entry in ``top_features_csv``, globs the sibling
    ``features-matrix_*.parquet`` (written by ``TrainingModel.extract_features``)
    next to the top-N CSV, slices it to the cross-scenario common-feature
    subset returned by :func:`find_common_features`, and stacks the rows
    across all scenarios into a single feature matrix. Pairwise Pearson
    correlations are then rendered as a square heatmap with a diverging
    colormap centred on 0.

    Args:
        top_features_csv (dict[str, str]): Mapping of column label → path to
            a ``top_{N}_features.csv`` file (one entry per scenario). The
            label is informational; correlations are computed on the stacked
            features matrices found next to each path.
        output_path (str | None, optional): Where to save the PNG. When
            ``None``, writes to ``{cwd}/common_features_correlation.png``.
            Defaults to ``None``.
        cmap (str, optional): Diverging colormap name. Defaults to
            ``"RdBu_r"``.

    Returns:
        plt.Axes: The heatmap axes.

    Raises:
        ValueError: If the sibling ``features-matrix_*.parquet`` next to any
            input path is missing or ambiguous (0 or >1 matches).
        KeyError: If any common feature is absent from a scenario's matrix.
    """
    common_df = find_common_features(list(top_features_csv.values()))
    common_features = list(common_df.index)

    blocks: list[pd.DataFrame] = []
    for path in top_features_csv.values():
        parent = os.path.dirname(path)
        matches = glob.glob(os.path.join(parent, "features-matrix_*.parquet"))
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly 1 features-matrix_*.parquet next to {path}, "
                f"found {len(matches)}."
            )
        matrix = pd.read_parquet(matches[0])
        missing = [f for f in common_features if f not in matrix.columns]
        if missing:
            shown = ", ".join(missing[:3]) + ("..." if len(missing) > 3 else "")
            raise KeyError(
                f"{len(missing)} common feature(s) missing from {matches[0]}: {shown}"
            )
        blocks.append(matrix[common_features])

    stacked = pd.concat(blocks, axis=0, ignore_index=True)
    corr = stacked.corr()
    short_labels = [shorten_feature_name(n) for n in corr.index]
    corr.index = short_labels
    corr.columns = short_labels

    n_features = len(common_features)
    side = max(6.0, 0.6 * n_features + 3)
    fig, ax = plt.subplots(figsize=(side, side))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        cbar_kws={"shrink": 0.8, "pad": 0.02},
        linewidths=0.5,
        square=True,
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=90)
    ax.set_title(
        f"Correlation across {len(top_features_csv)} scenarios "
        f"({n_features} common features)"
    )
    fig.tight_layout()

    out = output_path or os.path.join(os.getcwd(), "common_features_correlation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return ax
