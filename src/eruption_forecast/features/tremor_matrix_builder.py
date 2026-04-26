import os
from typing import Self
from datetime import timedelta

import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.utils.validation import validate_columns
from eruption_forecast.features.constants import (
    ID_COLUMN,
    ERUPTED_COLUMN,
    DATETIME_COLUMN,
    SECONDS_PER_DAY,
)


class TremorMatrixBuilder:
    """Build windowed tremor matrices aligned with label windows.

    Slices tremor time-series into fixed-size windows aligned with label windows,
    then concatenates them into a single tremor matrix suitable for downstream
    tsfresh feature extraction and model training.

    The tremor matrix contains one row per time sample within each window, with
    columns for window ID, datetime, and tremor metrics (RSAM/DSAR across
    frequency bands). Windows are validated to ensure correct sample counts.

    Attributes:
        tremor_df (pd.DataFrame): Tremor DataFrame with DatetimeIndex and tremor
            metric columns (rsam_*, dsar_*).
        label_df (pd.DataFrame): Label DataFrame with DatetimeIndex and columns
            'id' and/or 'is_erupted'.
        output_dir (str): Output directory path for saved CSVs.
        window_size (int): Window size in days.
        tremor_matrix_filename (str): Auto-generated filename for unified tremor matrix.
        matrix_tmp_dir (str): Directory for temporary per-window CSV files.
        overwrite (bool): Whether to overwrite existing output files.
        verbose (bool): Enable verbose logging.
        debug (bool): Enable debug mode.
        start_date (pd.Timestamp): Adjusted start date for matrix building.
        end_date (pd.Timestamp): Adjusted end date for matrix building.
        tremor_start_date_str (str): Tremor data start date in 'YYYY-MM-DD' format.
        tremor_end_date_str (str): Tremor data end date in 'YYYY-MM-DD' format.
        df (pd.DataFrame): Built tremor matrix (set after calling build()).
        csv (str | None): Path to saved tremor matrix CSV (set after calling build()).

    Args:
        tremor_df (pd.DataFrame): Tremor dataframe with DatetimeIndex.
            Index must be pd.DatetimeIndex. Can be built by running CalculateTremor
            or loaded from a calculated tremor CSV.
            Example location: output/tremor/tremor_VG.OJN.00.EHZ_2025-01-01-2025-09-28.csv
        label_df (pd.DataFrame): Label dataframe with DatetimeIndex and columns
            'id' and/or 'is_erupted'. Index must be pd.DatetimeIndex. Can be built
            using LabelBuilder or loaded from label CSV.
            Example location: output/labels/label_2025-01-01_2025-07-24_ws-2_step-6-hours_dtf-2_ie-0.csv
        output_dir (str | None, optional): Output directory path for saved CSVs.
            If None, defaults to ``root_dir/output/features``. Relative paths are
            resolved against ``root_dir`` (or ``os.getcwd()`` when ``root_dir`` is
            None). Absolute paths are used as-is. Defaults to None.
        window_size (int, optional): Window size in days. Defaults to 1.
        root_dir (str | None, optional): Anchor directory for resolving relative
            ``output_dir`` values. Defaults to None (uses ``os.getcwd()``).
        overwrite (bool, optional): Overwrite existing output files.
            Defaults to False.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        debug (bool, optional): Enable debug mode. Defaults to False.

    Raises:
        TypeError: If tremor_df or label_df index is not a DatetimeIndex.
        ValueError: If required label columns ('id') are missing or requested
            tremor columns do not exist.

    Examples:
        >>> # Prepare tremor data (10-minute intervals)
        >>> tremor_df = pd.read_csv("tremor.csv", index_col=0, parse_dates=True)
        >>> label_df = pd.read_csv("labels.csv", index_col=0, parse_dates=True)
        >>> builder = TremorMatrixBuilder(
        ...     tremor_df=tremor_df,
        ...     label_df=label_df,
        ...     output_dir="output/features",
        ...     window_size=1,  # 1-day windows
        ... ).build(
        ...     select_tremor_columns=["rsam_f0", "rsam_f1", "dsar_f0-f1"],
        ... )
        >>> tremor_matrix = builder.df
        >>> print(tremor_matrix.shape)
        (14400, 5)  # 100 windows × 144 samples/day, 5 columns
        >>>
        >>> # Access individual window
        >>> window_1 = tremor_matrix[tremor_matrix['id'] == 1]
        >>> print(window_1.shape)
        (144, 5)  # 1 day at 10-minute intervals
    """

    def __init__(
        self,
        tremor_df: pd.DataFrame,
        label_df: pd.DataFrame,
        output_dir: str | None = None,
        window_size: int = 1,
        root_dir: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the TremorMatrixBuilder with tremor data, labels, and window settings.

        Validates that both DataFrames carry a DatetimeIndex, derives output paths and
        filename, and sets result attributes to empty defaults. Calls validate() and
        create_directories() before returning.

        Args:
            tremor_df (pd.DataFrame): Tremor DataFrame with DatetimeIndex. Must contain
                tremor metric columns (rsam_*, dsar_*).
            label_df (pd.DataFrame): Label DataFrame with DatetimeIndex and at least
                an 'id' column.
            output_dir (str | None, optional): Directory for saved matrix CSVs.
                Defaults to ``root_dir/output/features``. Defaults to None.
            window_size (int, optional): Window size in days. Defaults to 1.
            root_dir (str | None, optional): Anchor directory for relative path resolution.
                Defaults to None (uses os.getcwd()).
            overwrite (bool, optional): Overwrite existing output files. Defaults to False.
            verbose (bool, optional): Enable verbose logging. Defaults to False.
            debug (bool, optional): Enable debug mode. Defaults to False.

        Raises:
            TypeError: If tremor_df or label_df index is not a pd.DatetimeIndex.
            ValueError: If required label columns are missing or tremor columns are absent.
        """
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        if not isinstance(tremor_df.index, pd.DatetimeIndex):
            raise TypeError("tremor_df.index is not a DatetimeIndex")
        if not isinstance(label_df.index, pd.DatetimeIndex):
            raise TypeError("label_df.index is not a DatetimeIndex")
        tremor_df = tremor_df.sort_index()
        label_df = label_df.sort_index()
        output_dir = resolve_output_dir(
            output_dir, root_dir, os.path.join("output", "features")
        )
        matrix_tmp_dir = os.path.join(output_dir, "tmp")
        tremor_start_date_str = tremor_df.index[0].strftime("%Y-%m-%d")
        tremor_end_date_str = tremor_df.index[-1].strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.tremor_df = tremor_df
        self.label_df = label_df
        self.output_dir = output_dir
        self.window_size = window_size
        self.matrix_tmp_dir = matrix_tmp_dir
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self.start_date = label_df.index.min() - timedelta(days=self.window_size)
        self.end_date = label_df.index.max()
        self.tremor_start_date_str = tremor_start_date_str
        self.tremor_end_date_str = tremor_end_date_str

        # ------------------------------------------------------------------
        # Will be set after build() method called
        # ------------------------------------------------------------------
        self.df: pd.DataFrame = pd.DataFrame()
        self.csv: str | None = None
        self.tremor_matrix_filename: str | None = None

        # ------------------------------------------------------------------
        # Validate and create directories
        # ------------------------------------------------------------------
        self.validate()
        self.create_directories()

    def validate(self) -> None:
        """Validate label and tremor DataFrame columns and date ranges.

        Checks that required label columns exist and adjusts start/end dates
        to fit within the available tremor data range. Called automatically
        during __init__.

        Raises:
            ValueError: If 'id' column is missing from the label DataFrame.

        Examples:
            >>> builder = TremorMatrixBuilder(...)
            >>> builder.validate()  # Called automatically during __init__
        """
        label_columns = self.label_df.columns.tolist()

        if ID_COLUMN not in label_columns:
            raise ValueError(
                f"Column '{ID_COLUMN}' not found in Label dataframe. "
                f"Columns available: {label_columns}"
            )

        # Ensuring label date within range of tremor date
        tremor_start_date = self.tremor_df.index.min()
        tremor_end_date = self.tremor_df.index.max()

        if self.start_date < tremor_start_date:
            self.start_date = tremor_start_date
            if self.verbose:
                logger.info(
                    f"start_date: {self.start_date}. Tremor start date: {tremor_start_date}"
                )
                logger.info(f"start_date updated to: {self.start_date}")

        if self.end_date > tremor_end_date:
            self.end_date = tremor_end_date
            if self.verbose:
                logger.info(
                    f"end_date: {self.end_date}. Tremor end date: {tremor_end_date}"
                )
                logger.info(f"end_date updated to: {self.end_date}")

        start_date_str = self.start_date.strftime("%Y-%m-%d")
        end_date_str = self.end_date.strftime("%Y-%m-%d")
        tremor_matrix_filename = f"tremor_matrix_unified_{start_date_str}_{end_date_str}_ws-{self.window_size}.csv"
        self.tremor_matrix_filename = tremor_matrix_filename

    def create_directories(self) -> None:
        """Create required output directories.

        Creates the main output directory and any required subdirectories
        for storing tremor matrix files. Called automatically during initialization.

        Examples:
            >>> builder = TremorMatrixBuilder(...)
            >>> builder.create_directories()  # Called automatically in __init__
        """
        ensure_dir(self.output_dir)

    def save_matrix_per_method(self, tremor_df: pd.DataFrame) -> Self:
        """Save each tremor metric column as a separate CSV file.

        Creates individual CSV files for each tremor method column
        (e.g., rsam_f0, dsar_f0-f1) in a subdirectory called
        'tremor_matrix_per_method' for easier debugging and analysis.

        Args:
            tremor_df (pd.DataFrame): Full tremor matrix containing 'id',
                'datetime', and tremor metric columns (rsam_*, dsar_*).

        Returns:
            Self: The TremorMatrixBuilder instance for method chaining.

        Examples:
            >>> builder = TremorMatrixBuilder(...)
            >>> builder.save_matrix_per_method(tremor_matrix)
            >>> # Creates files in output_dir/tremor_matrix_per_method/:
            >>> #   - tremor_matrix_rsam_f0.csv
            >>> #   - tremor_matrix_rsam_f1.csv
            >>> #   - tremor_matrix_dsar_f0-f1.csv
        """
        tremor_matrix_per_method_dir = os.path.join(
            self.output_dir, "tremor_matrix_per_method"
        )
        ensure_dir(tremor_matrix_per_method_dir)

        # Skip ID and datetime columns
        for column in tremor_df.columns.tolist():
            if column in [ID_COLUMN, DATETIME_COLUMN]:
                continue

            tremor_matrix_method_filename = f"tremor_matrix_{column}.csv"
            tremor_matrix_method_filepath = os.path.join(
                tremor_matrix_per_method_dir, tremor_matrix_method_filename
            )

            # Skip if file exists
            if not self.overwrite and (os.path.isfile(tremor_matrix_method_filepath)):
                continue

            tremor_df_matrix = tremor_df[[ID_COLUMN, DATETIME_COLUMN, column]]
            tremor_df_matrix.to_csv(tremor_matrix_method_filepath, index=False)

            if self.verbose:
                logger.info(
                    f"Tremor matrix {column} is saved to: {tremor_matrix_method_filepath}"
                )

        return self

    def _build_tremor_matrices(
        self,
        tremor_df: pd.DataFrame,
        filtered_label_df: pd.DataFrame,
        tremor_sampling_period: int,
        tremor_columns: list[str],
        save_tremor_matrix_per_id: bool = False,
    ) -> pd.DataFrame:
        """Build tremor matrices by slicing tremor data for each label window.

        Iterates through each label window, extracts the corresponding tremor
        time slice, validates sample count, adds window ID, and concatenates
        all windows into a unified matrix.

        Args:
            tremor_df (pd.DataFrame): Full tremor DataFrame with DatetimeIndex
                and tremor metric columns.
            filtered_label_df (pd.DataFrame): Filtered label DataFrame with
                DatetimeIndex and columns 'id' and/or 'is_erupted'.
            tremor_sampling_period (int): Sampling period in seconds (e.g., 600
                for 10-minute intervals).
            tremor_columns (list[str]): List of tremor column names to include
                in the output matrix.
            save_tremor_matrix_per_id (bool, optional): If True, saves individual
                CSV files for each window ID in the tmp/ directory. **Warning:**
                Generates many files. Defaults to False.

        Returns:
            pd.DataFrame: Unified tremor matrix with columns ['id', 'datetime',
                *tremor_columns]. Each row represents one time sample within a window.

        Raises:
            ValueError: If no valid tremor windows are found within the date range
                (when tremor data doesn't overlap with label windows).

        Examples:
            >>> matrices = builder._build_tremor_matrices(
            ...     tremor_df=tremor_data,
            ...     filtered_label_df=labels,
            ...     tremor_sampling_period=600,
            ...     tremor_columns=["rsam_f0", "rsam_f1"]
            ... )
            >>> print(matrices.shape)
            (14400, 4)  # 100 windows × 144 samples/day, 4 columns
            >>> print(matrices.columns.tolist())
            ['id', 'datetime', 'rsam_f0', 'rsam_f1']
        """
        tremor_matrices: list[pd.DataFrame] = []

        if save_tremor_matrix_per_id:
            ensure_dir(self.matrix_tmp_dir)

        # Check if filtered_label_df have "is_erupted" column
        # Added with None value if not exists.
        # This means the eruption is not happened yet.
        # Use in building tremor matrix for prediction
        if ERUPTED_COLUMN not in filtered_label_df.columns:
            filtered_label_df[ERUPTED_COLUMN] = None

        for (
            datetime_index,
            column_id,
            column_eruption,
        ) in filtered_label_df.itertuples():
            start_datetime = datetime_index - timedelta(days=self.window_size)
            end_datetime = datetime_index - timedelta(milliseconds=1)
            total_window = int(
                self.window_size * SECONDS_PER_DAY / tremor_sampling_period
            )

            tremor_df_sliced = tremor_df.loc[start_datetime:end_datetime]

            if len(tremor_df_sliced) == total_window:
                logger.debug(f"Label id={column_id}: accepted ({total_window} samples)")

                tremor_df_sliced = tremor_df_sliced.sort_index(ascending=True)
                tremor_df_sliced = tremor_df_sliced.reset_index()
                tremor_df_sliced[ID_COLUMN] = column_id

                # Rearrange column to: id, datetime, ...columns
                tremor_df_sliced = tremor_df_sliced[
                    [ID_COLUMN, DATETIME_COLUMN, *tremor_columns]
                ]

                tremor_matrices.append(tremor_df_sliced)

                # BE CAREFUL TO USE THIS FEATURE.
                # THIS WILL GENERATE TREMOR DATA GROUPED BY COLUMN "id" FROM LABEL DATASET
                # IT WILL GENERATED A LOT OF FILES DEPENDS ON THE WINDOW SIZE.
                if save_tremor_matrix_per_id:
                    start_date_str = start_datetime.strftime("%Y-%m-%d--%H-%M-%S")
                    end_date_str = end_datetime.strftime("%Y-%m-%d__%H-%M-%S")
                    suffix_filename = (
                        f"_eruption-{column_eruption}" if column_eruption else ""
                    )
                    matrix_tmp_filename = f"{column_id:05}_{start_date_str}_{end_date_str}{suffix_filename}.csv"
                    feature_tmp_filepath = os.path.join(
                        self.matrix_tmp_dir, matrix_tmp_filename
                    )
                    tremor_df_sliced.to_csv(feature_tmp_filepath, index=False)
                    filtered_label_df.loc[datetime_index, "feature_csv"] = (
                        matrix_tmp_filename
                    )
            else:
                logger.debug(
                    f"Window id={column_id}: skipped (expected {total_window} "
                    f"samples, got {len(tremor_df_sliced)})"
                )

        # Unified tremor matrix as one single dataframe
        if len(tremor_matrices) == 0:
            error_message = (
                f"Tremor data between date {self.start_date.strftime('%Y-%m-%d')} "
                f"to {self.end_date.strftime('%Y-%m-%d')} are not found. "
                f"Tremor data available from {self.tremor_start_date_str} to "
                f"{self.tremor_end_date_str}."
            )
            logger.error(error_message)
            raise ValueError(error_message)

        unified_tremor_matrix = pd.concat(tremor_matrices, ignore_index=False)

        return unified_tremor_matrix

    def build(
        self,
        select_tremor_columns: list[str] | None = None,
        tremor_matrix_filename: str | None = None,
        save_tremor_matrix_per_method: bool = True,
        save_tremor_matrix_per_id: bool = False,
    ) -> Self:
        """Build tremor matrix by aligning tremor data with label windows.

        Constructs a unified tremor matrix where each row represents one time
        sample, and rows are grouped by window ID from the label dataset.
        The matrix contains 'id', 'datetime', and tremor metric columns
        (RSAM, DSAR across frequency bands).

        This method performs the following steps:
        1. Iterates through each label window
        2. Slices tremor data for the window's date range
        3. Validates the sample count matches window_size expectations
        4. Adds the window 'id' column
        5. Concatenates all windows into a unified DataFrame
        6. Saves the result to CSV

        Args:
            select_tremor_columns (list[str] | None, optional): Subset of tremor
                column names to include (e.g., ["rsam_f0", "dsar_f0-f1"]).
                If None, all tremor columns are used. Defaults to None.
            tremor_matrix_filename (str | None, optional): Override the default
                auto-generated filename. If None, uses the auto-generated filename
                format: tremor_matrix_unified_<start>_<end>_ws-<size>.csv.
                Defaults to None.
            save_tremor_matrix_per_method (bool, optional): If True, saves
                individual CSVs for each tremor column method in the subdirectory
                'tremor_matrix_per_method/'. Useful for debugging individual metrics.
                Defaults to True.
            save_tremor_matrix_per_id (bool, optional): **WARNING: Generates
                many files!** If True, saves individual CSV files for each
                window ID in the tmp/ directory for debugging purposes.
                Defaults to False.

        Returns:
            Self: TremorMatrixBuilder instance for method chaining.
                Sets self.df and self.csv attributes.

        Raises:
            ValueError: If no valid tremor data is found for any label windows
                within the specified date range (when tremor and label date
                ranges don't overlap).
            ValueError: If select_tremor_columns contains column names that
                don't exist in tremor_df.

        Examples:
            >>> builder = TremorMatrixBuilder(
            ...     tremor_df=tremor_data,
            ...     label_df=labels,
            ...     output_dir="output/features",
            ...     window_size=1
            ... )
            >>> builder.build(
            ...     select_tremor_columns=["rsam_f0", "rsam_f1"],
            ...     save_tremor_matrix_per_method=True
            ... )
            >>> print(builder.df.shape)
            (14400, 4)  # 100 windows × 144 samples/window
            >>> print(builder.csv)
            "output/features/tremor_matrix_unified_2025-01-01_2025-09-28_ws-1.csv"
            >>>
            >>> # Method chaining example
            >>> matrix = TremorMatrixBuilder(tremor_df, label_df) \
            ...     .build(select_tremor_columns=["rsam_f0"]) \
            ...     .df
        """
        verbose = self.verbose

        # Build filename
        # Example filename: tremor_matrix_unified_2025-01-01_2025-09-28_ws-2.csv
        tremor_matrix_filename = tremor_matrix_filename or self.tremor_matrix_filename
        tremor_matrix_csv = os.path.join(self.output_dir, tremor_matrix_filename)

        # Skip if exists
        if not self.overwrite and (os.path.isfile(tremor_matrix_csv)):
            if verbose:
                logger.info(f"Tremor matrix already exists: {tremor_matrix_csv}")
            df: pd.DataFrame = pd.read_csv(tremor_matrix_csv)
            columns = df.columns.tolist()

            # Ensure select_tremor_columns exists in tremor matrix column
            if select_tremor_columns is None or all(
                item in columns for item in select_tremor_columns
            ):
                self.df = df
                self.csv = tremor_matrix_csv
                return self

            if verbose:
                logger.warning("Cannot skip, because there is column(s) different.")

        if verbose:
            logger.info("Create tremor matrix which grouped by label ID.")

        # TODO: Check sampling period consistency. Warn if not consistent
        # Both dataframes (label_df and tremor_df) index are using pd.DatetimeIndex
        tremor_df = self.tremor_df

        # Apply selected columns
        if select_tremor_columns is not None:
            validate_columns(tremor_df, select_tremor_columns)
            tremor_df = tremor_df[select_tremor_columns]

        filtered_label_df = self.label_df.loc[self.start_date : self.end_date]
        tremor_sampling_period = (tremor_df.index[1] - tremor_df.index[0]).seconds
        tremor_columns = tremor_df.columns.tolist()

        # Save tremor data grouped by label ID
        unified_tremor_matrix = self._build_tremor_matrices(
            tremor_df=tremor_df,
            filtered_label_df=filtered_label_df,
            tremor_sampling_period=tremor_sampling_period,
            tremor_columns=tremor_columns,
            save_tremor_matrix_per_id=save_tremor_matrix_per_id,
        )

        # Save all features and labels for training
        unified_tremor_matrix.to_csv(tremor_matrix_csv, index=False)

        # Save features per method/columns
        if save_tremor_matrix_per_method:
            self.save_matrix_per_method(unified_tremor_matrix)

        self.df = unified_tremor_matrix
        self.csv = tremor_matrix_csv

        logger.info(f"Unified tremor matrix saved to: {tremor_matrix_csv}")

        # Clear memory
        del unified_tremor_matrix

        return self
