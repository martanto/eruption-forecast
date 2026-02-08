# Standard library imports
import os
from datetime import timedelta
from typing import Self

# Third party imports
import pandas as pd

# Project imports
from eruption_forecast.features.constants import (
    DATETIME_COLUMN,
    ID_COLUMN,
    REQUIRED_LABEL_COLUMNS,
    SECONDS_PER_DAY,
)
from eruption_forecast.logger import logger
from eruption_forecast.utils import validate_columns


class TremorMatrixBuilder:
    """Builds tremor matrices and extract features.

    Slices tremor time-series into fixed-size windows aligned with label
    windows, then concatenates them into a single tremor matrix suitable
    for downstream model training with tsfresh.

    The feature matrix contains one row per time sample within each window,
    with columns for window ID, datetime, and tremor metrics (RSAM/DSAR).

    Args:
        tremor_df (pd.DataFrame): Tremor dataframe with DatetimeIndex.
            Index type of df_tremor is pd.DatetimeIndex.
            Can be build by running CalculateTremor or laad it from calculated tremor CSV.
            Example calculated tremor CSV location:
                output/tremor/tremor_VG.OJN.00.EHZ_2025-01-01-2025-09-28.csv
        label_df (pd.DataFrame): Label dataframe with DatetimeIndex and
            columns 'id' and 'is_erupted'. Index type of df_label is pd.DatetimeIndex.
            Can be build using LabelBuilder or load from label CSV.
            Example label CSV location:
                output/labels/label_2025-01-01_2025-07-24_ws-2_step-6-hours_dtf-2.csv
        output_dir (str): Output directory path for saved CSVs.
        window_size (int): Window size in days.
        overwrite (bool, optional): Overwrite existing output files.
            Defaults to False.
        verbose (bool, optional): Verbose logging. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.

    Raises:
        TypeError: If tremor_df or label_df index is not a DatetimeIndex.
        ValueError: If required label columns are missing or requested
            tremor columns do not exist.

    Example:
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
    """

    def __init__(
        self,
        tremor_df: pd.DataFrame,
        label_df: pd.DataFrame,
        output_dir: str,
        window_size: int,
        overwrite: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        # =========================
        # Set DEFAULT parameter
        # =========================
        if not isinstance(tremor_df.index, pd.DatetimeIndex):
            raise TypeError("tremor_df.index is not a DatetimeIndex")
        if not isinstance(label_df.index, pd.DatetimeIndex):
            raise TypeError("label_df.index is not a DatetimeIndex")
        tremor_df = tremor_df.sort_index()
        label_df.sort_index(inplace=True)
        matrix_tmp_dir = os.path.join(output_dir, "tmp")
        tremor_start_date_str = tremor_df.index[0].strftime("%Y-%m-%d")
        tremor_end_date_str = tremor_df.index[-1].strftime("%Y-%m-%d")
        tremor_matrix_filename = f"tremor_matrix_unified_{tremor_start_date_str}_{tremor_end_date_str}_ws-{window_size}.csv"

        # =========================
        # Set DEFAULT properties
        # =========================
        self.tremor_df = tremor_df
        self.label_df = label_df
        self.output_dir = output_dir
        self.window_size = window_size
        self.tremor_matrix_filename = tremor_matrix_filename
        self.matrix_tmp_dir = matrix_tmp_dir
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        # =========================
        # Set ADDITIONAL properties
        # =========================
        self.start_date = label_df.index[0] - timedelta(days=self.window_size)
        self.end_date = tremor_df.index[-1]
        self.tremor_start_date_str = tremor_start_date_str
        self.tremor_end_date_str = tremor_end_date_str

        # =========================
        # WIll be set after build() method called
        # =========================
        self.df: pd.DataFrame = pd.DataFrame()
        self.csv: str | None = None

        # =========================
        # Validate and create directories
        # =========================
        self.validate()
        self.create_directories()

    def validate(self) -> None:
        """Validate label and tremor dataframe columns and date ranges.

        Raises:
            ValueError: If required label columns are missing or requested
                tremor columns do not exist in the tremor dataframe.
        """
        label_columns = self.label_df.columns.tolist()

        for col in REQUIRED_LABEL_COLUMNS:
            if col not in label_columns:
                raise ValueError(
                    f"Column '{col}' not found in Label dataframe. "
                    f"Columns available: {label_columns}"
                )

        # Ensuring label date within range of tremor date
        tremor_start_date = self.tremor_df.index[0]
        tremor_end_date = self.tremor_df.index[-1]

        if self.start_date < tremor_start_date:
            self.start_date = tremor_start_date
            if self.verbose:
                logger.info(f"start_date updated to: {self.start_date}")

        if self.end_date > tremor_end_date:
            self.end_date = tremor_end_date
            if self.verbose:
                logger.info(f"end_date updated to: {self.end_date}")

    def create_directories(self) -> None:
        """Create required output directories."""
        os.makedirs(self.output_dir, exist_ok=True)

    def save_matrix_per_method(self, tremor_df: pd.DataFrame) -> Self:
        """Save each tremor metric column as a separate CSV file.

        Args:
            tremor_df (pd.DataFrame): Full features matrix containing
                id, datetime, and one or more tremor metric columns.
                Indexed by pd.DatetimeIndex.

        Returns:
            Self: The FeaturesBuilder instance for method chaining.
        """
        tremor_matrix_per_method_dir = os.path.join(
            self.output_dir, "tremor_matrix_per_method"
        )
        os.makedirs(tremor_matrix_per_method_dir, exist_ok=True)

        # SKipp ID and datetime columns
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
        tremor_matrices: list[pd.DataFrame] = []

        if save_tremor_matrix_per_id:
            os.makedirs(self.matrix_tmp_dir, exist_ok=True)

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
                tremor_df_sliced.reset_index(inplace=True)
                tremor_df_sliced[ID_COLUMN] = column_id

                # Rearrange column to: id, datetime, .. columns
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
                    matrix_tmp_filename = f"{column_id:05}_{start_date_str}_{end_date_str}_eruption-{column_eruption}.csv"
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
        """Build tremor matrix using tremor and label for each label window.

        Tremor matrix columns consist of "id", "datetime" and result from various methods
        such as RSAM and DSAR which processed for different frequency bands.
        The column "id" represent the "id" from label dataset.

        This method iterates through each label window, extracts the corresponding
        tremor data slice, validates sample count, and concatenates
        into a unified new tremor matrix.

        Args:
            select_tremor_columns (list[str], optional): Subset of tremor columns to
                use. If None, all columns are used. Defaults to None.
            tremor_matrix_filename (str, optional): Override default generated filename. Defaults to "features.csv".
            save_tremor_matrix_per_method (bool, optional): Save separate CSV per tremor
                column. Defaults to True.
            save_tremor_matrix_per_id (bool, optional): BE CAREFULL, IT WILL GENERATE A LOT OF FILES.
                Save individual windowed tremor CSVs for debugging. Defaults to False.
            verbose (bool, optional): Override instance verbose flag.
                Defaults to False.

        Raises:
            ValueError: If no valid tremor data found for any label windows.

        Example:
            >>> tremor_matrix = TremorMatrixBuilder(...).build(
            ...     save_tremor_matrix_per_method=True,
            ...     tremor_matrix_filename="sliced_tremor_matrix.csv",
            ... )

        Returns:
            Self: TremorMatrixBuilder instance
        """
        verbose = self.verbose

        # Build filename
        # Example filenmae: unified_tremor_matrix_2025-01-01_2025-09-28_ws-2.csv
        tremor_matrix_filename = tremor_matrix_filename or self.tremor_matrix_filename
        tremor_matrix_csv = os.path.join(self.output_dir, tremor_matrix_filename)

        # Skip if exists
        if not self.overwrite and (os.path.isfile(tremor_matrix_csv)):
            if verbose:
                logger.info(f"Tremor matrix {tremor_matrix_csv} already exists.")
            self.df = pd.read_csv(tremor_matrix_csv)
            return self

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
