import os
from typing import Self, Literal
from datetime import datetime, timedelta
from functools import cached_property

import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.window import construct_windows
from eruption_forecast.label.constants import (
    MIN_DATE_RANGE_DAYS,
    VALID_WINDOW_STEP_UNITS,
)
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.label.label_data import LabelData
from eruption_forecast.utils.date_utils import (
    sort_dates,
    to_datetime,
    normalize_dates,
)


class LabelBuilder:
    """Build labeled datasets for volcanic eruption forecasting.

    This class generates binary labels for supervised learning by creating sliding
    time windows from a date range and marking windows as "erupted" (1) or
    "not erupted" (0) based on known eruption dates and a configurable forecast
    lead time (day_to_forecast parameter).

    The labeling logic marks windows as positive (erupted=1) when their end time
    falls within [eruption_date - day_to_forecast, eruption_date]. This enables
    training models to predict eruptions N days in advance.

    Attributes:
        start_date (datetime.datetime): Start date for label generation.
        end_date (datetime.datetime): End date for label generation.
        start_date_str (str): Start date in YYYY-MM-DD format.
        end_date_str (str): End date in YYYY-MM-DD format.
        window_step (int): Step size between consecutive windows.
        window_step_unit (Literal["minutes", "hours"]): Unit for window_step.
        day_to_forecast (int): Days before eruption to start positive labeling.
        eruption_dates (list[str]): List of eruption dates in YYYY-MM-DD format.
        volcano_id (str): Volcano identifier used in filenames.
        n_days (int): Total days between start_date and end_date.
        output_dir (str): Root output directory path.
        label_dir (str): Directory for label CSV files.
        filename (str): Generated label filename.
        csv (str): Full path to the label CSV file.
        verbose (bool): Enable verbose logging.
        debug (bool): Enable debug logging.
        df (pd.DataFrame): Built labels DataFrame (access via property after build()).
        df_eruption (pd.DataFrame): Subset containing only erupted windows.
        df_eruptions (dict[str, pd.DataFrame]): Erupted windows grouped by eruption date.
        y (pd.Series): Label series (alias for df['is_erupted']).
        labels (pd.Series): Alias for y property.

    Examples:
        >>> # Basic usage with method chaining
        >>> builder = LabelBuilder(
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31",
        ...     window_step=12,
        ...     window_step_unit="hours",
        ...     day_to_forecast=2,
        ...     eruption_dates=["2020-06-15", "2020-09-20"],
        ...     volcano_id="VOLCANO_001",
        ...     verbose=True
        ... ).build().save()

        >>> # Access built labels
        >>> print(f"Total windows: {len(builder.df)}")
        >>> print(f"Positive labels: {len(builder.df_eruption)}")
        >>> print(f"Label balance: {builder.df['is_erupted'].mean():.2%}")

        >>> # Save to Excel instead of CSV
        >>> builder.save(file_type="xlsx")
    """

    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        volcano_id: str,
        output_dir: str | None = None,
        root_dir: str | None = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the LabelBuilder with configuration parameters.

        Sets up all parameters for label generation, validates inputs, and creates
        necessary output directories.

        Args:
            start_date (str | datetime.datetime): Start date in YYYY-MM-DD format
                or datetime object. Must be before end_date.
            end_date (str | datetime.datetime): End date in YYYY-MM-DD format
                or datetime object. Must be at least MIN_DATE_RANGE_DAYS after start_date.
            window_step (int): Step size between consecutive windows. Must be positive.
            window_step_unit (Literal["minutes", "hours"]): Time unit for window_step.
                Must be either "minutes" or "hours".
            day_to_forecast (int): Number of days before eruption to start labeling
                windows as positive. Must be positive and less than total date range.
            eruption_dates (list[str]): List of eruption dates in YYYY-MM-DD format.
                Dates are automatically sorted.
            volcano_id (str): Unique identifier for the volcano, used in output filenames.
            output_dir (str | None, optional): Output directory path. If None, defaults
                to "output" subdirectory. Relative paths are resolved against root_dir.
                Absolute paths are used as-is. Defaults to None.
            root_dir (str | None, optional): Anchor directory for resolving relative
                output_dir paths. If None, uses current working directory. Defaults to None.
            verbose (bool, optional): Enable informational logging. Defaults to False.
            debug (bool, optional): Enable debug-level logging. Defaults to False.

        Raises:
            ValueError: If start_date >= end_date, date range < MIN_DATE_RANGE_DAYS,
                window_step_unit not in VALID_WINDOW_STEP_UNITS, day_to_forecast <= 0,
                or day_to_forecast >= total days.

        Examples:
            >>> # Basic initialization
            >>> builder = LabelBuilder(
            ...     start_date="2020-01-01",
            ...     end_date="2020-12-31",
            ...     window_step=12,
            ...     window_step_unit="hours",
            ...     day_to_forecast=2,
            ...     eruption_dates=["2020-06-15"],
            ...     volcano_id="VOLCANO_001"
            ... )

            >>> # With custom output directory
            >>> builder = LabelBuilder(
            ...     start_date="2020-01-01",
            ...     end_date="2020-12-31",
            ...     window_step=30,
            ...     window_step_unit="minutes",
            ...     day_to_forecast=3,
            ...     eruption_dates=["2020-06-15", "2020-09-20"],
            ...     volcano_id="VOLCANO_002",
            ...     output_dir="custom_output",
            ...     verbose=True
            ... )
        """
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        start_date, end_date, start_date_str, end_date_str = normalize_dates(
            start_date, end_date
        )
        output_dir = resolve_output_dir(output_dir, root_dir, "output")
        label_dir = os.path.join(output_dir, "labels")

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.window_step = int(window_step)
        self.window_step_unit: Literal["minutes", "hours"] = window_step_unit
        self.day_to_forecast: int = int(day_to_forecast)
        self.eruption_dates: list[str] = sort_dates(eruption_dates)
        self.volcano_id: str = str(volcano_id)
        self.verbose: bool = bool(verbose)
        self.debug: bool = bool(debug)

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self._df: pd.DataFrame = pd.DataFrame()
        self._df_eruption: pd.DataFrame = pd.DataFrame()
        self._df_eruptions: dict[str, pd.DataFrame] = {}
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.n_days: int = (end_date - start_date).days
        self.output_dir = output_dir
        self.label_dir = label_dir
        self.filename = (
            f"label_{start_date_str}_{end_date_str}"
            f"_step-{window_step}-{window_step_unit}"
            f"_dtf-{day_to_forecast}.csv"
        )
        self.csv = os.path.join(label_dir, self.filename)

        # ------------------------------------------------------------------
        # Validate and create directories
        # ------------------------------------------------------------------
        self.validate()
        self.create_directories()

        # ------------------------------------------------------------------
        # Verbose and logging
        # ------------------------------------------------------------------
        if debug:
            logger.info("⚠️ Label Builder :: Debug mode is ON")

        if verbose:
            logger.info(f"Start Date (YYYY-MM-DD): {start_date_str}")
            logger.info(f"End Date (YYYY-MM-DD): {end_date_str}")
            logger.info(f"Window Step ({window_step_unit}): {window_step}")
            logger.info(f"Day To Forecast (days): {day_to_forecast}")
            logger.info(f"Volcano ID: {volcano_id}")

    def __repr__(self) -> str:
        return (
            f"LabelBuilder({self.start_date_str}, "
            f"{self.end_date_str}, "
            f"{self.window_step}, "
            f"{self.window_step_unit}, "
            f"{self.day_to_forecast}, "
            f"{self.eruption_dates}), "
            f"{self.volcano_id}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def df(self) -> pd.DataFrame:
        """Access the built labels DataFrame.

        Returns:
            pd.DataFrame: Labels DataFrame with DatetimeIndex and columns 'id' (int)
                and 'is_erupted' (int, values 0 or 1).

        Raises:
            ValueError: If DataFrame is empty because build() has not been called yet.

        Examples:
            >>> builder.build()
            >>> df = builder.df
            >>> print(df.columns.tolist())
            ['id', 'is_erupted']
            >>> print(df.index.name)
            'datetime'
        """
        if self._df.empty:
            raise ValueError("Please call 'build' method first to create labels")

        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the labels DataFrame with validation.

        Validates that the DataFrame has the required structure (DatetimeIndex,
        'id' and 'is_erupted' columns) before assignment.

        Args:
            df (pd.DataFrame): Labels DataFrame to set. Must have DatetimeIndex
                and columns 'id' and 'is_erupted'.

        Raises:
            TypeError: If df is not a pandas DataFrame or index is not DatetimeIndex.
            ValueError: If required columns are missing.

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame(
            ...     {'id': [0, 1], 'is_erupted': [0, 1]},
            ...     index=pd.DatetimeIndex(['2020-01-01', '2020-01-02'], name='datetime')
            ... )
            >>> builder.df = df
        """
        self.validate_columns(df)
        self._df = df

    @property
    def df_eruption(self) -> pd.DataFrame:
        """Access the erupted windows DataFrame.

        Returns only the rows where is_erupted == 1.

        Returns:
            pd.DataFrame: Subset of labels DataFrame containing only erupted windows.

        Raises:
            ValueError: If DataFrame is empty because build() has not been called yet.

        Examples:
            >>> builder.build()
            >>> erupted = builder.df_eruption
            >>> print(len(erupted))
            45
            >>> print((erupted['is_erupted'] == 1).all())
            True
        """
        if self._df_eruption.empty:
            raise ValueError(
                "Please call 'build' method first to create erupted labels"
            )

        return self._df_eruption

    @df_eruption.setter
    def df_eruption(self, df: pd.DataFrame) -> None:
        """Set the erupted labels DataFrame with validation.

        Validates DataFrame structure before assignment.

        Args:
            df (pd.DataFrame): Erupted labels DataFrame to set. Must have DatetimeIndex
                and columns 'id' and 'is_erupted'.

        Raises:
            TypeError: If df is not a pandas DataFrame or index is not DatetimeIndex.
            ValueError: If required columns are missing.

        Examples:
            >>> erupted_df = builder.df[builder.df['is_erupted'] == 1]
            >>> builder.df_eruption = erupted_df
        """
        self.validate_columns(df)
        self._df_eruption = df

    @property
    def df_eruptions(self) -> dict[str, pd.DataFrame]:
        """Access erupted windows grouped by eruption date.

        Returns a dictionary where keys are eruption dates (YYYY-MM-DD format)
        and values are DataFrames containing windows labeled as erupted for that
        specific eruption.

        Returns:
            dict[str, pd.DataFrame]: Dictionary mapping eruption dates to their
                corresponding erupted window DataFrames. Each DataFrame has the
                same structure as the main labels DataFrame.

        Raises:
            ValueError: If dictionary is empty because build() has not been called yet.

        Examples:
            >>> builder.build()
            >>> eruptions = builder.df_eruptions
            >>> print(eruptions.keys())
            dict_keys(['2020-06-15', '2020-09-20'])
            >>> print(len(eruptions['2020-06-15']))
            23
        """
        if len(self._df_eruptions) == 0:
            raise ValueError(
                "Please call 'build' method first to create erupted labels"
            )

        return self._df_eruptions

    @df_eruptions.setter
    def df_eruptions(self, df_dict: dict[str, pd.DataFrame]) -> None:
        """Set erupted windows dictionary with comprehensive validation.

        Validates that all keys are valid dates in YYYY-MM-DD format and all
        values are properly structured DataFrames with required columns.

        Args:
            df_dict (dict[str, pd.DataFrame]): Dictionary mapping eruption dates
                (YYYY-MM-DD format) to their erupted window DataFrames.

        Raises:
            TypeError: If df_dict is not a dict, keys are not strings, or values
                are not DataFrames.
            ValueError: If keys are not valid YYYY-MM-DD dates, DataFrames lack
                DatetimeIndex, or required columns are missing.

        Examples:
            >>> eruptions = {
            ...     "2020-06-15": df1,
            ...     "2020-09-20": df2
            ... }
            >>> builder.df_eruptions = eruptions
        """
        if not isinstance(df_dict, dict):
            raise TypeError("df_dict must be a dictionary")

        # Validate keys are strings and valid dates
        for key in df_dict.keys():
            if not isinstance(key, str):
                raise TypeError(
                    f"All keys must be strings. Got key with type: {type(key)}"
                )

            try:
                parsed_date = pd.to_datetime(key).strftime("%Y-%m-%d")
                if parsed_date != key:
                    raise ValueError(
                        f"Key '{key}' is not in YYYY-MM-DD format. Expected: {parsed_date}"
                    )
            except Exception as e:
                raise ValueError(
                    f"Key '{key}' is not a valid date in YYYY-MM-DD format"
                ) from e

        # Validate values are DataFrames with required structure
        for key, value in df_dict.items():
            if not isinstance(value, pd.DataFrame):
                raise TypeError(
                    f"All values must be pandas DataFrames. "
                    f"Got value for key '{key}' with type: {type(value)}"
                )

            if not isinstance(value.index, pd.DatetimeIndex):
                raise ValueError(
                    f"DataFrame for key '{key}' must have a DatetimeIndex. "
                    f"Got index type: {type(value.index)}"
                )

            if "id" not in value.columns:
                raise ValueError(
                    f"DataFrame for key '{key}' must have an 'id' column. "
                    f"Available columns: {value.columns.tolist()}"
                )

            if "is_erupted" not in value.columns:
                raise ValueError(
                    f"DataFrame for key '{key}' must have an 'is_erupted' column. "
                    f"Available columns: {value.columns.tolist()}"
                )

        self._df_eruptions.update(df_dict)

    @cached_property
    def y(self) -> pd.Series:
        """Extract the target label Series.

        Returns the 'is_erupted' column as a Series, commonly used as the
        target variable (y) in machine learning workflows.

        Returns:
            pd.Series: Series with name 'is_erupted' containing binary labels
                (0 for not erupted, 1 for erupted).

        Examples:
            >>> builder.build()
            >>> y = builder.y
            >>> print(y.name)
            'is_erupted'
            >>> print(y.value_counts())
            is_erupted
            0    1395
            1      45
            dtype: int64
        """
        df = self.df
        return pd.Series(df["is_erupted"], name="is_erupted")

    @cached_property
    def labels(self) -> pd.Series:
        """Alias for the y property.

        Provides an alternative name for accessing the target labels.

        Returns:
            pd.Series: Same as y property - the 'is_erupted' label series.

        Examples:
            >>> builder.build()
            >>> labels = builder.labels
            >>> assert (labels == builder.y).all()
        """
        return self.y

    def update_df_eruptions(self, df: pd.DataFrame) -> Self:
        """Update DataFrame with eruption labels based on configured eruption dates.

        For each eruption date, marks all windows falling between
        (eruption_date - day_to_forecast) and eruption_date as erupted (is_erupted=1).
        Also populates the df_eruptions dictionary with windows for each eruption.

        Args:
            df (pd.DataFrame): Labels DataFrame to update with eruption labels.
                Must have DatetimeIndex and 'is_erupted' column.

        Returns:
            Self: Updated instance for method chaining.

        Examples:
            >>> builder.update_df_eruptions(df)
            >>> print(df['is_erupted'].sum())
            45
        """
        # Update eruption value with 1
        if self.verbose:
            logger.info(f"Processing {len(self.eruption_dates)} eruption dates")

        for eruption in self.eruption_dates:
            try:
                day_of_eruption = to_datetime(eruption)

                # Move the start of eruption to number of day_to_forecast
                start_eruption = day_of_eruption - timedelta(days=self.day_to_forecast)

                # Set start time of eruption to 00:00:00
                start_eruption = start_eruption.replace(hour=0, minute=0, second=0)

                # Set end time of eruption to at 23:59:59
                end_eruption = day_of_eruption.replace(hour=23, minute=59, second=59)

                # Stop if eruption date is beyond the end date
                if end_eruption > self.end_date:
                    if self.debug:
                        logger.debug(
                            f"Eruption date {eruption} is beyond end date {self.end_date_str}, skipping"
                        )
                    continue

            except ValueError:
                raise ValueError(  # noqa: B904
                    f"Eruption date is {eruption}. Date of eruption must be in YYYY-MM-DD format."
                )

            # Set eruption value with 1 for range of eruption date (vectorized operation)
            df.loc[start_eruption:end_eruption, "is_erupted"] = 1

            if self.debug:
                logger.debug(
                    f"Labeled eruption window for {eruption}: "
                    f"{start_eruption.strftime('%Y-%m-%d %H:%M')} to "
                    f"{end_eruption.strftime('%Y-%m-%d %H:%M')}"
                )

            # Append eruption date as dict key with df_eruptions
            self.df_eruptions = {eruption: df.loc[start_eruption:end_eruption]}

        return self

    @staticmethod
    def validate_columns(df: pd.DataFrame) -> None:
        """Validate DataFrame structure and required columns.

        Ensures the DataFrame has the correct structure for label data:
        DatetimeIndex and both 'id' and 'is_erupted' columns.

        Args:
            df (pd.DataFrame): DataFrame to validate.

        Raises:
            TypeError: If df is not a pandas DataFrame or index is not DatetimeIndex.
            ValueError: If 'id' or 'is_erupted' columns are missing.

        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame(
            ...     {'id': [0, 1], 'is_erupted': [0, 1]},
            ...     index=pd.DatetimeIndex(['2020-01-01', '2020-01-02'])
            ... )
            >>> LabelBuilder.validate_columns(df)  # No exception

            >>> # Missing column raises error
            >>> bad_df = pd.DataFrame({'id': [0, 1]}, index=pd.DatetimeIndex(['2020-01-01', '2020-01-02']))
            >>> LabelBuilder.validate_columns(bad_df)  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            ValueError: df must have an 'is_erupted' column...
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(f"df index must be a DatetimeIndex, got {type(df.index)}")

        if "id" not in df.columns:
            raise ValueError(
                f"df must have an 'id' column. Available columns: {df.columns.tolist()}"
            )

        if "is_erupted" not in df.columns:
            raise ValueError(
                f"df must have an 'is_erupted' column. Available columns: {df.columns.tolist()}"
            )

    def validate(self) -> None:
        """Validate all label builder parameters.

        Performs comprehensive validation of all configuration parameters including:
        - Date range validity (start < end, minimum duration)
        - Window step unit validity
        - Day to forecast constraints

        Raises:
            ValueError: If any validation fails:
                - start_date >= end_date
                - Date range < MIN_DATE_RANGE_DAYS (7 days)
                - window_step_unit not in ['minutes', 'hours']
                - day_to_forecast <= 0
                - day_to_forecast >= total days in range

        Examples:
            >>> # Valid configuration passes
            >>> builder = LabelBuilder(
            ...     start_date="2020-01-01",
            ...     end_date="2020-01-15",
            ...     window_step=12,
            ...     window_step_unit="hours",
            ...     day_to_forecast=2,
            ...     eruption_dates=["2020-01-10"],
            ...     volcano_id="TEST"
            ... )

            >>> # Invalid date range fails
            >>> builder = LabelBuilder(
            ...     start_date="2020-01-01",
            ...     end_date="2020-01-03",  # Only 2 days
            ...     window_step=12,
            ...     window_step_unit="hours",
            ...     day_to_forecast=2,
            ...     eruption_dates=["2020-01-02"],
            ...     volcano_id="TEST"
            ... )  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            ValueError: Total days between start_date and end_date must be >= 7 days...
        """
        minimal_end_date = self.start_date + timedelta(days=MIN_DATE_RANGE_DAYS)

        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date must be less than end_date. "
                f"Got start_date={self.start_date_str}, end_date={self.end_date_str}"
            )

        if self.n_days < MIN_DATE_RANGE_DAYS:
            raise ValueError(
                f"Total days between start_date and end_date must be >= {MIN_DATE_RANGE_DAYS} days. "
                f"Got {self.n_days} days. "
                f"Parameter end_date should be at least {minimal_end_date.strftime('%Y-%m-%d')}"
            )

        if self.window_step_unit not in VALID_WINDOW_STEP_UNITS:
            raise ValueError(
                f"window_step_unit must be one of {VALID_WINDOW_STEP_UNITS}, "
                f"got '{self.window_step_unit}'"
            )

        if self.day_to_forecast <= 0:
            raise ValueError(f"day_to_forecast must be > 0, got {self.day_to_forecast}")

        if self.day_to_forecast >= self.n_days:
            raise ValueError(
                f"day_to_forecast must be less than {self.n_days} days, "
                f"got {self.day_to_forecast} days"
            )

    def create_directories(self) -> None:
        """Create output and label directories if they don't exist.

        Creates both the main output directory and the labels subdirectory.
        Uses os.makedirs with exist_ok=True to avoid errors if directories
        already exist.

        Examples:
            >>> builder = LabelBuilder(...)
            >>> builder.create_directories()
            >>> # Directories created: output_dir/ and output_dir/labels/
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def validate_eruption_dates(self) -> None:
        """Ensure at least one eruption exists between start and end dates.

        Checks that df_eruption is not empty, meaning at least one eruption
        date falls within the configured date range.

        Raises:
            ValueError: If no eruptions are recorded between start_date and end_date.
                The error message includes the provided eruption_dates list for debugging.

        Examples:
            >>> builder.build()
            >>> builder.validate_eruption_dates()  # Passes if eruptions exist

            >>> # No eruptions in range
            >>> builder = LabelBuilder(
            ...     start_date="2020-01-01",
            ...     end_date="2020-01-15",
            ...     eruption_dates=["2020-12-25"],  # Outside range
            ...     ...
            ... )
            >>> builder.build()  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            ValueError: No eruption between start date (2020-01-01) and end date (2020-01-15)...
        """
        if len(self.df_eruption) == 0:
            raise ValueError(
                f"No eruption recorded between date "
                f"{self.start_date_str} and {self.end_date_str}. "
                f"Your eruption_dates: {self.eruption_dates}"
            )

    def initiate_label(self) -> pd.DataFrame:
        """Initialize label DataFrame with all labels set to 0 (not erupted).

        Creates sliding time windows using construct_windows() based on the
        configured window_step and window_step_unit, then initializes all
        'is_erupted' values to 0. Eruption labels will be updated later by
        update_df_eruptions().

        Returns:
            pd.DataFrame: DataFrame with DatetimeIndex and 'is_erupted' column
                containing all zeros.

        Examples:
            >>> df = builder.initiate_label()
            >>> print(df.columns.tolist())
            ['is_erupted']
            >>> print(df['is_erupted'].unique())
            [0]
            >>> print(isinstance(df.index, pd.DatetimeIndex))
            True
        """
        df = construct_windows(
            start_date=self.start_date,
            end_date=self.end_date,
            window_step=self.window_step,
            window_step_unit=self.window_step_unit,
        )
        df["is_erupted"] = 0

        return df

    def save_eruption_dates(self) -> None:
        """Save eruption dates to a CSV file for reference.

        Creates an 'eruption_dates.csv' file in the label directory containing
        the volcano_id and all eruption dates. This provides a record of which
        eruptions were used for labeling.

        The output CSV has columns: id, volcano_id, eruption_date

        Examples:
            >>> builder.save_eruption_dates()
            >>> # Creates: label_dir/eruption_dates.csv
            >>> # Content:
            >>> # id,volcano_id,eruption_date
            >>> # 0,VOLCANO_001,2020-06-15
            >>> # 1,VOLCANO_001,2020-09-20
        """
        filename = os.path.join(self.label_dir, "eruption_dates.csv")

        df = pd.DataFrame(self.eruption_dates, columns=["eruption_date"])
        df.index.name = "id"
        df["volcano_id"] = self.volcano_id
        df = df[["volcano_id", "eruption_date"]]
        df.to_csv(filename, index=True)

        if self.verbose:
            logger.info(f"Eruption dates saved to {filename}")

    def from_csv(self, csv: str) -> pd.DataFrame:
        """Load label DataFrame from an existing CSV file.

        Uses the LabelData class to load and validate a previously saved
        label CSV file.

        Args:
            csv (str): Path to the label CSV file. Filename must follow the
                standard format: label_{start}_{end}_step-{X}-{unit}_dtf-{X}.csv

        Returns:
            pd.DataFrame: Loaded label DataFrame with DatetimeIndex and columns
                'id' and 'is_erupted'.

        Raises:
            ValueError: If file doesn't exist or filename format is invalid
                (raised by LabelData).

        Examples:
            >>> df = builder.from_csv("output/labels/label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")
            >>> print(df.columns.tolist())
            ['id', 'is_erupted']
        """
        label_data = LabelData(label_csv=csv)

        if self.verbose:
            logger.info(f"Label data loaded from {csv}")

        return label_data.df

    def build(self) -> Self:
        """Build labels based on eruption dates and window configuration.

        Main orchestration method that performs the complete label building workflow:
        1. Checks if label CSV already exists, loads it if found
        2. Otherwise, initializes new DataFrame with construct_windows()
        3. Creates sequential 'id' column for tremor data alignment
        4. Updates eruption labels using update_df_eruptions()
        5. Validates that at least one eruption exists in date range
        6. Saves to CSV if newly created

        Windows are marked as erupted (is_erupted=1) when their end time (index)
        falls within [eruption_date - day_to_forecast, eruption_date].

        Returns:
            Self: Instance with populated df, df_eruption, and df_eruptions properties.

        Raises:
            ValueError: If no eruptions fall within the start_date to end_date range.

        Examples:
            >>> builder = LabelBuilder(
            ...     start_date="2020-01-01",
            ...     end_date="2020-12-31",
            ...     window_step=12,
            ...     window_step_unit="hours",
            ...     day_to_forecast=2,
            ...     eruption_dates=["2020-06-15"],
            ...     volcano_id="VOLCANO_001"
            ... )
            >>> builder.build()
            >>> print(f"Total windows: {len(builder.df)}")
            Total windows: 1440
            >>> print(f"Erupted windows: {len(builder.df_eruption)}")
            Erupted windows: 48
            >>> print(f"Balance: {builder.df['is_erupted'].mean():.2%}")
            Balance: 3.33%
        """
        if self.verbose:
            logger.info(f"Building labels for {self.n_days} days")
            logger.info(f"Window step: {self.window_step} {self.window_step_unit}")
            logger.info(f"Day to forecast: {self.day_to_forecast} days")

        # Check if label csv is exists
        file_exists = os.path.isfile(self.csv)

        # Load or initiate the dataframe
        if file_exists:
            if self.verbose:
                logger.info(f"Loading existing labels from {self.csv}")
            df = self.from_csv(self.csv)
        else:
            if self.verbose:
                logger.info("Initiating new label DataFrame")
            df = self.initiate_label()

        if not file_exists:
            # Create id column to use as data ID reference with tremor data
            # such as RSAM, DSAR or MSNoise
            if self.debug:
                logger.debug(f"Creating ID column with {len(df)} rows")
            df["id"] = range(len(df))
            df = df[["id", "is_erupted"]].astype({"id": int, "is_erupted": int})

        self.update_df_eruptions(df)
        self.df = df

        df_eruption = df[df["is_erupted"] > 0]
        if df_eruption.empty:
            raise ValueError(
                f"No eruption between start date ({self.start_date_str}) and end date ({self.end_date_str}). "
                f"Your eruption_dates: {self.eruption_dates}. "
                f"Please change your start_date and end_date parameters."
            )

        if self.verbose:
            erupted_count = len(df_eruption)
            total_count = len(df)
            logger.info(
                f"Label building complete: {erupted_count} erupted windows out of {total_count} total "
                f"({erupted_count / total_count * 100:.2f}%)"
            )

        self.df_eruption = df_eruption

        if not file_exists:
            self.validate_eruption_dates()
            self.save()

        return self

    def save(self, file_type: Literal["csv", "xlsx"] = "csv") -> Self:
        """Save labels DataFrame to disk in CSV or Excel format.

        Saves the built labels DataFrame to a file with standardized filename:
        label_{start_date}_{end_date}_step-{window_step}-{unit}_dtf-{day_to_forecast}.{ext}

        Also saves a separate 'eruption_dates.csv' reference file.

        Args:
            file_type (Literal["csv", "xlsx"], optional): Output file format.
                Defaults to "csv".
                - "csv": Comma-separated values (lightweight, fast)
                - "xlsx": Excel workbook (for manual inspection)

        Returns:
            Self: Instance for method chaining.

        Examples:
            >>> # Save as CSV (default)
            >>> builder.build().save()
            >>> # File created: label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv

            >>> # Save as Excel
            >>> builder.build().save(file_type="xlsx")
            >>> # File created: label_2020-01-01_2020-12-31_step-12-hours_dtf-2.xlsx

            >>> # Method chaining
            >>> builder.build().save().save(file_type="xlsx")
            >>> # Creates both CSV and Excel files
        """
        df = self.df
        filepath = self.csv

        if file_type == "xlsx":
            filepath = self.csv.replace(".csv", ".xlsx")
            df.to_excel(filepath, index=True)

            # Update filepath as an excel file
            self.csv = filepath
            return self

        df.to_csv(filepath, index=True)

        if self.verbose:
            logger.info(f"Label saved to {filepath}")

        # Save eruption dates
        self.save_eruption_dates()

        return self
