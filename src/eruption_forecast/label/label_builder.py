# Standard library imports
import os
from datetime import datetime, timedelta
from functools import cached_property
from typing import Literal, Self

# Third party imports
import pandas as pd

# Project imports
from eruption_forecast.label.constants import (
    MIN_DATE_RANGE_DAYS,
    VALID_WINDOW_STEP_UNITS,
)
from eruption_forecast.label.label_data import LabelData
from eruption_forecast.logger import logger  # type: ignore[attr-defined]
from eruption_forecast.utils import (
    construct_windows,
    normalize_dates,
    sort_dates,
    to_datetime,
)


class LabelBuilder:
    """Build labeled datasets for volcanic eruption forecasting.

    This class generates binary labels for supervised learning by creating
    time windows from a date range and marking windows as "erupted" (1) or
    "not erupted" (0) based on known eruption dates. The labeling uses a
    "day_to_forecast" parameter to mark windows N days before eruptions as
    positive, enabling forecast model training.

    The class supports:
    - Configurable sliding time windows (size and step)
    - Multiple eruption dates
    - Flexible forecast lead time
    - CSV/XLSX export formats
    - Integration with TremorData via window IDs

    Attributes:
        start_date (datetime): Start date for label generation
        end_date (datetime): End date for label generation
        window_size (int): Size of each training window in days
        window_step (int): Step size between windows
        window_step_unit (Literal["minutes", "hours"]): Unit of window step
        day_to_forecast (int): Days before eruption to start positive labeling
        eruption_dates (list[str]): List of eruption dates in YYYY-MM-DD format
        volcano_id (str): Volcano identifier
        output_dir (str): Output directory for label files
        verbose (bool): Enable verbose logging
        debug (bool): Enable debug logging
        df (pd.DataFrame): Built labels dataframe (access via property)
        df_eruption (pd.DataFrame): Subset of erupted windows only
        df_eruptions (dict[str, pd.DataFrame]): Eruption windows by date

    Example:
        >>> # Create labels for 1-day windows with 12-hour steps
        >>> builder = LabelBuilder(
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31",
        ...     window_size=1,
        ...     window_step=12,
        ...     window_step_unit="hours",
        ...     day_to_forecast=2,
        ...     eruption_dates=["2020-06-15", "2020-09-20"],
        ...     volcano_id="VOLCANO_001",
        ...     verbose=True
        ... )
        >>> builder.build()
        >>> print(f"Total windows: {len(builder.df)}")
        >>> print(f"Erupted windows: {len(builder.df_eruption)}")
        >>> builder.save()  # Save to CSV

    Args:
        start_date (str | datetime): Start date in YYYY-MM-DD format.
        end_date (str | datetime): End date in YYYY-MM-DD format.
        window_size (int): Window size in days.
        window_step (int): Window step size.
        window_step_unit (Literal["minutes", "hours"]): Unit of window step.
        day_to_forecast (int): Day to forecast in days.
        eruption_dates (list[str]): Eruption dates in YYYY-MM-DD format.
        volcano_id (str): Volcano ID. To set and forecast ID.
        output_dir (Optional[str], optional): Output directory. Defaults to None.
        verbose (bool, optional): Verbose mode. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.
    """

    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        volcano_id: str,
        output_dir: str | None = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        start_date, end_date, start_date_str, end_date_str = normalize_dates(
            start_date, end_date
        )
        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        label_dir = os.path.join(output_dir, "labels")

        # Set DEFAULT properties
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.window_size: int = int(window_size)
        self.window_step: int = int(window_step)
        self.window_step_unit: Literal["minutes", "hours"] = window_step_unit
        self.day_to_forecast: int = int(day_to_forecast)
        self.eruption_dates: list[str] = sort_dates(eruption_dates)
        self.volcano_id: str = str(volcano_id)
        self.verbose: bool = bool(verbose)
        self.debug: bool = bool(debug)

        # Set ADDITIONAL properties
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
            f"_ws-{window_size}"
            f"_step-{window_step}-{window_step_unit}"
            f"_dtf-{day_to_forecast}.csv"
        )
        self.csv = os.path.join(label_dir, self.filename)

        # Validate
        self.validate()

        # Create directories
        self.create_directories()

        # Verbose and debugging
        if debug:
            logger.info("⚠️ Label Builder :: Debug mode is ON")

        if verbose:
            logger.info(f"Start Date (YYYY-MM-DD): {start_date_str}")
            logger.info(f"End Date (YYYY-MM-DD): {end_date_str}")
            logger.info(f"Window Size (days): {window_size}")
            logger.info(f"Window Step ({window_step_unit}): {window_step}")
            logger.info(f"Day To Forecast (days): {day_to_forecast}")
            logger.info(f"Volcano ID: {volcano_id}")

    def __repr__(self) -> str:
        return (
            f"LabelBuilder({self.start_date_str}, "
            f"{self.end_date_str}, "
            f"{self.window_size}, "
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
        """Labels DataFrame property

        Raises:
            ValueError: If dataframe is empty (build not called yet)

        Returns:
            pd.DataFrame: The labels dataframe
        """
        if self._df.empty:
            raise ValueError("Please call 'build' method first to create labels")

        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Labels DataFrame setter

        Asserts:
            df must be a pandas DataFrame
            df index must be a datetime index
            df must have an 'id' column
            df must have a 'is_erupted' column

        Args:
            df (pd.DataFrame): Labels to set

        Returns:
            None
        """
        self.validate_columns(df)
        self._df = df

    @property
    def df_eruption(self) -> pd.DataFrame:
        """Erupted Labels DataFrame property

        Raises:
            ValueError: If dataframe is empty (build not called yet)

        Returns:
            pd.DataFrame: The erupted labels dataframe
        """
        if self._df_eruption.empty:
            raise ValueError(
                "Please call 'build' method first to create erupted labels"
            )

        return self._df_eruption

    @df_eruption.setter
    def df_eruption(self, df: pd.DataFrame) -> None:
        """Erupted Labels DataFrame setter

        Asserts:
            df must be a pandas DataFrame
            df index must be a datetime index
            df must have an 'id' column
            df must have a 'is_erupted' column

        Args:
            df (pd.DataFrame): Labels to set

        Returns:
            None
        """
        self.validate_columns(df)
        self._df_eruption = df

    @property
    def df_eruptions(self) -> dict[str, pd.DataFrame]:
        """Erupted Windows DataFrame property

        Raises:
            ValueError: If dictionary is empty (build not called yet)

        Returns:
            dict[str, pd.DataFrame]:
                str: eruption date in YYYY-MM-DD format
                pd.DataFrame: eruption dataframe
        """
        if len(self._df_eruptions) == 0:
            raise ValueError(
                "Please call 'build' method first to create erupted labels"
            )

        return self._df_eruptions

    @df_eruptions.setter
    def df_eruptions(self, df_dict: dict[str, pd.DataFrame]) -> None:
        """Erupted Windows DataFrame setter

        Validates that df_dict is a dictionary with string keys (valid dates in YYYY-MM-DD format)
        and pandas DataFrame values with required columns.

        Args:
            df_dict (dict[str, pd.DataFrame]): Erupted Windows to set

        Raises:
            TypeError: If df_dict is not a dictionary or values are not DataFrames
            ValueError: If keys are not valid dates or DataFrames lack required columns

        Returns:
            None
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
        """Label property

        Returns:
            pd.Series
        """
        df = self.df
        return pd.Series(df["is_erupted"], name="is_erupted")

    @cached_property
    def labels(self) -> pd.Series:
        """Alias for 'y' property

        Returns:
            pd.Series
        """
        return self.y

    def update_df_eruptions(self, df: pd.DataFrame) -> Self:
        """Update DataFrame with eruption labels based on eruption dates.

        Args:
            df (pd.DataFrame): Labels DataFrame to update

        Returns:
            Self: Updated instance for method chaining
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
                    f"Eruption date is {eruption}. "
                    f"Date of eruption must be in YYYY-MM-DD format."
                )

            # Set eruption value with 1 for range of eruption date (vectorized operation)
            df.loc[start_eruption:end_eruption, "is_erupted"] = 1  # type: ignore[misc, index]

            if self.debug:
                logger.debug(
                    f"Labeled eruption window for {eruption}: "
                    f"{start_eruption.strftime('%Y-%m-%d %H:%M')} to "
                    f"{end_eruption.strftime('%Y-%m-%d %H:%M')}"
                )

            # Append eruption date as dict key with df_eruptions
            self.df_eruptions = {eruption: df.loc[start_eruption:end_eruption]}  # type: ignore[misc]

        return self

    @staticmethod
    def validate_columns(df: pd.DataFrame) -> None:
        """Validate DataFrame structure and required columns

        Args:
            df (pd.DataFrame): DataFrame to validate

        Raises:
            TypeError: If df is not a pandas DataFrame or index is not DatetimeIndex
            ValueError: If required columns are missing

        Returns:
            None
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
        """Validate properties

        Validates all label builder parameters including date ranges, window sizes,
        and forecasting parameters.

        Raises:
            ValueError: If any of the properties are invalid

        Returns:
            None
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

        if self.window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {self.window_size}")

        if self.window_size >= self.n_days:
            raise ValueError(
                f"window_size must be less than {self.n_days} days, "
                f"got {self.window_size} days"
            )

        if self.window_step_unit not in VALID_WINDOW_STEP_UNITS:
            raise ValueError(
                f"window_step_unit must be one of {VALID_WINDOW_STEP_UNITS}, "
                f"got '{self.window_step_unit}'"
            )

        # Maximum window overlap is the window size in hours
        if self.window_step_unit == "minutes":
            maximum_window_step = self.window_size * 60 * 24
        else:
            maximum_window_step = self.window_size * 24

        if not (0 < self.window_step <= maximum_window_step):
            raise ValueError(
                f"window_step must be between 1 and {maximum_window_step} "
                f"{self.window_step_unit}. "
                f"Got window_step={self.window_step}"
            )

        if self.day_to_forecast <= 0:
            raise ValueError(f"day_to_forecast must be > 0, got {self.day_to_forecast}")

        if self.day_to_forecast >= self.n_days:
            raise ValueError(
                f"day_to_forecast must be less than {self.n_days} days, "
                f"got {self.day_to_forecast} days"
            )

    def create_directories(self) -> None:
        """Create output and label directories if they don't exist

        Returns:
            None
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def validate_eruption_dates(self) -> None:
        """Ensure there is an eruption between start date and end date

        Raises:
            ValueError: If there is no eruption between start date and end date

        Returns:
            None
        """
        if len(self.df_eruption) == 0:
            raise ValueError(
                f"No eruption recorded between date "
                f"{self.start_date_str} and {self.end_date_str}. "
                f"Your eruption_dates: {self.eruption_dates}"
            )

    def initiate_label(self) -> pd.DataFrame:
        """Initialize label DataFrame with zeros (no eruption).

        Creates time windows using the configured window_step and window_step_unit,
        then initializes all labels to 0 (not erupted). The eruption labels will
        be updated later by update_df_eruptions().

        Returns:
            pd.DataFrame: DataFrame with datetime index and 'is_erupted' column (all 0s)

        Example:
            >>> df = builder.initiate_label()
            >>> print(df['is_erupted'].unique())
            [0]
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

        Creates an eruption_dates.csv file in the label directory containing
        the volcano ID and all eruption dates used for labeling.

        Returns:
            None

        Example output file:
            id,volcano_id,eruption_date
            0,VOLCANO_001,2020-06-15
            1,VOLCANO_001,2020-09-20
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

        Uses LabelData class to load and validate the CSV file.

        Args:
            csv (str): Path to the label CSV file

        Returns:
            pd.DataFrame: Loaded label dataframe with datetime index

        Example:
            >>> df = builder.from_csv("output/labels/label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2.csv")
        """
        label_data = LabelData(label_csv=csv)

        if self.verbose:
            logger.info(f"Label data loaded from {csv}")

        return label_data.df

    def build(self) -> Self:
        """Build labels based on eruption dates.

        Main method that orchestrates the label building process:
        1. Loads existing CSV if available, or initializes new DataFrame
        2. Creates ID column for tremor data reference
        3. Updates eruption labels using day_to_forecast parameter
        4. Validates that at least one eruption exists in date range
        5. Saves to CSV if newly created

        The method marks windows as "erupted" (1) starting from
        (eruption_date - day_to_forecast) through eruption_date.

        Returns:
            Self: Instance with populated df, df_eruption, and df_eruptions

        Raises:
            ValueError: If no eruptions fall within the start_date and end_date range

        Example:
            >>> builder.build()
            >>> print(f"Built {len(builder.df)} windows")
            >>> print(f"Positive labels: {len(builder.df_eruption)}")
        """
        if self.verbose:
            logger.info(f"Building labels for {self.n_days} days")
            logger.info(f"Window size: {self.window_size} days")
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
        """Save label file to disk.

        Saves the built labels DataFrame to a file with filename format:
        label_{start_date}_{end_date}_ws-{window_size}_step-{window_step}-{unit}_dtf-{day_to_forecast}.{ext}

        Also saves a separate eruption_dates.csv file for reference.

        Args:
            file_type (Literal["csv", "xlsx"]): File format to save. Defaults to "csv".
                - "csv": Comma-separated values
                - "xlsx": Excel workbook

        Returns:
            Self: Instance for method chaining

        Example:
            >>> builder.build().save()  # Save as CSV
            >>> builder.save(file_type="xlsx")  # Save as Excel
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
