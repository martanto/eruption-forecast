# Standard library imports
import os
from datetime import datetime, timedelta
from functools import cached_property
from typing import Literal, Optional, Self, Union

# Third party imports
import pandas as pd

from eruption_forecast.label.label_data import LabelData

# Project imports
from eruption_forecast.logger import logger
from eruption_forecast.utils import construct_windows, to_datetime


class LabelBuilder:
    """LabelBuilder class.

    Use this class to generate label for machine learning.
    Set value to 1 if eruption is recorded in the window.
    Set value to 0 if eruption is not recorded in the window.

    The label dataframe can be accessed through df property.
    The label series can be accessed through y property.

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
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        window_size: int,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        volcano_id: str,
        output_dir: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        start_date = to_datetime(start_date).replace(hour=0, minute=0, second=0)
        end_date = to_datetime(end_date).replace(hour=23, minute=59, second=59)
        start_date_str: str = start_date.strftime("%Y-%m-%d")
        end_date_str: str = end_date.strftime("%Y-%m-%d")
        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        label_dir = os.path.join(output_dir, "labels")

        # Set DEFAULT properties
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.window_size: int = int(window_size)
        self.window_step: int = int(window_step)
        self.window_step_unit: Literal["minutes", "hours"] = window_step_unit
        self.day_to_forecast: int = int(day_to_forecast)
        self.eruption_dates: list[str] = eruption_dates
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

        Asserts:
            self._df must not empty

        Returns:
            pd.DataFrame
        """
        assert not self._df.empty, "Please call 'build' method first to create labels"

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

        Asserts:
            self._df_eruption must not empty

        Returns:
            pd.DataFrame
        """
        assert (
            not self._df_eruption.empty
        ), "Please call 'build' method first to create erupted labels"

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

        Asserts:
            self._df_eruptions must not empty

        Returns:
            dict[str, pd.DataFrame]:
                str: eruption date in YYYY-MM-DD format
                pd.DataFrame: eruption dataframe
        """
        assert (
            len(self._df_eruptions) > 0
        ), "Please call 'build' method first to create erupted labels"

        return self._df_eruptions

    @df_eruptions.setter
    def df_eruptions(self, df_dict: dict[str, pd.DataFrame]) -> None:
        """Erupted Windows DataFrame setter

        Asserts:
            df_dict must be a dictionary
            df_dict must have string keys. And each key must be a valid date in YYYY-MM-DD format
            df_dict must have pandas DataFrame values

        Args:
            df_dict (dict[str, pd.DataFrame]): Erupted Windows to set

        Returns:
            None
        """
        assert isinstance(df_dict, dict), "df_dict must be a dictionary"

        # Ensuring keys are strings and valid dates
        assert all(
            isinstance(key, str) for key in df_dict.keys()
        ), "df_dict must have string keys"
        assert all(
            pd.to_datetime(key).strftime("%Y-%m-%d") == key for key in df_dict.keys()
        ), "df_dict must have string keys. And each key must be a valid date in YYYY-MM-DD format"
        assert all(
            isinstance(value, pd.DataFrame) for value in df_dict.values()
        ), "df_dict must have pandas DataFrame values"

        # Ensuring values are DataFrames with datetime index and required columns
        assert all(
            isinstance(value.index, pd.DatetimeIndex) for value in df_dict.values()
        ), "df index must be a datetime index"
        assert all("id" in value.columns for value in df_dict.values()), ValueError(
            f"df must have an 'id' column. Your columns"
        )
        assert all(
            "is_erupted" in value.columns for value in df_dict.values()
        ), "df must have an 'is_erupted' column"

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
        # Update eruption value with 1
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
                    continue

            except ValueError:
                raise ValueError(
                    f"Eruption date is {eruption}. "
                    f"Date of eruption must be in YYYY-MM-DD format."
                )

            # Set eruption value with 1 for range of eruption date
            for index, row in df.loc[start_eruption:end_eruption].iterrows():  # type: ignore
                df.loc[index, "is_erupted"] = 1

            # Append eruption date as dict key with df_eruptions
            self.df_eruptions = {eruption: df.loc[start_eruption:end_eruption]}

        return self

    def validate_columns(self, df: pd.DataFrame) -> None:
        """Validate columns

        Raises:
            AssertionError: If any of the columns are invalid

        Returns:
            None
        """
        assert isinstance(df, pd.DataFrame), "df must be a pandas DataFrame"
        assert isinstance(
            df.index, pd.DatetimeIndex
        ), "df index must be a datetime index"
        assert "id" in df.columns, "df must have an 'id' column"
        assert "is_erupted" in df.columns, "df must have an 'is_erupted' column"

    def validate(self) -> None:
        """Validate properties

        Raises:
            AssertionError: If any of the properties are invalid

        Returns:
            None
        """
        minimal_end_date = self.start_date + timedelta(days=7)

        assert self.start_date < self.end_date, "start_date must be less than end_date"

        assert self.n_days >= 7, (
            "Total days between start_date and end_date must be >= 7 days. "
            f"Parameter end_date at least {minimal_end_date.strftime('%Y-%m-%d')}"
        )

        assert self.window_size > 0, "window_size must be > 0"
        assert (
            self.window_size < self.n_days
        ), f"window_size must be less than {self.n_days} days)"

        assert self.window_step_unit in [
            "minutes",
            "hours",
        ], "window_step_unit must be 'minutes' or 'hours'"

        # Maximum window overlap is the window size in hours
        if self.window_step_unit == "minutes":
            maximum_window_step = self.window_size * 60 * 24
        else:
            maximum_window_step = self.window_size * 24

        assert 0 < self.window_step <= maximum_window_step, (
            f"window_step must be less than or equal to {maximum_window_step} "
            f"{self.window_step_unit}. \\n"
            f"window_step: {self.window_step}, maximum_window_step: {maximum_window_step}"
        )

        assert self.day_to_forecast > 0, "day_to_forecast must be > 0"
        assert (
            self.day_to_forecast < self.n_days
        ), f"day_to_forecast must be less than {self.n_days} days)"

        # Ensuring output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def assert_eruption_dates(self) -> Self:
        """Ensure there is an eruption between start date and end date

        Raises:
            AssertionError: If there is no eruption between start date and end date

        Returns:
            self
        """
        assert len(self.df_eruption) > 0, (
            f"No eruption recorded between date "
            f"{self.start_date_str} and {self.end_date_str}. "
            f"Your eruption_dates: {self.eruption_dates}"
        )

        return self

    def initiate_label(self) -> pd.DataFrame:
        """Initialize label values with zeros (no eruption)

        Returns:
            pd.DataFrame
        """
        df = construct_windows(
            self.window_step, self.window_step_unit, self.start_date, self.end_date
        )
        df["is_erupted"] = 0

        return df

    def save_eruption_dates(self) -> None:
        """Save eruption dates to csv file"""
        filename = os.path.join(self.label_dir, "eruption_dates.csv")

        df = pd.DataFrame(self.eruption_dates, columns=["eruption_date"])
        df.index.name = "id"
        df["volcano_id"] = self.volcano_id
        df = df[["volcano_id", "eruption_date"]]
        df.to_csv(filename, index=True)

        if self.verbose:
            logger.info(f"Eruption dates saved to {filename}")

    def from_csv(self, csv: str) -> pd.DataFrame:
        """Get label values dataframe

        Returns:
            pd.DataFrame
        """
        label_data = LabelData(label_csv=csv)

        if self.verbose:
            logger.info(f"Label data loaded from {csv}")

        return label_data.df

    def build(self) -> Self:
        """Build label based on eruption dates

        Returns:
            Self
        """

        # Check if label csv is exists
        file_exists = os.path.isfile(self.csv)

        # Load or initiate the dataframe
        df = self.from_csv(self.csv) if file_exists else self.initiate_label()

        if not file_exists:
            # Create id column to use as data ID reference with tremor data
            # such as RSAM, DSAR or MSNoise
            df["id"] = range(len(df))
            df = df[["id", "is_erupted"]].astype({"id": int, "is_erupted": int})

        self.update_df_eruptions(df)
        self.df = df
        self.df_eruption = df[df["is_erupted"] > 0]

        if not file_exists:
            self.assert_eruption_dates()
            self.save()

        return self

    def save(self, file_type: Literal["csv", "xlsx"] = "csv") -> Self:
        """Save label file

        Args:
            file_type (Literal["csv", "xlsx"]): File type to save. Defaults to "csv".

        Returns:
            Self
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
