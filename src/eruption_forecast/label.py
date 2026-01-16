# Standard library imports
import os
from datetime import datetime, timedelta
from functools import cached_property
from typing import Optional, Self

# Third party imports
import pandas as pd
from loguru import logger


class Label:
    """Label class.
    Use this class to generate label for machine learning.
    Set value to 1 if eruption is recorded in the window.
    Set value to 0 if eruption is not recorded in the window.

    The label dataframe can be accessed through df property.
    The label series can be accessed through y property.

    Args:
        start_date (str | datetime): Start date in YYYY-MM-DD format.
        end_date (str | datetime): End date in YYYY-MM-DD format.
        window_size (int): Window size in days.
        window_overlap (float): Window overlap in days. Range from 0 to 1.
        sampling_rate (float | int): Sampling rate in Hz.
        day_to_forecast (int): Day to forecast in days.
        eruption_dates (list[str]): Eruption dates in YYYY-MM-DD format.
        output_dir (Optional[str], optional): Output directory. Defaults to None.
        verbose (bool, optional): Verbose mode. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.
    """

    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int,
        window_overlap: float,
        sampling_rate: float | int,
        day_to_forecast: int,
        eruption_dates: list[str],
        output_dir: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        if isinstance(start_date, str):
            try:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start_date must be in YYYY-MM-DD format")

        if isinstance(end_date, str):
            try:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("end_date must be in YYYY-MM-DD format")

        start_date_str: str = start_date.strftime("%Y-%m-%d")
        end_date_str: str = end_date.strftime("%Y-%m-%d")
        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        label_dir = os.path.join(output_dir, "labels")

        # Set DEFAULT properties
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.window_size: int = window_size
        self.window_overlap: float = window_overlap
        self.sampling_rate: float = sampling_rate
        self.day_to_forecast: int = day_to_forecast
        self.eruption_dates: list[str] = eruption_dates
        self.verbose: bool = verbose
        self.debug: bool = debug

        # Set ADDITIONAL properties
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.n_days: int = (end_date - start_date).days
        self.df: pd.DataFrame = pd.DataFrame()
        self.df_erupted: pd.DataFrame = pd.DataFrame()
        self.output_dir = output_dir
        self.label_dir = label_dir
        self.filename = (
            f"label_{start_date_str}_{end_date_str}"
            f"_ws-{window_size}"
            f"_wo-{window_overlap}"
            f"_sr-{sampling_rate}"
            f"_dtf-{day_to_forecast}.csv"
        )
        self.filepath = os.path.join(label_dir, self.filename)

        # Validate
        self.validate()

        # Verbose and debugging
        if debug:
            logger.info("⚠️ Debug mode is ON")

        if verbose:
            logger.info(f"Start Date: {start_date_str}")
            logger.info(f"End Date: {end_date_str}")
            logger.info(f"Window Size: {window_size}")
            logger.info(f"Window Overlap: {window_overlap}")
            logger.info(f"Sampling Rate: {sampling_rate}")
            logger.info(f"Day To Forecast: {day_to_forecast}")
            logger.info(f"Label File: {self.filepath}")

    @cached_property
    def y(self) -> pd.Series:
        """Label property

        Returns:
            pd.Series
        """
        self.assert_eruption_dates()
        return pd.Series(self.df["is_erupted"], name="is_erupted")

    @cached_property
    def labels(self) -> pd.Series:
        """Alias for 'y' property

        Returns:
            pd.Series
        """
        return self.y

    def validate(self) -> None:
        """Asserting properties

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
        assert self.window_size < self.n_days, (
            f"window_size must be less than {self.n_days} days)"
        )

        assert 0 < self.window_overlap <= 1.0, (
            "window_overlap must be between 0 and/ or equal 1"
        )
        assert self.sampling_rate > 0, "sampling_rate must be > 0"

        assert self.day_to_forecast > 0, "day_to_forecast must be > 0"
        assert self.day_to_forecast < self.n_days, (
            f"day_to_forecast must be less than {self.n_days} days)"
        )

        # Ensuring output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def assert_eruption_dates(self) -> None:
        """Ensuring there is eruption between start date and end date

        Raises:
            AssertionError: If there is no eruption between start date and end date

        Returns:
            None
        """
        assert len(self.df_erupted) > 0, (
            f"No eruption recorded between date "
            f"{self.start_date_str} and {self.end_date_str}. "
            f"Your eruption_dates: {self.eruption_dates}"
        )

    def initiate_label(self) -> pd.DataFrame:
        """Initiate label value with zero values (no eruption)

        Returns:
            pd.DataFrame
        """
        number_of_sample_per_day = self.sampling_rate * 3600 * 24
        total_sample_point = number_of_sample_per_day * self.window_size
        non_overlap_sample_points = (
            total_sample_point
            if self.window_overlap == 1.0
            else (1 - self.window_overlap) * total_sample_point
        )
        non_overlap_in_seconds = timedelta(
            seconds=non_overlap_sample_points / self.sampling_rate
        )
        dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq=non_overlap_in_seconds,
            inclusive="both",
        )

        df = pd.DataFrame(index=dates)
        df["is_erupted"] = 0

        return df

    def create(self, save: bool = False) -> Self:
        """Create label based on eruption dates

        Args:
            save (bool): Save label file

        Returns:
            Self
        """
        df = self.initiate_label()

        eruption_dates = self.eruption_dates

        # Update eruption value with 1
        for eruption in eruption_dates:
            try:
                end_eruption = datetime.strptime(eruption, "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    f"Eruption date is {eruption}. "
                    f"Date of eruption must be in YYYY-MM-DD format."
                )
            start_eruption = end_eruption - timedelta(days=self.day_to_forecast)
            for index, row in df.loc[start_eruption:end_eruption].iterrows():  # type: ignore
                df.loc[index, "is_erupted"] = 1

        df["datetime"] = df.index
        df.reset_index(drop=True, inplace=True)
        df["id"] = df.index
        df = df[["id", "datetime", "is_erupted"]]

        self.df = df
        self.df_erupted = df[df["is_erupted"] > 0]

        self.assert_eruption_dates()

        return self

    def save(self) -> None:
        """Save label file"""
        self.df.to_csv(self.filepath, index=False)
