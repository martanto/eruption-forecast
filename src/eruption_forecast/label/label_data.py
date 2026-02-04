# Standard library imports
import os
from datetime import datetime
from functools import cached_property

# Third party imports
import pandas as pd

# Project imports
from eruption_forecast.label.constants import (
    DATE_FORMAT,
    DAY_TO_FORECAST_PREFIX,
    EXAMPLE_LABEL_FILENAME,
    LABEL_EXTENSION,
    LABEL_PREFIX,
    VALID_WINDOW_STEP_UNITS,
    WINDOW_SIZE_PREFIX,
    WINDOW_STEP_PREFIX,
)
from eruption_forecast.logger import logger  # type: ignore[attr-defined]
from eruption_forecast.utils import to_datetime


class LabelData:
    """Wrapper class for loading and parsing label CSV files.

    This class handles loading pre-built label CSV files and extracts metadata
    from the filename. The filename must follow the format:
    label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv

    Example filename: label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2.csv
        - ws-1: window_size = 1 day
        - step-12-hours: window_step = 12 hours
        - dtf-2: day_to_forecast = 2 days

    Attributes:
        label_csv (str): Path to the label CSV file
        start_date (datetime): Start date extracted from filename
        end_date (datetime): End date extracted from filename
        start_date_str (str): Start date string in YYYY-MM-DD format
        end_date_str (str): End date string in YYYY-MM-DD format
        window_size (int): Window size in days
        window_step (int): Window step size
        window_unit (str): Unit of window step ('hours' or 'minutes')
        day_to_forecast (int): Days before eruption to start labeling
        kwargs (dict): Dictionary of all extracted parameters
        df (pd.DataFrame): Cached label dataframe with datetime index

    Example:
        >>> label_data = LabelData("output/labels/label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2.csv")
        >>> print(label_data.window_size)
        1
        >>> print(label_data.parameters)
        {'start_date': datetime(2020, 1, 1), 'end_date': datetime(2020, 12, 31), ...}
        >>> df = label_data.df  # Load the dataframe
    """

    def __init__(self, label_csv: str) -> None:
        """Initialize LabelData with a label CSV file path.

        Args:
            label_csv (str): Path to the label CSV file. Filename must follow
                the format: label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv

        Raises:
            ValueError: If file doesn't exist or filename format is invalid
        """
        self.label_csv = label_csv

        self.validate()

        (
            prefix,
            start_date_str,
            end_date_str,
            window_size,
            window_step_and_unit,
            day_to_forecast,
        ) = self.basename.split("_")

        start_date = to_datetime(start_date_str)
        end_date = to_datetime(end_date_str)
        window_step_parts: list[str] = window_step_and_unit.split("-")

        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.start_date_str: str = start_date_str
        self.end_date_str: str = end_date_str
        self.window_size: int = int(window_size.split("-")[1])
        self.window_step: int = int(window_step_parts[1])
        self.window_unit: str = window_step_parts[2]
        self.day_to_forecast: int = int(day_to_forecast.split("-")[1])
        self.kwargs = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_date_str": self.start_date_str,
            "end_date_str": self.end_date_str,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "window_unit": self.window_unit,
            "day_to_forecast": self.day_to_forecast,
        }

    def validate(self) -> None:
        """Validate label filename format and components.

        Checks that the filename follows the expected format and that all
        components (dates, window size, step, day to forecast) are valid.

        Raises:
            ValueError: If file doesn't exist, filename format is invalid,
                or any component validation fails

        Example:
            Valid filename: label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2.csv
        """
        if not os.path.exists(self.label_csv):
            raise ValueError(f"Label file not found at {self.label_csv}")

        if not self.basename.startswith(LABEL_PREFIX):
            raise ValueError(
                f"Label filename is invalid. Filename should start with '{LABEL_PREFIX}'. "
                f"Example: {EXAMPLE_LABEL_FILENAME}"
            )

        if not self.filename.endswith(LABEL_EXTENSION):
            raise ValueError(
                f"Label file extension is invalid. Expected '{LABEL_EXTENSION}'"
            )

        parts = self.basename.split("_")
        if len(parts) != 6:
            raise ValueError(
                f"Label filename is invalid. "
                f"Expected format: label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv. "
                f": ws-1 -> window_size = 1 (days)"
                f": step-12-hours -> window_step = 12, window_step_unit = hours"
                f": dtf-2 -> day_to_forecast = 2 (days)"
                f"Got: {self.basename}{LABEL_EXTENSION}. Example: {EXAMPLE_LABEL_FILENAME}"
            )

        (
            _,
            start_date,
            end_date,
            window_size,
            window_step,
            day_to_forecast,
        ) = parts

        # Validating start date and end date
        try:
            datetime.strptime(start_date, DATE_FORMAT)
        except ValueError as e:
            raise ValueError(
                f"Start date is invalid. Expected format: {DATE_FORMAT}. Got: {start_date}"
            ) from e

        try:
            datetime.strptime(end_date, DATE_FORMAT)
        except ValueError as e:
            raise ValueError(
                f"End date is invalid. Expected format: {DATE_FORMAT}. Got: {end_date}"
            ) from e

        # Validating window size
        if not window_size.startswith(WINDOW_SIZE_PREFIX):
            raise ValueError(
                f"Window size is invalid. Expected format: {WINDOW_SIZE_PREFIX}X. Got: {window_size}"
            )

        if not window_size.split("-")[1].isdigit():
            raise ValueError(
                f"Window size should be integer in days. "
                f"Expected format: {WINDOW_SIZE_PREFIX}X where 'X' is an integer. Got: {window_size}"
            )

        # Validating window step.
        # Expected example: step-10-minutes or step-12-hours
        # window_steps = ["step", "10", "minutes"]
        window_steps = window_step.split("-")

        if len(window_steps) != 3:
            raise ValueError(
                f"Window step is invalid. Expected format: step-X-(hours/minutes). "
                f"Got: {window_step} (expected 3 parts, got {len(window_steps)})"
            )

        starts_with = window_steps[0]
        step = window_steps[1]
        step_unit = window_steps[2]

        if not window_step.startswith(WINDOW_STEP_PREFIX):
            raise ValueError(
                f"Window step is invalid. Expected format: {WINDOW_STEP_PREFIX}X-(hours/minutes). "
                f"Got: {window_step} (should start with '{WINDOW_STEP_PREFIX}')"
            )

        if not step.isdigit():
            raise ValueError(
                f"Window step is invalid. Step value must be numeric. "
                f"Got: {starts_with}-{step}-{step_unit}"
            )

        if step_unit not in VALID_WINDOW_STEP_UNITS:
            raise ValueError(
                f"Window step is invalid. Unit must be one of {VALID_WINDOW_STEP_UNITS}. "
                f"Got: {starts_with}-{step}-{step_unit}"
            )

        # Validating day to forecast
        if not day_to_forecast.startswith(DAY_TO_FORECAST_PREFIX):
            raise ValueError(
                f"Day to forecast is invalid. Expected format: {DAY_TO_FORECAST_PREFIX}X. Got: {day_to_forecast}"
            )

        if not day_to_forecast.split("-")[1].isdigit():
            raise ValueError(
                f"Day to forecast should be integer in days. "
                f"Expected format: {DAY_TO_FORECAST_PREFIX}X where 'X' is an integer. Got: {day_to_forecast}"
            )

    @cached_property
    def df(self) -> pd.DataFrame:
        """Get label dataframe from file

        Returns:
            pd.DataFrame: Label dataframe with datetime index
        """
        logger.debug(f"Loading label data from {self.label_csv}")
        df = pd.read_csv(self.label_csv, index_col="datetime", parse_dates=True)
        logger.debug(f"Loaded {len(df)} label rows")
        return df

    @cached_property
    def filename(self) -> str:
        """Get the filename from the full path.

        Returns:
            str: Basename of the label CSV file

        Example:
            >>> label_data.filename
            'label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2.csv'
        """
        return os.path.basename(self.label_csv)

    @cached_property
    def basename(self) -> str:
        """Get the basename without file extension.

        Returns:
            str: Filename without the .csv extension

        Example:
            >>> label_data.basename
            'label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2'
        """
        return self.filename.split(".")[0]

    @cached_property
    def filetype(self) -> str:
        """Get the file extension.

        Returns:
            str: File extension without the dot

        Example:
            >>> label_data.filetype
            'csv'
        """
        return self.filename.split(".")[1]

    @cached_property
    def parameters(self) -> dict[str, str | datetime | int]:
        """Get all parameters extracted from the label filename.

        Returns:
            dict[str, Union[str, datetime, int]]: Dictionary containing:
                - start_date (datetime): Start date
                - end_date (datetime): End date
                - start_date_str (str): Start date string
                - end_date_str (str): End date string
                - window_size (int): Window size in days
                - window_step (int): Window step value
                - window_unit (str): Window step unit ('hours' or 'minutes')
                - day_to_forecast (int): Days before eruption to label

        Example:
            >>> params = label_data.parameters
            >>> params['window_size']
            1
            >>> params['day_to_forecast']
            2
        """
        parameters: dict[str, str | datetime | int] = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_date_str": self.start_date_str,
            "end_date_str": self.end_date_str,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "window_unit": self.window_unit,
            "day_to_forecast": self.day_to_forecast,
        }

        return parameters
