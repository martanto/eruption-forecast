import os
from datetime import datetime
from functools import cached_property

import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.data_container import BaseDataContainer
from eruption_forecast.label.constants import (
    DATE_FORMAT,
    LABEL_PREFIX,
    LABEL_EXTENSION,
    WINDOW_STEP_PREFIX,
    DAY_TO_FORECAST_PREFIX,
    EXAMPLE_LABEL_FILENAME,
    VALID_WINDOW_STEP_UNITS,
)
from eruption_forecast.utils.date_utils import parse_label_filename


class LabelData(BaseDataContainer):
    """Wrapper class for loading and parsing label CSV files.

    This class handles loading pre-built label CSV files and extracts metadata
    from the standardized filename format. Use this class to load existing labels
    for model training or evaluation without rebuilding them.

    Inherits from :class:`BaseDataContainer`, providing `csv_path`, `start_date_str`,
    `end_date_str`, and `data` as part of the shared data-container interface.

    The filename must follow the format:
        label_{start_date}_{end_date}_step-{window_step}-{unit}_dtf-{day_to_forecast}.csv

    Attributes:
        label_csv (str): Path to the label CSV file.
        start_date (datetime.datetime): Start date extracted from filename.
        end_date (datetime.datetime): End date extracted from filename.
        start_date_str (str): Start date string in YYYY-MM-DD format.
        end_date_str (str): End date string in YYYY-MM-DD format.
        window_step (int): Window step size (e.g., 12 for 12 hours).
        window_unit (str): Unit of window step ('hours' or 'minutes').
        day_to_forecast (int): Days before eruption to start labeling as positive.
        df (pd.DataFrame): Cached label DataFrame with datetime index and columns
            'id' (int) and 'is_erupted' (0 or 1).
        data (pd.DataFrame): Alias for :attr:`df` — satisfies the
            :class:`BaseDataContainer` interface.
        filename (str): Basename of the label CSV file with extension.
        basename (str): Filename without extension.
        filetype (str): File extension without the dot.
        parameters (dict): Dictionary of all extracted parameters.

    Examples:
        >>> # Load existing label file
        >>> label_data = LabelData("output/labels/label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")
        >>> print(label_data.window_step)
        12
        >>> print(label_data.window_unit)
        'hours'
        >>> print(label_data.day_to_forecast)
        2

        >>> # Access the DataFrame
        >>> df = label_data.df
        >>> print(df.columns.tolist())
        ['id', 'is_erupted']
        >>> print(df.index.name)
        'datetime'

        >>> # Get all parameters as dict
        >>> params = label_data.parameters
        >>> print(params['start_date_str'])
        '2020-01-01'
    """

    def __init__(self, label_csv: str) -> None:
        """Initialize LabelData with a label CSV file path.

        Loads and validates the label CSV file, then parses all metadata from the
        filename according to the standard format.

        Args:
            label_csv (str): Path to the label CSV file. Filename must follow
                the format: label_{start}_{end}_step-{X}-{unit}_dtf-{X}.csv
                where start and end are YYYY-MM-DD dates, X are integers,
                and unit is 'hours' or 'minutes'.

        Raises:
            ValueError: If file doesn't exist, filename format is invalid,
                date format is incorrect, or any component validation fails.

        Examples:
            >>> # Valid initialization
            >>> label_data = LabelData("label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")

            >>> # Invalid - file not found
            >>> label_data = LabelData("nonexistent.csv")  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            ValueError: Label file not found at nonexistent.csv
        """
        super().__init__(csv_path=label_csv)
        self.label_csv = label_csv

        self.validate()

        parsed = parse_label_filename(self.basename)

        self.start_date: datetime = parsed["start_date"]
        self.end_date: datetime = parsed["end_date"]
        self._start_date_str: str = parsed["start_date_str"]
        self._end_date_str: str = parsed["end_date_str"]
        self.window_step: int = parsed["window_step"]
        self.window_unit: str = parsed["window_step_unit"]
        self.day_to_forecast: int = parsed["day_to_forecast"]

    @property
    def start_date_str(self) -> str:
        """Return the start date as an ISO-format string.

        Returns:
            str: Start date in ``"YYYY-MM-DD"`` format.
        """
        return self._start_date_str

    @property
    def end_date_str(self) -> str:
        """Return the end date as an ISO-format string.

        Returns:
            str: End date in ``"YYYY-MM-DD"`` format.
        """
        return self._end_date_str

    def validate(self) -> None:
        """Validate label filename format and all components.

        Performs comprehensive validation of the label CSV file:
        - File existence
        - Filename prefix and extension
        - Number of filename parts
        - Date format (YYYY-MM-DD)
        - Window step format (step-X-unit)
        - Day to forecast format (dtf-X)

        Raises:
            ValueError: If any validation check fails. Error messages include:
                - File not found
                - Invalid filename prefix (must start with 'label_')
                - Invalid file extension (must be '.csv')
                - Wrong number of filename parts (must be 5)
                - Invalid date format (must be YYYY-MM-DD)
                - Invalid window step format (must be step-X-unit)
                - Invalid window step value (must be numeric)
                - Invalid window unit (must be 'hours' or 'minutes')
                - Invalid day_to_forecast format (must be dtf-X where X is integer)

        Examples:
            >>> # Valid filename passes validation
            >>> label_data = LabelData("label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")
            >>> label_data.validate()  # No exception raised

            >>> # Invalid filename raises ValueError
            >>> label_data = LabelData("wrong_format.csv")  # doctest: +SKIP
            Traceback (most recent call last):
                ...
            ValueError: Label filename is invalid. Filename should start with 'label_'...
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
        if len(parts) != 5:
            raise ValueError(
                f"Label filename is invalid. "
                f"Expected format: label_YYYY-MM-DD_YYYY-MM-DD_step-X-unit_dtf-X.csv "
                f"(step-12-hours -> window_step=12, unit=hours; dtf-2 -> day_to_forecast=2). "
                f"Got: {self.basename}{LABEL_EXTENSION}. Example: {EXAMPLE_LABEL_FILENAME}"
            )

        (
            _,
            start_date,
            end_date,
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
        """Load and return the label DataFrame from CSV file.

        Reads the CSV file with 'datetime' as the index column, parsed as datetime.
        The DataFrame is cached after first access for performance.

        Returns:
            pd.DataFrame: Label DataFrame with DatetimeIndex and columns 'id' (int)
                and 'is_erupted' (int, values 0 or 1).

        Examples:
            >>> label_data = LabelData("label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")
            >>> df = label_data.df
            >>> print(df.index.name)
            'datetime'
            >>> print(df.columns.tolist())
            ['id', 'is_erupted']
            >>> print(df['is_erupted'].unique())
            [0 1]
        """
        logger.debug(f"Loading label data from {self.label_csv}")
        df = pd.read_csv(self.label_csv, index_col="datetime", parse_dates=True)
        logger.debug(f"Loaded {len(df)} label rows")
        return df

    @cached_property
    def filename(self) -> str:
        """Extract the filename from the full path.

        Derives the basename (with extension) from :attr:`label_csv` using
        ``os.path.basename``.

        Returns:
            str: Basename of the label CSV file including extension.

        Examples:
            >>> label_data = LabelData("/path/to/label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")
            >>> label_data.filename
            'label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv'
        """
        return os.path.basename(self.label_csv)

    @cached_property
    def basename(self) -> str:
        """Extract the basename without file extension.

        Strips the ``.csv`` extension from :attr:`filename` using
        ``os.path.splitext``.

        Returns:
            str: Filename without the .csv extension.

        Examples:
            >>> label_data = LabelData("label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")
            >>> label_data.basename
            'label_2020-01-01_2020-12-31_step-12-hours_dtf-2'
        """
        return os.path.splitext(self.filename)[0]

    @cached_property
    def filetype(self) -> str:
        """Extract the file extension without the leading dot.

        Derived from :attr:`filename` using ``os.path.splitext``, with the
        leading dot stripped (e.g., ``".csv"`` → ``"csv"``).

        Returns:
            str: File extension without the leading dot (e.g., 'csv').

        Examples:
            >>> label_data = LabelData("label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")
            >>> label_data.filetype
            'csv'
        """
        return os.path.splitext(self.filename)[1].lstrip(".")

    @cached_property
    def parameters(self) -> dict[str, str | datetime | int]:
        """Extract all parameters from the label filename.

        Assembles the parsed attributes (set during ``__init__``) into a single
        dictionary for convenient downstream access.

        Returns:
            dict[str, str | datetime | int]: Dictionary containing all parsed parameters:
                - start_date (datetime.datetime): Start date object
                - end_date (datetime.datetime): End date object
                - start_date_str (str): Start date in YYYY-MM-DD format
                - end_date_str (str): End date in YYYY-MM-DD format
                - window_step (int): Window step size value
                - window_unit (str): Window step unit ('hours' or 'minutes')
                - day_to_forecast (int): Days before eruption to start labeling

        Examples:
            >>> label_data = LabelData("label_2020-01-01_2020-12-31_step-12-hours_dtf-2.csv")
            >>> params = label_data.parameters
            >>> params['window_step']
            12
            >>> params['window_unit']
            'hours'
            >>> params['day_to_forecast']
            2
            >>> params['start_date_str']
            '2020-01-01'
        """
        parameters: dict[str, str | datetime | int] = {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "start_date_str": self.start_date_str,
            "end_date_str": self.end_date_str,
            "window_step": self.window_step,
            "window_unit": self.window_unit,
            "day_to_forecast": self.day_to_forecast,
        }

        return parameters

    @property
    def data(self) -> pd.DataFrame:
        """Alias for :attr:`df` — satisfies the :class:`BaseDataContainer` interface.

        Returns:
            pd.DataFrame: The label DataFrame.
        """
        return self.df
