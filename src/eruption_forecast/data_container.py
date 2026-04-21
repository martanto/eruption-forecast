"""Abstract base class for CSV-backed data containers."""

import os
from abc import ABC, abstractmethod
from functools import cached_property

import pandas as pd


class BaseDataContainer(ABC):
    """Abstract base for data containers backed by a CSV file.

    Defines the common interface shared by :class:`TremorData` and
    :class:`LabelData`: a CSV path, date-range string properties, and a
    DataFrame accessor.

    Attributes:
        csv (str | None): Path to the source CSV file, or ``None`` until
            assigned by a subclass.
    """

    def __init__(self) -> None:
        """Initialize the base data container with a null CSV path.

        Sets ``csv`` to ``None``; subclasses assign a valid path before any
        path-derived properties (``filename``, ``basename``, ``filetype``)
        are accessed.
        """
        self.csv: str | None = None

    @cached_property
    def filename(self) -> str:
        """Extract the filename from the full CSV path.

        Derives the basename (with extension) from :attr:`csv` using
        ``os.path.basename``.

        Returns:
            str: Basename of the CSV file including extension.
        """
        if self.csv is None:
            raise ValueError("No CSV path provided.")
        return os.path.basename(self.csv)

    @cached_property
    def basename(self) -> str:
        """Extract the filename without file extension.

        Strips the extension from :attr:`filename` using
        ``os.path.splitext``.

        Returns:
            str: Filename without the extension.
        """
        return os.path.splitext(self.filename)[0]

    @cached_property
    def filetype(self) -> str:
        """Extract the file extension without the leading dot.

        Derived from :attr:`filename` using ``os.path.splitext``, with the
        leading dot stripped (e.g., ``".csv"`` → ``"csv"``).

        Returns:
            str: File extension without the leading dot (e.g., 'csv').
        """
        return os.path.splitext(self.filename)[1].lstrip(".")

    @property
    @abstractmethod
    def start_date_str(self) -> str:
        """Return the start date of the data as an ISO-format string.

        Derived from the first timestamp in the underlying data source.

        Returns:
            str: Start date in ``"YYYY-MM-DD"`` format.
        """

    @property
    @abstractmethod
    def end_date_str(self) -> str:
        """Return the end date of the data as an ISO-format string.

        Derived from the last timestamp in the underlying data source.

        Returns:
            str: End date in ``"YYYY-MM-DD"`` format.
        """

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """Return the fully loaded data as a DataFrame.

        Index type and column names are determined by each concrete subclass.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
