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

    Args:
        csv (str): Path to the source CSV file.

    Attributes:
        csv (str): Path to the source CSV file.
    """

    def __init__(self, csv: str = "") -> None:
        """Store the CSV path.

        Args:
            csv (str, optional): Path to the source CSV file.
                Defaults to empty string for subclasses that set it later.
        """
        self.csv = csv

    @cached_property
    def filename(self) -> str:
        """Extract the filename from the full CSV path.

        Derives the basename (with extension) from :attr:`csv` using
        ``os.path.basename``.

        Returns:
            str: Basename of the CSV file including extension.
        """
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
        """Start date of the data as an ISO-format string.

        Returns:
            str: Start date in ``"YYYY-MM-DD"`` format.
        """

    @property
    @abstractmethod
    def end_date_str(self) -> str:
        """End date of the data as an ISO-format string.

        Returns:
            str: End date in ``"YYYY-MM-DD"`` format.
        """

    @property
    @abstractmethod
    def data(self) -> pd.DataFrame:
        """The fully loaded data as a DataFrame.

        Returns:
            pd.DataFrame: The loaded DataFrame. Implementation details (index,
                columns) are determined by each concrete subclass.
        """
