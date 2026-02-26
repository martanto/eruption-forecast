"""Abstract base class for CSV-backed data containers."""

from abc import ABC, abstractmethod

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
