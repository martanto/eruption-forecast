"""Container for loaded tremor time-series data.

A lightweight wrapper around a pandas DataFrame of tremor metrics (RSAM, DSAR,
and Shannon Entropy) produced by ``CalculateTremor``. It inherits from
``BaseDataContainer`` and adds sampling-rate validation, start/end date
properties, and support for loading data directly from a CSV file.

Key class:
    - ``TremorData``: Accepts a pre-loaded DataFrame or loads one from a CSV path
      via ``from_csv()``. Validates that the sampling interval matches the expected
      frequency and exposes ``start_date_str`` / ``end_date_str`` for downstream
      pipeline stages.
"""

import os
from typing import Self
from datetime import datetime
from functools import cached_property

import pandas as pd

from eruption_forecast.data_container import BaseDataContainer
from eruption_forecast.config.constants import DEFAULT_SAMPLING_FREQUENCY
from eruption_forecast.utils.validation import check_sampling_consistency


class TremorData(BaseDataContainer):
    """Container for tremor time-series data.

    Wraps a pandas DataFrame of tremor metrics (RSAM, DSAR, and Shannon
    Entropy) and exposes convenience properties for common metadata. Data can
    be supplied directly as a DataFrame or loaded from a CSV file.

    Inherits from :class:`BaseDataContainer`, providing ``csv``, ``start_date_str``,
    ``end_date_str``, and ``data`` as part of the shared data-container interface.

    Attributes:
        csv (str | None): Path to the source CSV file. Set externally after
            :meth:`from_csv` is used; ``None`` until assigned.
        verbose (bool): If True, emit informational log messages.
        debug (bool): If True, emit debug-level log messages.
        _df (pd.DataFrame): The underlying internal tremor DataFrame.

    Args:
        df (pd.DataFrame): Pre-loaded tremor DataFrame with a DatetimeIndex
            and metric columns (``rsam_*``, ``dsar_*``, ``entropy``).
        verbose (bool): If True, emit informational log messages.
            Defaults to False.
        debug (bool): If True, emit debug-level log messages.
            Defaults to False.

    Examples:
        >>> tremor = TremorData(df=my_df)
        >>> tremor = TremorData.from_csv("tremor.csv")
        >>> print(tremor.start_date_str, tremor.end_date_str)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the TremorData container with a pre-loaded DataFrame.

        Stores the provided DataFrame, initialises ``csv`` to ``None`` (via
        ``BaseDataContainer.__init__``), and configures logging flags.

        Args:
            df (pd.DataFrame): Pre-loaded tremor DataFrame with a DatetimeIndex
                and metric columns (``rsam_*``, ``dsar_*``, ``entropy``).
            verbose (bool, optional): Emit progress log messages. Defaults to False.
            debug (bool, optional): Emit debug log messages. Defaults to False.
        """
        super().__init__()
        self.df = df
        self.verbose = verbose
        self.debug = debug

    def __repr__(self) -> str:
        """Return a detailed string representation of this TremorData instance.

        Returns:
            str: A string showing the CSV path, DataFrame shape, and logging flags.
        """
        return (
            f"{self.__class__.__name__}(csv={self.csv}, df={self.df.shape}, "
            f"verbose={self.verbose}, debug={self.debug})"
        )

    @classmethod
    def from_csv(cls, tremor_csv: str) -> Self:
        """Load tremor data from a CSV file and return a new TremorData instance.

        Reads the CSV, parses the ``datetime`` column as the DatetimeIndex, and
        sorts the index in ascending order.

        Args:
            tremor_csv (str): Absolute or relative path to the tremor CSV file.
                Must contain a ``datetime`` column used as the DatetimeIndex.

        Returns:
            Self: A new :class:`TremorData` instance wrapping the loaded
                DataFrame.

        Raises:
            FileNotFoundError: If the file at ``tremor_csv`` does not exist.

        Examples:
            >>> tremor = TremorData.from_csv("output/OJN/tremor/tremor.csv")
        """
        if not os.path.exists(tremor_csv):
            raise FileNotFoundError(f"Tremor CSV file does not exist: {tremor_csv}")

        df = pd.read_csv(tremor_csv, index_col="datetime", parse_dates=True)
        df = df.sort_index()

        tremor_data = cls(df=df)

        return tremor_data

    @cached_property
    def columns(self) -> list[str]:
        """Return the list of DataFrame column names.

        Returns:
            list[str]: Column names, e.g. ``["rsam_f0", "rsam_f1", "dsar_f0-f1", "entropy"]``.
        """
        return self.df.columns.tolist()

    @cached_property
    def start_date(self) -> datetime:
        """Return the first timestamp in the tremor data.

        Returns:
            datetime: Start datetime derived from the first index entry.
        """
        start_date: datetime = self.df.index[0].to_pydatetime()
        return start_date

    @cached_property
    def end_date(self) -> datetime:
        """Return the last timestamp in the tremor data.

        Returns:
            datetime: End datetime derived from the last index entry.
        """
        end_date: datetime = self.df.index[-1].to_pydatetime()
        return end_date

    @cached_property
    def start_date_str(self) -> str:
        """Return the start date as an ISO-format string.

        Returns:
            str: Start date in ``"YYYY-MM-DD"`` format.
        """
        return self.start_date.strftime("%Y-%m-%d")

    @cached_property
    def end_date_str(self) -> str:
        """Return the end date as an ISO-format string.

        Returns:
            str: End date in ``"YYYY-MM-DD"`` format.
        """
        return self.end_date.strftime("%Y-%m-%d")

    @property
    def n_days(self) -> int:
        """Return the number of days spanned by the tremor data.

        Returns:
            int: Integer number of days between start and end date.
        """
        return int((self.end_date - self.start_date).days)

    @property
    def data(self) -> pd.DataFrame:
        """Return the tremor DataFrame, satisfying the :class:`BaseDataContainer` interface.

        Delegates to :attr:`df`.

        Returns:
            pd.DataFrame: The tremor DataFrame.
        """
        return self.df

    def check_consistency(self) -> tuple[bool, pd.DataFrame, pd.DataFrame, int | None]:
        """Check temporal sampling consistency of the tremor data.

        Validates that the DataFrame has a uniform 10-minute sampling
        interval throughout its DatetimeIndex.

        Returns:
            tuple: A 4-element tuple ``(is_consistent, consistent_df,
            inconsistent_df, sampling_rate)`` where:

            - ``is_consistent`` (bool): True if all intervals match the
              expected 10-minute frequency.
            - ``consistent_df`` (pd.DataFrame): Rows with consistent
              timestamps.
            - ``inconsistent_df`` (pd.DataFrame): Rows with unexpected
              gaps or duplicates.
            - ``sampling_rate`` (int | None): Sampling rate in seconds
              (600 for 10-minute data) or None if inconsistencies exist.

        Examples:
            >>> ok, df_ok, df_bad, rate = tremor.check_consistency()
            >>> if not ok:
            ...     print(f"Found {len(df_bad)} inconsistent rows")
        """
        return check_sampling_consistency(
            df=self.df,
            expected_freq=DEFAULT_SAMPLING_FREQUENCY,
            verbose=self.verbose,
        )
