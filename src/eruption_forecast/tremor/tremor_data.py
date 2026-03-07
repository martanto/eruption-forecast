import os
from datetime import datetime
from functools import cached_property

import pandas as pd

from eruption_forecast.data_container import BaseDataContainer
from eruption_forecast.config.constants import DEFAULT_SAMPLING_FREQUENCY
from eruption_forecast.utils.validation import check_sampling_consistency


class TremorData(BaseDataContainer):
    """Container for tremor time-series data.

    Wraps a pandas DataFrame of tremor metrics (RSAM, DSAR) and exposes
    convenience properties for common metadata. Data can be supplied
    directly as a DataFrame or loaded from a CSV file.

    Inherits from :class:`BaseDataContainer`, providing `csv_path`, `start_date_str`,
    `end_date_str`, and `data` as part of the shared data-container interface.

    Attributes:
        csv_path (str): Path to the source CSV file, set by
            :meth:`from_csv`. Empty string until data is loaded.
        verbose (bool): If True, emit informational log messages.
        debug (bool): If True, emit debug-level log messages.
        _df (pd.DataFrame): The underlying internal tremor DataFrame.
        data (pd.DataFrame): Alias for :attr:`df` satisfying the
            :class:`BaseDataContainer` interface.

    Args:
        df (pd.DataFrame | None): Pre-loaded tremor DataFrame
            with a DatetimeIndex. If None, an empty DataFrame is used until
            :meth:`from_csv` is called. Defaults to None.
        verbose (bool): If True, emit informational log messages.
            Defaults to False.
        debug (bool): If True, emit debug-level log messages.
            Defaults to False.

    Examples:
        >>> tremor = TremorData(df=my_df)
        >>> tremor = TremorData(verbose=True).from_csv("tremor.csv")
        >>> print(tremor.start_date_str, tremor.end_date_str)
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize the TremorData container with an optional DataFrame.

        Stores the provided DataFrame (or an empty one when df is None),
        initialises ``csv_path`` to an empty string, and configures logging flags.

        Args:
            df (pd.DataFrame | None, optional): Pre-loaded tremor DataFrame with
                DatetimeIndex and metric columns (rsam_*, dsar_*). Pass None to
                create an empty container and load data later via from_csv().
                Defaults to None.
            verbose (bool, optional): Emit progress log messages. Defaults to False.
            debug (bool, optional): Emit debug log messages. Defaults to False.
        """
        super().__init__(csv="")
        self.verbose = verbose
        self.debug = debug
        self.df = df if df is not None else pd.DataFrame()

    def __repr__(self) -> str:
        """Return a detailed string representation of this TremorData instance.

        Returns:
            str: A string showing the CSV path, DataFrame shape, and logging flags.
        """
        return (
            f"{self.__class__.__name__}(csv_path={self.csv_path}, df={self.df.shape}, "
            f"verbose={self.verbose}, debug={self.debug})"
        )

    def from_csv(self, tremor_csv: str) -> pd.DataFrame:
        """Load tremor data from a CSV file.

        Reads the CSV, parses the ``datetime`` column as the index, and
        sorts the index in ascending order. The CSV path is stored in
        ``self.csv_path`` for later reference.

        Args:
            tremor_csv (str): Absolute or relative path to the tremor CSV
                file. The file must contain a ``datetime`` column that will
                be used as the DatetimeIndex.

        Returns:
            pd.DataFrame: Tremor DataFrame sorted by DatetimeIndex with
                columns such as ``rsam_f0``, ``dsar_f0-f1``, etc.

        Raises:
            FileNotFoundError: If the file at ``tremor_csv`` does not exist.

        Examples:
            >>> tremor = TremorData()
            >>> df = tremor.from_csv("output/OJN/tremor/tremor.csv")
        """
        if not os.path.exists(tremor_csv):
            raise FileNotFoundError(f"Tremor CSV file does not exist: {tremor_csv}")

        df = pd.read_csv(tremor_csv, index_col="datetime", parse_dates=True)
        df = df.sort_index()
        self._df = df
        self.csv = tremor_csv
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Get the tremor DataFrame.

        Returns:
            pd.DataFrame: Tremor DataFrame with a DatetimeIndex and columns
                such as ``rsam_f0``, ``dsar_f0-f1``, etc.

        Raises:
            ValueError: If the DataFrame is empty (no data has been loaded).
        """
        if len(self._df) == 0:
            raise ValueError(
                "Tremor dataframe is empty. Load it using from_csv() or TremorData(df)."
            )
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the tremor DataFrame.

        Args:
            df (pd.DataFrame): Tremor DataFrame to set.
        """
        self._df = df

    @cached_property
    def columns(self) -> list[str]:
        """List of DataFrame column names.

        Returns:
            list[str]: Column names, e.g. ``["rsam_f0", "rsam_f1", "dsar_f0-f1"]``.
        """
        return self.df.columns.tolist()

    @cached_property
    def start_date(self) -> datetime:
        """First timestamp in the tremor data.

        Returns:
            datetime: Start datetime derived from the first index entry.
        """
        start_date: datetime = self.df.index[0].to_pydatetime()
        return start_date

    @cached_property
    def end_date(self) -> datetime:
        """Last timestamp in the tremor data.

        Returns:
            datetime: End datetime derived from the last index entry.
        """
        end_date: datetime = self.df.index[-1].to_pydatetime()
        return end_date

    @cached_property
    def start_date_str(self) -> str:
        """Start date as an ISO-format string.

        Returns:
            str: Start date in ``"YYYY-MM-DD"`` format.
        """
        return self.start_date.strftime("%Y-%m-%d")

    @cached_property
    def end_date_str(self) -> str:
        """End date as an ISO-format string.

        Returns:
            str: End date in ``"YYYY-MM-DD"`` format.
        """
        return self.end_date.strftime("%Y-%m-%d")

    @property
    def n_days(self) -> int:
        """Number of days spanned by the tremor data.

        Returns:
            int: Integer number of days between start and end date.
        """
        return int((self.end_date - self.start_date).days)

    @property
    def data(self) -> pd.DataFrame:
        """Alias for :attr:`df` — satisfies the :class:`BaseDataContainer` interface.

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
