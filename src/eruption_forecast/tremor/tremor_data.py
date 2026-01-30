# Standard library imports
import os
from datetime import datetime
from functools import cached_property
from typing import Optional, Tuple

# Third party imports
import numpy as np
import pandas as pd


class TremorData:
    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        self.verbose = verbose
        self.debug = debug
        self.csv: str = None
        self.df = df if df is not None else pd.DataFrame()

    def from_csv(self, tremor_csv: str) -> pd.DataFrame:
        """Load tremor data from csv file

        Args:
            tremor_csv (str): Path to tremor csv file

        Returns:
            self: Return self
        """
        assert os.path.exists(tremor_csv), ValueError(f"{tremor_csv} does not exist")

        df = pd.read_csv(self.csv, index_col="datetime", parse_dates=True)
        df.sort_index(inplace=True)
        self._df = df
        self.csv = tremor_csv
        return df

    @property
    def df(self) -> pd.DataFrame:
        """Load tremor dataframe"""
        assert len(self._df) > 0, ValueError(
            "Tremor dataframe is empty. Load it using from_csv() or TremorData(df)."
        )
        return self._df

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set tremor dataframe"""
        self._df = df

    @cached_property
    def columns(self) -> list[str]:
        """Get column names"""
        return self.df.columns.tolist()

    @cached_property
    def start_date(self) -> datetime:
        """Get start date of tremor data"""
        start_date: datetime = self.df.index[0].to_pydatetime()
        return start_date

    @cached_property
    def end_date(self) -> datetime:
        """Get end date of tremor data"""
        end_date: datetime = self.df.index[-1].to_pydatetime()
        return end_date

    @cached_property
    def start_date_str(self) -> str:
        """Get start date of tremor data as string"""
        return self.start_date.strftime("%Y-%m-%d")

    @cached_property
    def end_date_str(self) -> str:
        """Get end date of tremor data as string"""
        return self.end_date.strftime("%Y-%m-%d")

    @property
    def n_days(self) -> int:
        """Get number of days in tremor data"""
        return int((self.end_date - self.start_date).days)

    def check_sampling_consistency(
        self, tolerance: Optional[float] = 0.001
    ) -> Tuple[bool, int]:
        """Check if the tremor data has consistent sampling periods in seconds

        Args:
            tolerance (optional, float): Tolerance in seconds for considering sampling periods as equal (default: 0.001).

        Returns:
            bool: Return true if sampling period is consistent
            int: Return sampling period in seconds
        """
        df = self.df.copy()

        # Validate input
        assert len(df) > 2, ValueError(
            "DataFrame must have at least 2 rows to check sampling consistency"
        )
        assert isinstance(df.index, pd.DatetimeIndex), ValueError(
            "DataFrame index must be DatetimeIndex"
        )

        time_diffs = pd.Series(df.index).diff().dt.total_seconds()

        # Remove the first NaN value from diff
        time_diffs = pd.Series(time_diffs).dropna()

        expected_period = int(time_diffs.iloc[0])

        if len(time_diffs) == 0:
            return True, expected_period

        # Check if all periods are within tolerance of the expected period
        is_consistent = bool(np.all(np.abs(time_diffs - expected_period) <= tolerance))

        return is_consistent, expected_period
