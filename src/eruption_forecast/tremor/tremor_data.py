# Standard library imports
import os
from datetime import datetime
from functools import cached_property
from typing import Optional, Tuple

# Third party imports
import pandas as pd

# Project imports
from eruption_forecast.utils import check_sampling_consistency


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

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(csv={self.csv}, df={self.df}, "
            f"verbose={self.verbose}, debug={self.debug})"
        )

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

    def check_consistency(self) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
        """Check consistency of tremor data.

        Returns:
            bool: True if consistent. False otherwise.
            pd.DataFrame: Conisntency DataFrame with pd.DatetimeIndex.
            pd.DataFrame: Inconisntency DataFrame with pd.DatetimeIndex.
        """
        return check_sampling_consistency(
            df=self.df,
            verbose=self.verbose,
        )
