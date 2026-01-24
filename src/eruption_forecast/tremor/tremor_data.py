# Standard library imports
import os
from datetime import datetime
from functools import cached_property
from typing import Tuple, Union

# Third party imports
import numpy as np
import pandas as pd


class TremorData:
    def __init__(self, tremor_csv: str):
        self.tremor_csv = tremor_csv
        self.validate()

    def validate(self) -> None:
        """Validate tremor data

        Raises:
            ValueError: If tremor data is invalid
        """
        assert os.path.exists(self.tremor_csv), ValueError(
            f"{self.tremor_csv} does not exist"
        )

    @cached_property
    def df(self) -> pd.DataFrame:
        """Get tremor data as pandas DataFrame"""
        df = pd.read_csv(self.tremor_csv, index_col="datetime", parse_dates=True)
        df.sort_index(inplace=True)
        return df

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

    def validate_sampling_rate(self) -> Tuple[bool, int]:
        """Check if the tremor data has consistent sampling periods in seconds

        Returns:
            bool: Return true if sampling period is consistent
            int: Return sampling period in seconds
        """
        df = self.df.copy()
        df["_time"] = df.index

        first_datetime = df.index[0].to_pydatetime()
        second_datetime = df.index[1].to_pydatetime()
        timedelta_seconds = int((second_datetime - first_datetime).seconds)

        # Validate
        check = (
            all(df["_time"].diff()[1:] == np.timedelta64(timedelta_seconds, "s"))
            == True
        )

        if check:
            return True, timedelta_seconds

        return False, timedelta_seconds
