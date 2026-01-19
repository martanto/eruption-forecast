# Standard library imports
import os
from datetime import datetime
from functools import cached_property
from typing import Self

# Third party imports
import pandas as pd


class FeaturesBuilder:
    def __init__(
        self,
        tremor_csv: str,
        label_csv: str,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.tremor_csv = tremor_csv
        self.verbose = verbose
        self.debug = debug

        self.label_data: LabelData = LabelData(label_csv)

        if self.debug:
            logger.info("⚠️ Debug mode is ON")

        if self.verbose:
            logger.info(f"Tremor File: {self.tremor_csv}")
            logger.info(f"Label File: {self.label_csv}")

    @cached_property
    def df_tremor(self) -> pd.DataFrame:
        """Get tremor dataframe from file"""
        assert os.path.exists(self.tremor_csv), "Tremor file not found"
        return pd.read_csv(self.tremor_csv, index_col="datetime", parse_dates=True)

    @cached_property
    def df_label(self) -> pd.DataFrame:
        """Get label dataframe from file"""
        assert os.path.exists(self.label_csv), "Label file not found"
        return pd.read_csv(self.label_csv, index_col="datetime", parse_dates=True)

    def build(self) -> Self:
        """Build features from tremor and label data"""

        return self
