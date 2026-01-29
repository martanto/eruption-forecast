# Standard library imports
from datetime import datetime
from typing import Union

# Third party imports
import pandas as pd

# Project imports
from eruption_forecast.label.label_data import LabelData
from eruption_forecast.logger import logger
from eruption_forecast.tremor.tremor_data import TremorData


class FeaturesBuilder:
    """Features builder class

    Args:
        tremor_csv (str): Path to tremor CSV file.
        label_csv (str): Path to label CSV file.
        output_dir (str): Output directory path.
        verbose (bool, optional): Verbose mode. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.
    """

    def __init__(
        self,
        tremor_csv: str,
        label_csv: str,
        output_dir: str,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        tremor_data: TremorData = TremorData(tremor_csv=tremor_csv)
        label_data: LabelData = LabelData(label_csv=label_csv)

        # Set DEFAULT properties
        self.tremor_data: TremorData = tremor_data
        self.label_data: LabelData = label_data
        self.output_dir = output_dir
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.tremor_df: pd.DataFrame = tremor_data.df
        self.label_df: pd.DataFrame = label_data.df
        self.label_parameters: dict[str, Union[str, datetime, int]] = (
            label_data.parameters
        )

        # Verbose and debugging
        if self.debug:
            logger.info("⚠️ Debug mode is ON")

        if self.verbose:
            logger.info(f"Test")
