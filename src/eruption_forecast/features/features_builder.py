# Standard library imports
from datetime import datetime
from typing import Literal, Self, Union

# Third party imports
import pandas as pd
from loguru import logger

# Project imports
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.label.label_data import LabelData
from eruption_forecast.utils import construct_windows, validate_date_ranges


class FeaturesBuilder:
    """Features builder class

    Args:
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
        self.training_label: LabelData = label_data
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
