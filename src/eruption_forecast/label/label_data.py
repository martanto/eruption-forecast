# Standard library imports
import os
from datetime import datetime
from functools import cached_property
from typing import Union

# Third party imports
import pandas as pd

from eruption_forecast.utils import str_to_datetime


class LabelData:
    def __init__(self, label_csv: str):
        self.label_csv = label_csv

        # Example basename: label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2.csv
        # ws-1 -> window_size = 1 (days)
        # step-12-hours -> window_step = 12 (hours)
        # dtf-2 -> day_to_forecast = 2 (days)
        filename = os.path.basename(self.label_csv)
        self.filename = filename
        self.basename = filename.split(".")[0]
        self.filetype = filename.split(".")[1]

        self.validate()

        (
            prefix,
            start_date_str,
            end_date_str,
            window_size,
            window_step_and_unit,
            day_to_forecast,
        ) = self.basename.split("_")

        start_date = str_to_datetime(start_date_str)
        end_date = str_to_datetime(end_date_str)
        window_step_and_unit = window_step_and_unit.split("-")

        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.window_size = int(window_size.split("-")[1])
        self.window_step = int(window_step_and_unit[1])
        self.window_unit = window_step_and_unit[2]
        self.day_to_forecast = int(day_to_forecast.split("-")[1])
        self.parameters: dict[str, Union[str, datetime, int]] = {
            "start_date": start_date,
            "end_date": end_date,
            "start_date_str": start_date_str,
            "end_date_str": end_date_str,
            "window_size": self.window_size,
            "window_step": self.window_step,
            "window_unit": self.window_unit,
            "day_to_forecast": self.day_to_forecast,
        }

    def validate(self) -> None:
        """Validate label filename

        Example: label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2.csv
        """
        example_label_filename = (
            "label_2020-01-01_2020-12-31_ws-1_step-12-hours_dtf-2.csv"
        )

        assert os.path.exists(self.label_csv), ValueError(
            f"Label file not found at {self.label_csv}"
        )
        assert self.basename.startswith("label_"), ValueError(
            f"Label filename is invalid. Filename should start with 'label_'. "
            f"Example: {example_label_filename}"
        )
        assert self.filename.endswith(".csv"), ValueError(
            "Label file extension is invalid"
        )

        parts = self.basename.split("_")
        assert len(parts) == 6, ValueError(
            f"Label filename is invalid. "
            f"Expected format: label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv. "
            f": ws-1 -> window_size = 1 (days)"
            f": step-12-hours -> window_step = 12, window_step_unit = hours"
            f": dtf-2 -> day_to_forecast = 2 (days)"
            f"Got: {self.basename}.csv. Example: {example_label_filename}"
        )

        (
            _,
            start_date,
            end_date,
            window_size,
            window_step,
            day_to_forecast,
        ) = parts

        # Asserting start date and end date
        assert datetime.strptime(start_date, "%Y-%m-%d"), ValueError(
            f"Start date is invalid. Expected format: YYYY-MM-DD. Got: {start_date}"
        )
        assert datetime.strptime(end_date, "%Y-%m-%d"), ValueError(
            f"End date is invalid. Expected format: YYYY-MM-DD. Got: {end_date}"
        )

        # Asserting window size
        assert window_size.startswith("ws-"), ValueError(
            f"Window size is invalid. Expected format: ws-X. Got: ws-{window_size}"
        )
        assert window_size.split("-")[1].isdigit(), ValueError(
            f"Window size should be integer in days. "
            f"Expected format: ws-X where 'X' is an integer. Got: ws-{window_size}"
        )

        # Asserting window step.
        # Expected example: stap-10-minutes or step-12-hours
        # window_steps = ["step", "10", "minutes"]
        window_steps = window_step.split("-")
        starts_with = window_steps[0]
        step = window_steps[1]
        step_unit = window_steps[2]
        assert (
            (len(window_steps) == 3)
            and step.isdigit()
            and (window_step.startswith("step-"))
            and (step_unit in ["hours", "minutes"])
        ), ValueError(
            f"Window step is invalid. Expected format: step-X-(hours/minutes). "
            f"Got: {starts_with}-{step}-{step_unit}. {len(window_steps)}"
        )

        # Asserting day to forecast
        assert day_to_forecast.startswith("dtf-"), ValueError(
            f"Day to forecast is invalid. Expected format: dtf-X. Got: dtf-{day_to_forecast}"
        )
        assert day_to_forecast.split("-")[1].isdigit(), ValueError(
            f"Day to forecast should be integer in days. "
            f"Expected format: dtf-X where 'X' is an integer. Got: dtf-{day_to_forecast}"
        )

    @cached_property
    def df(self) -> pd.DataFrame:
        """Get label dataframe from file"""
        return pd.read_csv(self.label_csv, index_col="datetime", parse_dates=True)
