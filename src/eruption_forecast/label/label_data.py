# Standard library imports
import os
from datetime import datetime
from functools import cached_property

# Third party imports
import pandas as pd


class LabelData:
    def __init__(self, label_csv: str):
        self.label_csv = label_csv

        # Example basename: label_2020-01-01_2020-12-31_ws-1_step-12_sr-100_dtf-1.csv
        # ws-1 -> window_size = 1 (days)
        # step-12 -> window_step = 12 (hours)
        # sr-100 -> sampling_rate = 100 (Hz)
        # dtf-1 -> day_to_forecast = 1 (days)
        self.basename = os.path.basename(self.label_csv).split(".")[0]

        self.validate()

        (
            _,
            start_date,
            end_date,
            window_size,
            window_step,
            sampling_rate,
            day_to_forecast,
        ) = self.basename.split("_")

        self.start_date = str(start_date)
        self.end_date = str(end_date)
        self.start_date_obj = datetime.strptime(self.start_date, "%Y-%m-%d")
        self.end_date_obj = datetime.strptime(self.end_date, "%Y-%m-%d")
        self.window_size = int(window_size.split("-")[0])
        self.window_step = float(window_step.split("-")[0])
        self.sampling_rate = float(sampling_rate.split("-")[0])
        self.day_to_forecast = int(day_to_forecast.split("-")[0])

    def validate(self) -> None:
        """Validate label filename

        Example: label_2020-01-01_2020-12-31_ws-1_step-12_sr-100_dtf-1.csv
        """
        example_label_filename = (
            "label_2020-01-01_2020-12-31_ws-1_step-12_sr-100_dtf-1.csv"
        )

        assert os.path.exists(
            self.label_csv
        ), f"Label file not found at {self.label_csv}"
        assert self.basename.startswith(
            "label_"
        ), f"Label filename is invalid. Filename should start with 'label_'. Example: {example_label_filename}"
        assert self.basename.endswith(".csv"), "Label file extension is invalid"

        parts = self.basename.split("_")
        assert len(parts) == 7, (
            f"Label filename is invalid. Expected format: label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X_sr-X_dtf-X.csv. "
            f"Got: {self.basename}.csv. Example: {example_label_filename}"
        )

        (
            _,
            start_date,
            end_date,
            window_size,
            window_step,
            sampling_rate,
            day_to_forecast,
        ) = parts

        # Asserting start date and end date
        assert datetime.strptime(
            start_date, "%Y-%m-%d"
        ), f"Start date is invalid. Expected format: YYYY-MM-DD. Got: {start_date}"
        assert datetime.strptime(
            end_date, "%Y-%m-%d"
        ), f"End date is invalid. Expected format: YYYY-MM-DD. Got: {end_date}"

        # Asserting window size
        assert window_size.startswith(
            "ws-"
        ), f"Window size is invalid. Expected format: ws-X. Got: ws-{window_size}"
        assert window_size.split("-")[
            1
        ].isdigit(), f"Window size should be integer in days. Expected format: ws-X where 'X' is an integer. Got: ws-{window_size}"

        # Asserting window step
        assert window_step.startswith(
            "step-"
        ), f"Window step is invalid. Expected format: step-X. Got: step-{window_step}"
        assert window_step.split("-")[
            1
        ].isdigit(), f"Window step should be integer in hours. Expected format: step-X where 'X' is an integer. Got: step-{window_step}"

        # Asserting sampling rate
        assert sampling_rate.startswith(
            "sr-"
        ), f"Sampling rate is invalid. Expected format: sr-X. Got: sr-{sampling_rate}"
        assert sampling_rate.split("-")[
            1
        ].isdigit(), f"Sampling rate should be float in Hz. Expected format: sr-X where 'X' is a float or integer. Got: sr-{sampling_rate}"

        # Asserting day to forecast
        assert day_to_forecast.startswith(
            "dtf-"
        ), f"Day to forecast is invalid. Expected format: dtf-X. Got: dtf-{day_to_forecast}"
        assert day_to_forecast.split("-")[
            1
        ].isdigit(), f"Day to forecast should be integer in days. Expected format: dtf-X where 'X' is an integer. Got: dtf-{day_to_forecast}"

    @cached_property
    def df(self) -> pd.DataFrame:
        """Get label dataframe from file"""
        df = pd.read_csv(self.label_csv, index_col="datetime", parse_dates=True)
        assert not df.empty, "Label dataframe is empty"
        return df
