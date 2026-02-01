# Standard library imports
import os
from datetime import datetime, timedelta
from typing import Literal, Optional, Self, Union

import pandas as pd

# Project imports
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.logger import logger
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.utils import to_datetime


class ForecastModel:
    """Create forecast model from seismic data.

    CalculateTremor: Calculate Tremor data from seismic data.
    Build Label and extract features for training.
    Predict based on training data.

    Args:
        station (str): Seismic station code.
        channel (str): Seismic channel code.
        start_date (str | datetime): Start date in YYYY-MM-DD format.
        end_date (str | datetime): End date in YYYY-MM-DD format.
        window_size (int): Window size in days. Used to create label and training data.
        volcano_id (str): Volcano ID. To set and forecast ID.
        network (str): Seismic network code. Defaults to "VG".
        location (str): Seismic location code. Defaults to "00".
        output_dir (str): Directory for output files. Defaults to "output".
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        n_jobs (int): Number of parallel jobs to use. Defaults to 1.
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug mode. Defaults to False.
    """

    def __init__(
        self,
        station: str,
        channel: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        window_size: int,
        volcano_id: str,
        network: str = "VG",
        location: str = "00",
        output_dir: Optional[str] = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        start_date = to_datetime(start_date).replace(hour=0, minute=0, second=0)
        end_date = to_datetime(end_date).replace(hour=23, minute=59, second=59)
        start_date_str: str = start_date.strftime("%Y-%m-%d")
        end_date_str: str = end_date.strftime("%Y-%m-%d")
        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        network = network or "VG"
        location = location or "00"
        nslc = f"{network}.{station}.{location}.{channel}"
        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        station_dir = os.path.join(output_dir, nslc)
        training_dir = os.path.join(station_dir, "training")

        # Set DEFAULT properties
        self.station = station
        self.channel = channel
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.window_size: int = window_size
        self.volcano_id = volcano_id
        self.network = network
        self.location = location
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.start_date_minus_window_size = start_date - timedelta(days=window_size)
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.nslc = nslc
        self.station_dir = station_dir
        self.training_dir = training_dir
        self.kwargs = {
            "station": station,
            "channel": channel,
            "network": network,
            "location": location,
            "start_date": start_date,
            "end_date": end_date,
            "output_dir": output_dir,
            "overwrite": overwrite,
            "n_jobs": n_jobs,
        }

        # Will be set after calculate() method called
        self.CalculateTremor: Optional[CalculateTremor] = None
        self.TremorData: Optional[TremorData] = None
        self.tremor_data: pd.DataFrame = None
        self.tremor_csv: Optional[str] = None

        # WIll be set after train() method called
        self.LabelBuilder: Optional[LabelBuilder] = None
        self.label_data: pd.DataFrame = None
        self.label_csv: Optional[str] = None

        # Base filename without extension
        self.basename: Optional[str] = None
        self._id = f"{start_date_str}_{end_date_str}_{nslc}_ws-{window_size}"

        # Validate

        # Verbose and debugging
        if debug:
            logger.info("⚠️ Forecast Model :: Debug mode is ON")

        if verbose:
            logger.info(f"Start Date (YYYY-MM-DD): {start_date_str}")
            logger.info(f"End Date (YYYY-MM-DD): {end_date_str}")
            logger.info(f"Volcano ID: {self.volcano_id}")
            logger.info(f"Network: {self.network}")
            logger.info(f"Station: {self.station}")
            logger.info(f"Location: {self.location}")
            logger.info(f"Channel: {self.channel}")
            logger.info(f"Output Dir: {self.output_dir}")

    def load_tremor_data(self, tremor_csv: str) -> Self:
        """Load calculate tremor data from CSV file

        Args:
            tremor_csv (str): Tremor CSV file

        Returns:
            self (Self)
        """
        tremor_data = TremorData()
        self.TremorData = TremorData()
        self.TremorData.csv = tremor_csv
        self.tremor_data = tremor_data.from_csv(tremor_csv)
        self.tremor_csv = tremor_csv
        return self

    def calculate(
        self,
        source: Literal["sds", "fdsn"] = "sds",
        methods: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        interpolate: bool = True,
        value_multiplier: Optional[float] = None,
        cleanup_tmp_dir: bool = False,
        plot_tmp: bool = True,
        save_plot: bool = True,
        overwrite_plot: bool = False,
        sds_dir: Optional[str] = None,
        client_url: str = "https://service.iris.edu",
        verbose: Optional[bool] = None,
        debug: Optional[bool] = None,
    ) -> Self:
        """Calculate Tremor Data

        Args:
            source (optional, Literal["sds", "fdsn"]): Seismic data source
            methods (Optional[str]): Calculation methods to apply.
            filename_prefix (Optional[str]): Prefix for generated filenames.
            remove_outlier_method (bool): If True, removes outliers from the data. Defaults to True.
            interpolate (bool): If True, interpolates the data. Defaults to True.
            value_multiplier (Optional[float]): Scaling factor for seismic values.
            cleanup_tmp_dir (bool): If True, deletes temporary directory after use. Defaults to False.
            plot_tmp (bool): If True, plot temporary results for quick view.
            save_plot (bool): If True, save tremor results for quick view.
            overwrite_plot (bool): If True, overwrite existing plot files. Defaults to False.
            sds_dir (str): SDS directory location. Must be provided if source is 'sds'.
            client_url (str): URL to FDSN service. Default to https://service.iris.edu
            verbose (bool): If True, enables verbose logging. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.

        Returns:
            self (Self): ForecastModel object
        """
        verbose = verbose or self.verbose
        debug = debug or self.debug

        # Update "start_date" key arguments
        kwargs = self.kwargs
        kwargs["start_date"] = self.start_date_minus_window_size

        calculate = CalculateTremor(
            methods=methods,
            filename_prefix=filename_prefix,
            remove_outlier_method=remove_outlier_method,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_tmp_dir=cleanup_tmp_dir,
            plot_tmp=plot_tmp,
            save_plot=save_plot,
            overwrite_plot=overwrite_plot,
            **kwargs,
        )

        if verbose:
            calculate.verbose = True

        if debug:
            calculate.debug = True

        self.CalculateTremor = calculate

        if source == "sds":
            assert sds_dir is not None, ValueError(
                f"You choose 'sds' as source please provid 'sds_dir' paramater."
                f"Example: calculate(source='sds', sds_dir='converted')"
            )
            assert os.path.isdir(sds_dir), ValueError(f"SDS dir {sds_dir} not exists.")

            calculate = calculate.from_sds(sds_dir=sds_dir).run()

        # TODO: Get data from FDSN services
        if source == "fdsn":
            # calculate = calculate.from_fdsn(client_url=client_url).run()
            logger.error(f"FDSN source is not yet supported. Client url: {client_url}")
            raise

        tremor_data = TremorData(calculate.df)
        df_tremor = tremor_data.df
        self.TremorData = tremor_data
        self.TremorData.csv = calculate.csv
        self.tremor_csv = calculate.csv

        # Update self.start_date and self.end_date based on calculated tremor data
        if self.start_date_minus_window_size < tremor_data.start_date:
            self.start_date_minus_window_size = tremor_data.start_date
            if self.verbose:
                logger.info(
                    f"start_date_minus_window_size parameter: {self.start_date_minus_window_size} updated to "
                    f"tremor start date: {tremor_data.start_date}"
                )

        if self.end_date > tremor_data.end_date:
            self.end_date = tremor_data.end_date
            if self.verbose:
                logger.info(
                    f"start_date parameter: {self.end_date} updated to "
                    f"tremor end date: {tremor_data.end_date}"
                )

        # Update tremor data based on update self.start_date or self.end_date
        self.tremor_data = df_tremor.loc[
            self.start_date_minus_window_size : self.end_date
        ]

        return self

    def train(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        output_dir: Optional[str] = None,
        tremor_columns: Optional[list[str]] = None,
        verbose: Optional[bool] = None,
        debug: Optional[bool] = None,
    ) -> Self:
        """Build training model.

        Args:
            window_step (int): Window step size.
            window_step_unit (Literal["minutes", "hours"]): Unit of window step.
            day_to_forecast (int): Day to forecast in days.
            eruption_dates (list[str]): Eruption dates in YYYY-MM-DD format.
            output_dir (Optional[str], optional): Output directory. Defaults to None.
            tremor_columns (Optional[list[str]], optional): Columns to select. Defaults to None.
            verbose (bool): If True, enables verbose logging. Defaults to False.
            debug (bool): If True, enables debug mode. Defaults to False.

        Returns:
            self (Self): ForecastModel object
        """
        verbose = verbose or self.verbose
        debug = debug or self.debug

        output_dir = output_dir or self.station_dir

        label_builder = LabelBuilder(
            start_date=self.start_date,
            end_date=self.end_date,
            window_size=self.window_size,
            window_step=window_step,
            window_step_unit=window_step_unit,
            day_to_forecast=day_to_forecast,
            eruption_dates=eruption_dates,
            volcano_id=self.volcano_id,
            output_dir=output_dir,
            verbose=verbose,
            debug=debug,
        ).build()

        df_label = label_builder.df

        self.LabelBuilder = label_builder
        self.label_csv = label_builder.csv

        # Build output directory
        basename = os.path.basename(label_builder.csv).split(".csv")[0]
        self.basename = basename

        # Set training and features matrix directory
        training_label_dir = os.path.join(self.training_dir, basename)
        features_matrix_dir = os.path.join(training_label_dir, "features_matrix")

        os.makedirs(training_label_dir, exist_ok=True)
        os.makedirs(features_matrix_dir, exist_ok=True)

        # Load tremor and select specific columns
        df_tremor = self.tremor_data.copy()
        if tremor_columns is not None:
            for column in tremor_columns:
                assert column in df_tremor.columns, ValueError(
                    f"Column {column} not exists in tremor data"
                )
            df_tremor = df_tremor[tremor_columns]
        df_tremor.sort_index(ascending=True, inplace=True)

        # Ensuring tremor data is within label data range
        tremor_start_date_obj: pd.Timestamp = df_tremor.index[0]
        tremor_end_date_obj: pd.Timestamp = df_tremor.index[-1]

        label_start_date_obj: pd.Timestamp = df_label.index[0]
        label_end_date_obj: pd.Timestamp = df_label.index[-1]

        assert tremor_start_date_obj <= label_start_date_obj, ValueError(
            f"Training start date ({tremor_start_date_obj}) should be after/equal "
            f"to tremor start date ({label_start_date_obj}). "
            f"Change your training start date after/equal {tremor_start_date_obj}."
        )
        assert tremor_end_date_obj >= label_end_date_obj, ValueError(
            f"Training end date ({tremor_end_date_obj}) should be before/equal "
            f"to tremor end date ({label_start_date_obj}). "
            f"Change your training end date before/equal {tremor_end_date_obj}."
        )

        # Omitting label data (df) based on window step
        df_label = df_label.loc[self.start_date :]
        self.label_data = df_label

        return self

    def predict(self):
        return self
