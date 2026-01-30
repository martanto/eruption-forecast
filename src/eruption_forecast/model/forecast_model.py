# Standard library imports
import os
from datetime import datetime
from typing import Literal, Optional, Self, Union

# Project imports
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.logger import logger
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

        # Set DEFAULT properties
        self.station = station
        self.channel = channel
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.volcano_id = volcano_id
        self.network = network
        self.location = location
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.start_date_str = start_date_str
        self.end_date_str = end_date_str
        self.nslc = nslc
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
            "verbose": verbose,
            "debug": debug,
        }

        # Will be set after calculate() method called
        self.calculate_tremor: Optional[CalculateTremor] = None
        self.tremor_data: Optional[TremorData] = None
        self.tremor_csv: Optional[str] = None
        self.station_dir: Optional[str] = None

        # WIll be set after train() method called
        self.label_builder: Optional[LabelBuilder] = None
        self.label_csv: Optional[str] = None

        # Validate
        self.validate()

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

    def validate(self) -> None:
        """Validate properties

        Raises:
            AssertionError: If any of the properties are invalid

        Returns:
            None
        """
        return None

    def load_tremor_data(self, tremor_csv: str) -> Self:
        """Load calculate tremor data from CSV file

        Args:
            tremor_csv (str): Tremor CSV file

        Returns:
            self (Self)
        """
        tremor_data = TremorData().from_csv(tremor_csv)
        self.tremor_data = tremor_data
        return self

    def calculate(
        self,
        source: Literal["sds", "fdsn"] = "sds",
        methods: Optional[str] = None,
        filename_prefix: Optional[str] = None,
        remove_outliers: bool = True,
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
            remove_outliers (bool): If True, removes outliers from the data. Defaults to True.
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
        calculate = CalculateTremor(
            methods=methods,
            filename_prefix=filename_prefix,
            remove_outliers=remove_outliers,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_tmp_dir=cleanup_tmp_dir,
            plot_tmp=plot_tmp,
            save_plot=save_plot,
            overwrite_plot=overwrite_plot,
            **self.kwargs,
        )

        if verbose:
            calculate.verbose = True

        if debug:
            calculate.debug = True

        self.calculate_tremor = calculate

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

        self.tremor_data = TremorData(calculate.df)
        self.tremor_data.csv = calculate.tremor_csv
        self.tremor_csv = calculate.tremor_csv
        self.station_dir = calculate.station_dir

        return self

    def train(
        self,
        window_size: int,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        day_to_forecast: int,
        eruption_dates: list[str],
        output_dir: Optional[str] = None,
    ) -> Self:
        """Build training model.

        Args:
            window_size (int): Window size in days.
            window_step (int): Window step size.
            window_step_unit (Literal["minutes", "hours"]): Unit of window step.
            day_to_forecast (int): Day to forecast in days.
            eruption_dates (list[str]): Eruption dates in YYYY-MM-DD format.
            output_dir (Optional[str], optional): Output directory. Defaults to None.

        Returns:
            self (Self): ForecastModel object
        """

        output_dir = output_dir or self.station_dir

        label_builder = LabelBuilder(
            start_date=self.start_date,
            end_date=self.end_date,
            window_size=window_size,
            window_step=window_step,
            window_step_unit=window_step_unit,
            day_to_forecast=day_to_forecast,
            eruption_dates=eruption_dates,
            volcano_id=self.volcano_id,
            output_dir=output_dir,
            verbose=self.verbose,
            debug=self.debug,
        ).build()

        self.label_builder = label_builder
        self.label_csv = label_builder.csv

        return self

    def predict(self):
        return self
