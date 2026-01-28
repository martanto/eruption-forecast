# Standard library imports
import glob
import os
import shutil
from datetime import datetime, timedelta
from functools import cached_property
from multiprocessing import Pool
from typing import Literal, Optional, Self, Tuple, Union

# Third party imports
import matplotlib.dates as mdates
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from obspy import Stream, UTCDateTime

# Project imports
import eruption_forecast
from eruption_forecast.dsar import DSAR
from eruption_forecast.rsam import RSAM
from eruption_forecast.sds import SDS
from eruption_forecast.utils import to_datetime


class CalculateTremor:
    """Calculate Tremor Data from seismic data.

    Args:
        station (str): Seismic station code.
        channel (str): Seismic channel code.
        start_date (str): Start date for data processing (YYYY-MM-DD).
        end_date (Optional[str]): End date for data processing (YYYY-MM-DD).
        network (str): Seismic network code. Defaults to "VG".
        location (str): Seismic location code. Defaults to "00".
        methods (Optional[str]): Calculation methods to apply.
        output_dir (str): Directory for output files. Defaults to "output".
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        filename_prefix (Optional[str]): Prefix for generated filenames.
        n_jobs (int): Number of parallel jobs to use. Defaults to 1.
        remove_outliers (bool): If True, removes outliers from the data. Defaults to True.
        interpolate (bool): If True, interpolates the data. Defaults to True.
        value_multiplier (Optional[float]): Scaling factor for seismic values.
        cleanup_tmp_dir (bool): If True, deletes temporary directory after use. Defaults to False.
        plot_tmp (bool): If True, plot temporary results for quick view.
        plot_tremor (bool): If True, plot tremor results for quick view.
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug mode. Defaults to False.
    """

    def __init__(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        station: str,
        channel: str,
        network: str = "VG",
        location: str = "00",
        methods: Optional[str] = None,
        output_dir: str = "output",
        overwrite: bool = False,
        filename_prefix: Optional[str] = None,
        n_jobs: int = 1,
        remove_outliers: bool = True,
        interpolate: bool = True,
        value_multiplier: Optional[float] = None,
        cleanup_tmp_dir: bool = False,
        plot_tmp: bool = False,
        plot_tremor: bool = False,
        overwrite_plot: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        network = network or "VG"
        location = location or "00"
        nslc = f"{network}.{station}.{location}.{channel}"
        output_dir = os.path.join(os.getcwd(), output_dir)
        station_dir = os.path.join(output_dir, nslc)
        forecast_dir = os.path.join(station_dir, "forecast")
        tremor_dir = os.path.join(station_dir, "tremor")
        figures_dir = os.path.join(station_dir, "figures")

        # Set DEFAULT properties
        self.station = station.upper()
        self.channel = channel.upper()
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.network = network or "VG"
        self.location = location or "00"
        self.methods: list[str] = (
            [methods] if isinstance(methods, str) else list(methods or ["rsam", "dsar"])
        )
        self.output_dir: str = output_dir
        self.station_dir: str = station_dir
        self.forecast_dir: str = forecast_dir
        self.tremor_dir: str = tremor_dir
        self.overwrite = overwrite
        self.filename_prefix = filename_prefix
        self.n_jobs = n_jobs
        self.remove_outliers = remove_outliers
        self.interpolate = interpolate
        self.value_multiplier = value_multiplier
        self.cleanup_tmp_dir = cleanup_tmp_dir
        self.plot_tmp = plot_tmp
        self.plot_tremor = plot_tremor
        self.overwrite_plot = overwrite_plot
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.df: pd.DataFrame = pd.DataFrame()
        self.start_date_str: str = start_date.strftime("%Y-%m-%d")
        self.end_date_str: str = end_date.strftime("%Y-%m-%d")
        self.start_date_utc_datetime = UTCDateTime(self.start_date)
        self.end_date_utc_datetime = UTCDateTime(self.end_date)
        self.freq_bands: list[tuple[float, float]] = [
            (0.01, 0.1),
            (0.1, 2),
            (2, 5),
            (4.5, 8),
            (8, 16),
        ]

        self.current_datetime = datetime.now()
        self.dates: pd.DatetimeIndex = pd.date_range(
            start=self.start_date, end=self.end_date
        )
        self.n_days: int = len(self.dates)
        self.nslc = nslc
        self.tmp_dir: str = os.path.join(self.tremor_dir, "tmp")
        self.log_dir = os.path.join(self.station_dir, "logs")
        self.sds: Optional[SDS] = None
        self.tmp_files: list[str] = []
        self.figures_dir = figures_dir
        self.figures_tmp_dir = os.path.join(figures_dir, "tmp")
        self._source: Optional[str] = None
        self._sds_dir: Optional[str] = None
        self._client_url = "https://service.iris.edu"

        # Validate
        self.validate()

        # Verbose and debugging
        if debug:
            logger.info("⚠️ Debug mode is ON")

        if self.verbose:
            logger.info(f"Version: {eruption_forecast.__version__}")
            logger.info(f"Running on {self.n_jobs} job(s)")
            logger.info(f"NSLC: {self.nslc}")
            logger.info(f"Start Date: {self.start_date_str}")
            logger.info(f"End Date: {self.end_date_str}")
            logger.info(f"Total Days: {self.n_days}")
            logger.info(f"Output Directory: {self.output_dir}")
            logger.info(f"Station Directory: {self.station_dir}")
            logger.info(f"Tremor Directory: {self.tremor_dir}")
            logger.info(f"Overwrite: {self.overwrite}")
            logger.info(f"Freq Bands: {self.freq_bands_alias}")

    def change_freq_bands(self, freq_bands: list[tuple[float, float]]) -> Self:
        """Change freq bands default values

        Return:
            Self
        """
        self.freq_bands = freq_bands
        return self

    @property
    def freq_bands_alias(self) -> dict[str, tuple[float, float]]:
        """Freq band with alias.

        Returns:
            dict[str, tuple[float, float]]: Contains band alias and freq min and max.

        Example:
            {
                'f0': (0.01, 0.1),
                'f1': (0.1, 2),
                'f2': (2, 5),
                'f3': (4.5, 8),
                'f4': (8, 16)
            }
        """
        names = {}
        for index, freq_band in enumerate(self.freq_bands):
            names[f"f{index}"] = (freq_band[0], freq_band[1])
        return names

    @cached_property
    def filename(self) -> str:
        default_filename = f"{self.nslc}_{self.start_date_str}_{self.end_date_str}"
        return (
            default_filename
            if self.filename_prefix is None
            else f"{self.filename_prefix}_{default_filename}"
        )

    @cached_property
    def jobs(self) -> list[tuple[int, datetime]]:
        """Generate jobs for multiprocessing

        Returns:
            list[tuple[int, datetime]]: List of job index and datetime
        """
        return [(job_index, date) for job_index, date in enumerate(self.dates)]

    def create_temporary_dir(self) -> Self:
        """Create temporary directory.

        Returns:
            None
        """
        if self.cleanup_tmp_dir:
            if self.verbose:
                logger.info(f"Cleaning up temp dir: {self.tmp_dir}")
            shutil.rmtree(self.tmp_dir)

        os.makedirs(self.tmp_dir, exist_ok=True)
        return self

    @logger.catch
    def create_directories(self) -> None:
        """Create the directories.

        Returns:
            None
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.station_dir, exist_ok=True)
        os.makedirs(self.tremor_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        if self.plot_tmp:
            os.makedirs(self.figures_tmp_dir, exist_ok=True)

        if self.debug:
            os.makedirs(self.log_dir, exist_ok=True)

    def validate(self) -> None:
        """Assert the input parameters.

        Raises:
            ValueError: If start date is after end date
            AssertionError: If window overlap is not between 0 and 1
            AssertionError: If method is not found

        Returns:
            None
        """
        assert (
            self.n_jobs > 0
        ), f"Number of jobs must be greater than 0. Your value: {self.n_jobs}"
        assert (
            self.start_date_utc_datetime < self.end_date_utc_datetime
        ), f"Start date {self.start_date_str} must be before end date {self.end_date_str}"

        for method in self.methods:
            assert (
                method in self.methods
            ), f"Method '{method}' not found. Choose between: {self.methods}"

        self.create_directories()

    def stream(self, date: Optional[datetime] = None) -> Stream:
        """Get the stream for a specific date.

        Args:
            date (datetime, optional): Date. Defaults to None.

        Returns:
            Stream: Stream
        """
        assert date is not None, "Date must be provided"
        assert self._source in ["sds", "fdsn"], (
            f"❌ Please choose a data source. Use `from_sds` or from `from_fdsn` method "
            f"to determine the data source before `run`."
        )

        stream = Stream()

        if self._source == "sds" and self.sds:
            stream = self.sds.get(date)
        if self._source == "fdsn":
            stream = Stream()

        return stream.detrend(type="demean")

    def from_sds(self, sds_dir: str) -> Self:
        """Set the data source to Seiscomp Data Structure (SDS).
        https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html

        Args:
            sds_dir (str): Root SDS directory

        Returns:
            Self
        """
        self._source = "sds"
        self._sds_dir = sds_dir
        self.sds = SDS(
            sds_dir,
            network=self.network,
            station=self.station,
            channel=self.channel,
            location=self.location,
            verbose=self.verbose,
            debug=self.debug,
        )
        return self

    def from_fdsn(self, client_url: Optional[str] = None) -> Self:
        """Set the data source to FDSN.

        Args:
            client_url (Optional[str], optional): Client URL. Defaults to None.

        Returns:
            Self
        """
        self._source = "fdsn"
        self._client_url = client_url or self._client_url
        return self

    def run(self) -> Self:
        """Run the calculation based on n_jobs.

        Returns:
            Self
        """
        self.create_temporary_dir()

        if self.n_jobs == 1:
            for job in self.jobs:
                self.run_job(*job)

        if self.n_jobs > 1:
            if self.verbose:
                logger.info(f"Running on {self.n_jobs} job(s)")

            pool = Pool(self.n_jobs)
            pool.starmap(self.run_job, self.jobs)
            pool.close()
            pool.join()

        # Merge calculated tremor CSV files from tmp dir
        _, df = self.concat_tremor_data(self.tmp_dir, self.tremor_dir)
        self.df = df

        if self.plot_tremor:
            self.plot(df, plot_type="tremor")

        return self

    def run_job(self, job_index: int, date: datetime) -> None:
        """Run a job for a specific date.

        Args:
            job_index (int): Job index
            date (datetime): Date to run the job

        Returns:
            None
        """
        date_str = date.strftime("%Y-%m-%d")
        temp_file = os.path.join(self.tmp_dir, f"{date_str}.csv")

        logger.info(f"Running Jobs ID: {job_index}. Date: {date_str}")

        if not self.overwrite and os.path.exists(temp_file):
            self.tmp_files.append(temp_file)

            if self.verbose:
                logger.info(f"{date_str} :: File CSV loaded {temp_file}")

            return None

        df = self.calculate(date)

        # save tremor data
        df.to_csv(
            temp_file,
            index=True,
            index_label="datetime",
        )

        # plot tremor data
        if self.plot_tmp:
            self.plot(df, plot_type="tmp")

        # save tremor data
        self.tmp_files.append(temp_file)

        if self.verbose:
            logger.info(
                f"{date_str} :: File CSV saved {os.path.join(self.tmp_dir, date_str)}"
            )

        return None

    def calculate(self, date: datetime) -> pd.DataFrame:
        """Calculate tremor data.
        This method calculates the tremor data using Real Seismic Amplitude Measurement (RSAM) and Displacement Seismic Amplitude Ratio (DSAR).

        Args:
            date (datetime): Date to calculate

        Returns:
            pd.DataFrame: Tremor data
        """
        stream = self.stream(date)
        date_str = date.strftime("%Y-%m-%d")

        # Frequency bands
        freq_bands = self.freq_bands_alias

        datetime_index = pd.date_range(
            start=date, end=date + timedelta(days=1), freq="10min", inclusive="left"
        )

        df = pd.DataFrame(index=datetime_index)

        # Initialize DSAR
        dsar = DSAR(
            remove_outliers=self.remove_outliers, verbose=self.verbose, debug=self.debug
        )

        for method in self.methods:
            if method == "rsam":
                for band_name, freq_band in freq_bands.items():
                    column_name = f"rsam_{band_name}"

                    df[column_name] = self.calculate_rsam(
                        stream=stream.copy(),
                        freq_min=freq_band[0],
                        freq_max=freq_band[1],
                    ).values

                    if self.verbose:
                        logger.info(
                            f"{date_str} :: RSAM ({column_name}) calculation finished"
                        )

            if method == "dsar":
                if len(freq_bands) < 2:
                    logger.warning(
                        f"{date_str} :: DSAR needs at least 2 frequencies to calculate. "
                        f"Your current freq_bands are {self.freq_bands}"
                        f"Set using change_freq_bands(). "
                        f"Example: change_freq_bands([(0.01, 0.1), (1.0, 2.0)])"
                    )
                    continue

                # Inetgrating
                stream_integrate = stream.copy().integrate()

                # Anchoring data to start time
                trace = stream_integrate[0]
                trace.data = trace.data - trace.data[0]
                stream_integrate = Stream(trace)

                # Filter and integrate stream based on freq_bands. Then save it to a list
                filtered_streams: list[dict[str, str | Stream]] = []
                for band_name, freq_band in freq_bands.items():
                    if self.debug:
                        logger.info(f"{date_str} :: DSAR Calculating {freq_band}")

                    filter_stream = stream_integrate.copy().filter(
                        "bandpass",
                        freqmin=freq_band[0],
                        freqmax=freq_band[1],
                        corners=4,
                    )

                    filtered_stream_dict: dict[str, str | Stream] = {
                        "band_name": band_name,
                        "filtered_stream": filter_stream,
                    }
                    filtered_streams.append(filtered_stream_dict)

                # Calculate DSAR
                first_dsar = None
                len_freq_bands = len(freq_bands)
                for index, filtered_stream in enumerate(filtered_streams):
                    if index < (len_freq_bands - 1):
                        first_band_name = filtered_stream["band_name"]
                        second_band_name = filtered_streams[index + 1]["band_name"]
                        column_name = f"dsar_{first_band_name}-{second_band_name}"

                        first_stream = filtered_stream["filtered_stream"]
                        if first_dsar is not None:
                            first_stream = first_dsar

                        second_stream = filtered_streams[index + 1]["filtered_stream"]

                        # Use the new DSAR class
                        df[column_name] = dsar.calculate(
                            first_stream=first_stream,
                            second_stream=second_stream,
                            value_multiplier=self.value_multiplier or 1.0,
                        ).values

                        first_dsar = dsar.first_dsar

                        if self.verbose:
                            logger.info(
                                f"{date_str} :: DSAR ({column_name}) calculation finished"
                            )

                        del filtered_stream

        return df

    def calculate_rsam(
        self, stream: Stream, freq_min: float, freq_max: float
    ) -> pd.Series:
        series = (
            RSAM(
                stream=stream,
                verbose=self.verbose,
                debug=self.debug,
            )
            .apply_filter(
                freq_min=freq_min,
                freq_max=freq_max,
            )
            .calculate(
                value_multiplier=self.value_multiplier or 1.0,
                remove_outliers=self.remove_outliers,
                interpolate=True,
            )
        )

        return series

    def plot(
        self,
        df: pd.DataFrame,
        plot_type: Literal["tmp", "tremor"] = "tmp",
    ) -> None:
        """Plot tremor data

        Args:
            df (pd.DataFrame): Tremor data
            plot_type (Literal["tmp", "tremor"]): Type of plot to be saved. Defaults to "tmp".

        Returns:
            None
        """
        overwrite = self.overwrite_plot or self.overwrite
        start_date: pd.Timestamp = df.index[0]
        end_date: pd.Timestamp = df.index[-1]

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # Save plot to tmp directory
        figure_dir = self.figures_tmp_dir if plot_type == "tmp" else self.figures_dir
        figure_name = (
            start_date_str
            if plot_type == "tmp"
            else f"tremor_{start_date_str}_{end_date_str}"
        )
        unix_timestamp = int(datetime.now().timestamp())
        filepath = os.path.join(figure_dir, f"{figure_name}_{unix_timestamp}.png")

        # Define date locator and formatter based on plot type
        date_locator = (
            mdates.HourLocator(interval=2)
            if plot_type == "tmp"
            else mdates.DayLocator(interval=14)
        )
        date_formatter = (
            mdates.DateFormatter("%H:%M")
            if plot_type == "tmp"
            else mdates.DateFormatter("%Y-%m-%d")
        )

        if os.path.exists(filepath) and not overwrite:
            logger.info(f"{start_date_str} :: Plot already exists at {filepath}")
            return

        columns = df.columns
        n_rows = len(columns)
        fig, axs = plt.subplots(
            nrows=n_rows, ncols=1, figsize=(10, 1.2 * n_rows), sharex=True
        )

        for index, column in enumerate(columns):
            ax = axs[index] if n_rows > 1 else axs
            ax.plot(
                df.index,
                df[column],
                color="black",
                linewidth=1,
                label=column,
                alpha=0.8,
            )
            ax.set_xlim(start_date, end_date)
            ax.legend(loc="upper left", fontsize=8, frameon=False)

            ax.xaxis.set_major_locator(date_locator)
            ax.xaxis.set_major_formatter(date_formatter)
            for label in ax.get_xticklabels(which="major"):
                label.set(rotation=30, horizontalalignment="right", fontsize=8)

            if index == (n_rows - 1):
                ax.set_xlabel(
                    start_date_str if plot_type == "tmp" else self.nslc, fontsize=10
                )

        plt.tight_layout()
        plt.savefig(filepath, dpi=100)
        plt.close()

        if self.verbose:
            logger.info(f"{start_date_str} :: Plot saved to {filepath}")

    @staticmethod
    def concat_tremor_data(
        tmp_dir: str, tremor_dir: Optional[str] = None
    ) -> Tuple[str, pd.DataFrame]:
        """Concatenate calculated tremor data from tmp dir to tremor dir.

        Args:
            tmp_dir (str): Temporary dir where calculated tremor saved
            tremor_dir (str, optional): Directory where tremor data will be saved. Defaults to None.

        Returns:
            Tuple[str, pd.DataFrame]: Tremor data location and tremor data
        """
        assert os.path.isdir(tmp_dir), f"Directory {tmp_dir} does not exist"

        files = glob.glob(os.path.join(tmp_dir, "*.csv"))
        assert len(files) > 0, f"File(s) not found in {tmp_dir}"

        tremor_dir = tmp_dir.replace("tmp", "") if tremor_dir is None else tremor_dir
        os.makedirs(tremor_dir, exist_ok=True)

        df = pd.concat(
            [
                pd.read_csv(file, index_col="datetime", parse_dates=True)
                for file in sorted(files)
            ],
            sort=True,
        )

        tremor_filepath = os.path.join(tremor_dir, "tremor.csv")
        df.to_csv(tremor_filepath, index=True)

        logger.info(f"Tremor data saved to {tremor_filepath}")

        return tremor_filepath, df
