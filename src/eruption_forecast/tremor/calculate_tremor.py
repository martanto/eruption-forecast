# Standard library imports
import glob
import os
import shutil
from datetime import datetime, timedelta
from functools import cached_property, lru_cache
from multiprocessing import Pool
from typing import Optional, Self, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime

# Project imports
import eruption_forecast
from eruption_forecast.logger import logger
from eruption_forecast.plot import plot_tremor
from eruption_forecast.rsam import RSAM
from eruption_forecast.sds import SDS
from eruption_forecast.utils import calculate_window_metrics, to_datetime


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
        save_plot (bool): If True, save tremor results for quick view.
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
        save_plot: bool = False,
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

        # TODO: Add shanon entropy, and kurtosis
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
        self.save_plot = save_plot
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
        assert isinstance(freq_bands, list), ValueError(
            "freq_bands must be a list. Example [(0.1,1.0),(1.0,2.5),(2.0,5.0)]"
        )

        for freqs in freq_bands:
            assert isinstance(freqs, tuple), ValueError(
                f"Frequencies must be a tuple. Consist of freq minimum and maximum. "
                f"Example (0.1,1.0). Your values are: {freqs}"
            )
            assert len(freqs) == 2, ValueError(
                f"Frequencies must have two elements. Example (0.1,1.0)."
                f"Your values are: {freqs}"
            )

            freq_min, freq_max = freqs
            assert isinstance(freq_min, float) or isinstance(freq_min, int), ValueError(
                f"Freq minimum must be float or int. Your value is: {freq_min}"
            )
            assert isinstance(freq_max, float) or isinstance(freq_max, int), ValueError(
                f"Freq maximum must be float or int. Your value is: {freq_max}"
            )
            assert freq_min < freq_max, ValueError()

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

    def get_stream(self, date: Optional[datetime] = None) -> Stream:
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

        if self.save_plot:
            plot_tremor(
                df=df,
                interval=14,
                interval_unit="days",
                figure_dir=self.figures_dir,
                title=self.nslc,
                overwrite=self.overwrite,
                verbose=self.verbose,
            )

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
            plot_tremor(
                df=df,
                interval=2,
                interval_unit="hours",
                filename=f"{date_str}.png",
                figure_dir=self.figures_tmp_dir,
                title=date_str,
                overwrite=self.overwrite,
                verbose=self.verbose,
            )

        # save tremor data
        self.tmp_files.append(temp_file)

        if self.verbose:
            logger.info(
                f"{date_str} :: File CSV saved {os.path.join(self.tmp_dir, date_str)}"
            )

        return None

    @lru_cache(maxsize=128)
    def calculate(self, date: datetime) -> pd.DataFrame:
        """Calculate tremor data.
        This method calculates the tremor data using Real Seismic Amplitude Measurement (RSAM) and Displacement Seismic Amplitude Ratio (DSAR).

        Args:
            date (datetime): Date to calculate

        Returns:
            pd.DataFrame: Tremor data
        """
        stream = self.get_stream(date)
        date_str = date.strftime("%Y-%m-%d")

        # Build tremor DataFrame index
        datetime_index = pd.date_range(
            start=date, end=date + timedelta(days=1), freq="10min", inclusive="left"
        )

        # Init tremor DataFrame without tremor values
        df = pd.DataFrame(index=datetime_index)

        for method in self.methods:
            if method == "rsam":
                df = self.calculate_rsam(date_str, df, stream)

            if method == "dsar":
                if len(self.freq_bands_alias) < 2:
                    logger.warning(
                        f"{date_str} :: DSAR needs at least 2 frequencies to calculate. "
                        f"Your current freq_bands are {self.freq_bands}"
                        f"Set using change_freq_bands(). "
                        f"Example: change_freq_bands([(0.01, 0.1), (1.0, 2.0)])"
                    )
                    continue
                df = self.calculate_dsar(date_str, df, stream)

        return df

    def calculate_dsar(
        self, date_str: str, df: pd.DataFrame, stream: Stream
    ) -> pd.DataFrame:
        """Calculate DSAR for a given date

        Args:
            date_str (str): Date to calculate DSAR for
            df (pd.DataFrame): Tremor data with datetime index
            stream (Stream): Obspy Stream object

        Returns:
            pd.DataFrame: DSAR data
        """
        needs_original = "rsam" in self.methods and self.methods.index(
            "dsar"
        ) < self.methods.index("rsam")

        if needs_original:
            stream_integrated = stream.copy().integrate()
        else:
            stream_integrated = stream.integrate()

        trace: Trace = stream_integrated[0]
        trace.data = trace.data - trace.data[0]
        stream_integrated = Stream(trace)

        # Sequential processing: filter -> calculate -> free -> repeat
        prev_series = None
        prev_band_name = None
        freq_bands = self.freq_bands_alias

        for band_name, freq_band in freq_bands.items():
            if self.debug:
                logger.info(f"{date_str} :: DSAR Calculating {freq_band}")

            # Filter a copy
            filtered_stream = stream_integrated.copy().filter(
                "bandpass",
                freqmin=freq_band[0],
                freqmax=freq_band[1],
                corners=4,
            )

            # Extract the amplitude series immediately
            current_series = calculate_window_metrics(
                trace=filtered_stream[0],
                window_duration_minutes=10,
                metric_function=np.mean,
                remove_outliers=self.remove_outliers,
                minimum_completion_ratio=0.3,
                absolute_value=True,
            )

            # Free the filtered stream immediately
            del filtered_stream

            # Calculate DSAR ratio if we have previous series
            if prev_series is not None:
                column_name = f"dsar_{prev_band_name}-{band_name}"
                dsar_series = prev_series / current_series

                if self.value_multiplier and self.value_multiplier > 1:
                    dsar_series = dsar_series * self.value_multiplier

                df[column_name] = dsar_series.values

                if self.verbose:
                    logger.info(
                        f"{date_str} :: DSAR ({column_name}) calculation finished"
                    )

            # Store current for next iteration (only Series, not full Stream)
            prev_series = current_series
            prev_band_name = band_name

        # Clean up
        del prev_series, stream_integrated
        return df

    def calculate_rsam(
        self, date_str: str, df: pd.DataFrame, stream: Stream
    ) -> pd.DataFrame:
        """Calculate RSAM for a given date

        Args:
            date_str (str): Date to calculate RSAM for
            df (pd.DataFrame): Tremor data with datetime index
            stream (Stream): Obspy Stream object

        Returns:
            pd.DataFrame: Tremor data
        """
        assert isinstance(df.index, pd.DatetimeIndex), ValueError(
            f"Index of dataframe should be pd.DatetimeIndex"
        )

        freq_bands = self.freq_bands_alias

        for band_name, freq_band in freq_bands.items():
            column_name = f"rsam_{band_name}"

            stream_copy = stream.copy()

            df[column_name] = (
                RSAM(
                    stream=stream_copy,
                    verbose=self.verbose,
                    debug=self.debug,
                )
                .apply_filter(
                    freq_min=freq_band[0],
                    freq_max=freq_band[1],
                )
                .calculate(
                    value_multiplier=self.value_multiplier or 1.0,
                    remove_outliers=self.remove_outliers,
                    interpolate=True,
                )
                .values
            )

            del stream_copy

            if self.verbose:
                logger.info(f"{date_str} :: RSAM ({column_name}) calculation finished")

        return df

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
