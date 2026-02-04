# Standard library imports
import glob
import os
import shutil
from datetime import datetime, timedelta
from functools import cached_property
from multiprocessing import Pool
from typing import Literal, Self

# Third party imports
import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime

# Project imports
import eruption_forecast
from eruption_forecast.logger import logger
from eruption_forecast.plot import plot_tremor
from eruption_forecast.sds import SDS
from eruption_forecast.tremor.rsam import RSAM
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
        n_jobs (int): Number of parallel jobs to use. Defaults to 1.
        remove_outlier_method (Literal["maximum", "all"], optional): Remove outlier method. Defaults to "maximum".
        interpolate (bool): If True, interpolates the data. Defaults to True.
        value_multiplier (Optional[float]): Scaling factor for seismic values.
        cleanup_tmp_dir (bool): If True, deletes temporary directory after use. Defaults to False.
        plot_tmp (bool): If True, plot temporary results for quick view.
        save_plot (bool): If True, save tremor results for quick view.
        overwrite_plot (bool): If True, overwrite existing plot files. Defaults to False.
        filename_prefix (Optional[str]): Prefix for generated filenames.
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug mode. Defaults to False.
    """

    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        station: str,
        channel: str,
        network: str = "VG",
        location: str = "00",
        methods: str | None = None,
        output_dir: str = "output",
        overwrite: bool = False,
        n_jobs: int = 1,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        interpolate: bool = True,
        value_multiplier: float | None = None,
        cleanup_tmp_dir: bool = False,
        plot_tmp: bool = False,
        save_plot: bool = False,
        overwrite_plot: bool = False,
        filename_prefix: str | None = None,
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
        figures_dir = os.path.join(tremor_dir, "figures")

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
        self.remove_outlier_method = remove_outlier_method
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
        self.tmp_dir: str = os.path.join(tremor_dir, "tmp")
        self.sds: SDS | None = None
        self.tmp_files: list[str] = []
        self.figures_dir = figures_dir
        self._filename = f"{self.nslc}_{self.start_date_str}_{self.end_date_str}.csv"
        self._source: str | None = None
        self._sds_dir: str | None = None
        self._client_url = "https://service.iris.edu"
        self.csv = os.path.join(tremor_dir, self.filename)

        # Validate
        self.validate()

        # Verbose and debugging
        if debug:
            logger.info("⚠️ Calculate Tremor :: Debug mode is ON")

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

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(Version: {eruption_forecast.__version__}). "
            f"Running on {self.n_jobs} job(s) from {self.start_date_str} "
            f"to {self.end_date_str} ({self.n_days} days). Using {self.nslc} "
            f"and frequency bands: {self.freq_bands_alias}. "
            f"Tremor calculation saved to {self.csv}"
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(start_date={self.start_date_str}, "
            f"end_date={self.end_date_str}, station={self.station}, network={self.network}, "
            f"channel={self.channel}, location={self.location}, methods={self.methods}, "
            f"output_dir={self.output_dir}, overwrite={self.overwrite}, n_jobs={self.n_jobs}, "
            f"remove_outlier_method={self.remove_outlier_method}, interpolate={self.interpolate}, "
            f"value_multiplier={self.value_multiplier}, cleanup_tmp_dir={self.cleanup_tmp_dir}, "
            f"plot_tmp={self.plot_tmp}, save_plot={self.save_plot}, overwrite_plot={self.overwrite_plot}, "
            f"csv={self.csv}, verbose={self.verbose}, debug={self.debug})"
        )

    def change_freq_bands(self, freq_bands: list[tuple[float, float]]) -> Self:
        """Change frequency bands default values.

        Args:
            freq_bands (list[tuple[float, float]]): List of frequency band tuples.

        Returns:
            Self: CalculateTremor object.

        Raises:
            TypeError: If freq_bands is not a list or contains non-tuple elements
            ValueError: If frequency values are invalid
        """
        if not isinstance(freq_bands, list):
            raise TypeError(
                "freq_bands must be a list. Example [(0.1,1.0),(1.0,2.5),(2.0,5.0)]"
            )

        for freqs in freq_bands:
            if not isinstance(freqs, tuple):
                raise TypeError(
                    f"Frequencies must be a tuple. Consist of freq minimum and maximum. "
                    f"Example (0.1,1.0). Your values are: {freqs}"
                )
            if len(freqs) != 2:
                raise ValueError(
                    f"Frequencies must have two elements. Example (0.1,1.0). "
                    f"Your values are: {freqs}"
                )

            freq_min, freq_max = freqs
            if not isinstance(freq_min, (float, int)):
                raise TypeError(
                    f"Freq minimum must be float or int. Your value is: {freq_min}"
                )
            if not isinstance(freq_max, (float, int)):
                raise TypeError(
                    f"Freq maximum must be float or int. Your value is: {freq_max}"
                )
            if freq_min >= freq_max:
                raise ValueError(
                    f"Freq minimum must be less than freq maximum. "
                    f"Got: {freq_min} >= {freq_max}"
                )

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

    @property
    def filename(self) -> str:
        """Return defaut filename"""
        # Example: tremor_VG.OJN.00.EHZ_2025-01-01_2025-12-31
        return (
            f"tremor_{self._filename}"
            if self.filename_prefix is None
            else f"{self.filename_prefix}_{self._filename}"
        )

    @filename.setter
    def filename(self, filename: str) -> None:
        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        self._filename = filename

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
            os.makedirs(self.figures_dir, exist_ok=True)

    def validate(self) -> None:
        """Validate input parameters.

        Raises:
            ValueError: If n_jobs is invalid, dates are invalid, or method is not found

        Returns:
            None
        """
        if self.n_jobs <= 0:
            raise ValueError(
                f"Number of jobs must be greater than 0. Your value: {self.n_jobs}"
            )
        if self.start_date_utc_datetime >= self.end_date_utc_datetime:
            raise ValueError(
                f"Start date {self.start_date_str} must be before end date {self.end_date_str}"
            )

        valid_methods = ["rsam", "dsar"]
        for method in self.methods:
            if method not in valid_methods:
                raise ValueError(
                    f"Method '{method}' not found. Choose between: {valid_methods}"
                )

        self.create_directories()

    def get_stream(self, date: datetime | None = None) -> Stream:
        """Get the stream for a specific date.

        Args:
            date (datetime, optional): Date. Defaults to None.

        Returns:
            Stream: Stream

        Raises:
            ValueError: If date is None or data source not set
        """
        if date is None:
            raise ValueError("Date must be provided")
        if self._source not in ["sds", "fdsn"]:
            raise ValueError(
                "Please choose a data source. Use `from_sds()` or `from_fdsn()` method "
                "to determine the data source before `run()`."
            )

        stream = Stream()

        if self._source == "sds" and self.sds:
            stream = self.sds.get(date)
        if self._source == "fdsn":
            stream = Stream()

        return stream

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

    def from_fdsn(self, client_url: str | None = None) -> Self:
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
        csv = self.csv

        if not self.overwrite and os.path.isfile(csv):
            logger.info(f"Load tremor from file: {csv}")
            self.df = pd.read_csv(csv, index_col=0, parse_dates=True)
            return self

        self.create_temporary_dir()

        if self.n_jobs == 1:
            for job in self.jobs:
                temp_file = self.run_job(*job)
                if temp_file is not None:
                    self.tmp_files.append(temp_file)

        if self.n_jobs > 1:
            if self.verbose:
                logger.info(f"Running on {self.n_jobs} job(s)")

            with Pool(self.n_jobs) as pool:
                tmp_files = pool.starmap(self.run_job, self.jobs)
                tmp_files = [tmp_file for tmp_file in tmp_files if tmp_file is not None]
                self.tmp_files.extend(tmp_files)

        # Merge calculated tremor CSV files from tmp dir
        df = self.concat_tremor_data(self.tmp_dir, self.tremor_dir)

        start_date = df.index[0].strftime("%Y-%m-%d")
        end_date = df.index[-1].strftime("%Y-%m-%d")

        # Update filename with latest datetime index from df
        filename = f"tremor_{self.nslc}_{start_date}-{end_date}"
        csv = os.path.join(self.tremor_dir, f"{filename}.csv")

        df.to_csv(csv, index=True)

        logger.info(f"Tremor data saved to {csv}")

        if self.save_plot:
            plot_tremor(
                df=df,
                interval=14,
                interval_unit="days",
                figure_dir=self.tremor_dir,
                filename=filename,
                title=self.nslc,
                overwrite=self.overwrite or self.overwrite_plot,
                verbose=self.verbose,
            )

        self.df = df
        self.filename = filename
        self.csv = csv

        return self

    @logger.catch
    def run_job(self, job_index: int, date: datetime) -> str | None:
        """Run a job for a specific date.

        Args:
            job_index (int): Job index
            date (datetime): Date to run the job

        Returns:
            str: CSV filepath
            None: Not saved, if dataframe is empty
        """
        date_str = date.strftime("%Y-%m-%d")
        temp_file = os.path.join(self.tmp_dir, f"{date_str}.csv")
        temp_plot = os.path.join(self.figures_dir, f"{date_str}.png")

        logger.info(f"Running Jobs ID: {job_index}. Date: {date_str}")

        can_skip = (
            not self.overwrite
            and os.path.exists(temp_file)
            and (not self.plot_tmp and os.path.exists(temp_plot))
        )

        if can_skip:
            if self.verbose:
                logger.info(f"{date_str} :: File CSV loaded {temp_file}")
            return temp_file

        df = self.calculate(date)

        # Pass if df is empty. No data to process
        if df.empty:
            return None

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
                figure_dir=self.figures_dir,
                title=date_str,
                overwrite=self.overwrite,
                verbose=self.verbose,
            )

        if self.verbose:
            logger.info(
                f"{date_str} :: File CSV saved {os.path.join(self.tmp_dir, date_str)}"
            )

        return temp_file

    def calculate(self, date: datetime) -> pd.DataFrame:
        """Calculate tremor data.
        This method calculates the tremor data using Real Seismic Amplitude Measurement (RSAM) and Displacement Seismic Amplitude Ratio (DSAR).

        Args:
            date (datetime): Date to calculate

        Returns:
            pd.DataFrame: Tremor data
        """
        stream = self.get_stream(date)

        # Return empty dataframe if stream is empty
        # (no traces found or miniseed file not exists)
        if len(stream) == 0:
            return pd.DataFrame()

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

        if self.verbose:
            logger.info(f"{date_str} :: Calculation finished.")

        return df

    def calculate_dsar(
        self, date_str: str, df: pd.DataFrame, stream: Stream
    ) -> pd.DataFrame:
        """Calculate Displacement Seismic Amplitude Ratio (DSAR) for a given date.

        DSAR is calculated as the ratio of mean absolute amplitudes between consecutive
        frequency bands after integration of the seismic signal. The integration converts
        velocity to displacement, and ratios are computed between adjacent frequency bands.

        Args:
            date_str (str): Date to calculate DSAR for (YYYY-MM-DD format)
            df (pd.DataFrame): Tremor data with DatetimeIndex
            stream (Stream): ObsPy Stream object containing seismic trace

        Returns:
            pd.DataFrame: DataFrame with DSAR columns added (dsar_f0-f1, dsar_f1-f2, etc.)

        Raises:
            ValueError: If stream is empty or has no traces
            TypeError: If dataframe index is not DatetimeIndex
        """
        # Validate inputs
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be DatetimeIndex")

        if len(stream) == 0:
            raise ValueError(f"{date_str} :: Stream is empty, cannot calculate DSAR")

        # Determine if we need to preserve original stream for RSAM calculation
        needs_original = "rsam" in self.methods and self.methods.index(
            "dsar"
        ) < self.methods.index("rsam")

        # Integrate stream to convert velocity to displacement
        if needs_original:
            stream_integrated = stream.copy().integrate()
        else:
            stream_integrated = stream.integrate()

        # Remove DC component (subtract first value)
        trace: Trace = stream_integrated[0]
        trace.data = trace.data - trace.data[0]
        stream_integrated = Stream(trace)

        # Sequential processing: filter -> calculate -> free -> repeat
        # This approach minimizes memory usage by processing one band at a time
        prev_series: pd.Series | None = None
        prev_band_name: str | None = None
        freq_bands = self.freq_bands_alias

        for band_name, freq_band in freq_bands.items():
            if self.debug:
                logger.debug(
                    f"{date_str} :: DSAR - Processing frequency band {band_name}: {freq_band}"
                )

            # Apply bandpass filter to stream copy
            filtered_stream = stream_integrated.copy().filter(
                "bandpass",
                freqmin=freq_band[0],
                freqmax=freq_band[1],
                corners=4,
            )

            # Extract amplitude series with outlier removal
            current_series = calculate_window_metrics(
                trace=filtered_stream[0],
                window_duration_minutes=10,
                metric_function=np.mean,
                remove_outlier_method=self.remove_outlier_method,
                minimum_completion_ratio=0.3,
                absolute_value=True,
                value_multiplier=1.0,  # Don't multiply yet, do it once on ratio
            )

            # Interpolate missing values
            current_series = current_series.interpolate(method="linear")

            # Free filtered stream immediately to reduce memory footprint
            del filtered_stream

            # Calculate DSAR ratio between consecutive frequency bands
            if prev_series is not None:
                column_name = f"dsar_{prev_band_name}-{band_name}"

                # Calculate ratio: low_freq / high_freq
                # Replace inf and -inf values (from division by zero) with NaN
                dsar_series = prev_series / current_series
                dsar_series = dsar_series.replace([np.inf, -np.inf], np.nan)

                # Apply value multiplier if specified
                if self.value_multiplier and self.value_multiplier > 1:
                    dsar_series = dsar_series * self.value_multiplier

                # Store in dataframe
                df[column_name] = dsar_series.values

                if self.verbose:
                    logger.debug(
                        f"{date_str} :: DSAR ({column_name}) calculation completed."
                    )

            # Store current series for next iteration
            prev_series = current_series
            prev_band_name = band_name

        # Clean up memory
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

        Raises:
            TypeError: If dataframe index is not DatetimeIndex
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index of dataframe should be pd.DatetimeIndex")

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
                    remove_outlier_method=self.remove_outlier_method,
                    interpolate=True,
                )
                .values
            )

            del stream_copy

            if self.verbose:
                logger.debug(
                    f"{date_str} :: RSAM ({column_name}) calculation completed."
                )

        return df

    @staticmethod
    def concat_tremor_data(tmp_dir: str, tremor_dir: str | None = None) -> pd.DataFrame:
        """Concatenate calculated tremor data from tmp dir to tremor dir.

        Args:
            tmp_dir (str): Temporary dir where calculated tremor saved
            tremor_dir (str, optional): Directory where tremor data will be saved. Defaults to None.

        Returns:
            pd.DataFrame: Tremor data

        Raises:
            FileNotFoundError: If tmp_dir doesn't exist or no CSV files found
        """
        if not os.path.isdir(tmp_dir):
            raise FileNotFoundError(f"Directory {tmp_dir} does not exist")

        files = glob.glob(os.path.join(tmp_dir, "*.csv"))
        if len(files) == 0:
            raise FileNotFoundError(f"No CSV files found in {tmp_dir}")

        tremor_dir = tmp_dir.replace("tmp", "") if tremor_dir is None else tremor_dir
        os.makedirs(tremor_dir, exist_ok=True)

        df = pd.concat(
            [
                pd.read_csv(file, index_col="datetime", parse_dates=True)
                for file in sorted(files)
            ],
            sort=True,
        )

        return df.sort_index()
