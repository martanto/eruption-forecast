import os
import glob
import shutil
from typing import Self, Literal
from datetime import datetime, timedelta
from functools import cached_property
from multiprocessing import Pool

import numpy as np
import pandas as pd
from obspy import Trace, Stream, UTCDateTime

import eruption_forecast
from eruption_forecast.logger import logger
from eruption_forecast.sources.sds import SDS
from eruption_forecast.tremor.rsam import RSAM
from eruption_forecast.sources.fdsn import FDSN
from eruption_forecast.utils.window import calculate_window_metrics
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.utils.date_utils import to_datetime
from eruption_forecast.plots.tremor_plots import plot_tremor
from eruption_forecast.tremor.shanon_entropy import ShanonEntropy


class CalculateTremor:
    """Calculate tremor metrics (RSAM and DSAR) from raw seismic data.

    Processes seismic waveform data from SDS archives or FDSN web services,
    computing Real-time Seismic Amplitude Measurement (RSAM) and Displacement
    Seismic Amplitude Ratio (DSAR) across configurable frequency bands at
    10-minute sampling intervals.

    Supports multiprocessing for parallel daily calculations and produces
    time-series CSV files suitable for downstream feature extraction and
    eruption forecasting.

    Default frequency bands: (0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz

    Attributes:
        station (str): Seismic station code in uppercase.
        channel (str): Seismic channel code in uppercase.
        start_date (datetime): Start date for data processing.
        end_date (datetime): End date for data processing.
        network (str): Seismic network code.
        location (str): Seismic location code.
        methods (list[str]): Calculation methods to apply (rsam, dsar).
        output_dir (str): Root output directory.
        station_dir (str): Station-specific output directory.
        forecast_dir (str): Forecast output directory.
        tremor_dir (str): Tremor data output directory.
        overwrite (bool): Whether to overwrite existing files.
        filename_prefix (str | None): Custom prefix for output filenames.
        n_jobs (int): Number of parallel jobs for daily processing.
        remove_outlier_method (Literal["maximum", "all"]): Outlier removal strategy.
        interpolate (bool): Whether to interpolate gaps in data.
        value_multiplier (float | None): Scaling factor for seismic values.
        cleanup_daily_dir (bool): Whether to delete daily directory after merging.
        plot_daily (bool): Whether to plot intermediate daily results.
        save_plot (bool): Whether to save final tremor plot to disk.
        overwrite_plot (bool): Whether to overwrite existing plot files.
        verbose (bool): If True, enables verbose logging.
        debug (bool): If True, enables debug-level logging.
        df (pd.DataFrame): Calculated tremor DataFrame.
        start_date_str (str): Start date in "YYYY-MM-DD" format.
        end_date_str (str): End date in "YYYY-MM-DD" format.
        freq_bands (list[tuple[float, float]]): Frequency bands for analysis.
        nslc (str): Network.Station.Location.Channel identifier.
        daily_dir (str): Directory for daily CSV files.
        csv (str): Path to final merged tremor CSV.
        SDS (SDS | None): SDS data reader (set after from_sds()).
        daily_files (list[str]): List of daily CSV file paths.

    Args:
        start_date (str | datetime): Start date for data processing (YYYY-MM-DD).
        end_date (str | datetime): End date for data processing (YYYY-MM-DD).
        station (str): Seismic station code (e.g., "OJN").
        channel (str): Seismic channel code (e.g., "EHZ").
        network (str): Seismic network code. Defaults to "VG".
        location (str): Seismic location code. Defaults to "00".
        methods (list[str] | None): Calculation methods to apply
            (e.g., ``["rsam", "dsar", "entropy"]``). If None, defaults to all
            three methods. Defaults to None.
        output_dir (str | None): Directory for output files.
            If None, defaults to ``root_dir/output``. Relative paths are resolved
            against ``root_dir`` (or ``os.getcwd()`` when ``root_dir`` is None).
            Absolute paths are used as-is. Defaults to None.
        root_dir (str | None): Anchor directory for resolving relative
            ``output_dir`` values. Defaults to None (uses ``os.getcwd()``).
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        n_jobs (int): Number of parallel jobs for daily processing. Defaults to 1.
        remove_outlier_method (Literal["maximum", "all"]): Outlier removal strategy.
            "maximum" removes only the single maximum outlier; "all" removes all outliers.
            Defaults to "maximum".
        interpolate (bool): If True, interpolates gaps in the data. Defaults to False.
        value_multiplier (float | None): Scaling factor applied to seismic values.
            Defaults to None (no scaling).
        cleanup_daily_dir (bool): If True, deletes the daily directory after
            merging daily results. Defaults to False.
        plot_daily (bool): If True, plots intermediate daily results for inspection.
            Defaults to False.
        save_plot (bool): If True, saves the final tremor plot to disk. Defaults to False.
        overwrite_plot (bool): If True, overwrites existing plot files. Defaults to False.
        filename_prefix (str | None): Custom prefix for output filenames.
            Defaults to None (auto-generated).
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug-level logging. Defaults to False.

    Raises:
        ValueError: If n_jobs <= 0, start_date >= end_date, or invalid method specified.

    Examples:
        >>> tremor = CalculateTremor(
        ...     start_date="2025-01-01",
        ...     end_date="2025-01-31",
        ...     station="OJN",
        ...     channel="EHZ",
        ...     n_jobs=4,
        ... ).from_sds(sds_dir="/data/sds").run()
        >>> print(tremor.df.head())
        >>> print(f"Saved to: {tremor.csv}")
    """

    def __init__(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        station: str,
        channel: str,
        network: str = "VG",
        location: str = "00",
        methods: list[str] | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
        overwrite: bool = False,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        interpolate: bool = False,
        value_multiplier: float | None = None,
        cleanup_daily_dir: bool = False,
        plot_daily: bool = False,
        save_plot: bool = False,
        overwrite_plot: bool = False,
        filename_prefix: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ):
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)
        network = network or "VG"
        location = location or "00"
        nslc = f"{network}.{station}.{location}.{channel}"
        output_dir = resolve_output_dir(output_dir, root_dir, "output")
        station_dir = os.path.join(output_dir, nslc)
        forecast_dir = os.path.join(station_dir, "forecast")
        tremor_dir = os.path.join(station_dir, "tremor")
        figures_dir = os.path.join(tremor_dir, "figures")

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.station = station.upper()
        self.channel = channel.upper()
        self.start_date: datetime = start_date
        self.end_date: datetime = end_date
        self.network = network or "VG"
        self.location = location or "00"

        # TODO: Add kurtosis
        self.methods: list[str] = methods or ["rsam", "dsar", "entropy"]
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
        self.cleanup_daily_dir = cleanup_daily_dir
        self.plot_daily = plot_daily
        self.save_plot = save_plot
        self.overwrite_plot = overwrite_plot
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
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
        self.figures_dir = figures_dir
        self.current_datetime = datetime.now()
        self.dates: pd.DatetimeIndex = pd.date_range(
            start=self.start_date, end=self.end_date
        )
        self.n_days: int = len(self.dates)
        self.nslc = nslc
        self.daily_dir: str = os.path.join(tremor_dir, "daily")
        self._filename = f"{self.nslc}_{self.start_date_str}_{self.end_date_str}.csv"
        self.csv = os.path.join(tremor_dir, self.filename)

        # ------------------------------------------------------------------
        # Will be set after from_sds() called
        # ------------------------------------------------------------------
        self.SDS: SDS | None = None
        self._sds_dir: str | None = None

        # ------------------------------------------------------------------
        # Will be set after from_fdsn() called
        # ------------------------------------------------------------------
        self.FDSN: FDSN | None = None
        self._client_url = "https://service.iris.edu"

        # ------------------------------------------------------------------
        # Will be set after from_sds() or from_fdsn() called
        # ------------------------------------------------------------------
        self._source: str | None = None

        # ------------------------------------------------------------------
        # Will be set after run() called
        # ------------------------------------------------------------------
        self.daily_files: list[str] = []

        # ------------------------------------------------------------------
        # Validate and create directories
        # ------------------------------------------------------------------
        self.validate()

        # ------------------------------------------------------------------
        # Verbose and logging
        # ------------------------------------------------------------------
        if debug:
            logger.info("⚠️ Calculate Tremor :: Debug mode is ON")

        if verbose:
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
            f"value_multiplier={self.value_multiplier}, cleanup_daily_dir={self.cleanup_daily_dir}, "
            f"plot_daily={self.plot_daily}, save_plot={self.save_plot}, overwrite_plot={self.overwrite_plot}, "
            f"csv={self.csv}, verbose={self.verbose}, debug={self.debug})"
        )

    def change_freq_bands(self, freq_bands: list[tuple[float, float]]) -> Self:
        """Change frequency bands for tremor calculation.

        Updates the default frequency bands used for RSAM and DSAR calculations.
        Each band is specified as a (min_freq, max_freq) tuple in Hz.

        Args:
            freq_bands (list[tuple[float, float]]): List of frequency band tuples.
                Each tuple must contain exactly two numeric values (freq_min, freq_max)
                where freq_min < freq_max. Example: [(0.1, 1.0), (1.0, 2.5), (2.0, 5.0)].

        Returns:
            Self: The CalculateTremor instance for method chaining.

        Raises:
            TypeError: If freq_bands is not a list, or contains non-tuple elements,
                or frequency values are not numeric.
            ValueError: If a tuple doesn't have exactly 2 elements, or freq_min >= freq_max.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.change_freq_bands([(0.1, 1.0), (1.0, 5.0), (5.0, 10.0)])
            >>> print(tremor.freq_bands_alias)
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
                    f"Freq minimum must be less than freq maximum. Got: {freq_min} >= {freq_max}"
                )

        self.freq_bands = freq_bands
        return self

    @property
    def freq_bands_alias(self) -> dict[str, tuple[float, float]]:
        """Get frequency bands with f0, f1, ... aliases.

        Returns:
            dict[str, tuple[float, float]]: Dictionary mapping band aliases
                (f0, f1, f2, ...) to (freq_min, freq_max) tuples.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> print(tremor.freq_bands_alias)
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
        """Get the tremor CSV output filename.

        Returns:
            str: Filename for the tremor CSV output, optionally prefixed
                with filename_prefix.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-12-31", station="OJN", channel="EHZ")
            >>> print(tremor.filename)
            tremor_VG.OJN.00.EHZ_2025-01-01_2025-12-31.csv
        """
        # Example: tremor_VG.OJN.00.EHZ_2025-01-01_2025-12-31
        return (
            f"tremor_interpolated_{self._filename}"
            if self.filename_prefix is None
            else f"{self.filename_prefix}_{self._filename}"
        )

    @filename.setter
    def filename(self, filename: str) -> None:
        """Set the tremor CSV output filename.

        Args:
            filename (str): Filename to set. '.csv' extension is added if missing.
        """
        if not filename.endswith(".csv"):
            filename = f"{filename}.csv"
        self._filename = filename

    @cached_property
    def jobs(self) -> list[tuple[int, datetime]]:
        """Generate jobs for multiprocessing.

        Creates a list of (job_index, date) tuples for parallel processing,
        one per day in the date range.

        Returns:
            list[tuple[int, datetime]]: List of (job_index, date) tuples.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-03", station="OJN", channel="EHZ")
            >>> print(len(tremor.jobs))  # 3 days
        """
        return [(job_index, date) for job_index, date in enumerate(self.dates)]

    def create_daily_dir(self) -> Self:
        """Create daily directory for temporary CSV files.

        If cleanup_daily_dir is True, removes any existing daily directory first.
        Creates the directory if it doesn't exist.

        Returns:
            Self: The CalculateTremor instance for method chaining.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.create_daily_dir()
        """
        if self.cleanup_daily_dir:
            if self.verbose:
                logger.info(f"Cleaning up temp dir: {self.daily_dir}")
            shutil.rmtree(self.daily_dir)

        os.makedirs(self.daily_dir, exist_ok=True)
        return self

    def create_directories(self) -> None:
        """Create all required output directories.

        Creates output_dir, station_dir, tremor_dir, daily_dir, and optionally
        figures_dir (if plot_daily is True).

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.create_directories()
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.station_dir, exist_ok=True)
        os.makedirs(self.tremor_dir, exist_ok=True)
        os.makedirs(self.daily_dir, exist_ok=True)

        if self.plot_daily:
            os.makedirs(self.figures_dir, exist_ok=True)

    def validate(self) -> None:
        """Validate input parameters and create directories.

        Checks that n_jobs > 0, start_date < end_date, and all methods are valid.
        Creates required output directories.

        Raises:
            ValueError: If n_jobs <= 0, start_date >= end_date, or invalid method specified.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.validate()  # Automatically called in __init__
        """
        if self.n_jobs <= 0:
            raise ValueError(
                f"Number of jobs must be greater than 0. Your value: {self.n_jobs}"
            )
        if self.start_date_utc_datetime >= self.end_date_utc_datetime:
            raise ValueError(
                f"Start date {self.start_date_str} must be before end date {self.end_date_str}"
            )

        valid_methods = ["rsam", "dsar", "entropy"]
        if not isinstance(self.methods, list):
            raise ValueError(f"Methods must be a list. Your value: {self.methods}")

        for method in self.methods:
            if method not in valid_methods:
                raise ValueError(
                    f"Method '{method}' not found. Choose between: {valid_methods}"
                )

        self.create_directories()

    def get_stream(self, date: datetime | None = None) -> Stream:
        """Get the seismic stream for a specific date.

        Retrieves seismic data from the configured data source (SDS or FDSN)
        for the specified date.

        Args:
            date (datetime): Date for which to retrieve the stream.

        Returns:
            Stream: ObsPy Stream object containing seismic traces for the date.

        Raises:
            ValueError: If date is None or data source not set via from_sds() or from_fdsn().

        Examples:
            >>> from datetime import datetime
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.from_sds("/data/sds")
            >>> stream = tremor.get_stream(datetime(2025, 1, 1))
        """
        if date is None:
            raise ValueError("Date must be provided")
        if self._source not in ["sds", "fdsn"]:
            raise ValueError(
                "Please choose a data source. Use `from_sds()` or `from_fdsn()` method "
                "to determine the data source before `run()`."
            )

        stream = Stream()

        if self._source == "sds" and self.SDS:
            stream = self.SDS.get(date)
        if self._source == "fdsn" and self.FDSN:
            stream = self.FDSN.get(date)

        return stream

    def from_sds(self, sds_dir: str) -> Self:
        """Set the data source to SeisComP Data Structure (SDS).

        Configures the calculator to read seismic data from an SDS archive.
        SDS is a standard directory structure for organizing seismic waveform data.
        See: https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html

        Args:
            sds_dir (str): Root directory of the SDS archive.

        Returns:
            Self: The CalculateTremor instance for method chaining.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.from_sds("/data/sds").run()
        """
        self._source = "sds"
        self._sds_dir = sds_dir
        self.SDS = SDS(
            sds_dir,
            network=self.network,
            station=self.station,
            channel=self.channel,
            location=self.location,
            verbose=self.verbose,
            interpolate=self.interpolate,
            debug=self.debug,
        )
        return self

    def from_fdsn(self, client_url: str | None = None) -> Self:
        """Set the data source to FDSN web services.

        Configures the calculator to fetch seismic data from an FDSN web service.
        FDSN (International Federation of Digital Seismograph Networks) provides
        standardized web services for accessing seismic data.

        Args:
            client_url (str | None): FDSN client URL. If None, defaults to
                "https://service.iris.edu". Defaults to None.

        Returns:
            Self: The CalculateTremor instance for method chaining.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.from_fdsn("https://service.iris.edu").run()
        """
        self._source = "fdsn"
        self._client_url = client_url or self._client_url
        self.FDSN = FDSN(
            client_url=self._client_url,
            network=self.network,
            station=self.station,
            channel=self.channel,
            location=self.location,
            verbose=self.verbose,
            debug=self.debug,
        )
        return self

    def run(self) -> Self:
        """Execute tremor calculation workflow.

        Orchestrates the full calculation: loads existing CSV if available and
        overwrite=False, otherwise processes daily data (in parallel if n_jobs > 1),
        merges results, and saves the final CSV. Optionally saves a plot.

        Returns:
            Self: The CalculateTremor instance with populated df and csv attributes.

        Raises:
            ValueError: If tremor CSV cannot be loaded when overwrite=False.

        Examples:
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-03", station="OJN", channel="EHZ", n_jobs=4)
            >>> tremor.from_sds("/data/sds").run()
            >>> print(tremor.df.head())
            >>> print(f"Saved to: {tremor.csv}")
        """
        csv = self.csv

        if not self.overwrite and os.path.isfile(csv):
            logger.info(f"Load tremor from file: {csv}")
            try:
                self.df = pd.read_csv(csv, index_col=0, parse_dates=True)
                return self
            except ValueError:
                raise ValueError(f"Could not load tremor from file: {csv}")

        self.create_daily_dir()

        if self.n_jobs == 1:
            for job in self.jobs:
                temp_file = self.run_job(*job)
                if temp_file is not None:
                    self.daily_files.append(temp_file)

        if self.n_jobs > 1:
            if self.verbose:
                logger.info(f"Running on {self.n_jobs} job(s)")

            with Pool(self.n_jobs) as pool:
                daily_files = pool.starmap(self.run_job, self.jobs)
                daily_files = [
                    daily_file for daily_file in daily_files if daily_file is not None
                ]
                self.daily_files.extend(daily_files)

        # Merge calculated tremor CSV files from daily dir
        df = self.concat_tremor_data(self.daily_dir, self.tremor_dir)

        start_date = df.index[0].strftime("%Y-%m-%d")
        end_date = df.index[-1].strftime("%Y-%m-%d")

        # Save non-interpolated dataframe
        non_interpolated_filename = (
            f"tremor_non-interpolated_{self.nslc}_{start_date}-{end_date}.csv"
        )
        csv_non_interpolated = os.path.join(self.tremor_dir, non_interpolated_filename)
        df.to_csv(csv_non_interpolated, index=True)

        # Update filename with latest datetime index from df.
        # Set _filename directly (not via the property setter) to avoid the
        # property re-adding the "tremor_" prefix and producing a double prefix.
        self._filename = f"{self.nslc}_{start_date}-{end_date}.csv"
        csv = os.path.join(self.tremor_dir, self.filename)

        # Handle missing data to all columns
        df = df.interpolate(method="time", limit_direction="both")
        df.to_csv(csv, index=True)

        logger.info(f"Interpolated tremor data saved to {csv}")

        if self.save_plot:
            plot_tremor(
                df=df,
                interval=14,
                interval_unit="days",
                figure_dir=self.tremor_dir,
                filename=self.filename,
                title=self.nslc,
                overwrite=self.overwrite or self.overwrite_plot,
                verbose=self.verbose,
            )

        self.df = df
        self.csv = csv

        return self

    @logger.catch
    def run_job(self, job_index: int, date: datetime) -> str | None:
        """Execute tremor calculation for a single day.

        Processes one day of seismic data, calculates RSAM and DSAR metrics,
        saves results to a daily CSV file, and optionally plots the data.
        Skips processing if the file already exists and overwrite=False.

        Args:
            job_index (int): Job index for logging purposes.
            date (datetime): Date to process.

        Returns:
            str | None: Path to the saved CSV file, or None if the DataFrame is empty
                (no data available for that day).

        Examples:
            >>> from datetime import datetime
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.from_sds("/data/sds")
            >>> csv_path = tremor.run_job(0, datetime(2025, 1, 1))
        """
        date_str = date.strftime("%Y-%m-%d")
        daily_file = os.path.join(self.daily_dir, f"{date_str}.csv")
        temp_plot = os.path.join(self.figures_dir, f"{date_str}.png")

        logger.info(f"Running Jobs ID: {job_index}. Date: {date_str}")

        can_skip = (
            not self.overwrite
            and os.path.exists(daily_file)
            and (not self.plot_daily or os.path.exists(temp_plot))
        )

        if can_skip:
            logger.info(f"{date_str} :: File Exists: {daily_file}")
            return daily_file

        df = self.calculate(date)

        # Pass if df is empty. No data to process
        if df.empty:
            return None

        # save tremor data
        df.to_csv(
            daily_file,
            index=True,
            index_label="datetime",
        )

        # plot tremor data
        if self.plot_daily:
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
            logger.info(f"{date_str} :: File CSV saved to {daily_file}")

        return daily_file

    def calculate(self, date: datetime) -> pd.DataFrame:
        """Calculate tremor metrics for a single day.

        Computes each enabled method (RSAM, DSAR, Shannon Entropy) for the specified
        date across all configured frequency bands. Returns an empty DataFrame if no
        seismic data is available for that day.

        Args:
            date (datetime): Date to process.

        Returns:
            pd.DataFrame: Tremor DataFrame with DatetimeIndex (10-minute intervals)
                and columns for each enabled method: RSAM (``rsam_f0``, ``rsam_f1``, …),
                DSAR (``dsar_f0-f1``, …), and Shannon Entropy (``entropy``).
                Returns an empty DataFrame if no seismic data is available.

        Examples:
            >>> from datetime import datetime
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> tremor.from_sds("/data/sds")
            >>> df = tremor.calculate(datetime(2025, 1, 1))
            >>> print(df.head())
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

            if method == "entropy":
                df = self.calculate_entropy(date_str, df, stream)

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
            # current_series = current_series.interpolate(method="time")

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
                if self.value_multiplier and self.value_multiplier != 1.0:
                    dsar_series = dsar_series * self.value_multiplier

                # Store in dataframe
                df[column_name] = dsar_series.to_numpy()

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
        """Calculate Real Seismic Amplitude Measurement (RSAM) for a given date.

        Computes RSAM (mean absolute amplitude) for each configured frequency band
        and adds columns (rsam_f0, rsam_f1, ...) to the DataFrame.

        Args:
            date_str (str): Date string in "YYYY-MM-DD" format (for logging).
            df (pd.DataFrame): Tremor DataFrame with DatetimeIndex to populate.
            stream (Stream): ObsPy Stream object containing seismic trace.

        Returns:
            pd.DataFrame: DataFrame with RSAM columns added.

        Raises:
            TypeError: If DataFrame index is not DatetimeIndex.

        Examples:
            >>> from datetime import datetime
            >>> from obspy import read
            >>> tremor = CalculateTremor(start_date="2025-01-01", end_date="2025-01-02", station="OJN", channel="EHZ")
            >>> stream = read("seismic_data.mseed")
            >>> df = pd.DataFrame(index=pd.date_range("2025-01-01", periods=144, freq="10min"))
            >>> df = tremor.calculate_rsam("2025-01-01", df, stream)
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
                    interpolate=False,
                )
                .to_numpy()
            )

            del stream_copy

            if self.verbose:
                logger.debug(
                    f"{date_str} :: RSAM ({column_name}) calculation completed."
                )

        return df

    def calculate_entropy(
        self, date_str: str, df: pd.DataFrame, stream: Stream
    ) -> pd.DataFrame:
        """Calculate Shannon entropy for a given date.

        Computes windowed Shannon entropy over the full broadband seismic signal
        (without per-band filtering) and stores the result in a single ``entropy``
        column of the DataFrame.

        Args:
            date_str (str): Date string in "YYYY-MM-DD" format (used for logging).
            df (pd.DataFrame): Tremor DataFrame with DatetimeIndex to populate.
            stream (Stream): ObsPy Stream object containing the seismic trace.

        Returns:
            pd.DataFrame: DataFrame with an ``entropy`` column added.

        Raises:
            TypeError: If the DataFrame index is not a DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index of dataframe should be pd.DatetimeIndex")

        column_name = "entropy"

        stream_copy = stream.copy()

        df[column_name] = (
            ShanonEntropy(
                stream=stream_copy,
                verbose=self.verbose,
                debug=self.debug,
            )
            .calculate(
                remove_outlier_method=self.remove_outlier_method,
                interpolate=False,
            )
            .to_numpy()
        )

        del stream_copy

        if self.verbose:
            logger.debug(f"{date_str} :: Entropy calculation completed.")

        return df

    @staticmethod
    def concat_tremor_data(
        daily_dir: str, tremor_dir: str | None = None
    ) -> pd.DataFrame:
        """Concatenate daily tremor CSV files into a single DataFrame.

        Reads all CSV files from the daily directory, concatenates them,
        and returns a sorted DataFrame. Used to merge daily results after
        parallel processing.

        Args:
            daily_dir (str): Directory containing daily CSV files.
            tremor_dir (str | None): Directory where merged tremor data will be saved.
                If None, defaults to daily_dir with "daily" removed. Defaults to None.

        Returns:
            pd.DataFrame: Concatenated tremor DataFrame sorted by DatetimeIndex.

        Raises:
            FileNotFoundError: If daily_dir doesn't exist or contains no CSV files.

        Examples:
            >>> df = CalculateTremor.concat_tremor_data("/output/tremor/daily")
            >>> print(df.head())
        """
        if not os.path.isdir(daily_dir):
            raise FileNotFoundError(f"Directory {daily_dir} does not exist")

        files = glob.glob(os.path.join(daily_dir, "*.csv"))
        if len(files) == 0:
            raise FileNotFoundError(f"No CSV files found in {daily_dir}")

        tremor_dir = (
            daily_dir.replace("daily", "") if tremor_dir is None else tremor_dir
        )
        os.makedirs(tremor_dir, exist_ok=True)

        df = pd.concat(
            [
                pd.read_csv(file, index_col="datetime", parse_dates=True)
                for file in sorted(files)
            ],
            sort=True,
        )

        return df.sort_index()
