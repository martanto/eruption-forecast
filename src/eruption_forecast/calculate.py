# Standard library imports
import os
import shutil
from datetime import datetime, timedelta, timezone
from functools import cached_property
from multiprocessing import Pool
from typing import Optional, Self, Any

# Third party imports
import numpy as np
import pandas as pd
from loguru import logger
from obspy import Stream, UTCDateTime

# Project imports
import eruption_forecast

from .dsar import DSAR
from .rsam import RSAM
from .sds import SDS
from .utils import delete_outliers, get_windows_information


class Calculate:
    """Calculate Tremor Data from seismic data.

    Args:
        station (str): Seismic station code.
        channel (str): Seismic channel code.
        start_date (str): Start date for data processing (YYYY-MM-DD).
        end_date (Optional[str]): End date for data processing (YYYY-MM-DD).
        window_size (Optional[int]): Size of the processing window in minutes. Defaults to 3.
        window_overlap (Optional[float]): Overlap between windows as a fraction. Defaults to 0.75.
        day_to_forecast (Optional[int]): Number of days to forecast ahead. Defaults to 1.
        network (str): Seismic network code. Defaults to "VG".
        location (str): Seismic location code. Defaults to "00".
        methods (Optional[str]): Calculation methods to apply.
        output_dir (str): Directory for output files. Defaults to "output".
        tremor_dir (str): Directory for tremor data. Defaults to "tremor".
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        filename_prefix (Optional[str]): Prefix for generated filenames.
        n_jobs (int): Number of parallel jobs to use. Defaults to 1.
        volcano_code (Optional[str]): Code representing the volcano.
        handle_zero_as_gap (bool): If True, treats zero values as data gaps. Defaults to True.
        remove_outliers (bool): If True, removes outliers from the data. Defaults to True.
        value_multiplier (Optional[float]): Scaling factor for seismic values.
        interpolate (bool): If True, interpolates missing data points. Defaults to False.
        cleanup_tmp_dir (bool): If True, deletes temporary directory after use. Defaults to False.
        verbose (bool): If True, enables verbose logging. Defaults to False.
        debug (bool): If True, enables debug mode. Defaults to False.
    """
    def __init__(
        self,
        station: str,
        channel: str,
        start_date: str,
        end_date: Optional[str] = None,
        window_size: Optional[int] = 3,
        window_overlap: Optional[float] = 0.75,
        day_to_forecast: Optional[int] = 1,
        network: str = "VG",
        location: str = "00",
        methods: Optional[str] = None,
        output_dir: str = "output",
        tremor_dir: str = "tremor",
        overwrite: bool = False,
        filename_prefix: Optional[str] = None,
        n_jobs: int = 1,
        volcano_code: Optional[str] = None,
        handle_zero_as_gap: bool = True,
        remove_outliers: bool = True,
        value_multiplier: Optional[float] = None,
        interpolate: bool = False,
        cleanup_tmp_dir: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        end_date = end_date or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        network = network or "VG"
        location = location or "00"
        nslc = f"{network}.{station}.{location}.{channel}"
        output_dir = os.path.join(os.getcwd(), output_dir, "forecast")
        station_dir = os.path.join(output_dir, nslc)

        # Set DEFAULT properties
        self.station = station.upper()
        self.channel = channel.upper()
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.window_size: int = window_size or 3
        self.window_overlap: float = window_overlap or 0.75
        self.day_to_forecast: int = day_to_forecast or 1
        self.network = network or "VG"
        self.location = location or "00"
        self.methods: list[str] = (
            [methods] if isinstance(methods, str) else list(methods or ["rsam", "dsar"])
        )
        self.output_dir: str = output_dir
        self.station_dir: str = station_dir
        self.tremor_dir: str = os.path.join(station_dir, tremor_dir)
        self.overwrite = overwrite
        self.filename_prefix = filename_prefix
        self.n_jobs = n_jobs
        self.volcano_code = volcano_code
        self.handle_zero_as_gap = handle_zero_as_gap
        self.remove_outliers = remove_outliers
        self.value_multiplier = value_multiplier
        self.interpolate = interpolate
        self.cleanup_tmp_dir = cleanup_tmp_dir
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        try:
            self.start_date_obj: datetime = datetime.strptime(start_date, "%Y-%m-%d")
            self.end_date_obj: datetime = datetime.strptime(end_date, "%Y-%m-%d")
            self.start_date_utc_datetime = UTCDateTime(self.start_date_obj)
            self.end_date_utc_datetime = UTCDateTime(self.end_date_obj)
        except ValueError:
            raise ValueError(f"Start date and end date must be in format YYYY-MM-DD")

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
        self.tmp_dir: str = os.path.join(self.tremor_dir, "_tmp")
        self.log_dir = os.path.join(self.station_dir, "logs")
        self.results: list[Any] = []
        self.sds: Optional[SDS] = None
        self.tmp_files: list[str] = []
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
            logger.info(f"Start Date: {self.start_date}")
            logger.info(f"End Date: {self.end_date}")
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
            dict[str, tuple[float, float]]: Dict contains band alias and freq bands

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
        default_filename = f"{self.nslc}_{self.start_date}_{self.end_date}"
        return (
            default_filename
            if self.filename_prefix is None
            else f"{self.filename_prefix}_{default_filename}"
        )

    @cached_property
    def jobs(self) -> list[tuple[int, datetime]]:
        return [(job_index, date) for job_index, date in enumerate(self.dates)]

    @logger.catch
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
        if self.start_date_utc_datetime > self.end_date_utc_datetime:
            raise ValueError(
                f"Start date {self.start_date} must be before end date {self.end_date}"
            )

        assert 0 < self.window_overlap <= 1, (
            f"Window overlap must be between 0 and 1. Default 0.75."
        )

        for method in self.methods:
            assert method in self.methods, (
                f"Method '{method}' not found. Choose between: {self.methods}"
            )

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

        if self._source == "sds" and self.sds:
            return self.sds.get(date)
        if self._source == "fdsn":
            return Stream()

        return Stream()

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
            return self

        if self.verbose:
            logger.info(f"Running on {self.n_jobs} job(s)")

        pool = Pool(self.n_jobs)
        pool.starmap(self.run_job, self.jobs)
        pool.close()
        pool.join()

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
        df.to_csv(
            temp_file,
            index=True,
            index_label="datetime",
        )

        self.tmp_files.append(temp_file)

        if self.verbose:
            logger.info(
                f"{date_str} :: File CSV saved {os.path.join(self.tmp_dir, date_str)}"
            )

        return None

    def calculate(self, date: datetime) -> pd.DataFrame:
        """Calculate tremor data.
        Those method calculate the tremor data using Real Seismic Amplitude Measurement (RSAM) and Displacement Seismic Amplitude Ratio (DSAR).

        Args:
            date (datetime): Date to calculate

        Returns:
            pd.DataFrame: Tremor data
        """
        stream = self.stream(date).detrend(type="demean")
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
                        date=date,
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
                len_freq_bands = len(freq_bands)
                for index, filtered_stream in enumerate(filtered_streams):
                    if index < (len_freq_bands - 1):
                        first_band_name = filtered_stream["band_name"]
                        second_band_name = filtered_streams[index + 1]["band_name"]
                        column_name = f"dsar_{first_band_name}-{second_band_name}"

                        first_stream = filtered_stream["filtered_stream"]
                        second_stream = filtered_streams[index + 1]["filtered_stream"]

                        # Use the new DSAR class
                        df[column_name] = dsar.calculate(
                            first_stream=first_stream,
                            second_stream=second_stream,
                            value_multiplier=self.value_multiplier or 1.0,
                        ).values

                        if self.verbose:
                            logger.info(
                                f"{date_str} :: DSAR ({column_name}) calculation finished"
                            )

        return df

    def calculate_rsam(
        self, stream: Stream, freq_min: float, freq_max: float, date: datetime
    ) -> pd.Series:
        date_str = date.strftime("%Y-%m-%d")

        rsam: RSAM = (
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
            )
        )

        series = rsam.series.interpolate(method="linear")

        return series
