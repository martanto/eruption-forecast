import os
from typing import Optional, Self

import shutil
from multiprocessing import Pool

import eruption_forecast
import pandas as pd
import numpy as np
from obspy import UTCDateTime, Stream, Trace
from datetime import datetime, timezone, timedelta
from functools import cached_property
from loguru import logger
from .sds import SDS


class Calculate:
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
        cleanup_tmp_dir: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        # Set DEFAULT parameter
        if methods is None:
            methods = ["rsam", "dsar"]
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
        self.window_size: int = window_size
        self.window_overlap: float = window_overlap
        self.day_to_forecast: int = day_to_forecast
        self.network = network or "VG"
        self.location = location or "00"
        self.methods: list[str] = [methods] if isinstance(methods, str) else methods
        self.output_dir: str = output_dir
        self.station_dir: str = station_dir
        self.tremor_dir: str = os.path.join(station_dir, tremor_dir)
        self.overwrite = overwrite
        self.filename_prefix = filename_prefix
        self.n_jobs = n_jobs
        self.volcano_code = volcano_code
        self.handle_zero_as_gap = handle_zero_as_gap
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

        self.current_datetime = datetime.now()
        self.dates: pd.DatetimeIndex = pd.date_range(
            start=self.start_date, end=self.end_date
        )
        self.n_days: int = len(self.dates)
        self.nslc = nslc
        self.tmp_dir: str = os.path.join(self.tremor_dir, "_tmp")
        self.log_dir = os.path.join(self.station_dir, "logs")
        self.results = []
        self.sds: SDS = SDS
        self.dfs = []
        self._source: Optional[str] = None
        self._sds_dir: Optional[str] = None
        self._client_url = "https://service.iris.edu"

        # Validate
        self._assert()

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

    @cached_property
    def filename(self) -> str:
        default_filename = f"{self.nslc}_{self.start_date}_{self.end_date}"
        return (
            default_filename
            if self.filename_prefix is None
            else f"{self.filename_prefix}_{default_filename}"
        )

    @cached_property
    def jobs(self):
        return [(job_index, date) for job_index, date in enumerate(self.dates)]

    @logger.catch
    def create_temporary_dir(self):
        if self.cleanup_tmp_dir:
            if self.verbose:
                logger.info(f"Cleaning up temp dir: {self.tmp_dir}")
            shutil.rmtree(self.tmp_dir)

        os.makedirs(self.tmp_dir, exist_ok=True)
        return self

    def _check_directory(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.station_dir, exist_ok=True)
        os.makedirs(self.tremor_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        if self.debug:
            os.makedirs(self.log_dir, exist_ok=True)

    def _assert(self):
        if self.start_date_utc_datetime > self.end_date_utc_datetime:
            raise ValueError(
                f"Start date {self.start_date} must be before end date {self.end_date}"
            )

        assert (
            0 < self.window_overlap <= 1
        ), f"Window overlap must be between 0 and 1. Default 0.75."

        for method in self.methods:
            assert (
                method in self.methods
            ), f"Method '{method}' not found. Choose between: {self.methods}"

        self._check_directory()

    def stream(self, date: datetime = None) -> Stream:
        assert self._source in ["sds", "fdsn"], (
            f"❌ Please choose a data source. Use `from_sds` or from `from_fdsn` method "
            f"to determine the data source before `run`."
        )

        if self._source == "sds":
            return self.sds.get(date)
        if self._source == "fdsn":
            return Stream()

        return Stream()

    def from_sds(self, sds_dir: str) -> Self:
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
        self._source = "fdsn"
        self._client_url = client_url or self._client_url
        return self

    def run(self):
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

    def run_job(self, job_index: int, date: datetime):
        date_str = date.strftime("%Y-%m-%d")
        temp_file = os.path.join(self.tmp_dir, f"{date_str}.csv")

        if self.debug:
            logger.debug(f"Job index {job_index}. Date: {date_str}")

        if not self.overwrite and os.path.exists(temp_file):
            df = pd.read_csv(temp_file, index_col=0, parse_dates=True)
            self.dfs.append(df)

            if self.verbose:
                logger.info(f"{date_str} :: File CSV loaded {temp_file}")

            return True

        if self.verbose:
            logger.debug(f"Jobs ID: {job_index}. Date: {date_str}")

        stream = self.calculate(date)

        return True

    def calculate(self, date: datetime):
        stream = self.stream(date).detrend(type="demean")

        rsam_freq_bands: list[tuple[float, float]] = [
            (0.01, 0.1),
            (0.1, 2),
            (2, 5),
            (4.5, 8),
            (8, 16),
        ]

        datetime_index = pd.date_range(
            start=date,
            end=date + timedelta(days=1),
            freq = "10min", inclusive = "left"
        )

        df = pd.DataFrame(index=datetime_index)

        for method in self.methods:
            if method == "rsam":
                for rsam_freq_band in rsam_freq_bands:
                    self.calculate_rsam(stream.copy(), freq_bands=rsam_freq_band)

        return stream

    @staticmethod
    def trace_to_series(trace: Trace) -> pd.Series:
        index_time = pd.date_range(
            start=trace.stats.starttime.datetime,
            periods=trace.stats.npts,
            freq="{}ms".format(trace.stats.delta * 1000),
        )

        series = pd.Series(
            data=np.abs(trace.data),
            index=index_time,
            name="values",
            dtype=trace.data.dtype,
        )

        series.index.name = "datetime"

        return series

    def calculate_rsam(
        self, stream: Stream, freq_bands: Optional[tuple[float, float]] = None
    ):
        if freq_bands:
            if self.debug:
                logger.debug(f"RSAM Filtering using {freq_bands}")

            freq_min, freq_max = freq_bands
            stream = stream.filter(
                "bandpass", freqmin=freq_min, freqmax=freq_max, corners=4
            )

        series = self.trace_to_series(stream[0]).resample(rule="10min")

