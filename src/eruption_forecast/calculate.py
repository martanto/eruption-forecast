import os
import sys
from typing import Optional, Self

import numpy as np
from multiprocessing import Pool

import eruption_forecast
import pandas as pd
from obspy import read, UTCDateTime
from datetime import datetime, timezone
from functools import cached_property
from loguru import logger


class Calculate:
    def __init__(
        self,
        station: str,
        channel: str,
        start_date: str,
        end_date: Optional[str] = None,
        network: str = "VG",
        location: str = "00",
        methods: list[str] | Optional[str] = None,
        output_dir: str = "output",
        tremor_dir: str = "tremor",
        overwrite: bool = False,
        filename_prefix: Optional[str] = None,
        n_jobs: int = 1,
        cleanup_tmp_dir: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.station = station.upper()
        self.channel = channel.upper()
        self.network = network.upper()
        self.location = location

        self.nslc = f"{network}.{station}.{location}.{channel}"

        methods = ["rsam", "dsar"] if methods is None else methods
        self.methods: list[str] = [methods] if isinstance(methods, str) else methods

        self.start_date: str = start_date
        self.end_date: str = end_date or datetime.now(tz=timezone.utc).strftime(
            "%Y-%m-%d"
        )

        try:
            self.start_date_obj: datetime = datetime.strptime(start_date, "%Y-%m-%d")
            self.end_date_obj: datetime = datetime.strptime(end_date, "%Y-%m-%d")
            self.start_date_utc_datetime = UTCDateTime(self.start_date_obj)
            self.end_date_utc_datetime = UTCDateTime(self.end_date_obj)
        except ValueError:
            raise ValueError(f"❌ Start date and end date must be in format YYYY-MM-DD")

        self.current_datetime = datetime.now()
        self.dates: pd.DatetimeIndex = pd.date_range(
            start=self.start_date, end=self.end_date
        )
        self.n_days: int = len(self.dates)
        self.overwrite = overwrite
        self.filename_prefix: str = filename_prefix
        self.cleanup_tmp_dir = cleanup_tmp_dir
        self.n_jobs = n_jobs

        self.verbose = verbose
        self.debug = debug

        logger.info(f"Version: {eruption_forecast.__version__}")

        self.output_dir: str = os.path.join(os.getcwd(), output_dir, "forecast")
        self.station_dir: str = os.path.join(self.output_dir, self.nslc)
        self.tremor_dir: str = os.path.join(self.station_dir, tremor_dir)
        self.tmp_dir: str = os.path.join(self.tremor_dir, "_tmp")
        self.log_dir = os.path.join(self.station_dir, "logs")

        self._assert()
        self._check_directory()
        self._source: Optional[str] = None
        self._sds_dir: Optional[str] = None
        self._client_url = "https://service.iris.edu"

        if debug:
            logger.info("⚠️ Debug mode is ON")

    @cached_property
    def filename(self) -> str:
        default_filename = (
            f"{self.filename_prefix}_{self.nslc}_{self.start_date}_{self.end_date}"
        )
        return (
            default_filename
            if self.filename_prefix is None
            else f"{self.filename_prefix}_{default_filename}"
        )

    @cached_property
    def jobs(self):
        return [(index, date) for index, date in enumerate(self.dates)]

    def _check_directory(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.station_dir, exist_ok=True)
        os.makedirs(self.tremor_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)

        if self.debug:
            os.makedirs(self.log_dir, exist_ok=True)

    def _assert(self):
        for method in self.methods:
            assert (
                method in self.methods
            ), f"❌ Method '{method}' not found. Choose from: {self.methods}"

    def _run(self, index: int, date: datetime):
        # if self.verbose:
            # logger.remove()
            # logger.info(f"Index {index}. Date: {date}")

        if self.debug:
            logger.debug(f"Index {index}. Date: {date}")

        return True

    def run(self):
        assert self._source in ["sds", "fdsn"], (
            f"❌ Please choose a data source. Use `from_sds` or from `from_fdsn` method "
            f"to determine the data source before `run`."
        )

        if self.verbose:
            logger.info(f"Running on {self.n_jobs} jobs")
            logger.info(f"NSLC: {self.nslc}")
            logger.info(f"Start Date: {self.start_date}")
            logger.info(f"End Date: {self.end_date}", end="\n\n")

        if self.n_jobs > 1:
            pool = Pool(self.n_jobs)
            pool.starmap(self._run, self.jobs)
            pool.close()
            pool.join()
            return self

        for job in self.jobs:
            self._run(*job)
        return self

    def from_sds(self, sds_dir: str) -> Self:
        self._source = "sds"
        self._sds_dir = sds_dir
        return self

    def from_fdsn(self, client_url: Optional[str] = None) -> Self:
        self._source = "fdsn"
        self._client_url = client_url or self._client_url
        return self

    def rsam(self):
        pass

    def dsar(self):
        pass
