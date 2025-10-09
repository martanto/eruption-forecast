import os
import eruption_forecast
import pandas as pd
from obspy import read, UTCDateTime
from datetime import datetime


class Calculate:
    def __init__(
        self,
        station: str,
        channel: str,
        network: str = "VG",
        location: str = "00",
        source: str = "sds",
        methods: list[str] | str = None,
        start_date: str = None,
        end_date: str = None,
        output_dir: str = "output",
        tremor_dir: str = "tremor",
        overwrite: bool = False,
        filename_prefix: str = None,
        n_jobs: int = 2,
        cleanup_tmp_dir: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        self.station = station
        self.channel = channel
        self.network = network
        self.location = location

        self.nslc = f"{network}.{station}.{location}.{channel}"

        methods = ["rsam", "dsar"] if methods is None else methods
        self.methods: list[str] = [methods] if isinstance(methods, str) else methods

        self.start_date: str = start_date
        self.end_date: str = end_date

        try:
            self.start_date_obj: datetime = datetime.strptime(start_date, "%Y-%m-%d")
            self.end_date_obj: datetime = datetime.strptime(end_date, "%Y-%m-%d")
            self.start_date_utc_datetime = UTCDateTime(self.start_date_obj)
            self.end_date_utc_datetime = UTCDateTime(self.end_date_obj)
        except ValueError:
            raise ValueError(f"❌ Start date and end date must be in format YYYY-MM-DD")

        self.overwrite = overwrite
        self.filename_prefix: str = filename_prefix
        self.cleanup_tmp_dir = cleanup_tmp_dir
        self.n_jobs = n_jobs
        self.source = source

        self.verbose = verbose
        self.debug = debug

        if debug:
            print("⚠️ Debug mode is ON")

        print(f"Version: {eruption_forecast.__version__}")

        self._assert()
        self._check_directory(output_dir, tremor_dir)

    @property
    def filename(self) -> str:
        default_filename = (
            f"{self.filename_prefix}_{self.nslc}_{self.start_date}_{self.end_date}"
        )
        return (
            default_filename
            if self.filename_prefix is None
            else f"{self.filename_prefix}_{default_filename}"
        )

    def _check_directory(self, output_dir: str, tremor_dir: str) -> None:
        self.output_dir: str = os.path.join(os.getcwd(), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.tremor_dir: str = os.path.join(self.output_dir, tremor_dir)
        os.makedirs(self.tremor_dir, exist_ok=True)

        self.station_dir: str = os.path.join(self.tremor_dir, self.nslc)
        os.makedirs(self.station_dir, exist_ok=True)

        self.tmp_dir: str = os.path.join(self.station_dir, "_tmp")
        os.makedirs(self.tmp_dir, exist_ok=True)

    def _assert(self):
        for method in self.methods:
            assert (
                method in self.methods
            ), f"❌ Method '{method}' not found. Choose from: {self.methods}"

        assert self.source in [
            "file",
            "sds",
            "server",
        ], f"❌ Source '{self.source}' not found. Choose from: {self.source}"

    def run(self):
        for method in self.methods:
            if method == "rsam":
                self.rsam()
            if method == "dsar":
                self.dsar()

    def rsam(self):
        pass

    def dsar(self):
        pass
