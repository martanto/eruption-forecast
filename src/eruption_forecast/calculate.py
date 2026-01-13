# Standard library imports
import os
import shutil
from datetime import datetime, timedelta, timezone
from functools import cached_property
from multiprocessing import Pool
from typing import Optional, Self

# Third party imports
import numpy as np
import pandas as pd
from loguru import logger
from obspy import Stream, UTCDateTime

# Project imports
import eruption_forecast

from .rsam import RSAM
from .sds import SDS
from .utils import delete_outliers, get_windows_information


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
        remove_outliers: bool = True,
        value_multiplier: Optional[float] = None,
        interpolate: bool = False,
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
        self.remove_outlier = remove_outliers
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
        self.results = []
        self.sds: SDS = SDS
        self.tmp_files: list[str] = []
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
        """Freq band with alias

        Returns:
            dict[str, tuple[float, float]]: Dict contains name and freq bands

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

    def run_job(self, job_index: int, date: datetime) -> None:
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

    def calculate(self, date: datetime):
        stream = self.stream(date).detrend(type="demean")
        date_str = date.strftime("%Y-%m-%d")

        freq_bands = self.freq_bands_alias

        datetime_index = pd.date_range(
            start=date, end=date + timedelta(days=1), freq="10min", inclusive="left"
        )

        df = pd.DataFrame(index=datetime_index)

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

                # Filtering
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
                        first_filtered_stream = filtered_stream["filtered_stream"]
                        second_filtered_stream = filtered_streams[index + 1][
                            "filtered_stream"
                        ]

                        first_band_name = filtered_stream["band_name"]
                        second_band_name = filtered_streams[index + 1]["band_name"]
                        column_name = f"dsar_{first_band_name}-{second_band_name}"

                        df[column_name] = self.calculate_dsar(
                            first_filtered_stream=first_filtered_stream,
                            second_filtered_stream=second_filtered_stream,
                            date=date,
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
                value_multiplier=self.value_multiplier,
                remove_outlier=self.remove_outlier,
            )
        )

        series = rsam.series.interpolate(method="linear")

        return series

    def calculate_dsar(
        self,
        first_filtered_stream: Stream,
        second_filtered_stream: Stream,
        date: datetime,
    ) -> pd.Series:

        date_str = date.strftime("%Y-%m-%d")
        first_trace = first_filtered_stream[0]
        second_trace = second_filtered_stream[0]

        first_data = first_trace.data
        second_data = second_trace.data

        windows = get_windows_information(first_trace)
        total_window = windows["total_window"]
        sample_per_ten_minute = windows["samples_per_ten_minute"]

        indices = []
        data = []
        for index_window in range(total_window):
            next_index = index_window + 1
            first_index = int(index_window * sample_per_ten_minute)
            last_index = int(next_index * sample_per_ten_minute)

            first_ten_minutes_data = first_data[first_index:last_index]
            second_ten_minutes_data = second_data[first_index:last_index]

            length_ten_minutes_data = len(first_ten_minutes_data)
            minimum_samples = int(np.ceil(0.3 * length_ten_minutes_data))

            if self.debug and (length_ten_minutes_data == 0):
                logger.debug(f"{date_str} :: 10 minutes data empty for index_window = {index_window} or ten_minutes_data[{first_index}:{last_index}]")

            if self.remove_outlier and (length_ten_minutes_data > minimum_samples):
                first_ten_minutes_data = delete_outliers(first_ten_minutes_data)
                second_ten_minutes_data = delete_outliers(second_ten_minutes_data)

            index = date + timedelta(minutes=index_window * 10)

            dsar = np.nan
            if length_ten_minutes_data > 0:
                mean_first_ten_minutes_data = np.mean(abs(first_ten_minutes_data))
                mean_second_ten_minutes_data = np.mean(abs(second_ten_minutes_data))
                dsar = mean_first_ten_minutes_data / mean_second_ten_minutes_data

            indices.append(index)
            data.append(dsar)

        series = pd.Series(data=data, index=indices, name="datetime")
        series = series.interpolate(method="linear")

        if self.value_multiplier > 1:
            series = series.apply(lambda values: values * self.value_multiplier)

        return series
