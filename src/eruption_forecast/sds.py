# Standard library imports
import os
from datetime import datetime
from typing import Any, List

# Third party imports
from obspy import ObsPyReadingError, Stream, Trace, read

# Project imports
from eruption_forecast.logger import logger


class SDS:
    """SeisComP Data Structure (SDS) class.

    Get more information about SDS format from:
    https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html

    Args:
        sds_dir (str): Path to SDS directory.
        station (str): Station name.
        channel (str): Channel name.
        network (str, optional): Network name. Defaults to "VG".
        location (str, optional): Location name. Defaults to "00".
        verbose (bool, optional): Verbose mode. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.

    Attributes:
        sds_dir (str): Path to SDS directory.
        station (str): Station name.
        channel (str): Channel name.
        network (str): Network name.
        location (str): Location name.
        verbose (bool): Verbose mode.
        debug (bool): Debug mode.
        nslc (str): NSLC name.
        files (List[dict[str, Any]]): List of files.

    Methods:
        get_filepath(date: datetime) -> str: Get filepath for SDS data.
        load_stream(filepath: str, date_str: str) -> Stream: Get Stream data from SDS.
        get(date: datetime) -> Stream: Get Stream data from SDS.
    """

    def __init__(
        self,
        sds_dir: str,
        station: str,
        channel: str,
        network: str = "VG",
        location: str = "00",
        verbose: bool = False,
        debug: bool = False,
    ):
        self.sds_dir = sds_dir
        self.station = station.upper()
        self.channel = channel.upper()
        self.network = network.upper()
        self.location = location.upper()
        self.verbose = verbose
        self.debug = debug

        self.nslc = f"{network}.{station}.{location}.{channel}"
        self.files: List[dict[str, Any]] = []

    def get_filepath(self, date: datetime) -> str:
        """Get filepath for SDS data.

        Args:
            date (datetime): Date to get filepath for.
        """
        year = date.year
        julian_day = date.strftime("%j")
        data_dir = os.path.join(
            self.sds_dir, str(year), self.network, self.station, f"{self.channel}.D"
        )
        filename = f"{self.nslc}.D.{year}.{julian_day}"
        return os.path.join(data_dir, filename)

    def load_stream(self, filepath: str, date_str: str) -> Stream:
        """Get Stream data from SDS.

        Args:
            filepath (str): Path to SDS file.
            date_str (str): Date to get data from.
        """
        try:
            stream = read(filepath, format="MSEED")

            file = {
                "date": date_str,
                "file": filepath,
                "length": len(stream),
            }

            stream = stream.merge(fill_value="interpolate")
            self.files.append(file)
            return stream
        except ObsPyReadingError as e:
            if self.debug:
                logger.error(f"{datetime.now()} :: {filepath}\n{e}")
            return Stream()

    def get(self, date: datetime) -> Stream:
        """Get Stream data from SDS.

        Args:
            date (datetime): Date to get data from.
        """
        date_str = date.strftime("%Y-%m-%d")
        filepath = self.get_filepath(date)
        if not os.path.exists(filepath):
            if self.debug:
                logger.warning(f"{date_str} :: Data not exists in {filepath}")
            return Stream()

        stream = self.load_stream(filepath, date_str)

        if len(stream) == 0:
            logger.warning(f"{date_str} :: No trace(s) found in {filepath}")
        elif self.verbose:
            trace: Trace = stream[0]
            data_length = len(trace.data)
            logger.info(f"{date_str} :: Stream loaded {filepath}")
            logger.info(
                f"{date_str} :: {len(stream)} trace(s) found. Total data {data_length}"
            )

        return stream
