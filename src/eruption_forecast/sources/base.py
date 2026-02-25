"""Abstract base class for seismic data sources."""

from abc import ABC, abstractmethod
from datetime import datetime

from obspy import Stream


class SeismicDataSource(ABC):
    """Abstract base for seismic data sources (SDS, FDSN, etc.).

    Defines the common interface for all seismic data adapters. Concrete
    subclasses must implement :meth:`get` to retrieve a day's worth of
    seismic waveform data.

    Attributes:
        network (str): Seismic network code. Set by concrete subclasses.
        station (str): Seismic station code. Set by concrete subclasses.
        location (str): Location code. Set by concrete subclasses.
        channel (str): Channel code. Set by concrete subclasses.
        nslc (str): Combined ``Network.Station.Location.Channel`` identifier.
            Set by concrete subclasses as ``f"{network}.{station}.{location}.{channel}"``.
    """

    nslc: str

    @abstractmethod
    def get(self, date: datetime) -> Stream:
        """Retrieve seismic stream for a specific date.

        Args:
            date (datetime): Date for which to retrieve data.

        Returns:
            Stream: ObsPy Stream containing seismic waveform data, or an
                empty Stream if no data is available.
        """

    def _make_log_prefix(self, date: datetime) -> str:
        """Build a standard log prefix string for a given date.

        Args:
            date (datetime): Date to format.

        Returns:
            str: Log prefix in the format ``"YYYY-MM-DD :: {nslc}"``.
        """
        return f"{date.strftime('%Y-%m-%d')} :: {self.nslc}"
