"""FDSN web service adapter with transparent local SDS caching.

This module provides the ``FDSN`` class, which retrieves seismic waveform data
from any FDSN-compliant web service using the ObsPy FDSN client. To avoid
repeated network requests, each successfully downloaded day is written to a
local SDS archive and served from cache on subsequent runs.

Key class:
    - ``FDSN``: Implements ``SeismicDataSource.get(date)`` — checks the local SDS
      cache first, downloads from the FDSN endpoint if not cached, saves the result
      as a miniSEED file in SDS format, and returns the ObsPy ``Stream``.
    - ``download_dir`` is created automatically if it does not exist.
    - Configurable via ``client_url`` (defaults to IRIS), ``network``, ``station``,
      ``location``, and ``channel`` parameters.

Reference: https://www.fdsn.org/about/
"""

import os
from datetime import datetime

from obspy import Stream, UTCDateTime, read
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.fdsn.header import FDSNException

from eruption_forecast.logger import logger
from eruption_forecast.sources.sds import SDS
from eruption_forecast.sources.base import SeismicDataSource
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.utils.date_utils import normalize_dates


class FDSN(SeismicDataSource):
    """Retrieve seismic data from an FDSN web service.

    FDSN (International Federation of Digital Seismograph Networks) provides
    standardised web services for accessing seismic waveform data. This class
    wraps the ObsPy FDSN client and adds transparent local caching via an SDS
    archive so that each day's data is only downloaded once.

    More information: https://www.fdsn.org/about/

    Args:
        station (str): Station code (e.g., "OJN").
        channel (str): Channel code (e.g., "EHZ").
        network (str, optional): Network code. Defaults to "VG".
        location (str, optional): Location code. Defaults to "00".
        client_url (str, optional): FDSN web-service URL. Defaults to
            ``"https://service.iris.edu"``.
        download_dir (str, optional): Local directory for caching downloaded
            miniSEED files. Created automatically if it does not exist.
            Defaults to ``"downloads"`` relative to the current working directory.
        overwrite (bool, optional): If True, re-download data even when a cached
            file already exists. Defaults to False.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        debug (bool, optional): Enable debug logging. Defaults to False.

    Attributes:
        station (str): Station code.
        channel (str): Channel code.
        network (str): Network code.
        location (str): Location code.
        channel_type (str): str = Channel type. Defaults to "D".,
        download_dir (str): Local directory used as the SDS cache root.
        overwrite (bool): Whether cached files are overwritten on each download.
        verbose (bool): Whether verbose logging is enabled.
        debug (bool): Whether debug logging is enabled.
        client (FDSNClient): ObsPy FDSN client instance.
        SDS (SDS): SDS archive used as the local cache for downloaded data.
        nslc (str): Network.Station.Location.Channel identifier.

    Examples:
        >>> fdsn = FDSN(station="OJN", channel="EHZ", client_url="https://service.iris.edu")
        >>> from datetime import datetime
        >>> stream = fdsn.get(datetime(2025, 1, 1))
    """

    def __init__(
        self,
        station: str,
        channel: str,
        network: str,
        location: str,
        channel_type: str = "D",
        client_url: str | None = None,
        download_dir: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize the FDSN adapter with station metadata and cache settings.

        Creates an ObsPy FDSN client for the given URL, ensures the local download
        directory exists, and initialises an SDS instance to serve as the local
        miniSEED cache for downloaded data.

        Args:
            station (str): Seismic station code (e.g., "OJN").
            channel (str): Channel code (e.g., "EHZ").
            network (str, optional): Seismic network code. Defaults to "VG".
            location (str, optional): Location code. Defaults to "00".
            channel_type (str, optional): Set channel type. Defaults to "D".
            client_url (str | None, optional): FDSN web service base URL.
                Defaults to "https://service.iris.edu".
            download_dir (str | None, optional): Local directory used as the SDS
                cache for downloaded miniSEED files. Created automatically if absent.
                Defaults to ``<cwd>/downloads``.
            overwrite (bool, optional): Re-download data even when a cached SDS file
                exists. Defaults to False.
            verbose (bool, optional): Emit progress log messages. Defaults to False.
            debug (bool, optional): Emit debug log messages. Defaults to False.
        """
        client_url = client_url or "https://service.iris.edu"
        client = FDSNClient(client_url)
        download_dir = download_dir or os.path.join(os.getcwd(), "downloads")

        # Ensure the cache directory exists before SDS is initialised,
        # because SDS.__init__ raises FileNotFoundError if the directory is absent.
        ensure_dir(download_dir)

        self.station = station
        self.channel = channel
        self.network = network
        self.location = location
        self.channel_type = channel_type
        self.download_dir = download_dir
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        self.client = client
        self.SDS: SDS = SDS(
            sds_dir=self.download_dir,
            station=self.station,
            channel=self.channel,
            network=self.network,
            location=self.location,
            channel_type=self.channel_type,
            verbose=verbose,
            debug=debug,
        )
        self.nslc = self.SDS.nslc

    def get(self, date: datetime) -> Stream:
        """Retrieve seismic stream for a specific date from an FDSN web service.

        Checks the local SDS cache first; downloads from the FDSN web service
        only when the cached file is absent or ``overwrite`` is True. Downloaded
        streams are saved to the SDS archive so that subsequent calls for the
        same date skip the network request.

        Args:
            date (datetime): Date for which to retrieve data. The time part is
                ignored — the full calendar day (00:00:00–23:59:59) is fetched.

        Returns:
            Stream: ObsPy Stream containing seismic waveform data for the
                requested date, or an empty Stream if no data is available.

        Raises:
            TypeError: If date is not a datetime object.

        Examples:
            >>> from datetime import datetime
            >>> fdsn = FDSN(station="OJN", channel="EHZ")
            >>> stream = fdsn.get(datetime(2025, 1, 1))
            >>> print(len(stream))  # Number of traces loaded
        """
        if not isinstance(date, datetime):
            raise TypeError("Date must be a datetime object")

        prefix = self.SDS._make_log_prefix(date)

        start_date, end_date, _, _ = normalize_dates(date, date)
        start_date_utc = UTCDateTime(start_date)
        end_date_utc = UTCDateTime(end_date)

        filepath = self.SDS.get_filepath(date)
        if os.path.exists(filepath) and not self.overwrite:
            stream: Stream = read(filepath, format="MSEED")
            return stream

        # Try to download seismic data from FDSN network
        stream = Stream()
        try:
            stream = self.client.get_waveforms(
                network=self.network,
                station=self.station,
                channel=self.channel,
                location=self.location,
                starttime=start_date_utc,
                endtime=end_date_utc,
            )

            if len(stream) == 0:
                if self.verbose:
                    logger.info(f"{prefix} Waveforms retrieved. No trace(s) found.")
                return Stream()

            if self.verbose:
                logger.info(
                    f"{prefix} Waveforms retrieved. Trace(s) found: {len(stream)}"
                )

            self.SDS.save(stream, date)
            return stream

        except FDSNException as e:
            if self.debug:
                logger.error(f"{prefix} Download failed. Error: {e}")
            return stream
