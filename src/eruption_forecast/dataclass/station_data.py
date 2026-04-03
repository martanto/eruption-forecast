from dataclasses import field, dataclass


@dataclass
class StationData:
    """Immutable container for seismic station identity codes.

    Holds the five codes that uniquely identify a seismic channel within a
    network archive (NSLC scheme) and normalises them to uppercase on
    construction. ``nslc`` and ``snlc`` identifiers are derived automatically.

    Args:
        station (str): Station code (e.g., ``"OJN"``). Must be non-empty.
        channel (str): Channel code (e.g., ``"EHZ"``). Must be non-empty.
        network (str): Network code (e.g., ``"VG"``). Must be non-empty.
        location (str | None, optional): Location code (e.g., ``"00"``). Both
            ``None`` and ``""`` are accepted and stored as an empty string.
            Defaults to ``None``.
        channel_type (str, optional): Data type suffix appended to the channel
            directory name in SDS archives. Defaults to ``"D"``.

    Attributes:
        station (str): Station code (uppercase).
        channel (str): Channel code (uppercase).
        network (str): Network code (uppercase).
        location (str): Location code (uppercase). Always a ``str``; ``None``
            input is coerced to ``""``.
        channel_type (str): Channel type (uppercase).
        nslc (str): Standard ``Network.Station.Location.Channel`` identifier
        nslct (str): Standard ``Network.Station.Location.Channel.ChannelType``
            identifier, used by some legacy SDS filename conventions.

    Raises:
        ValueError: If ``station``, ``channel``, or ``network`` is empty or
            not a string.
        ValueError: If ``location`` is not a string and not ``None``.

    Examples:
        >>> sd = StationData(station="OJN", channel="EHZ", network="VG", location="00")
        >>> sd.nslc
        'VG.OJN.00.EHZ'
        >>> sd.snlc
        'OJN.VG.00.EHZ'
    """

    station: str
    channel: str
    network: str
    location: str = ""
    channel_type: str = "D"

    nslc: str = field(init=False, repr=False)
    nslct: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs, normalise codes to uppercase, and derive identifiers.

        Called automatically by the dataclass machinery after ``__init__``.
        Raises ``ValueError`` for any empty or wrongly-typed code, coerces
        ``location=None`` to ``""``, uppercases all codes, then builds
        ``nslc`` and ``nslct``.

        Raises:
            ValueError: If ``station``, ``channel``, or ``network`` is empty or
                not a string.
            ValueError: If ``location`` is not a string and not ``None``.
        """
        if not self.station or not isinstance(self.station, str):
            raise ValueError("station must be a non-empty string")
        if not self.channel or not isinstance(self.channel, str):
            raise ValueError("channel must be a non-empty string")
        if not self.network or not isinstance(self.network, str):
            raise ValueError("network must be a non-empty string")
        if self.location is None:
            self.location = ""
        elif not isinstance(self.location, str):
            raise ValueError("location must be a string or None")

        self.station = self.station.upper()
        self.channel = self.channel.upper()
        self.network = self.network.upper()
        self.location = self.location.upper()
        self.channel_type = self.channel_type.upper()

        self.nslc = f"{self.network}.{self.station}.{self.location}.{self.channel}"
        self.nslct = f"{self.nslc}.{self.channel_type}"
