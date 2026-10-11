"""General research settings."""

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters


class Setup(Parameters):
    """Directory and station defaults for local workflows."""

    sds_dir: str = ""
    output_dir: str = "output"
    tremor_dir: str = ""
    root_dir: str | None = None
    network: str = "VG"
    station: str = "OJN"
    location: str = "00"
    channel: str = "EHZ"
    channel_type: str = "D"
    n_jobs: int = Field(default=1, ge=1)
    window_size: int = Field(default=2, ge=1)
    verbose: bool = False
