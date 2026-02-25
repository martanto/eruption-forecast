"""Get data source for eruption forecast"""

from eruption_forecast.sources.sds import SDS
from eruption_forecast.sources.base import SeismicDataSource
from eruption_forecast.sources.fdsn import FDSN


__all__ = [
    "SeismicDataSource",
    "SDS",
    "FDSN",
]
