"""Get data source for eruption forecast"""

from eruption_forecast.sources.sds import SDS
from eruption_forecast.sources.fdsn import FDSN


__all__ = [
    "SDS",
    "FDSN",
]