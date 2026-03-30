"""Seismic data source adapters for reading waveform data in eruption forecasting.

This package provides a unified interface for retrieving seismic waveform data
from different data archives and web services. All adapters implement the
``SeismicDataSource`` abstract base class, exposing a common ``get(date)`` method
that returns an ObsPy ``Stream`` for a single day.

Key classes:
    - ``SeismicDataSource``: Abstract base class defining the ``get(date) -> Stream``
      contract that all concrete source adapters must implement.
    - ``SDS``: Reads miniSEED files from a local SeisComP Data Structure (SDS)
      archive. Supports data interpolation for gaps and handles missing files
      gracefully.
    - ``FDSN``: Downloads seismic data from any FDSN-compliant web service via the
      ObsPy FDSN client. Caches downloaded day files locally as SDS miniSEED so
      that subsequent runs skip the network request entirely.

Both ``SDS`` and ``FDSN`` are used internally by ``CalculateTremor`` and are
also available for direct use when reading seismic streams outside the pipeline.
"""

from eruption_forecast.sources.sds import SDS
from eruption_forecast.sources.base import SeismicDataSource
from eruption_forecast.sources.fdsn import FDSN


__all__ = [
    "SeismicDataSource",
    "SDS",
    "FDSN",
]
