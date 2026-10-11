"""Request dependencies without import-time state initialization."""

from fastapi import Request

from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.repositories.database import Database


def database(request: Request) -> Database:
    """Return the app-scoped repository.

    Example:
        ``Depends(database)`` injects the repository into a route.
    """
    return request.app.state.database


def settings(request: Request) -> AppSettings:
    """Return application paths.

    Example:
        ``Depends(settings)`` supplies isolated runtime paths.
    """
    return request.app.state.settings
