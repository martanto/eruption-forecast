"""Requests and results for selecting backend-machine directories."""

from pydantic import Field

from eruption_forecast_app.schemas.common import Parameters


class DirectoryPickerRequest(Parameters):
    """Optional starting directory for the native folder dialog."""

    initial_path: str | None = Field(default=None, max_length=4096)


class DirectoryPickerResult(Parameters):
    """An absolute directory path, or null when the user cancels."""

    path: str | None
