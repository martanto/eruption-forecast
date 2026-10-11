"""Job submission and lifecycle contracts."""

from typing import Any, Literal

from pydantic import Field, BaseModel

from eruption_forecast_app.schemas.common import Parameters
from eruption_forecast_app.schemas.scenario import Configuration


JobKind = Literal[
    "availability",
    "calculation",
    "training",
    "prediction",
    "evaluation",
    "explanation",
    "scenario",
]
JobState = Literal[
    "queued",
    "running",
    "succeeded",
    "partial",
    "failed",
    "cancelling",
    "cancelled",
    "interrupted",
]


class JobRequest(Parameters):
    kind: JobKind
    configuration: Configuration


class Job(BaseModel):
    id: str
    kind: JobKind
    state: JobState
    created_at: str
    updated_at: str
    configuration: dict[str, Any]
    resolved: dict[str, Any]
    workspace: str
    error: str | None = None
    pid: int | None = None
    retry_of: str | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
