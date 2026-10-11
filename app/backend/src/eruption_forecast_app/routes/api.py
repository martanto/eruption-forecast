"""Local configuration, job, artifact, and system routes."""

import os
import json
from typing import Any, Annotated
from pathlib import Path

import yaml
import psutil
from fastapi import Query, Depends, APIRouter, HTTPException
from pydantic import Field, BaseModel
from fastapi.responses import Response, FileResponse

from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.schemas.job import Job, JobRequest
from eruption_forecast_app.dependencies import database, settings
from eruption_forecast_app.schemas.path import (
    DirectoryPickerResult,
    DirectoryPickerRequest,
)
from eruption_forecast_app.schemas.setup import Setup
from eruption_forecast_app.schemas.common import Parameters
from eruption_forecast_app.schemas.scenario import Configuration, resolve
from eruption_forecast_app.repositories.database import Database
from eruption_forecast_app.services.path_service import roots, checked, validate_setup
from eruption_forecast_app.services.artifact_service import preview, discover
from eruption_forecast_app.services.directory_picker import (
    DirectoryPickerBusy,
    DirectoryPickerUnavailable,
    pick_directory,
    cancel_directory_picker,
)
from eruption_forecast_app.controllers.job_controller import submit
from eruption_forecast_app.services.parameter_catalog import catalog


router = APIRouter(prefix="/api/v1")
DB = Annotated[Database, Depends(database)]
Runtime = Annotated[AppSettings, Depends(settings)]


class PathRequest(Parameters):
    path: str


class ImportRequest(Parameters):
    content: str = Field(max_length=1_000_000)


class Profile(Parameters):
    name: str = Field(min_length=1, max_length=120)
    configuration: Configuration


class Logs(BaseModel):
    text: str
    cursor: int


@router.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Check API readiness.

    Example:
        GET /api/v1/health
    """
    return {"status": "ready"}


@router.get("/parameters", tags=["configuration"])
def parameters() -> dict[str, Any]:
    """Return the complete form catalog.

    Example:
        GET /api/v1/parameters
    """
    return catalog()


@router.get("/settings", response_model=Setup, tags=["configuration"])
def get_setup(db: DB) -> Any:
    """Read directory and station defaults.

    Example:
        GET /api/v1/settings
    """
    return db.value("setup", Setup().model_dump())


@router.put("/settings", response_model=Setup, tags=["configuration"])
def save_setup(value: Setup, db: DB) -> Any:
    """Validate and save approved directory roots.

    Example:
        PUT /api/v1/settings with a Setup JSON body.
    """
    result = validate_setup(value.model_dump())
    db.set_value("setup", result)
    return result


@router.post("/paths/validate", tags=["configuration"])
def validate_path(value: PathRequest, db: DB, runtime: Runtime) -> dict:
    """Inspect a path within configured roots.

    Example:
        POST /api/v1/paths/validate with {"path": "output"}.
    """
    path = checked(value.path, roots(db, runtime), must_exist=False)
    return {
        "path": str(path),
        "exists": path.exists(),
        "directory": path.is_dir(),
        "writable": os.access(path if path.exists() else path.parent, os.W_OK),
    }


@router.delete("/paths/pick-directory", tags=["configuration"])
def cancel_folder_selection() -> dict[str, bool]:
    """Cancel the backend's active native folder dialog, if any."""
    cancel_directory_picker()
    return {"cancelled": True}


@router.post("/paths/pick-directory", response_model=DirectoryPickerResult, tags=["configuration"])
def select_directory(value: DirectoryPickerRequest) -> DirectoryPickerResult:
    """Open the backend computer's native directory dialog.

    Args:
        value: Optional initial absolute directory.

    Returns:
        An absolute path, or null when the dialog is cancelled.

    Raises:
        HTTPException: A dialog is already open or selection is unavailable.
    """
    try:
        return DirectoryPickerResult(path=pick_directory(value.initial_path))
    except DirectoryPickerBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    except DirectoryPickerUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/paths/browse", tags=["configuration"])
def browse(db: DB, runtime: Runtime, path: str | None = None) -> list[dict]:
    """Browse configured roots, never arbitrary filesystem locations.

    Example:
        GET /api/v1/paths/browse
    """
    allowed = roots(db, runtime)
    targets = (
        sorted(
            checked(path, allowed).iterdir(),
            key=lambda p: (not p.is_dir(), p.name.lower()),
        )
        if path
        else allowed
    )
    result = []
    for target in targets[:500]:
        if target.exists():
            checked(target, allowed)
            result.append(
                {"name": target.name, "path": str(target), "directory": target.is_dir()}
            )
    return result


@router.get("/configurations", tags=["configuration"])
def profiles(db: DB) -> list[dict]:
    """List named saved research configurations.

    Example:
        GET /api/v1/configurations
    """
    return db.value("profiles", [])


@router.post("/configurations", response_model=Profile, tags=["configuration"])
def save_profile(value: Profile, db: DB) -> Profile:
    """Save a profile, replacing the matching name.

    Example:
        POST /api/v1/configurations with a named configuration.
    """
    values = [p for p in db.value("profiles", []) if p["name"] != value.name]
    values.append(value.model_dump(exclude_unset=True, mode="json"))
    db.set_value("profiles", values)
    return value


@router.post("/configurations/resolve", tags=["configuration"])
def resolved(value: Configuration) -> dict:
    """Inspect inherited settings before submission.

    Example:
        POST /api/v1/configurations/resolve with Configuration JSON.
    """
    return resolve(value)


@router.post(
    "/configurations/import",
    response_model=Configuration,
    response_model_exclude_unset=True,
    tags=["configuration"],
)
def import_config(value: ImportRequest) -> Configuration:
    """Validate an application JSON or YAML configuration.

    Example:
        POST /api/v1/configurations/import with a content string.
    """
    return Configuration.model_validate(yaml.safe_load(value.content))


@router.post("/configurations/export", tags=["configuration"])
def export_config(value: Configuration, format: str = "yaml") -> Response:
    """Download a reproducible configuration.

    Example:
        POST /api/v1/configurations/export?format=json
    """
    data = value.model_dump(mode="json", exclude_unset=True)
    if format not in {"json", "yaml"}:
        raise ValueError("Export format must be json or yaml")
    content = (
        json.dumps(data, indent=2)
        if format == "json"
        else yaml.safe_dump(data, sort_keys=False)
    )
    return Response(
        content,
        media_type="application/json" if format == "json" else "application/yaml",
        headers={"Content-Disposition": f'attachment; filename="forecast-ui.{format}"'},
    )


@router.get(
    "/configurations/example",
    response_model=Configuration,
    response_model_exclude_unset=True,
    tags=["configuration"],
)
def example() -> Configuration:
    """Provide a small OJN preset inspired by forecast-scenario.py.

    Example:
        GET /api/v1/configurations/example
    """
    return Configuration.model_validate(
        {
            "name": "OJN smoke workflow",
            "setup": {
                "sds_dir": r"D:\Data\OJN",
                "network": "VG",
                "station": "OJN",
                "location": "00",
                "channel": "EHZ",
            },
            "calculation": {
                "constructor": {
                    "start_date": "2025-03-01",
                    "end_date": "2025-03-22",
                    "methods": ["rsam", "dsar", "entropy"],
                    "save_plot": True,
                    "interpolate": True,
                }
            },
            "training": {
                "constructor": {
                    "start_date": "2025-03-01",
                    "end_date": "2025-03-22",
                    "classifiers": "lite-rf",
                    "eruption_dates": ["2025-03-20"],
                    "top_n_features": 5,
                },
                "label": {"window_step": 6, "window_step_unit": "hours"},
                "fit": {"seeds": 1, "resample_method": "under"},
            },
            "prediction": {
                "constructor": {"start_date": "2025-03-15", "end_date": "2025-03-22"},
                "label": {"window_step": 6, "window_step_unit": "hours"},
            },
            "evaluation": {"constructor": {"eruption_dates": ["2025-03-20"]}},
            "explanation": {"constructor": {"eruption_dates": ["2025-03-20"]}},
            "scenarios": [
                {
                    "id": "ojn-smoke",
                    "name": "OJN smoke",
                    "description": "Small real-data workflow; not a validated scientific experiment.",
                }
            ],
        }
    )


@router.post("/jobs", response_model=Job, status_code=202, tags=["jobs"])
def create_job(value: JobRequest, db: DB, runtime: Runtime) -> dict:
    """Validate and queue independent or scenario work.

    Example:
        POST /api/v1/jobs with kind and configuration.
    """
    return submit(value, db, runtime)


@router.get("/jobs", response_model=list[Job], tags=["jobs"])
def jobs(db: DB, limit: int = Query(100, ge=1, le=500)) -> list[dict]:
    """List persisted job history.

    Example:
        GET /api/v1/jobs
    """
    return db.jobs(limit)


@router.get("/jobs/{identifier}", response_model=Job, tags=["jobs"])
def job(identifier: str, db: DB) -> dict:
    """Read job status and stage events.

    Example:
        GET /api/v1/jobs/job-id
    """
    return db.job(identifier)


@router.post("/jobs/{identifier}/cancel", response_model=Job, tags=["jobs"])
def cancel(identifier: str, db: DB) -> dict:
    """Cancel queued work or request process-tree termination.

    Example:
        POST /api/v1/jobs/job-id/cancel
    """
    value = db.cancel(identifier)
    if value is None:
        raise HTTPException(409, "Only queued or running jobs can be cancelled")
    return value


@router.post(
    "/jobs/{identifier}/retry", response_model=Job, status_code=202, tags=["jobs"]
)
def retry(identifier: str, db: DB, runtime: Runtime) -> dict:
    """Create a new attempt with the original workspace and caches.

    Example:
        POST /api/v1/jobs/job-id/retry
    """
    original = db.job(identifier)
    if original["state"] in {"queued", "running", "cancelling"}:
        raise HTTPException(409, "Wait for the original job to finish before retrying")
    return submit(
        JobRequest(kind=original["kind"], configuration=original["configuration"]),
        db,
        runtime,
        original["workspace"],
        identifier,
    )


@router.get("/jobs/{identifier}/logs", response_model=Logs, tags=["jobs"])
def logs(
    identifier: str, db: DB, runtime: Runtime, cursor: int = Query(0, ge=0)
) -> dict:
    """Read a bounded log chunk using a byte cursor.

    Example:
        GET /api/v1/jobs/job-id/logs?cursor=0
    """
    db.job(identifier)
    path = runtime.state / "jobs" / identifier / "execution.log"
    if not path.exists():
        return {"text": "", "cursor": cursor}
    with path.open("rb") as stream:
        stream.seek(min(cursor, path.stat().st_size))
        content = stream.read(64000)
        position = stream.tell()
    return {"text": content.decode("utf-8", errors="replace"), "cursor": position}


@router.post("/artifacts/discover", tags=["artifacts"])
def index_artifacts(value: PathRequest, db: DB, runtime: Runtime) -> list[dict]:
    """Index an existing configured output or tremor directory.

    Example:
        POST /api/v1/artifacts/discover with a configured path.
    """
    directory = checked(value.path, roots(db, runtime))
    if not directory.is_dir():
        raise ValueError("Select a directory to discover artifacts")
    return discover(directory, db)


@router.get("/artifacts", tags=["artifacts"])
def artifacts(db: DB, job_id: str | None = None) -> list[dict]:
    """List artifact metadata, including existing local outputs.

    Example:
        GET /api/v1/artifacts?job_id=job-id
    """
    return db.artifacts(job_id)


def _artifact(identifier: str, db: Database, runtime: AppSettings) -> Path:
    with db.connect() as connection:
        row = connection.execute(
            "SELECT path FROM artifacts WHERE id=?", (identifier,)
        ).fetchone()
    if not row:
        raise KeyError(identifier)
    return checked(row[0], roots(db, runtime))


@router.get("/artifacts/{identifier}/download", tags=["artifacts"])
def download(identifier: str, db: DB, runtime: Runtime) -> FileResponse:
    """Download or display a registered file.

    Example:
        GET /api/v1/artifacts/artifact-id/download
    """
    path = _artifact(identifier, db, runtime)
    return FileResponse(
        path,
        filename=path.name,
        content_disposition_type="inline"
        if path.suffix in {".png", ".pdf"}
        else "attachment",
    )


@router.get("/artifacts/{identifier}/preview", tags=["artifacts"])
def table(
    identifier: str,
    db: DB,
    runtime: Runtime,
    limit: int = Query(200, ge=1, le=2000),
    offset: int = Query(0, ge=0, le=10000),
    chart: bool = False,
) -> dict:
    """Preview bounded tabular data.

    Example:
        GET /api/v1/artifacts/artifact-id/preview?chart=true
    """
    return preview(_artifact(identifier, db, runtime), limit, offset, chart)


@router.get("/system", tags=["system"])
def system(db: DB, runtime: Runtime) -> dict:
    """Report actual machine resources and worker health.

    Example:
        GET /api/v1/system
    """
    memory = psutil.virtual_memory()
    disks = []
    for path in roots(db, runtime):
        existing = path
        while not existing.exists():
            existing = existing.parent
        usage = psutil.disk_usage(str(existing))
        disks.append(
            {
                "path": str(path),
                "total": usage.total,
                "free": usage.free,
                "percent": usage.percent,
            }
        )
    counts: dict[str, int] = {}
    with db.connect() as connection:
        counts = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT state,count(*) FROM jobs GROUP BY state"
            )
        }
    return {
        "cpu_percent": psutil.cpu_percent(),
        "cpu_count": psutil.cpu_count(),
        "package_cpu_budget": max(1, (psutil.cpu_count() or 1) - 2),
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "percent": memory.percent,
        },
        "disks": disks,
        "queue": counts,
        "worker": db.value("worker", {}),
    }
