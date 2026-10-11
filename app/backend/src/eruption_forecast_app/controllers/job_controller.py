"""Resolve and validate configurations before scientific work is queued."""

from uuid import uuid4
from pathlib import Path
from datetime import datetime

from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.schemas.job import JobRequest
from eruption_forecast_app.schemas.scenario import resolve
from eruption_forecast_app.repositories.database import Database
from eruption_forecast_app.services.path_service import roots, checked, validate_setup


def _validate_stage(
    stage: str, value: dict, allowed: list[Path], upstream: bool = False
) -> None:
    constructor = value["constructor"]
    if stage in {"calculation", "training", "prediction"}:
        start, end = constructor.get("start_date"), constructor.get("end_date")
        if not start or not end:
            raise ValueError(
                f"{stage}.constructor: start_date and end_date are required"
            )
        if datetime.fromisoformat(start) >= datetime.fromisoformat(end):
            raise ValueError(f"{stage}.constructor.end_date: must be after start_date")
        if (
            stage in {"training", "prediction"}
            and not value.get("load_features", {}).get("features_matrix_path")
            and (datetime.fromisoformat(end) - datetime.fromisoformat(start)).days < 7
        ):
            raise ValueError(
                f"{stage}.constructor.end_date: label building requires at least seven days between start and end"
            )
    for key, path in constructor.items():
        if key in {"tremor_data", "model"} and path:
            checked(path, allowed)
        if key in {"output_dir", "root_dir"} and path:
            checked(path, allowed, must_exist=False)
    if stage == "training" and not constructor.get("eruption_dates"):
        raise ValueError(
            "training.constructor.eruption_dates: provide known eruption dates"
        )
    if (
        stage in {"training", "prediction"}
        and not constructor.get("tremor_data")
        and not upstream
    ):
        raise ValueError(f"{stage}.constructor.tremor_data: select a tremor CSV")
    if (
        stage in {"prediction", "evaluation", "explanation"}
        and not constructor.get("model")
        and not upstream
    ):
        raise ValueError(
            f"{stage}.constructor.model: select an upstream model artifact"
        )
    if (
        stage == "training"
        and value["label"].get("builder") == "dynamic"
        and not value["label"].get("days_before_eruption")
    ):
        raise ValueError(
            "training.label.days_before_eruption: required for dynamic labels"
        )
    for key in ("features_matrix_path", "label_features_csv", "select_features"):
        path = value.get("load_features", {}).get(key)
        if isinstance(path, str) and path:
            checked(path, allowed)
    for key in ("select_features",):
        path = value.get("features", {}).get(key)
        if isinstance(path, str) and path:
            checked(path, allowed)
    if stage == "prediction" and value["use_features_from"] == "files":
        if not all(
            value["load_features"].get(key)
            for key in ("features_matrix_path", "label_features_csv")
        ):
            raise ValueError(
                "prediction.load_features: select both feature matrix and label CSV for files mode"
            )
    if stage == "calculation":
        if not value["freq_bands"] or any(
            lo <= 0 or lo >= hi for lo, hi in value["freq_bands"]
        ):
            raise ValueError(
                "calculation.freq_bands: each band requires 0 < lower < upper"
            )
        if value["source"] == "sds":
            if not value.get("sds_dir"):
                raise ValueError("calculation.sds_dir: select an SDS directory")
            checked(value["sds_dir"], allowed)


def submit(
    request: JobRequest,
    db: Database,
    runtime: AppSettings,
    workspace: str | None = None,
    retry_of: str | None = None,
) -> dict:
    """Validate and enqueue a reproducible job.

    Example:
        >>> submit(job_request, database, AppSettings())
    """
    effective = resolve(request.configuration)
    effective["setup"] = validate_setup(effective["setup"])
    saved = db.value("setup")
    if saved is None:
        db.set_value("setup", effective["setup"])
    allowed = roots(db, runtime)
    checked(effective["setup"]["output_dir"], allowed, must_exist=False)
    effective["calculation"]["sds_dir"] = (
        effective["calculation"]["sds_dir"] or effective["setup"]["sds_dir"]
    )
    if request.kind == "availability":
        constructor = effective["calculation"]["constructor"]
        if not effective["setup"]["sds_dir"]:
            raise ValueError("setup.sds_dir: select an SDS directory")
        checked(effective["setup"]["sds_dir"], allowed)
        if not constructor.get("start_date") or not constructor.get("end_date"):
            raise ValueError("calculation.constructor: set the availability date range")
        if datetime.fromisoformat(constructor["start_date"]) > datetime.fromisoformat(
            constructor["end_date"]
        ):
            raise ValueError("Availability end date must be on or after start date")
    elif request.kind == "scenario":
        if not effective["scenarios"]:
            raise ValueError("scenarios: add at least one scenario")
        if not effective["training"]["constructor"].get("tremor_data"):
            _validate_stage("calculation", effective["calculation"], allowed)
        for scenario in effective["scenarios"]:
            for stage in ("training", "prediction", "evaluation", "explanation"):
                if stage == "evaluation" and not effective["evaluate_model"]:
                    continue
                if stage == "explanation" and not effective["explain_model"]:
                    continue
                _validate_stage(stage, scenario[stage], allowed, upstream=True)
    else:
        _validate_stage(request.kind, effective[request.kind], allowed)
    directory = workspace or str(
        Path(effective["setup"]["output_dir"]) / "ui-runs" / uuid4().hex[:12]
    )
    checked(directory, allowed, must_exist=False)
    return db.create_job(
        request.kind,
        request.configuration.model_dump(exclude_unset=True, mode="json"),
        effective,
        directory,
        retry_of,
    )
