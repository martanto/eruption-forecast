"""Scientific subprocess entry point; never imported by HTTP routes."""

import sys
import json
from pathlib import Path

from loguru import logger

from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.repositories.database import Database
from eruption_forecast_app.services.pipeline.stages import (
    train,
    explain,
    predict,
    evaluate,
    calculate,
    availability,
)
from eruption_forecast_app.services.artifact_service import discover


def execute(identifier: str, runtime: AppSettings) -> None:
    """Run one persisted attempt and record partial artifacts on failure.

    Example:
        >>> execute('job-id', AppSettings())
    """
    db = Database(runtime.database)
    job = db.job(identifier)
    config = job["resolved"]
    workspace = Path(job["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)
    logger.configure(extra={"job_id": identifier})
    failed = []
    output_roots = {workspace}

    def stage(name: str, function, *args, scenario: str = ""):
        if name != "availability":
            settings, directory, *remaining = args
            selected = settings["constructor"].get("output_dir")
            if (
                selected
                and Path(selected).resolve()
                != Path(config["setup"]["output_dir"]).resolve()
            ):
                directory = (
                    Path(selected)
                    / "ui-runs"
                    / workspace.name
                    / Path(directory).relative_to(workspace)
                )
                directory.mkdir(parents=True, exist_ok=True)
            output_roots.add(Path(directory))
            args = (settings, directory, *remaining)
        db.event(identifier, name, "running", scenario=scenario)
        logger.info("Job {} · {} · {}", identifier, scenario or "standalone", name)
        try:
            result = function(*args)
            db.event(identifier, name, "succeeded", scenario=scenario)
            return result
        except Exception as exc:
            db.event(identifier, name, "failed", str(exc), scenario)
            raise

    try:
        if job["kind"] == "availability":
            stage(
                "availability",
                availability,
                config["calculation"],
                config["setup"],
                workspace,
            )
        elif job["kind"] == "calculation":
            stage("calculation", calculate, config["calculation"], workspace)
        elif job["kind"] == "scenario":
            tremor = config["training"]["constructor"].get("tremor_data")
            if not tremor:
                tremor = stage(
                    "calculation", calculate, config["calculation"], workspace
                ).df
            for scenario in config["scenarios"]:
                directory = workspace / "scenarios" / scenario["id"]
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    scenario_tremor = (
                        scenario["training"]["constructor"].get("tremor_data") or tremor
                    )
                    training = stage(
                        "training",
                        train,
                        scenario["training"],
                        directory,
                        scenario_tremor,
                        scenario=scenario["name"],
                    )
                    prediction = stage(
                        "prediction",
                        predict,
                        scenario["prediction"],
                        directory,
                        None
                        if scenario["prediction"]["constructor"].get("model")
                        else training,
                        scenario["prediction"]["constructor"].get("tremor_data")
                        or scenario_tremor,
                        scenario=scenario["name"],
                    )
                    for name, enabled, function in (
                        ("evaluation", config["evaluate_model"], evaluate),
                        ("explanation", config["explain_model"], explain),
                    ):
                        if enabled:
                            settings = scenario[name]
                            upstream = (
                                training
                                if settings["model_source"] == "training"
                                else prediction
                            )
                            if not settings["constructor"].get("eruption_dates"):
                                settings["constructor"]["eruption_dates"] = scenario[
                                    "training"
                                ]["constructor"]["eruption_dates"]
                            stage(
                                name,
                                function,
                                settings,
                                directory,
                                None
                                if settings["constructor"].get("model")
                                else upstream,
                                scenario=scenario["name"],
                            )
                except Exception as exc:
                    logger.exception("Scenario {} failed", scenario["name"])
                    failed.append(f"{scenario['name']}: {exc}")
                    db.event(
                        identifier, "scenario", "failed", str(exc), scenario["name"]
                    )
            if failed:
                state = (
                    "failed" if len(failed) == len(config["scenarios"]) else "partial"
                )
                db.update(identifier, state, "\n".join(failed))
                return
        else:
            function = {
                "training": train,
                "prediction": predict,
                "evaluation": evaluate,
                "explanation": explain,
            }[job["kind"]]
            stage(job["kind"], function, config[job["kind"]], workspace)
        db.update(identifier, "succeeded")
    except Exception as exc:
        logger.exception("Job {} failed", identifier)
        db.update(identifier, "failed", str(exc))
    finally:
        artifacts = []
        for directory in output_roots:
            artifacts.extend(discover(directory, db, identifier))
        (runtime.state / "jobs" / identifier / "artifacts.json").write_text(
            json.dumps(artifacts, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    execute(sys.argv[1], AppSettings())
