"""Verify independent HTTP stage jobs using existing OJN smoke inputs."""

import os
import json
from pathlib import Path

from fastapi.testclient import TestClient
from eruption_forecast_app.main import create_app
from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.workers.runner import run_worker


def main() -> None:
    """Run independent training, prediction, evaluation and explanation jobs."""
    directory = Path("app/backend/tests/output/sds-smoke/pipeline").resolve()
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    config = report["configuration"]
    runtime = AppSettings(state=directory / "state")
    os.environ["ERUPTION_APP_STATE"] = str(runtime.state)
    config["training"]["constructor"]["tremor_data"] = str(
        next((directory / "results").rglob("tremor-interpolated*.csv"))
    )
    config["prediction"]["constructor"]["tremor_data"] = config["training"][
        "constructor"
    ]["tremor_data"]
    completed = []
    with TestClient(create_app(runtime)) as client:
        for kind in ("training", "prediction", "evaluation", "explanation"):
            submitted = client.post(
                "/api/v1/jobs", json={"kind": kind, "configuration": config}
            )
            submitted.raise_for_status()
            job = submitted.json()
            print(f"Running independent {kind}: {job['id']}", flush=True)
            run_worker(runtime, once=True)
            result = client.get(f"/api/v1/jobs/{job['id']}").json()
            completed.append(
                {
                    "kind": kind,
                    "id": result["id"],
                    "state": result["state"],
                    "error": result["error"],
                }
            )
            (directory / "standalone-report.json").write_text(
                json.dumps(completed, indent=2), encoding="utf-8"
            )
            assert result["state"] == "succeeded", result["error"]
            artifacts = client.get(
                "/api/v1/artifacts", params={"job_id": job["id"]}
            ).json()
            if kind == "training":
                config["prediction"]["constructor"]["model"] = next(
                    item["path"]
                    for item in artifacts
                    if item["path"].endswith(".TrainingModel.pkl")
                )
            elif kind == "prediction":
                model = next(
                    item["path"]
                    for item in artifacts
                    if item["path"].endswith(".PredictionModel.pkl")
                )
                config["evaluation"]["constructor"]["model"] = model
                config["explanation"]["constructor"]["model"] = model
    print(json.dumps(completed, indent=2))


if __name__ == "__main__":
    main()
