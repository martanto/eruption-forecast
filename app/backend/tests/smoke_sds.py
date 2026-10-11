"""Run an isolated real SDS availability and calculation smoke test.

Example:
    uv run python app/backend/tests/smoke_sds.py
"""

import os
import json
from pathlib import Path

from fastapi.testclient import TestClient
from eruption_forecast_app.main import create_app
from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.workers.runner import run_worker


def main() -> None:
    """Run real jobs against the user-provided OJN archive."""
    output = Path("app/backend/tests/output/sds-smoke").resolve()
    runtime = AppSettings(state=output / "state")
    os.environ["ERUPTION_APP_STATE"] = str(runtime.state)
    with TestClient(create_app(runtime)) as client:
        config = client.get("/api/v1/configurations/example").json()
        config["setup"]["output_dir"] = str(output / "results")
        response = client.put("/api/v1/settings", json=config["setup"])
        response.raise_for_status()
        results = []
        for kind in ("availability", "calculation"):
            submitted = client.post(
                "/api/v1/jobs", json={"kind": kind, "configuration": config}
            )
            submitted.raise_for_status()
            job = submitted.json()
            run_worker(runtime, once=True)
            completed = client.get(f"/api/v1/jobs/{job['id']}").json()
            results.append(
                {
                    "kind": kind,
                    "id": job["id"],
                    "state": completed["state"],
                    "error": completed["error"],
                    "artifacts": client.get(
                        "/api/v1/artifacts", params={"job_id": job["id"]}
                    ).json(),
                }
            )
            print(json.dumps(results[-1], indent=2))
        (output / "report.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        if any(value["state"] != "succeeded" for value in results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
