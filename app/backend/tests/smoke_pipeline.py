"""Exercise the real model stages using the OJN tremor smoke output."""

import os
import json
from pathlib import Path

from fastapi.testclient import TestClient
from eruption_forecast_app.main import create_app
from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.workers.runner import run_worker


def main() -> None:
    """Run a small scenario through the actual queued scientific adapters."""
    base = Path("app/backend/tests/output/sds-smoke").resolve()
    output = base / "pipeline"
    runtime = AppSettings(state=output / "state")
    os.environ["ERUPTION_APP_STATE"] = str(runtime.state)
    with TestClient(create_app(runtime)) as client:
        config = client.get("/api/v1/configurations/example").json()
        config["setup"].update(output_dir=str(output / "results"))
        config["calculation"]["constructor"].update(
            start_date="2025-03-15", end_date="2025-03-23", save_plot=False
        )
        config["training"]["constructor"].update(
            start_date="2025-03-15",
            end_date="2025-03-23",
            cv_splits=2,
            top_n_features=8,
        )
        existing = list((output / "results").rglob("tremor-interpolated*.csv"))
        if existing:
            config["training"]["constructor"]["tremor_data"] = str(existing[-1])
        config["training"]["fit"].update(resample_method=None)
        config["training"]["features"] = {
            "select_tremor_columns": ["rsam_f2", "rsam_f3"],
            "select_features": [
                "rsam_f2__mean",
                "rsam_f2__variance",
                "rsam_f2__maximum",
                "rsam_f2__minimum",
                "rsam_f3__mean",
                "rsam_f3__variance",
                "rsam_f3__maximum",
                "rsam_f3__minimum",
            ],
        }
        config["prediction"]["constructor"].update(
            start_date="2025-03-15", end_date="2025-03-23"
        )
        config["prediction"]["use_features_from"] = "training"
        config["explanation"]["plot"] = {
            "plot_per_seed": False,
            "plot_aggregate": True,
            "max_display": 3,
        }
        client.put("/api/v1/settings", json=config["setup"]).raise_for_status()
        submitted = client.post(
            "/api/v1/jobs", json={"kind": "scenario", "configuration": config}
        )
        submitted.raise_for_status()
        job = submitted.json()
        print(f"Running scenario {job['id']}", flush=True)
        run_worker(runtime, once=True)
        completed = client.get(f"/api/v1/jobs/{job['id']}").json()
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(
            json.dumps(completed, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "id": job["id"],
                    "state": completed["state"],
                    "error": completed["error"],
                    "events": completed["events"],
                },
                indent=2,
            )
        )
        if completed["state"] != "succeeded":
            raise SystemExit(1)
        cache_files = list(Path(completed["workspace"]).rglob("*.TrainingModel.pkl"))
        original_times = {str(path): path.stat().st_mtime_ns for path in cache_files}
        retry = client.post(f"/api/v1/jobs/{job['id']}/retry")
        retry.raise_for_status()
        run_worker(runtime, once=True)
        retried = client.get(f"/api/v1/jobs/{retry.json()['id']}").json()
        assert retried["state"] == "succeeded", retried["error"]
        assert original_times and all(
            Path(path).stat().st_mtime_ns == stamp
            for path, stamp in original_times.items()
        ), "Training cache was recomputed during retry"
        assert client.get(
            "/api/v1/artifacts", params={"job_id": retry.json()["id"]}
        ).json()
        (output / "retry-report.json").write_text(
            json.dumps(retried, indent=2), encoding="utf-8"
        )
        print(
            f"Retry succeeded with unchanged training cache: {retry.json()['id']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
