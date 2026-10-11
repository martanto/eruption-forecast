"""Application contracts, queue guarantees and filesystem containment."""

import ast
import importlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from pydantic import ValidationError

from eruption_forecast_app.generate_parameters import ROOT, SPECS
from eruption_forecast_app.main import create_app
from eruption_forecast_app.repositories.database import Database
from eruption_forecast_app.schemas.scenario import Configuration, resolve
from eruption_forecast_app.schemas.stages import Training
from eruption_forecast_app.services.artifact_service import preview
from eruption_forecast_app.services.path_service import checked
from eruption_forecast_app.settings import AppSettings


@pytest.fixture
def client(tmp_path):
    runtime = AppSettings(state=tmp_path / "state")
    with TestClient(create_app(runtime)) as value:
        output = tmp_path / "output"
        output.mkdir()
        assert value.put("/api/v1/settings", json={"output_dir": str(output)}).status_code == 200
        yield value


def test_openapi_is_reproducible_and_side_effect_free(tmp_path):
    runtime = AppSettings(state=tmp_path / "absent")
    generated = create_app(runtime).openapi()
    assert not runtime.state.exists()
    assert generated == json.loads((ROOT / "openapi.json").read_text())
    assert "JobRequest" in generated["components"]["schemas"]


@pytest.mark.parametrize("stage", list(SPECS))
def test_every_public_parameter_is_represented(stage):
    source, methods = SPECS[stage]
    tree = ast.parse((ROOT / "src/eruption_forecast" / source).read_text(encoding="utf-8"))
    module = importlib.import_module(f"eruption_forecast_app.schemas.{stage}")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in methods:
            expected = {arg.arg for arg in node.args.args} - {"self", "df", "label_df"}
            model = getattr(module, methods[node.name])
            assert expected == set(model.model_fields)
            defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
            for arg, default in zip(node.args.args, defaults, strict=True):
                if arg.arg in expected and default is not None and not isinstance(default, ast.Name):
                    assert model.model_fields[arg.arg].default == ast.literal_eval(default)


def test_sparse_overrides_preserve_explicit_nulls_and_general_defaults():
    value = Configuration.model_validate({"setup": {"n_jobs": 3}, "training": {"fit": {"seeds": 8}}, "scenarios": [{"id": "a", "training": {"constructor": {"n_jobs": None}, "fit": {"seeds": 2, "resample_method": None}}}, {"id": "b"}]})
    result = resolve(value)
    assert result["scenarios"][0]["training"]["constructor"]["n_jobs"] == 3
    assert result["scenarios"][0]["training"]["fit"]["seeds"] == 2
    assert result["scenarios"][0]["training"]["fit"]["resample_method"] is None
    assert result["scenarios"][1]["training"]["fit"]["seeds"] == 8
    assert result["training"]["constructor"]["save_model"] is True


@pytest.mark.parametrize("value", [{"fit": {"seeds": 0}}, {"features": {"minimum_completion": 1.5}}, {"constructor": {"classifiers": "unknown"}}, {"fit": {"typo": True}}])
def test_reject_invalid_scientific_parameters(value):
    with pytest.raises(ValidationError):
        Training.model_validate(value)


def test_duplicate_scenario_ids_rejected():
    with pytest.raises(ValidationError):
        Configuration.model_validate({"scenarios": [{"id": "a"}, {"id": "a"}]})


def test_path_traversal_rejected(tmp_path):
    root = tmp_path / "approved"
    root.mkdir()
    with pytest.raises(ValueError, match="outside"):
        checked(root / "../secret", [root], must_exist=False)


def test_queue_claim_is_atomic_and_only_one_job_runs(tmp_path):
    db = Database(tmp_path / "queue.sqlite3")
    db.initialize()
    for index in range(4):
        db.create_job("calculation", {}, {}, f"output/{index}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda _: db.claim(), range(4)))
    assert sum(item is not None for item in claimed) == 1
    assert sum(job["state"] == "queued" for job in db.jobs()) == 3


def test_cancellation_cannot_be_overwritten_by_success(tmp_path):
    db = Database(tmp_path / "queue.sqlite3")
    db.initialize()
    job = db.create_job("calculation", {}, {}, "output")
    db.update(job["id"], "cancelling")
    db.update(job["id"], "succeeded")
    assert db.job(job["id"])["state"] == "cancelled"


def test_routes_and_unknown_resource(client):
    assert client.get("/api/v1/health").json() == {"status": "ready"}
    assert client.get("/api/v1/parameters").status_code == 200
    assert client.get("/api/v1/jobs/missing").status_code == 404
    assert client.get("/docs").status_code == 200
    assert client.get("/api/v1/system").status_code == 200


def test_invalid_submission_does_not_enqueue(client):
    response = client.post("/api/v1/jobs", json={"kind": "training", "configuration": {"setup": client.get("/api/v1/settings").json()}})
    assert response.status_code == 422
    assert client.get("/api/v1/jobs").json() == []


def test_cross_origin_submission_rejected(client):
    assert client.post("/api/v1/jobs", json={}, headers={"Origin": "https://untrusted.example"}).status_code == 403


def test_yaml_round_trip_preserves_sparse_configuration(client):
    configuration = {"name": "Experiment", "training": {"fit": {"seeds": 2}}}
    exported = client.post("/api/v1/configurations/export", json=configuration)
    imported = client.post("/api/v1/configurations/import", json={"content": exported.text})
    assert imported.status_code == 200
    assert imported.json()["training"]["fit"]["seeds"] == 2


def test_bounded_csv_preview_and_chart(tmp_path):
    path = tmp_path / "tremor.csv"
    path.write_text("datetime,rsam_f0\n" + "\n".join(f"2025-03-16T00:{i % 60:02d}:00,{i}" for i in range(100)))
    assert len(preview(path, 5, 5)["rows"]) == 5
    chart = preview(path, 10, chart=True)
    assert len(chart["rows"]) <= 10
    assert chart["total"] == 100


def test_artifact_discovery_does_not_load_pickles(client):
    directory = Path(client.get("/api/v1/settings").json()["output_dir"])
    (directory / "untrusted.pkl").write_bytes(b"not a pickle")
    response = client.post("/api/v1/artifacts/discover", json={"path": str(directory)})
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_submit_cancel_and_retry_with_persistent_history(client):
    configuration = client.get("/api/v1/configurations/example").json()
    configuration["setup"] = client.get("/api/v1/settings").json()
    configuration["calculation"]["source"] = "fdsn"
    response = client.post("/api/v1/jobs", json={"kind": "calculation", "configuration": configuration})
    assert response.status_code == 202
    job = response.json()
    assert client.post(f"/api/v1/jobs/{job['id']}/cancel").json()["state"] == "cancelled"
    retried = client.post(f"/api/v1/jobs/{job['id']}/retry")
    assert retried.status_code == 202
    assert retried.json()["workspace"] == job["workspace"]
    assert retried.json()["retry_of"] == job["id"]


def test_retries_share_artifacts_without_losing_attempt_history(tmp_path):
    db = Database(tmp_path / "queue.sqlite3")
    db.initialize()
    path = tmp_path / "tremor.csv"
    path.write_text("datetime,value\n2025-03-16,1")
    first = db.artifact(path, "first")
    second = db.artifact(path, "retry")
    assert first["id"] == second["id"]
    assert db.artifacts("first")[0]["id"] == first["id"]
    assert db.artifacts("retry")[0]["id"] == first["id"]


def test_parquet_preview_applies_offset_and_limit(tmp_path):
    path = tmp_path / "features.parquet"
    pd.DataFrame({"value": range(40)}).to_parquet(path)
    values = preview(path, limit=5, offset=10)
    assert [row["value"] for row in values["rows"]] == list(range(10, 15))
    assert values["total"] == 40


def test_cancellation_survives_executor_pid_registration(tmp_path):
    db = Database(tmp_path / "queue.sqlite3")
    db.initialize()
    job = db.create_job("calculation", {}, {}, "output")
    db.update(job["id"], "cancelling")
    db.update(job["id"], "running", pid=123)
    assert db.job(job["id"])["state"] == "cancelling"


def test_plot_aliases_are_forwarded_with_explicit_forecast_precedence():
    value = Configuration.model_validate({"prediction": {"plot": {"threshold": 0.8, "title": "Forecast"}}, "scenarios": [{"id": "a", "prediction": {"forecast": {"plot_threshold": 0.6}}}]})
    result = resolve(value)
    assert result["prediction"]["forecast"]["plot_threshold"] == 0.8
    assert result["prediction"]["forecast"]["plot_title"] == "Forecast"
    assert result["scenarios"][0]["prediction"]["forecast"]["plot_threshold"] == 0.6
