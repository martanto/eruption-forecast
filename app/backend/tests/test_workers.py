"""Worker recovery and scenario failure isolation."""

from pathlib import Path
import os
import sys
import time
import subprocess

import psutil

from eruption_forecast_app.repositories.database import Database
from eruption_forecast_app.schemas.scenario import Configuration, resolve
from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.workers import executor
from eruption_forecast_app.workers.runner import run_worker
from eruption_forecast_app.workers.cancellation import terminate_tree


def test_restart_marks_abandoned_attempt_interrupted(tmp_path):
    runtime = AppSettings(state=tmp_path / "state")
    db = Database(runtime.database)
    db.initialize()
    job = db.create_job("training", {}, {}, str(tmp_path / "results"))
    db.update(job["id"], "running")
    run_worker(runtime, once=True)
    assert db.job(job["id"])["state"] == "interrupted"
    assert db.value("worker")["stopped"] is True


def test_scenario_failure_does_not_prevent_later_scenario(tmp_path, monkeypatch):
    runtime = AppSettings(state=tmp_path / "state")
    db = Database(runtime.database)
    db.initialize()
    workspace = tmp_path / "results" / "run"
    configuration = Configuration.model_validate({
        "setup": {"output_dir": str(tmp_path / "results")},
        "training": {"constructor": {"tremor_data": "existing.csv", "eruption_dates": ["2025-03-20"]}},
        "evaluate_model": False,
        "explain_model": False,
        "scenarios": [{"id": "bad", "name": "Bad", "training": {"constructor": {"classifiers": "rf"}}}, {"id": "good", "name": "Good", "training": {"constructor": {"classifiers": "gb"}}}],
    })
    calls = []

    def train(config, directory, tremor):
        calls.append(config["constructor"]["classifiers"])
        if calls[-1] == "rf":
            raise ValueError("Training failure")
        Path(directory, "completed.txt").write_text("real adapter would write here")
        return object()

    monkeypatch.setattr(executor, "train", train)
    monkeypatch.setattr(executor, "predict", lambda *args: object())
    job = db.create_job("scenario", configuration.model_dump(), resolve(configuration), str(workspace))
    (runtime.state / "jobs" / job["id"]).mkdir(parents=True)
    executor.execute(job["id"], runtime)
    completed = db.job(job["id"])
    assert completed["state"] == "partial"
    assert calls == ["rf", "gb"]
    assert any(event["stage"] == "prediction" and event["scenario"] == "Good" and event["state"] == "succeeded" for event in completed["events"])
    assert db.artifacts(job["id"])


def test_cancellation_terminates_real_owned_process_tree(tmp_path):
    child = tmp_path / "child.py"
    child.write_text("import time\ntime.sleep(60)\n")
    parent = tmp_path / "parent.py"
    parent.write_text("import os\nimport subprocess\nimport sys\nimport time\nfrom pathlib import Path\nchild = subprocess.Popen([sys.executable, sys.argv[1]], creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)\nPath(sys.argv[2]).write_text(str(child.pid))\ntime.sleep(60)\n")
    marker = tmp_path / "child.pid"
    process = subprocess.Popen([sys.executable, str(parent), str(child), str(marker)], creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.exists(), "Parent did not create its child"
        child_pid = int(marker.read_text())
        terminate_tree(process.pid)
        process.wait(timeout=5)
        assert not psutil.pid_exists(child_pid) or psutil.Process(child_pid).status() == psutil.STATUS_ZOMBIE
    finally:
        if process.poll() is None:
            terminate_tree(process.pid)
            process.wait(timeout=5)
