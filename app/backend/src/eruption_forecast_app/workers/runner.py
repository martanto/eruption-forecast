"""Consume SQLite jobs without doing scientific work in the API."""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

import psutil
from loguru import logger

from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.workers.cancellation import terminate_tree
from eruption_forecast_app.repositories.database import Database, timestamp
from eruption_forecast_app.services.artifact_service import discover


def run_worker(runtime: AppSettings, once: bool = False) -> None:
    """Process queued work with a persistent heartbeat and ownership lock.

    Example:
        >>> run_worker(AppSettings(), once=True)
    """
    db = Database(runtime.database)
    db.initialize()
    owner = {"pid": os.getpid(), "started": psutil.Process().create_time()}
    with db.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT value FROM settings WHERE key='worker'"
        ).fetchone()
        prior = json.loads(row[0]) if row else {}
        if prior.get("pid") and psutil.pid_exists(prior["pid"]):
            process = psutil.Process(prior["pid"])
            if abs(process.create_time() - prior.get("started", 0)) < 0.1:
                raise RuntimeError("A worker already owns this application database")
        connection.execute(
            "INSERT INTO settings VALUES ('worker',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (json.dumps(owner),),
        )
        stale = [
            dict(r)
            for r in connection.execute(
                "SELECT * FROM jobs WHERE state IN ('running','cancelling')"
            )
        ]
    for job in stale:
        if job["pid"] and psutil.pid_exists(job["pid"]):
            process = psutil.Process(job["pid"])
            if "eruption_forecast_app.workers.executor" in " ".join(process.cmdline()):
                terminate_tree(job["pid"])
        db.update(
            job["id"], "interrupted", "Worker stopped before this attempt completed"
        )
    child = None
    try:
        while True:
            db.set_value("worker", {**owner, "heartbeat": timestamp()})
            job = db.claim()
            if not job:
                if once:
                    return
                time.sleep(1)
                continue
            directory = runtime.state / "jobs" / job["id"]
            directory.mkdir(parents=True, exist_ok=True)
            for name, value in (
                ("configuration", job["configuration"]),
                ("resolved-configuration", job["resolved"]),
            ):
                (directory / f"{name}.json").write_text(
                    json.dumps(value, indent=2), encoding="utf-8"
                )
            environment = {
                **os.environ,
                "ERUPTION_APP_STATE": str(runtime.state),
                "MPLBACKEND": "Agg",
                "PYTHONUNBUFFERED": "1",
            }
            with (directory / "execution.log").open("ab", buffering=0) as stream:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "eruption_forecast_app.workers.executor",
                        job["id"],
                    ],
                    cwd=runtime.root,
                    env=environment,
                    stdout=stream,
                    stderr=stream,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                db.update(job["id"], "running", pid=child.pid)
                while child.poll() is None:
                    db.set_value(
                        "worker",
                        {**owner, "heartbeat": timestamp(), "job_id": job["id"]},
                    )
                    if db.job(job["id"])["state"] == "cancelling":
                        terminate_tree(child.pid)
                        db.update(job["id"], "cancelled")
                    time.sleep(0.5)
            status = db.job(job["id"])["state"]
            if status == "running":
                db.update(
                    job["id"],
                    "failed",
                    f"Executor exited with code {child.returncode}; see job logs",
                )
            elif status == "cancelling":
                db.update(job["id"], "cancelled")
            artifacts = discover(Path(job["workspace"]), db, job["id"])
            manifest = directory / "artifacts.json"
            if not manifest.exists():
                manifest.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
            child = None
            if once:
                return
    finally:
        if child and child.poll() is None:
            terminate_tree(child.pid)
            db.update(job["id"], "interrupted", "Worker shut down")
        db.set_value("worker", {"heartbeat": timestamp(), "stopped": True})
        logger.info("Worker stopped")
