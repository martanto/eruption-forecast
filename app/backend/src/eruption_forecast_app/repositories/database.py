"""Short SQLite transactions shared by API and worker processes."""

import json
import sqlite3
from uuid import uuid4
from typing import Any
from pathlib import Path
from datetime import UTC, datetime


def timestamp() -> str:
    """Return a UTC lifecycle timestamp.

    Example:
        >>> timestamp().endswith('+00:00')
        True
    """
    return datetime.now(UTC).isoformat()


class Database:
    """SQLite repository with atomic queue claims."""

    def __init__(self, path: Path):
        self.path = path

    def connect(self) -> sqlite3.Connection:
        """Open a bounded connection.

        Example:
            >>> connection = Database(Path('.app-state/app.sqlite3')).connect()
        """
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        """Create local metadata tables.

        Example:
            >>> Database(Path('.app-state/app.sqlite3')).initialize()
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                (Path(__file__).parent / "migrations/001_initial.sql").read_text()
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS job_artifacts (job_id TEXT NOT NULL, artifact_id TEXT NOT NULL, PRIMARY KEY(job_id,artifact_id))"
            )
            db.execute(
                "INSERT OR IGNORE INTO job_artifacts SELECT job_id,id FROM artifacts WHERE job_id IS NOT NULL"
            )

    def value(self, key: str, default: Any = None) -> Any:
        """Read a saved configuration value.

        Example:
            >>> database.value('setup', {})
        """
        with self.connect() as db:
            row = db.execute(
                "SELECT value FROM settings WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else default

    def set_value(self, key: str, value: Any) -> None:
        """Persist JSON settings atomically.

        Example:
            >>> database.set_value('setup', {'n_jobs': 1})
        """
        with self.connect() as db:
            db.execute(
                "INSERT INTO settings VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value)),
            )

    def create_job(
        self,
        kind: str,
        configuration: dict,
        resolved: dict,
        workspace: str,
        retry_of: str | None = None,
    ) -> dict:
        """Enqueue an immutable configuration snapshot.

        Example:
            >>> database.create_job('calculation', {}, {}, '/output')
        """
        identifier = uuid4().hex
        now = timestamp()
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs(id,kind,state,created_at,updated_at,configuration,resolved,workspace,retry_of) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    kind,
                    "queued",
                    now,
                    now,
                    json.dumps(configuration),
                    json.dumps(resolved),
                    workspace,
                    retry_of,
                ),
            )
        return self.job(identifier)

    def job(self, identifier: str) -> dict:
        """Read a job and its persisted events.

        Example:
            >>> database.job('job-id')
        """
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identifier,)).fetchone()
            if not row:
                raise KeyError(identifier)
            result = dict(row)
            result["events"] = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM events WHERE job_id=? ORDER BY id", (identifier,)
                )
            ]
        for key in ("configuration", "resolved"):
            result[key] = json.loads(result[key])
        return result

    def jobs(self, limit: int = 100) -> list[dict]:
        """List recent jobs without loading scientific artifacts.

        Example:
            >>> database.jobs(20)
        """
        with self.connect() as db:
            ids = [
                r[0]
                for r in db.execute(
                    "SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
                )
            ]
        return [self.job(identifier) for identifier in ids]

    def update(
        self,
        identifier: str,
        state: str,
        error: str | None = None,
        pid: int | None = None,
    ) -> None:
        """Persist a lifecycle transition without overriding cancellation.

        Example:
            >>> database.update('job-id', 'failed', 'Missing input')
        """
        with self.connect() as db:
            current = db.execute(
                "SELECT state FROM jobs WHERE id=?", (identifier,)
            ).fetchone()
            if (
                current
                and current[0] == "cancelling"
                and state in {"succeeded", "partial", "failed"}
            ):
                state = "cancelled"
            elif current and current[0] == "cancelling" and state == "running":
                state = "cancelling"
            db.execute(
                "UPDATE jobs SET state=?,error=?,pid=?,updated_at=? WHERE id=?",
                (state, error, pid, timestamp(), identifier),
            )

    def claim(self) -> dict | None:
        """Claim one job while enforcing one active heavy job globally.

        Example:
            >>> database.claim()
        """
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM jobs WHERE state IN ('running','cancelling')"
            ).fetchone():
                return None
            row = db.execute(
                "SELECT id FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE jobs SET state='running',updated_at=? WHERE id=?",
                (timestamp(), row[0]),
            )
        return self.job(row[0])

    def cancel(self, identifier: str) -> dict | None:
        """Cancel atomically with respect to worker claims and completion."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state FROM jobs WHERE id=?", (identifier,)
            ).fetchone()
            if row is None:
                raise KeyError(identifier)
            if row[0] not in {"queued", "running", "cancelling"}:
                return None
            state = "cancelled" if row[0] == "queued" else "cancelling"
            db.execute(
                "UPDATE jobs SET state=?,updated_at=? WHERE id=?",
                (state, timestamp(), identifier),
            )
        return self.job(identifier)

    def event(
        self,
        identifier: str,
        stage: str,
        state: str,
        detail: str = "",
        scenario: str = "",
    ) -> None:
        """Append an actual stage transition.

        Example:
            >>> database.event('job-id', 'fit', 'running')
        """
        with self.connect() as db:
            db.execute(
                "INSERT INTO events(job_id,stage,state,detail,scenario,created_at) VALUES (?,?,?,?,?,?)",
                (identifier, stage, state, detail, scenario, timestamp()),
            )

    def artifact(self, path: Path, job_id: str | None = None) -> dict:
        """Register a file without unpickling it.

        Example:
            >>> database.artifact(Path('output/tremor.csv'))
        """
        resolved = str(path.resolve())
        with self.connect() as db:
            db.execute(
                "INSERT INTO artifacts(id,path,job_id,created_at) VALUES (?,?,?,?) ON CONFLICT(path) DO NOTHING",
                (uuid4().hex, resolved, job_id, timestamp()),
            )
            artifact = dict(
                db.execute(
                    "SELECT * FROM artifacts WHERE path=?", (resolved,)
                ).fetchone()
            )
            if job_id:
                db.execute(
                    "INSERT OR IGNORE INTO job_artifacts VALUES (?,?)",
                    (job_id, artifact["id"]),
                )
            return artifact

    def artifacts(self, job_id: str | None = None) -> list[dict]:
        """List registered outputs.

        Example:
            >>> database.artifacts()
        """
        with self.connect() as db:
            if job_id:
                return [
                    dict(r)
                    for r in db.execute(
                        "SELECT a.* FROM artifacts a JOIN job_artifacts j ON a.id=j.artifact_id WHERE j.job_id=? ORDER BY a.path",
                        (job_id,),
                    )
                ]
            return [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM artifacts ORDER BY created_at DESC LIMIT 5000"
                )
            ]
