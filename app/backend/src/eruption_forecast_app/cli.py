"""Local application commands.

Example:
    uv run python -m eruption_forecast_app dev
"""

import os
import sys
import json
import subprocess
from pathlib import Path

import click
import uvicorn

from eruption_forecast_app.main import create_app
from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.workers.runner import run_worker
from eruption_forecast_app.workers.cancellation import terminate_tree


@click.group()
def cli() -> None:
    """Manage the local eruption forecasting application."""


@cli.command()
@click.option("--port", default=8000, type=click.IntRange(1, 65535))
def serve(port: int) -> None:
    """Serve the API and compiled frontend on localhost."""
    uvicorn.run(create_app(), host="127.0.0.1", port=port)


@cli.command()
@click.option("--once", is_flag=True, help="Process at most one job.")
def worker(once: bool) -> None:
    """Consume persistent queued jobs."""
    run_worker(AppSettings(), once=once)


@cli.command()
@click.option("--port", default=8000, type=click.IntRange(1, 65535))
def dev(port: int) -> None:
    """Start the local API and worker; run Vite separately."""
    process = subprocess.Popen(
        [sys.executable, "-m", "eruption_forecast_app", "worker"],
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        uvicorn.run(create_app(), host="127.0.0.1", port=port)
    finally:
        terminate_tree(process.pid)


@cli.command("export-openapi")
@click.option("--output", default="openapi.json", type=click.Path(path_type=Path))
@click.option("--check", is_flag=True, help="Fail if the committed contract differs.")
def export_openapi(output: Path, check: bool) -> None:
    """Generate the API contract without initializing runtime state."""
    content = json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    if check:
        if not output.exists() or output.read_text(encoding="utf-8") != content:
            raise click.ClickException("OpenAPI contract is stale; run export-openapi")
    else:
        output.write_text(content, encoding="utf-8")
    click.echo(f"OpenAPI {'verified' if check else 'exported'}: {output}")
