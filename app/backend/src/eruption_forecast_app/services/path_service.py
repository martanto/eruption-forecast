"""Resolve file access beneath explicitly configured roots."""

from pathlib import Path

from eruption_forecast_app.settings import AppSettings
from eruption_forecast_app.repositories.database import Database


def roots(db: Database, runtime: AppSettings) -> list[Path]:
    """Return approved local roots.

    Example:
        >>> roots(database, AppSettings())
    """
    setup = db.value("setup", {})
    configured = [
        setup.get(key) for key in ("sds_dir", "output_dir", "tremor_dir", "root_dir")
    ]
    return [
        runtime.state.resolve(),
        *(Path(value).resolve() for value in configured if value),
    ]


def checked(path: str | Path, allowed: list[Path], must_exist: bool = True) -> Path:
    """Resolve a path and reject traversal outside approved roots.

    Raises:
        ValueError: The target is outside approved roots or does not exist.

    Example:
        >>> checked('output/file.csv', [Path('output').resolve()])
    """
    target = Path(path).expanduser().resolve()
    if not any(target == root or target.is_relative_to(root) for root in allowed):
        raise ValueError(
            "Path is outside configured SDS, tremor, output, or application-state directories"
        )
    if must_exist and not target.exists():
        raise ValueError(f"Path does not exist: {target}")
    return target


def validate_setup(values: dict) -> dict:
    """Validate and normalize user-selected root directories.

    Example:
        >>> validate_setup({'output_dir': 'output'})['output_dir']
    """
    result = dict(values)
    for key in ("sds_dir", "tremor_dir", "output_dir", "root_dir"):
        if not result.get(key):
            continue
        path = Path(result[key]).expanduser().resolve()
        if key != "output_dir" and not path.is_dir():
            raise ValueError(f"{key}: select an existing directory")
        if key == "output_dir":
            existing = path
            while not existing.exists():
                existing = existing.parent
            if not existing.is_dir():
                raise ValueError("output_dir: parent must be a directory")
        result[key] = str(path)
    for key in ("network", "station", "location", "channel", "channel_type"):
        value = result.get(key, "").upper()
        if any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in value):
            raise ValueError(f"{key}: use letters, numbers, underscore or hyphen")
        result[key] = value
    if (
        not result.get("station")
        or not result.get("network")
        or not result.get("channel")
    ):
        raise ValueError("Network, station, and channel are required")
    return result
