"""Runtime paths, independent of the scientific output directory."""

import os
from pathlib import Path
from dataclasses import field, dataclass


def project_root() -> Path:
    """Resolve the checkout, or the working directory for an installed wheel."""
    if os.environ.get("ERUPTION_PROJECT_ROOT"):
        return Path(os.environ["ERUPTION_PROJECT_ROOT"]).resolve()
    checkout = Path(__file__).resolve().parents[4]
    return checkout if (checkout / "app/frontend").is_dir() else Path.cwd().resolve()


@dataclass
class AppSettings:
    """Local state paths; override ``ERUPTION_APP_STATE`` for isolated runs."""

    root: Path = field(default_factory=project_root)
    state: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ERUPTION_APP_STATE", ".app-state")
        ).resolve()
    )

    @property
    def database(self) -> Path:
        """SQLite metadata database."""
        return self.state / "app.sqlite3"

    @property
    def frontend(self) -> Path:
        """Compiled frontend assets."""
        return self.root / "app/frontend/dist"
