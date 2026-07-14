import os
import json
from typing import Any, Self, Literal
from datetime import datetime
from dataclasses import field, dataclass

import yaml

from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.config.base_config import BaseConfig


@dataclass
class EvaluationConfig(BaseConfig):
    """Configuration for ``EvaluationModel.__init__`` parameters.

    Captures the serializable arguments accepted by
    ``EvaluationModel.__init__``. The upstream ``model`` parameter is
    intentionally omitted because it is always a live
    :class:`TrainingModel` / :class:`PredictionModel` instance that cannot be
    round-tripped through YAML/JSON.

    ``version`` and ``saved_at`` are inherited from :class:`BaseConfig` so
    every stage config stamps the installed package release identically.

    Attributes:
        eruption_dates (list[str]): Ground-truth eruption dates in
            ``"YYYY-MM-DD"`` format. Defaults to ``[]``.
        overwrite (bool): Re-run and overwrite cached output files. Defaults
            to ``False``.
        output_dir (str | None): Root output directory. ``None`` resolves to
            ``root_dir/output``. Defaults to ``None``.
        root_dir (str | None): Anchor directory for resolving relative paths.
            ``None`` falls back to ``os.getcwd()``. Defaults to ``None``.
        prefix_config (str | None): Discriminator slugified into the
            ``save_config()`` filename, inserted before ``.config`` (e.g.
            ``"scenario 1"`` → ``evaluation.scenario-1.config.yaml``).
            ``None`` keeps the default filename. Defaults to ``None``.
        n_jobs (int): Number of parallel workers. Defaults to ``1``.
        verbose (bool): Emit detailed progress logs. Defaults to ``False``.
    """

    eruption_dates: list[str] = field(default_factory=list)
    overwrite: bool = False
    output_dir: str | None = None
    root_dir: str | None = None
    prefix_config: str | None = None
    n_jobs: int = 1
    save_model: bool = False
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the evaluation configuration to a plain dictionary.

        Places ``version`` and ``saved_at`` at the head of the dictionary so
        the serialised YAML/JSON layout matches the ``ForecastConfig`` style.

        Returns:
            dict[str, Any]: Flat dictionary ready for YAML/JSON serialisation.
        """
        data: dict[str, Any] = {
            "version": self.version,
            "saved_at": self.saved_at,
        }
        for f in (
            "eruption_dates",
            "overwrite",
            "output_dir",
            "root_dir",
            "prefix_config",
            "n_jobs",
            "save_model",
            "verbose",
        ):
            data[f] = getattr(self, f)
        return data

    def save(self, path: str, fmt: Literal["yaml", "json"] = "yaml") -> str:
        """Save the evaluation configuration to *path*.

        The parent directory is created automatically when it does not exist.
        ``saved_at`` is refreshed to the current time before writing.

        Args:
            path (str): Destination file path.
            fmt (Literal["yaml", "json"]): Output format. Defaults to
                ``"yaml"``.

        Returns:
            str: The path where the file was written.
        """
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        self.saved_at = datetime.now().isoformat(timespec="seconds")
        data = self.to_dict()

        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# eruption-forecast EvaluationModel configuration\n")
                yaml.safe_dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

        return path

    @classmethod
    def load(cls, path: str) -> Self:
        """Load an evaluation configuration from *path*.

        The format (YAML or JSON) is detected from the file extension.

        Args:
            path (str): Source file path (``.yaml``/``.yml`` or ``.json``).

        Returns:
            EvaluationConfig: A fully populated config instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")

        ext = os.path.splitext(path)[1].lower()
        with open(path, encoding="utf-8") as f:
            if ext == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f)

        return cls.from_dict(data or {})
