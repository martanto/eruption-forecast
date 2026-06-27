import os
import json
from typing import Any, Self, Literal
from datetime import datetime
from dataclasses import field, dataclass

import yaml

from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.config.base_config import BaseConfig


@dataclass
class PredictionConfig(BaseConfig):
    """Configuration for ``PredictionModel.__init__`` parameters.

    Captures every argument accepted by ``PredictionModel.__init__`` so that a
    prediction instance can snapshot its construction state to YAML/JSON.
    Non-serializable inputs are reduced to a string handle when possible:

    - ``model`` is stored as a path string when the user supplied one,
      ``None`` when a live :class:`ClassifierEnsemble` / :class:`SeedEnsemble`
      instance was passed.
    - ``tremor_data`` is stored as the CSV path when the user supplied one,
      ``None`` when a pre-loaded ``pd.DataFrame`` was passed.

    Attributes:
        model (str | None): Path to a trained-model artefact (``.pkl`` /
            ``.json`` / registry CSV) or ``None`` when a live ensemble was
            passed. Defaults to ``None``.
        tremor_data (str | None): CSV path passed to ``TremorData``. ``None``
            when the caller supplied a pre-loaded DataFrame. Defaults to
            ``None``.
        start_date (str): Forecast period start date in ``"YYYY-MM-DD"`` or
            ISO-8601 format. Defaults to ``""``.
        end_date (str): Forecast period end date. Defaults to ``""``.
        window_size (int): Sliding window size in days. Defaults to ``2``.
        overwrite (bool): Overwrite existing output files. Defaults to
            ``False``.
        output_dir (str | None): Root output directory. ``None`` resolves to
            ``root_dir/output``. Defaults to ``None``.
        root_dir (str | None): Anchor directory for resolving relative paths.
            ``None`` falls back to ``os.getcwd()``. Defaults to ``None``.
        n_jobs (int): Number of parallel workers. Defaults to ``1``.
        verbose (bool): Emit detailed progress logs. Defaults to ``False``.
        version (str): Schema version string.
        saved_at (str): ISO-8601 timestamp set at save time.
    """

    model: str | None = None
    tremor_data: str | None = None
    start_date: str = ""
    end_date: str = ""
    window_size: int = 2
    overwrite: bool = False
    output_dir: str | None = None
    root_dir: str | None = None
    n_jobs: int = 1
    verbose: bool = False

    version: str = "1.0"
    saved_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the prediction configuration to a plain dictionary.

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
            "model",
            "tremor_data",
            "start_date",
            "end_date",
            "window_size",
            "overwrite",
            "output_dir",
            "root_dir",
            "n_jobs",
            "verbose",
        ):
            data[f] = getattr(self, f)
        return data

    def save(self, path: str, fmt: Literal["yaml", "json"] = "yaml") -> str:
        """Save the prediction configuration to *path*.

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
                f.write("# eruption-forecast PredictionModel configuration\n")
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
        """Load a prediction configuration from *path*.

        The format (YAML or JSON) is detected from the file extension.

        Args:
            path (str): Source file path (``.yaml``/``.yml`` or ``.json``).

        Returns:
            PredictionConfig: A fully populated config instance.

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
