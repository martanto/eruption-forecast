import os
import json
from typing import Any, Self, Literal
from datetime import datetime
from dataclasses import field, dataclass

import yaml

from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.config.base_config import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for ``TrainingModel.__init__`` parameters.

    Captures every argument accepted by ``TrainingModel.__init__`` so that a
    training instance can snapshot its construction state to YAML/JSON. The
    ``tremor_data`` field is stored as a string when the caller passed a CSV
    path, and ``None`` when a pre-loaded ``pd.DataFrame`` was supplied —
    pandas DataFrames are not round-trippable through YAML/JSON.

    Attributes:
        tremor_data (str | None): CSV path passed to ``TremorData``. ``None``
            when the caller supplied a pre-loaded DataFrame. Defaults to
            ``None``.
        start_date (str): Training period start date in ``"YYYY-MM-DD"`` or
            ISO-8601 format. Defaults to ``""``.
        end_date (str): Training period end date. Defaults to ``""``.
        classifiers (str | list[str]): Classifier key(s) to train (e.g.
            ``["rf", "xgb"]``). Defaults to ``"rf"``.
        eruption_dates (list[str]): Known eruption dates in ``"YYYY-MM-DD"``
            format. Defaults to ``[]``.
        window_size (int): Sliding window size in days. Defaults to ``2``.
        cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified"]):
            Cross-validation strategy. Defaults to ``"shuffle-stratified"``.
        cv_splits (int): Number of CV folds. Defaults to ``5``.
        top_n_features (int): Top-N features retained per seed after
            feature selection. Defaults to ``20``.
        include_eruption_date (bool): If ``True``, the eruption date itself
            is labeled as erupted. Defaults to ``False``.
        nslc (str | None): ``Network.Station.Location.Channel`` identifier.
            Used to scope the content-addressable cache identity to a single
            station-channel. ``None`` for standalone runs. Defaults to ``None``.
        output_dir (str | None): Root output directory. ``None`` resolves to
            ``root_dir/output``. Defaults to ``None``.
        root_dir (str | None): Anchor directory for resolving relative paths.
            ``None`` falls back to ``os.getcwd()``. Defaults to ``None``.
        overwrite (bool): Re-run and overwrite cached feature and model
            files. Defaults to ``False``.
        n_jobs (int): Number of parallel outer workers for seed-level
            parallelism. Defaults to ``1``.
        n_grids (int): Parallel workers used inside ``GridSearchCV`` and
            ``FeatureSelector``. Defaults to ``1``.
        verbose (bool): Emit detailed progress logs. Defaults to ``False``.
        version (str): Schema version string.
        saved_at (str): ISO-8601 timestamp set at save time.
    """

    tremor_data: str | None = None
    start_date: str = ""
    end_date: str = ""
    classifiers: str | list[str] = "rf"
    eruption_dates: list[str] = field(default_factory=list)
    window_size: int = 2
    cv_strategy: Literal["shuffle", "stratified", "shuffle-stratified"] = (
        "shuffle-stratified"
    )
    cv_splits: int = 5
    top_n_features: int = 20
    include_eruption_date: bool = False
    nslc: str | None = None
    output_dir: str | None = None
    root_dir: str | None = None
    overwrite: bool = False
    n_jobs: int = 1
    n_grids: int = 1
    verbose: bool = False

    version: str = "1.0"
    saved_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert the training configuration to a plain dictionary.

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
            "tremor_data",
            "start_date",
            "end_date",
            "classifiers",
            "eruption_dates",
            "window_size",
            "cv_strategy",
            "cv_splits",
            "top_n_features",
            "include_eruption_date",
            "nslc",
            "output_dir",
            "root_dir",
            "overwrite",
            "n_jobs",
            "n_grids",
            "verbose",
        ):
            data[f] = getattr(self, f)
        return data

    def save(self, path: str, fmt: Literal["yaml", "json"] = "yaml") -> str:
        """Save the training configuration to *path*.

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
                f.write("# eruption-forecast TrainingModel configuration\n")
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
        """Load a training configuration from *path*.

        The format (YAML or JSON) is detected from the file extension.

        Args:
            path (str): Source file path (``.yaml``/``.yml`` or ``.json``).

        Returns:
            TrainingConfig: A fully populated config instance.

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
