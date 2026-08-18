import os
import json
from typing import Any, Self, Literal
from datetime import datetime
from dataclasses import field, dataclass

import yaml

from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.config.base_config import BaseConfig


@dataclass
class LabelConfig(BaseConfig):
    """Configuration for ``LabelBuilder.__init__`` parameters.

    Captures every argument accepted by
    :class:`~eruption_forecast.label.label_builder.LabelBuilder` so a label
    build can snapshot its construction state to YAML/JSON alongside its
    output CSV. ``start_date`` and ``end_date`` are always stored as strings
    (ISO-8601 when the caller passed a ``datetime``) so the serialised form
    round-trips cleanly through YAML/JSON.

    ``version`` and ``saved_at`` are inherited from :class:`BaseConfig` so
    every stage config stamps the installed package release identically.

    Attributes:
        start_date (str): Label period start date in ``"YYYY-MM-DD"`` or
            ISO-8601 format. Defaults to ``""``.
        end_date (str): Label period end date. Defaults to ``""``.
        window_step (int): Step size between consecutive windows. Defaults
            to ``1``.
        window_step_unit (Literal["minutes", "hours"]): Unit of
            ``window_step``. Defaults to ``"hours"``.
        day_to_forecast (int): Number of days before an eruption to start
            positive labelling. Defaults to ``2``.
        eruption_dates (list[str]): Known eruption dates in ``"YYYY-MM-DD"``
            format. Defaults to ``[]``.
        volcano_id (str | None): Volcano identifier used in output filenames.
            ``None`` triggers ``LabelBuilder`` to derive one from the date
            range. Defaults to ``None``.
        include_eruption_date (bool): When ``True`` the eruption date counts
            as one of the ``day_to_forecast`` positive days; when ``False``
            it is marked positive as an *additional* day. Defaults to
            ``True``.
        output_dir (str | None): Root output directory. ``None`` resolves to
            ``root_dir/output``. Defaults to ``None``.
        root_dir (str | None): Anchor directory for resolving relative paths.
            ``None`` falls back to ``os.getcwd()``. Defaults to ``None``.
        verbose (bool): Emit informational logs. Defaults to ``False``.
        debug (bool): Emit debug-level logs. Defaults to ``False``.
    """

    start_date: str = ""
    end_date: str = ""
    window_step: int = 1
    window_step_unit: Literal["minutes", "hours"] = "hours"
    day_to_forecast: int = 2
    eruption_dates: list[str] = field(default_factory=list)
    volcano_id: str | None = None
    include_eruption_date: bool = True
    output_dir: str | None = None
    root_dir: str | None = None
    verbose: bool = False
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the label configuration to a plain dictionary.

        Places ``version`` and ``saved_at`` at the head of the dictionary so
        the serialised YAML/JSON layout matches the other stage configs.

        Returns:
            dict[str, Any]: Flat dictionary ready for YAML/JSON serialisation.
        """
        data: dict[str, Any] = {
            "version": self.version,
            "saved_at": self.saved_at,
        }
        for f in (
            "start_date",
            "end_date",
            "window_step",
            "window_step_unit",
            "day_to_forecast",
            "eruption_dates",
            "volcano_id",
            "include_eruption_date",
            "output_dir",
            "root_dir",
            "verbose",
            "debug",
        ):
            data[f] = getattr(self, f)
        return data

    def save(self, path: str, fmt: Literal["yaml", "json"] = "yaml") -> str:
        """Save the label configuration to *path*.

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
                f.write("# eruption-forecast LabelBuilder configuration\n")
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
        """Load a label configuration from *path*.

        The format (YAML or JSON) is detected from the file extension.

        Args:
            path (str): Source file path (``.yaml``/``.yml`` or ``.json``).

        Returns:
            LabelConfig: A fully populated config instance.

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
