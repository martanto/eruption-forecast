from typing import Any, Self
from datetime import datetime
from dataclasses import field, asdict, fields, dataclass
from importlib.metadata import PackageNotFoundError, version as _pkg_version


def _package_version() -> str:
    """Resolve the installed ``eruption-forecast`` distribution version.

    Falls back to ``"0.0.0"`` when the distribution is not installed (e.g.
    a source checkout run without ``uv sync``) so downstream serialisation
    never blows up on a missing metadata entry.

    Returns:
        str: The installed package version, or ``"0.0.0"`` when unavailable.
    """
    try:
        return _pkg_version("eruption-forecast")
    except PackageNotFoundError:
        return "0.0.0"


@dataclass
class BaseConfig:
    """Base serialization mixin for eruption-forecast config dataclasses.

    Provides ``to_dict`` and ``from_dict`` so each config section avoids
    repeating the same boilerplate serialization logic. Every subclass
    inherits ``version`` (auto-resolved from the installed package
    distribution) and ``saved_at`` (refreshed on every ``save()`` call by
    the top-level configs), so the two metadata keys are always available
    downstream without per-subclass boilerplate.

    Attributes:
        version (str): Package version string. Defaults to the installed
            ``eruption-forecast`` version, falling back to ``"0.0.0"`` when
            the distribution metadata is unavailable.
        saved_at (str): ISO-8601 timestamp set at instance-construction
            time and refreshed on every top-level ``save()`` call.
    """

    version: str = field(default_factory=_package_version)
    saved_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert this config section to a plain dictionary.

        Uses ``dataclasses.asdict`` to recursively serialize all fields.

        Returns:
            dict[str, Any]: A flat dictionary of all field values.
        """
        return asdict(self)

    def to_init_kwargs(self) -> dict[str, Any]:
        """Convert to a dict suitable for splatting into ``__init__``.

        Same as :meth:`to_dict` but strips the persistence-metadata keys
        (``version``, ``saved_at``) that live on every subclass but are
        never accepted by the corresponding model constructor / stage
        method. Used by ``ForecastModel.from_config`` and
        ``ForecastModel.run`` when replaying captured sections.

        Returns:
            dict[str, Any]: Serialized fields minus persistence metadata.
        """
        d = self.to_dict()
        d.pop("version", None)
        d.pop("saved_at", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create an instance from a plain dictionary.

        Unknown keys are silently ignored so that older config files remain
        forward-compatible.

        Args:
            data (dict[str, Any]): Dictionary of field names to values.

        Returns:
            Self: A new instance populated from *data*.
        """
        valid = {k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        return cls(**valid)
