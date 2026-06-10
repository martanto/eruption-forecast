from typing import Any, Self
from dataclasses import asdict, fields, dataclass


@dataclass
class BaseConfig:
    """Base serialization mixin for eruption-forecast config dataclasses.

    Provides ``to_dict`` and ``from_dict`` so each config section avoids
    repeating the same boilerplate serialization logic.
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert this config section to a plain dictionary.

        Uses ``dataclasses.asdict`` to recursively serialize all fields.

        Returns:
            dict[str, Any]: A flat dictionary of all field values.
        """
        return asdict(self)

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
