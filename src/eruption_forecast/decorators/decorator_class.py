import json
from typing import Any, Literal
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass

import yaml


class SerializationWrapper:
    """Utility class for serializing Python objects to JSON or YAML.

    Provides static methods to recursively convert arbitrary Python objects
    into JSON/YAML-compatible types and write them to disk.

    Examples:
        >>> SerializationWrapper.save_to_file(
        ...     {"key": "value"}, "output/config.json"
        ... )
    """

    @staticmethod
    def _prepare_value(value: Any) -> Any:
        """Recursively convert a value to a JSON/YAML-serializable type.

        Handles dataclasses (via ``asdict``), objects with ``__dict__``,
        lists/tuples, dicts, and primitive types. Falls back to ``str()``
        for anything else.

        Args:
            value (Any): The value to convert.

        Returns:
            Any: A JSON/YAML-serializable representation of ``value``.
        """
        if is_dataclass(value):
            return asdict(value)
        elif hasattr(value, "__dict__"):
            return value.__dict__
        elif isinstance(value, (list, tuple)):
            return [SerializationWrapper._prepare_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: SerializationWrapper._prepare_value(v) for k, v in value.items()}
        elif isinstance(value, (str, int, float, bool, type(None))):
            return value
        else:
            return str(value)

    @staticmethod
    def save_to_file(
        data: dict, filepath: str | Path, save_as: Literal["json", "yaml"] | None = None
    ) -> None:
        """Save a dictionary to a JSON or YAML file.

        Creates parent directories automatically. Values are passed through
        :meth:`_prepare_value` before serialisation to ensure compatibility.

        Args:
            data (dict): Dictionary to serialise and write to disk.
            filepath (str | Path): Destination file path. Parent directories
                are created if they do not exist.
            save_as (Literal["json", "yaml"] | None, optional): Output
                format. If None, defaults to ``"json"``. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If ``save_as`` is not ``"json"`` or ``"yaml"``.

        Examples:
            >>> SerializationWrapper.save_to_file(
            ...     {"model": "rf", "score": 0.92},
            ...     "output/results.yaml",
            ...     save_as="yaml",
            ... )
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        prepared_data = SerializationWrapper._prepare_value(data)

        save_as = save_as or "json"

        if save_as.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(prepared_data, f, indent=2, default=str)
        elif save_as.lower() == "yaml":
            with open(filepath, "w") as f:
                yaml.dump(prepared_data, f, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {save_as}. Use 'json' or 'yaml'")

        print(f"Saved to: {filepath}")


class AutoSaveDict(dict):
    """Dictionary that automatically persists itself to disk on every update.

    Wraps the built-in ``dict`` and overrides mutating methods
    (``__setitem__``, ``__delitem__``, ``update``) so the file is rewritten
    after each change. The saved file includes a ``timestamp`` and a
    ``data`` key containing the dictionary contents.

    Args:
        filepath (str): Destination file path for persistence. The file is
            written immediately on construction.
        save_as (Literal["json", "yaml"], optional): Serialisation format.
            Defaults to ``"json"``.
        *args: Positional arguments forwarded to ``dict.__init__``.
        **kwargs: Keyword arguments forwarded to ``dict.__init__``.

    Examples:
        >>> config = AutoSaveDict("outputs/app_config.json", save_as="json")
        >>> config["app_name"] = "My Application"
        >>> config["version"] = "1.0.0"
        >>> config["settings"] = {"debug": False, "port": 8080}
    """

    def __init__(
        self, filepath: str, save_as: Literal["json", "yaml"] = "json", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filepath = filepath
        self.save_as = save_as
        self._save()

    def _save(self) -> None:
        """Persist the current dictionary state to the configured file."""
        data = {"timestamp": datetime.now().isoformat(), "data": dict(self)}
        SerializationWrapper.save_to_file(data, self.filepath, self.save_as)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an item and immediately save the dictionary to disk."""
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key: Any) -> None:
        """Delete an item and immediately save the dictionary to disk."""
        super().__delitem__(key)
        self._save()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the dictionary and immediately save to disk."""
        super().update(*args, **kwargs)
        self._save()
