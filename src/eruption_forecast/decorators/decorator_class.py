import json
from typing import Any, Literal
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass

import yaml


class SerializationWrapper:
    """Utility class for serializing Python objects to JSON or YAML formats.

    Provides static methods to recursively convert arbitrary Python objects into
    JSON/YAML-compatible types and write them to disk with automatic directory creation.

    Examples:
        >>> SerializationWrapper.save_to_file(
        ...     {"key": "value"}, "output/config.json"
        ... )
        Saved to: output/config.json

        >>> data = {"model": "RandomForest", "accuracy": 0.95, "timestamp": datetime.now()}
        >>> SerializationWrapper.save_to_file(data, "results.yaml", save_as="yaml")
        Saved to: results.yaml
    """

    @staticmethod
    def _prepare_value(value: Any) -> Any:
        """Recursively convert a value to a JSON/YAML-serializable type.

        Handles dataclasses (via `asdict`), objects with `__dict__`, lists, tuples,
        dicts, and primitive types. Falls back to `str()` for unsupported types.

        Args:
            value (Any): The value to convert to a serializable format.

        Returns:
            Any: A JSON/YAML-serializable representation of `value`. Returns the original
                value if already serializable, or converted/stringified version otherwise.

        Examples:
            >>> from dataclasses import dataclass
            >>> @dataclass
            ... class Point:
            ...     x: int
            ...     y: int
            >>> SerializationWrapper._prepare_value(Point(1, 2))
            {'x': 1, 'y': 2}

            >>> SerializationWrapper._prepare_value([1, "text", {"key": "value"}])
            [1, 'text', {'key': 'value'}]
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
        """Save a dictionary to a JSON or YAML file with automatic directory creation.

        Recursively prepares all values for serialization using `_prepare_value()` and
        writes the data to disk in the specified format. Parent directories are created
        automatically if they don't exist.

        Args:
            data (dict): Dictionary to serialize and write to disk.
            filepath (str | Path): Destination file path. Parent directories are
                created automatically if they do not exist.
            save_as (Literal["json", "yaml"] | None, optional): Output serialization
                format. If None, defaults to "json". Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If `save_as` is not "json" or "yaml".

        Examples:
            >>> SerializationWrapper.save_to_file(
            ...     {"model": "rf", "score": 0.92},
            ...     "output/results.yaml",
            ...     save_as="yaml",
            ... )
            Saved to: output/results.yaml

            >>> config = {"app": "forecaster", "version": "1.0", "debug": True}
            >>> SerializationWrapper.save_to_file(config, "config.json")
            Saved to: config.json
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
    """Dictionary that automatically persists itself to disk on every mutation.

    Extends the built-in `dict` and overrides mutating methods (`__setitem__`,
    `__delitem__`, `update`) to automatically save the dictionary to a file after
    each modification. The saved file includes a timestamp and the dictionary data.

    Attributes:
        filepath (str): Destination file path for automatic persistence.
        save_as (Literal["json", "yaml"]): Serialization format for the output file.

    Args:
        filepath (str): Destination file path for persistence. The file is written
            immediately upon construction and after every modification.
        save_as (Literal["json", "yaml"], optional): Serialization format.
            Defaults to "json".
        *args: Positional arguments forwarded to `dict.__init__`.
        **kwargs: Keyword arguments forwarded to `dict.__init__`.

    Examples:
        >>> config = AutoSaveDict("outputs/app_config.json", save_as="json")
        Saved to: outputs/app_config.json
        >>> config["app_name"] = "My Application"
        Saved to: outputs/app_config.json
        >>> config["version"] = "1.0.0"
        Saved to: outputs/app_config.json
        >>> config["settings"] = {"debug": False, "port": 8080}
        Saved to: outputs/app_config.json

        >>> state = AutoSaveDict("state.yaml", save_as="yaml", initial_data=42)
        Saved to: state.yaml
        >>> state.update({"key1": "value1", "key2": 100})
        Saved to: state.yaml
    """

    def __init__(
        self, filepath: str, save_as: Literal["json", "yaml"] = "json", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filepath = filepath
        self.save_as = save_as
        self._save()

    def _save(self) -> None:
        """Persist the current dictionary state to the configured file.

        Creates a wrapper dict with `timestamp` and `data` keys, then delegates
        serialization to `SerializationWrapper.save_to_file()`.

        Returns:
            None
        """
        data = {"timestamp": datetime.now().isoformat(), "data": dict(self)}
        SerializationWrapper.save_to_file(data, self.filepath, self.save_as)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an item and immediately save the dictionary to disk.

        Args:
            key (Any): Dictionary key to set.
            value (Any): Value to associate with the key.

        Returns:
            None
        """
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key: Any) -> None:
        """Delete an item and immediately save the dictionary to disk.

        Args:
            key (Any): Dictionary key to delete.

        Returns:
            None

        Raises:
            KeyError: If the key does not exist in the dictionary.
        """
        super().__delitem__(key)
        self._save()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the dictionary and immediately save to disk.

        Args:
            *args (Any): Positional arguments forwarded to `dict.update()`.
            **kwargs (Any): Keyword arguments forwarded to `dict.update()`.

        Returns:
            None
        """
        super().update(*args, **kwargs)
        self._save()
