import json
from typing import Any, Literal
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass

import yaml


class SerializationWrapper:
    """Utility class for serializing Python objects to JSON or YAML"""

    @staticmethod
    def _prepare_value(value: Any) -> Any:
        """Convert value to JSON/YAML serializable format"""
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
    ):
        """
        Save data to JSON or YAML file

        Args:
            data (dict): Dictionary to save
            filepath (str | Path): Path to output file
            save_as (Literal["json","yaml"]: Save as 'json' or 'yaml'
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
        print(f"Saved to: {filepath}")


class AutoSaveDict(dict):
    """Dictionary that automatically saves to file on updates

    Examples:
        >>> config = AutoSaveDict('outputs/app_config.json', format='json')
        >>> config['app_name'] = 'My Application'
        >>> config['version'] = '1.0.0'
        >>> config['settings'] = {
        >>>     'debug': False,
        >>>     'port': 8080
        >>> }
    """

    def __init__(
        self, filepath: str, save_as: Literal["json", "yaml"] = "json", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.filepath = filepath
        self.save_as = save_as
        self._save()

    def _save(self):
        """Save current state to file"""
        data = {"timestamp": datetime.now().isoformat(), "data": dict(self)}
        SerializationWrapper.save_to_file(data, self.filepath, self.save_as)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._save()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._save()

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._save()
