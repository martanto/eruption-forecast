"""Pipeline configuration dataclasses for eruption-forecast.

Each pipeline stage has a dedicated config section. The top-level
``PipelineConfig`` holds all sections and knows how to save/load itself
as YAML or JSON.
"""

from __future__ import annotations

import os
import json
from typing import Any, Literal
from datetime import datetime
from dataclasses import field, asdict, fields, dataclass


@dataclass
class ModelConfig:
    """Configuration for ``ForecastModel.__init__`` parameters.

    Stores the core model initialization parameters so the pipeline can
    be reconstructed from a saved configuration file.
    """

    station: str = ""
    channel: str = ""
    start_date: str = ""
    end_date: str = ""
    window_size: int = 1
    volcano_id: str = ""
    network: str = "VG"
    location: str = "00"
    output_dir: str | None = None
    root_dir: str | None = None
    overwrite: bool = False
    n_jobs: int = 1
    verbose: bool = False
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the config section to a plain dictionary.

        Returns:
            dict[str, Any]: A flat dictionary of all field values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Create a ``ModelConfig`` from a plain dictionary.

        Unknown keys are silently ignored so that older config files remain
        forward-compatible.

        Args:
            data (dict[str, Any]): Dictionary of field names to values.

        Returns:
            ModelConfig: A new instance populated from *data*.
        """
        valid = {k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        return cls(**valid)


@dataclass
class CalculateConfig:
    """Configuration for ``ForecastModel.calculate()`` parameters.

    Captures every argument accepted by ``calculate()`` so the tremor
    calculation step can be replayed identically.
    """

    source: str = "sds"
    sds_dir: str | None = None
    methods: str | None = None
    filename_prefix: str | None = None
    remove_outlier_method: str = "maximum"
    interpolate: bool = True
    value_multiplier: float | None = None
    cleanup_daily_dir: bool = False
    plot_daily: bool = False
    save_plot: bool = False
    overwrite_plot: bool = False
    client_url: str = "https://service.iris.edu"
    n_jobs: int | None = None
    verbose: bool = False
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the config section to a plain dictionary.

        Returns:
            dict[str, Any]: A flat dictionary of all field values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalculateConfig:
        """Create a ``CalculateConfig`` from a plain dictionary.

        Unknown keys are silently ignored.

        Args:
            data (dict[str, Any]): Dictionary of field names to values.

        Returns:
            CalculateConfig: A new instance populated from *data*.
        """
        valid = {k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        return cls(**valid)


@dataclass
class BuildLabelConfig:
    """Configuration for ``ForecastModel.build_label()`` parameters.

    Stores label-building parameters so eruption windows can be re-created
    with identical settings.
    """

    window_step: int = 12
    window_step_unit: str = "hours"
    day_to_forecast: int = 2
    eruption_dates: list[str] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    tremor_columns: list[str] | None = None
    verbose: bool = False
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the config section to a plain dictionary.

        Returns:
            dict[str, Any]: A flat dictionary of all field values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BuildLabelConfig:
        """Create a ``BuildLabelConfig`` from a plain dictionary.

        Unknown keys are silently ignored.

        Args:
            data (dict[str, Any]): Dictionary of field names to values.

        Returns:
            BuildLabelConfig: A new instance populated from *data*.
        """
        valid = {k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        return cls(**valid)


@dataclass
class ExtractFeaturesConfig:
    """Configuration for ``ForecastModel.extract_features()`` parameters.

    Captures tsfresh feature extraction settings for reproducible runs.
    """

    select_tremor_columns: list[str] | None = None
    save_tremor_matrix_per_method: bool = True
    save_tremor_matrix_per_id: bool = False
    exclude_features: list[str] | None = None
    use_relevant_features: bool = False
    overwrite: bool = False
    n_jobs: int | None = None
    verbose: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the config section to a plain dictionary.

        Returns:
            dict[str, Any]: A flat dictionary of all field values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtractFeaturesConfig:
        """Create an ``ExtractFeaturesConfig`` from a plain dictionary.

        Unknown keys are silently ignored.

        Args:
            data (dict[str, Any]): Dictionary of field names to values.

        Returns:
            ExtractFeaturesConfig: A new instance populated from *data*.
        """
        valid = {k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        return cls(**valid)


@dataclass
class TrainConfig:
    """Configuration for ``ForecastModel.train()`` parameters.

    Stores classifier and cross-validation settings so a training run can be
    replayed. Complex objects such as ``grid_params`` are intentionally
    excluded because they are not trivially serialisable.
    """

    classifier: str = "rf"
    cv_strategy: str = "shuffle"
    random_state: int = 0
    total_seed: int = 500
    with_evaluation: bool = True
    number_of_significant_features: int = 20
    sampling_strategy: float = 0.75
    save_all_features: bool = False
    plot_significant_features: bool = False
    n_jobs: int | None = None
    overwrite: bool = False
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the config section to a plain dictionary.

        Returns:
            dict[str, Any]: A flat dictionary of all field values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        """Create a ``TrainConfig`` from a plain dictionary.

        Unknown keys are silently ignored.

        Args:
            data (dict[str, Any]): Dictionary of field names to values.

        Returns:
            TrainConfig: A new instance populated from *data*.
        """
        valid = {k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        return cls(**valid)


@dataclass
class ForecastConfig:
    """Configuration for ``ForecastModel.forecast()`` parameters.

    Captures the forecasting window and output settings so a prediction run
    can be replayed on new data.
    """

    start_date: str = ""
    end_date: str = ""
    window_size: int = 1
    window_step: int = 12
    window_step_unit: str = "hours"
    save_predictions: bool = True
    save_plot: bool = True
    n_jobs: int | None = None
    overwrite: bool = False
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the config section to a plain dictionary.

        Returns:
            dict[str, Any]: A flat dictionary of all field values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastConfig:
        """Create a ``ForecastConfig`` from a plain dictionary.

        Unknown keys are silently ignored.

        Args:
            data (dict[str, Any]): Dictionary of field names to values.

        Returns:
            ForecastConfig: A new instance populated from *data*.
        """
        valid = {k: v for k, v in data.items() if k in {f.name for f in fields(cls)}}
        return cls(**valid)


@dataclass
class PipelineConfig:
    """Full pipeline configuration container.

    Holds one optional section per pipeline stage plus metadata. Only
    sections that were actually called appear when the config is saved, so a
    partial run produces a partial YAML that can still be loaded and
    continued.

    Attributes:
        version (str): Schema version string.
        saved_at (str): ISO-8601 timestamp set at save time.
        model (ModelConfig): Core model initialization parameters.
        calculate (CalculateConfig | None): Tremor calculation parameters.
        build_label (BuildLabelConfig | None): Label-building parameters.
        extract_features (ExtractFeaturesConfig | None): Feature-extraction parameters.
        train (TrainConfig | None): Training parameters.
        forecast (ForecastConfig | None): Forecasting parameters.
    """

    version: str = "1.0"
    saved_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    calculate: CalculateConfig | None = None
    build_label: BuildLabelConfig | None = None
    extract_features: ExtractFeaturesConfig | None = None
    train: TrainConfig | None = None
    forecast: ForecastConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the full config to a nested plain dictionary.

        Sections that are ``None`` are omitted so the serialized output only
        contains stages that were actually executed.

        Returns:
            dict[str, Any]: Nested dictionary ready for YAML/JSON serialisation.
        """
        data: dict[str, Any] = {
            "version": self.version,
            "saved_at": self.saved_at,
            "model": self.model.to_dict(),
        }
        for section_name in (
            "calculate",
            "build_label",
            "extract_features",
            "train",
            "forecast",
        ):
            section = getattr(self, section_name)
            if section is not None:
                data[section_name] = section.to_dict()
        return data

    def save(self, path: str, fmt: Literal["yaml", "json"] = "yaml") -> str:
        """Save the pipeline configuration to *path*.

        The parent directory is created automatically when it does not exist.
        ``saved_at`` is refreshed to the current time before writing.

        Args:
            path (str): Destination file path.
            fmt (Literal["yaml", "json"]): Output format. Defaults to ``"yaml"``.

        Returns:
            str: The path where the file was written.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.saved_at = datetime.now().isoformat(timespec="seconds")
        data = self.to_dict()

        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            import yaml  # lazy import — PyYAML is a transitive dependency

            with open(path, "w", encoding="utf-8") as f:
                f.write("# eruption-forecast pipeline configuration\n")
                yaml.safe_dump(
                    data,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

        return path

    @classmethod
    def load(cls, path: str) -> PipelineConfig:
        """Load a pipeline configuration from *path*.

        The format (YAML or JSON) is detected from the file extension.

        Args:
            path (str): Source file path (``.yaml``/``.yml`` or ``.json``).

        Returns:
            PipelineConfig: A fully populated config instance.

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
                import yaml  # lazy import

                data = yaml.safe_load(f)

        config = cls(
            version=data.get("version", "1.0"),
            saved_at=data.get("saved_at", ""),
            model=ModelConfig.from_dict(data.get("model", {})),
        )

        section_map: dict[str, Any] = {
            "calculate": CalculateConfig,
            "build_label": BuildLabelConfig,
            "extract_features": ExtractFeaturesConfig,
            "train": TrainConfig,
            "forecast": ForecastConfig,
        }
        for section_name, section_cls in section_map.items():
            if section_name in data:
                setattr(config, section_name, section_cls.from_dict(data[section_name]))

        return config
