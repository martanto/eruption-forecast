"""Pipeline configuration dataclasses for eruption-forecast.

Each pipeline stage has a dedicated config section. The top-level
``PipelineConfig`` holds all sections and knows how to save/load itself
as YAML or JSON.
"""

import os
import json
from typing import Any, Self, Literal
from datetime import datetime
from dataclasses import field, asdict, fields, dataclass

import yaml

from eruption_forecast.utils.pathutils import ensure_dir


@dataclass
class _ConfigBase:
    """Base serialization mixin for pipeline config dataclasses.

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


@dataclass
class ModelConfig(_ConfigBase):
    """Configuration for ``ForecastModel.__init__`` parameters.

    Stores the core model initialization parameters so the pipeline can
    be reconstructed from a saved configuration file.

    Attributes:
        station (str): Seismic station code (e.g. ``"OJN"``). Defaults to ``""``.
        channel (str): Seismic channel code (e.g. ``"EHZ"``). Defaults to ``""``.
        window_size (int): Duration in days of each tremor window. Defaults to ``1``.
        volcano_id (str): Unique volcano identifier used in output filenames. Defaults to ``""``.
        network (str): Seismic network code. Defaults to ``""``.
        location (str): Seismic location code. Defaults to ``""``.
        output_dir (str | None): Base output directory. ``None`` resolves to
            ``root_dir/output``. Defaults to ``None``.
        root_dir (str | None): Anchor directory for resolving relative paths.
            ``None`` falls back to ``os.getcwd()``. Defaults to ``None``.
        overwrite (bool): Whether to overwrite existing output files. Defaults to ``False``.
        n_jobs (int): Number of parallel workers. Defaults to ``1``.
        verbose (bool): Enable verbose logging. Defaults to ``False``.
        debug (bool): Enable debug-level logging. Defaults to ``False``.
    """

    station: str = ""
    channel: str = ""
    window_size: int = 1
    volcano_id: str = ""
    network: str = ""
    location: str = ""
    output_dir: str | None = None
    root_dir: str | None = None
    overwrite: bool = False
    n_jobs: int = 1
    verbose: bool = False
    debug: bool = False


@dataclass
class CalculateConfig(_ConfigBase):
    """Configuration for ``ForecastModel.calculate()`` parameters.

    Captures every argument accepted by ``calculate()`` so the tremor
    calculation step can be replayed identically.

    Attributes:
        start_date (str): Tremor calculation start date in ``"YYYY-MM-DD"`` format.
            Defaults to ``""``.
        end_date (str): Tremor calculation end date in ``"YYYY-MM-DD"`` format.
            Defaults to ``""``.
        source (str): Seismic data source — ``"sds"`` or ``"fdsn"``. Defaults to ``"sds"``.
        sds_dir (str | None): Path to the SDS archive directory. Required when
            ``source="sds"``. Defaults to ``None``.
        methods (list[str] | None): Tremor calculation methods to apply (e.g.
            ``["rsam", "dsar", "entropy"]``). ``None`` enables all methods.
            Defaults to ``None``.
        filename_prefix (str | None): Optional prefix for generated filenames.
            Defaults to ``None``.
        remove_outlier_method (str): Outlier removal strategy — ``"all"`` or
            ``"maximum"``. Defaults to ``"maximum"``.
        remove_tremor_anomalies (bool): Whether to remove anomalies after tremor
            calculation using Z-score analysis. Defaults to ``False``.
        interpolate (bool): Whether to interpolate gaps in the tremor data.
            Defaults to ``True``.
        value_multiplier (float | None): Scaling factor applied to tremor values.
            ``None`` disables scaling. Defaults to ``None``.
        cleanup_daily_dir (bool): Whether to delete the daily temporary directory
            after merging. Defaults to ``False``.
        plot_daily (bool): Whether to generate a plot for each daily result.
            Defaults to ``False``.
        save_plot (bool): Whether to save the merged tremor plot. Defaults to ``False``.
        overwrite_plot (bool): Whether to overwrite existing plot files.
            Defaults to ``False``.
        client_url (str): FDSN web-service base URL. Used when ``source="fdsn"``.
            Defaults to ``"https://service.iris.edu"``.
        n_jobs (int | None): Parallel workers for this stage. ``None`` inherits
            from ``ForecastModel.n_jobs``. Defaults to ``None``.
        verbose (bool): Enable verbose logging. Defaults to ``False``.
        debug (bool): Enable debug-level logging. Defaults to ``False``.
    """

    start_date: str = ""
    end_date: str = ""
    source: str = "sds"
    sds_dir: str | None = None
    methods: list[str] | None = None
    filename_prefix: str | None = None
    remove_outlier_method: str = "maximum"
    remove_tremor_anomalies: bool = False
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


@dataclass
class BuildLabelConfig(_ConfigBase):
    """Configuration for ``ForecastModel.build_label()`` parameters.

    Stores label-building parameters so eruption windows can be re-created
    with identical settings.

    Attributes:
        window_step (int): Step size between consecutive label windows.
            Defaults to ``12``.
        window_step_unit (str): Unit of ``window_step`` — ``"minutes"`` or
            ``"hours"``. Defaults to ``"hours"``.
        day_to_forecast (int): Number of days before an eruption whose windows
            are labeled positive (``is_erupted=1``). Defaults to ``2``.
        eruption_dates (list[str]): Known eruption dates in ``"YYYY-MM-DD"``
            format. Defaults to ``[]``.
        start_date (str | None): Override for the training period start date.
            ``None`` uses ``ForecastModel.start_date``. Defaults to ``None``.
        end_date (str | None): Override for the training period end date.
            ``None`` uses ``ForecastModel.end_date``. Defaults to ``None``.
        include_eruption_date (bool): If ``True``, the eruption date itself is
            labeled as erupted (not excluded from the positive window).
            Defaults to ``False``.
        tremor_columns (list[str] | None): Subset of tremor columns to retain
            for labeling. ``None`` keeps all columns. Defaults to ``None``.
        builder (str): Label builder type — ``"standard"`` or ``"dynamic"``.
            Defaults to ``"standard"``.
        days_before_eruption (int | None): Days before each eruption to start
            its window. Required when ``builder="dynamic"``. Defaults to ``None``.
        verbose (bool): Enable verbose logging. Defaults to ``False``.
        debug (bool): Enable debug-level logging. Defaults to ``False``.
    """

    window_step: int = 12
    window_step_unit: str = "hours"
    day_to_forecast: int = 2
    eruption_dates: list[str] = field(default_factory=list)
    start_date: str | None = None
    end_date: str | None = None
    include_eruption_date: bool = False
    tremor_columns: list[str] | None = None
    builder: str = "standard"
    days_before_eruption: int | None = None
    verbose: bool = False
    debug: bool = False


@dataclass
class ExtractFeaturesConfig(_ConfigBase):
    """Configuration for ``ForecastModel.extract_features()`` parameters.

    Captures tsfresh feature extraction settings for reproducible runs.

    Attributes:
        select_tremor_columns (list[str] | None): Tremor columns to use for feature
            extraction. ``None`` uses all available columns. Defaults to ``None``.
        save_tremor_matrix_per_method (bool): Whether to save a separate tremor-matrix
            CSV for each tremor column. Defaults to ``True``.
        save_tremor_matrix_per_id (bool): Whether to save one tremor-matrix CSV per
            label window. Intended for debugging only — can generate many files.
            Defaults to ``False``.
        exclude_features (list[str] | None): tsfresh feature calculator names to skip
            during extraction. ``None`` excludes nothing. Defaults to ``None``.
        use_relevant_features (bool): If ``True``, applies tsfresh relevance filtering
            to retain only statistically significant features. Requires labels.
            Defaults to ``False``.
        overwrite (bool): Whether to overwrite existing feature files.
            Defaults to ``False``.
        n_jobs (int | None): Parallel workers for tsfresh extraction. ``None``
            inherits from ``ForecastModel.n_jobs``. Defaults to ``None``.
        verbose (bool | None): Enable verbose logging. ``None`` inherits from
            ``ForecastModel.verbose``. Defaults to ``None``.
    """

    select_tremor_columns: list[str] | None = None
    save_tremor_matrix_per_method: bool = True
    save_tremor_matrix_per_id: bool = False
    exclude_features: list[str] | None = None
    use_relevant_features: bool = False
    overwrite: bool = False
    n_jobs: int | None = None
    verbose: bool | None = None


@dataclass
class TrainConfig(_ConfigBase):
    """Configuration for ``ForecastModel.train()`` parameters.

    Stores classifier and cross-validation settings so a training run can be
    replayed. Complex objects such as ``grid_params`` are intentionally
    excluded because they are not trivially serialisable.

    Attributes:
        classifiers (list[str]): Ordered list of classifier keys to train
            (e.g. ``["rf", "xgb"]``). Defaults to ``["rf"]``.
        cv_strategy (str): Cross-validation strategy — ``"shuffle"``,
            ``"stratified"``, ``"shuffle-stratified"``, or ``"timeseries"``.
            Defaults to ``"shuffle-stratified"``.
        random_state (int): Starting random seed. Seeds run from
            ``random_state`` to ``random_state + total_seed - 1``.
            Defaults to ``0``.
        total_seed (int): Number of independent training runs (seeds).
            Defaults to ``500``.
        with_evaluation (bool): If ``True``, performs an 80/20 train/test split
            and computes per-seed evaluation metrics. If ``False``, trains on the
            full dataset without metrics. Defaults to ``False``.
        number_of_significant_features (int): Top-N features retained per seed
            after feature selection. Defaults to ``20``.
        sampling_strategy (float): Sampling ratio forwarded to the resampler.
            Defaults to ``0.75``.
        resample_method (Literal["under", "over", "auto"] | None): Resampling
            strategy for class balancing. ``"under"`` applies
            ``RandomUnderSampler``, ``"over"`` applies ``RandomOverSampler``,
            ``None`` skips resampling, and ``"auto"`` inspects the class ratio
            from the loaded labels — if the minority (eruption) class is below 10 %
            of all samples, ``"under"`` is used; otherwise resampling is skipped.
            Defaults to ``"auto"``.
        save_all_features (bool): Whether to save all ranked features per seed
            in addition to the top-N. Defaults to ``False``.
        plot_significant_features (bool): Whether to save a feature-importance
            plot per seed. Defaults to ``False``.
        n_jobs (int | None): Parallel workers for multi-seed dispatch. ``None``
            inherits from ``ForecastModel.n_jobs``. Defaults to ``None``.
        grid_search_n_jobs (int): Parallel jobs inside each ``GridSearchCV``
            call (inner loop). Uses the ``loky`` backend so it is safe for
            nested parallelism. Enforce ``n_jobs × grid_search_n_jobs ≤
            cpu_count``. Defaults to ``1``.
        overwrite (bool): Whether to overwrite existing training output files.
            Defaults to ``False``.
        verbose (bool): Enable verbose logging. Defaults to ``False``.
        plot_shap (bool): Whether to generate SHAP explanation plots per seed.
            Defaults to ``False``.
        save_model (bool): Whether to serialise the ``ForecastModel`` instance
            to disk after training completes. Defaults to ``True``.
        use_gpu (bool): Enable GPU acceleration for XGBoost via ``device="cuda:<gpu_id>"``.
            Forces ``n_jobs=1`` to prevent GPU memory contention across parallel
            seed workers. Has no effect on other classifiers. Defaults to ``False``.
        gpu_id (int): GPU device index to use when use_gpu is True. Use 0 for the
            first GPU, 1 for the second, etc. Defaults to ``0``.
    """

    classifiers: list[str] = field(default_factory=lambda: ["rf"])
    cv_strategy: str = "shuffle"
    random_state: int = 0
    total_seed: int = 500
    with_evaluation: bool = False
    number_of_significant_features: int = 20
    sampling_strategy: float = 0.75
    resample_method: Literal["under", "over", "auto"] | None = "auto"
    save_all_features: bool = False
    plot_significant_features: bool = False
    n_jobs: int | None = None
    grid_search_n_jobs: int = 1
    overwrite: bool = False
    verbose: bool = False
    plot_shap: bool = False
    save_model: bool = True
    use_gpu: bool = False
    gpu_id: int = 0


@dataclass
class ForecastConfig(_ConfigBase):
    """Configuration for ``ForecastModel.forecast()`` parameters.

    Captures the forecasting window and output settings so a prediction run
    can be replayed on new data.

    Attributes:
        start_date (str): Forecast period start date in ``"YYYY-MM-DD"`` format.
            Defaults to ``""``.
        end_date (str): Forecast period end date in ``"YYYY-MM-DD"`` format.
            Defaults to ``""``.
        window_step (int): Step size between consecutive forecast windows.
            Defaults to ``12``.
        window_step_unit (str): Unit of ``window_step`` — ``"minutes"`` or
            ``"hours"``. Defaults to ``"hours"``.
        save_predictions (bool): Whether to save the prediction DataFrame as CSV.
            Defaults to ``True``.
        threshold (float, optional): Threshold for classifying eruption
            probability as positive. Defaults to ``0.7``.
        n_jobs (int | None): Parallel workers for feature extraction during
            forecasting. ``None`` inherits from ``ForecastModel.n_jobs``.
            Defaults to ``None``.
        overwrite (bool): Whether to overwrite existing forecast output files.
            Defaults to ``False``.
        verbose (bool): Enable verbose logging. Defaults to ``False``.
    """

    start_date: str = ""
    end_date: str = ""
    window_step: int = 12
    window_step_unit: str = "hours"
    save_predictions: bool = True
    threshold: float = 0.7
    n_jobs: int | None = None
    overwrite: bool = False
    verbose: bool = False


@dataclass
class PipelineConfig(_ConfigBase):
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
        ensure_dir(os.path.dirname(os.path.abspath(path)))
        self.saved_at = datetime.now().isoformat(timespec="seconds")
        data = self.to_dict()

        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
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
    def load(cls, path: str) -> Self:
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
