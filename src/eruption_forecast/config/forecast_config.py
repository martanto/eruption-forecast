"""Pipeline configuration dataclasses for the ``ForecastModel`` API.

Each pipeline stage (``calculate``, ``train``, ``predict``, ``evaluate``) has
a dedicated config section. The top-level ``ForecastConfig`` holds all
sections and knows how to save/load itself as YAML or JSON.

This module targets ``eruption_forecast.model.forecast.ForecastModel``.
"""

import os
import json
from typing import Any, Self, Literal
from datetime import datetime
from dataclasses import field, dataclass

import yaml

from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.config.base_config import BaseConfig


@dataclass
class BaseForecastConfig(BaseConfig):
    """Configuration for ``ForecastModel.__init__`` parameters.

    Mirrors the constructor surface of
    :class:`eruption_forecast.model.forecast.ForecastModel` so a pipeline can
    be reconstructed from a saved configuration file.

    Attributes:
        station (str): Seismic station code (e.g. ``"OJN"``). Defaults to ``""``.
        channel (str): Seismic channel code (e.g. ``"EHZ"``). Defaults to ``""``.
        network (str): Seismic network code. Defaults to ``""``.
        location (str): Seismic location code. Defaults to ``""``.
        day_to_forecast (int): Number of days before an eruption used as the
            forecast horizon. Stored at the model level because
            ``calculate()`` reads ``self.day_to_forecast`` when adjusting the
            tremor start date. Defaults to ``2``.
        output_dir (str | None): Base output directory. ``None`` resolves to
            ``root_dir/output``. Defaults to ``None``.
        root_dir (str | None): Anchor directory for resolving relative paths.
            ``None`` falls back to ``os.getcwd()``. Defaults to ``None``.
        overwrite (bool): Whether to overwrite existing output files.
            Defaults to ``False``.
        n_jobs (int): Number of parallel workers. Defaults to ``1``.
        verbose (bool): Enable verbose logging. Defaults to ``False``.
    """

    station: str = ""
    channel: str = ""
    network: str = ""
    location: str = ""
    day_to_forecast: int = 2
    output_dir: str | None = None
    root_dir: str | None = None
    overwrite: bool = False
    n_jobs: int = 1
    verbose: bool = False


@dataclass
class ForecastCalculateConfig(BaseConfig):
    """Configuration for ``ForecastModel.calculate()`` parameters.

    Captures every argument accepted by ``calculate()`` so the tremor
    calculation step can be replayed identically.

    Attributes:
        start_date (str): Tremor calculation start date in ``"YYYY-MM-DD"``
            format. Defaults to ``""``.
        end_date (str): Tremor calculation end date in ``"YYYY-MM-DD"``
            format. Defaults to ``""``.
        source (Literal["sds", "fdsn"]): Seismic data source. Defaults to ``"sds"``.
        methods (str | list[str] | None): Tremor calculation methods to apply
            (e.g. ``["rsam", "dsar", "entropy"]``). ``None`` enables all
            methods. Defaults to ``None``.
        remove_outlier_method (Literal["all", "maximum"]): Outlier removal
            strategy. Defaults to ``"maximum"``.
        remove_tremor_anomalies (bool): Whether to remove anomalies after
            tremor calculation using Z-score analysis. Defaults to ``False``.
        interpolate (bool): Whether to interpolate gaps in the tremor data.
            Defaults to ``True``.
        value_multiplier (float | None): Scaling factor applied to tremor
            values. ``None`` disables scaling. Defaults to ``None``.
        cleanup_daily_dir (bool): Whether to delete the daily temporary
            directory after merging. Defaults to ``False``.
        plot_daily (bool): Whether to generate a plot for each daily result.
            Defaults to ``False``.
        save_plot (bool): Whether to save the merged tremor plot. Defaults to
            ``False``.
        overwrite_plot (bool): Whether to overwrite existing plot files.
            Defaults to ``False``.
        sds_dir (str | None): Path to the SDS archive directory. Required when
            ``source="sds"``. Defaults to ``None``.
        client_url (str): FDSN web-service base URL. Used when
            ``source="fdsn"``. Defaults to ``"https://service.iris.edu"``.
        minimum_completion_ratio (float): Minimum fraction of expected samples
            per day for a daily file to be accepted. Defaults to ``0.3``.
        overwrite (bool | None): Overwrite existing output files. ``None``
            inherits from ``ForecastModel.overwrite``. Defaults to ``None``.
        n_jobs (int | None): Parallel workers for this stage. ``None``
            inherits from ``ForecastModel.n_jobs``. Defaults to ``None``.
        verbose (bool | None): Enable verbose logging. ``None`` inherits from
            ``ForecastModel.verbose``. Defaults to ``None``.
    """

    start_date: str = ""
    end_date: str = ""
    source: Literal["sds", "fdsn"] = "sds"
    methods: str | list[str] | None = None
    remove_outlier_method: Literal["all", "maximum"] = "maximum"
    remove_tremor_anomalies: bool = False
    interpolate: bool = True
    value_multiplier: float | None = None
    cleanup_daily_dir: bool = False
    plot_daily: bool = False
    save_plot: bool = False
    overwrite_plot: bool = False
    sds_dir: str | None = None
    client_url: str = "https://service.iris.edu"
    minimum_completion_ratio: float = 0.3
    overwrite: bool | None = None
    n_jobs: int | None = None
    verbose: bool | None = None


@dataclass
class ForecastTrainConfig(BaseConfig):
    """Configuration for ``ForecastModel.train()`` parameters.

    ``train()`` fuses label building, feature extraction, and multi-seed
    fitting into a single stage.

    Attributes:
        start_date (str): Training period start date. Defaults to ``""``.
        end_date (str): Training period end date. Defaults to ``""``.
        eruption_dates (list[str]): Known eruption dates in ``"YYYY-MM-DD"``
            format. Defaults to ``[]``.
        window_step (int): Step size between consecutive label windows.
            Defaults to ``12``.
        window_step_unit (Literal["minutes", "hours"]): Unit of
            ``window_step``. Defaults to ``"hours"``.
        label_builder (Literal["standard", "dynamic"]): Label builder type.
            Defaults to ``"standard"``.
        days_before_eruption (int | None): Days before each eruption to start
            its window. Required when ``label_builder="dynamic"``. Defaults to
            ``None``.
        classifiers (str | list[str]): Classifier key(s) to train (e.g.
            ``["rf", "xgb"]``). Defaults to ``"rf"``.
        cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified"]):
            Cross-validation strategy. Defaults to ``"shuffle-stratified"``.
        cv_splits (int): Number of CV splits. Defaults to ``5``.
        scoring (str): Sklearn scoring identifier passed to ``GridSearchCV``.
            Defaults to ``"balanced_accuracy"``.
        number_of_features (int): Top-N features retained per seed after
            feature selection. Defaults to ``20``.
        include_eruption_date (bool): If ``True``, the eruption date itself is
            labeled as erupted. Defaults to ``True``.
        select_tremor_columns (list[str] | None): Subset of tremor columns to
            use for feature extraction. ``None`` keeps all columns. Defaults
            to ``None``.
        save_tremor_matrix_per_method (bool): Whether to save a separate
            tremor-matrix CSV per tremor column. Defaults to ``True``.
        exclude_features (list[str] | None): tsfresh feature calculator names
            to skip during extraction. ``None`` excludes nothing. Defaults to
            ``None``.
        minimum_completion (float): Minimum fraction of valid samples per
            window required to keep the window. Defaults to ``1.0``.
        seeds (int): Number of independent training seeds. Defaults to ``10``.
        resample_method (Literal["under", "over", "auto"] | None): Resampling
            strategy for class balancing. Defaults to ``"auto"``.
        minority_threshold (float): Minority-class ratio below which
            ``resample_method="auto"`` triggers undersampling. Defaults to
            ``0.15``.
        sampling_strategy (str | float): Sampling ratio forwarded to the
            resampler. Defaults to ``0.75``.
        plot_features (bool): Whether to save feature-importance plots.
            Defaults to ``True``.
        output_dir (str | None): Override for the training output directory.
            ``None`` uses ``ForecastModel.station_dir``. Defaults to ``None``.
        overwrite (bool | None): Overwrite existing training output files.
            ``None`` inherits from ``ForecastModel.overwrite``. Defaults to
            ``None``.
        n_jobs (int | None): Parallel workers for multi-seed dispatch.
            ``None`` inherits from ``ForecastModel.n_jobs``. Defaults to
            ``None``.
        n_grids (int): Number of grid-search iterations per seed. Defaults to
            ``1``.
        use_cache (bool): Whether to consult the content-addressable training
            cache before refitting. Defaults to ``True``.
        verbose (bool | None): Enable verbose logging. ``None`` inherits from
            ``ForecastModel.verbose``. Defaults to ``None``.
    """

    start_date: str = ""
    end_date: str = ""
    eruption_dates: list[str] = field(default_factory=list)
    window_step: int = 12
    window_step_unit: Literal["minutes", "hours"] = "hours"
    label_builder: Literal["standard", "dynamic"] = "standard"
    days_before_eruption: int | None = None
    classifiers: str | list[str] = "rf"
    cv_strategy: Literal[
        "shuffle", "stratified", "shuffle-stratified"
    ] = "shuffle-stratified"
    cv_splits: int = 5
    scoring: str = "balanced_accuracy"
    number_of_features: int = 20
    include_eruption_date: bool = True
    select_tremor_columns: list[str] | None = None
    save_tremor_matrix_per_method: bool = True
    exclude_features: list[str] | None = None
    minimum_completion: float = 1.0
    seeds: int = 10
    resample_method: Literal["under", "over", "auto"] | None = "auto"
    minority_threshold: float = 0.15
    sampling_strategy: str | float = 0.75
    plot_features: bool = True
    output_dir: str | None = None
    overwrite: bool | None = None
    n_jobs: int | None = None
    n_grids: int = 1
    use_cache: bool = True
    verbose: bool | None = None


@dataclass
class ForecastPredictConfig(BaseConfig):
    """Configuration for ``ForecastModel.predict()`` parameters.

    Captures the forecasting window and output settings. Variadic
    ``**plot_kwargs`` are deliberately omitted because they may carry
    non-serialisable matplotlib objects.

    Attributes:
        start_date (str): Forecast period start date. Defaults to ``""``.
        end_date (str): Forecast period end date. Defaults to ``""``.
        window_step (int): Step size between consecutive forecast windows.
            Defaults to ``12``.
        window_step_unit (Literal["minutes", "hours"]): Unit of
            ``window_step``. Defaults to ``"hours"``.
        save_seed_result (bool): Whether to save the per-seed prediction
            DataFrame as CSV. Defaults to ``True``.
        plot_threshold (float): Threshold above which the forecast is
            classified positive on the plot. Defaults to ``0.5``.
        plot_title (str | None): Optional plot title override. Defaults to
            ``None``.
        plot_pdf (bool): Whether to save the forecast plot as a PDF alongside
            the PNG. Defaults to ``True``.
        output_dir (str | None): Override for the prediction output
            directory. ``None`` uses ``ForecastModel.station_dir``. Defaults
            to ``None``.
        overwrite (bool | None): Overwrite existing forecast output files.
            ``None`` inherits from ``ForecastModel.overwrite``. Defaults to
            ``None``.
        n_jobs (int | None): Parallel workers for feature extraction.
            ``None`` inherits from ``ForecastModel.n_jobs``. Defaults to
            ``None``.
        use_cache (bool): Whether to consult the content-addressable
            prediction cache before re-running. Defaults to ``True``.
        verbose (bool | None): Enable verbose logging. ``None`` inherits from
            ``ForecastModel.verbose``. Defaults to ``None``.
    """

    start_date: str = ""
    end_date: str = ""
    window_step: int = 12
    window_step_unit: Literal["minutes", "hours"] = "hours"
    save_seed_result: bool = True
    plot_threshold: float = 0.5
    plot_title: str | None = None
    plot_pdf: bool = True
    output_dir: str | None = None
    overwrite: bool | None = None
    n_jobs: int | None = None
    use_cache: bool = True
    verbose: bool | None = None


@dataclass
class ForecastEvaluateConfig(BaseConfig):
    """Configuration for ``ForecastModel.evaluate()`` parameters.

    Attributes:
        model (Literal["training", "prediction"]): Which model in the current
            pipeline to evaluate. Defaults to ``"prediction"``.
        eruption_dates (list[str] | None): Ground-truth eruption dates in
            ``"YYYY-MM-DD"`` format. ``None`` falls back to the dates captured
            during ``train()``. Defaults to ``None``.
        plot_per_seed (bool): Render per-seed evaluation plots. Defaults to
            ``False``.
        plot_aggregate (bool): Render aggregate plots per classifier.
            Defaults to ``True``.
        output_dir (str | None): Root output directory for evaluation
            artefacts. ``None`` uses ``ForecastModel.station_dir``. Defaults
            to ``None``.
        overwrite (bool | None): Overwrite cached evaluation results. ``None``
            inherits from ``ForecastModel.overwrite``. Defaults to ``None``.
        n_jobs (int | None): Parallel workers. ``None`` inherits from
            ``ForecastModel.n_jobs``. Defaults to ``None``.
        verbose (bool | None): Enable verbose logging. ``None`` inherits from
            ``ForecastModel.verbose``. Defaults to ``None``.
    """

    model: Literal["training", "prediction"] = "prediction"
    eruption_dates: list[str] | None = None
    plot_per_seed: bool = False
    plot_aggregate: bool = True
    output_dir: str | None = None
    overwrite: bool | None = None
    n_jobs: int | None = None
    verbose: bool | None = None


@dataclass
class ForecastConfig(BaseConfig):
    """Full ``ForecastModel`` configuration container.

    Holds one optional section per pipeline stage plus metadata. Only
    sections that were actually called appear when the config is saved, so a
    partial run produces a partial YAML that can still be loaded and
    continued.

    Attributes:
        version (str): Schema version string.
        saved_at (str): ISO-8601 timestamp set at save time.
        model (BaseForecastConfig): Core model initialization parameters.
        calculate (ForecastCalculateConfig | None): Tremor calculation parameters.
        train (ForecastTrainConfig | None): Training parameters.
        predict (ForecastPredictConfig | None): Prediction parameters.
        evaluate (ForecastEvaluateConfig | None): Evaluation parameters.
    """

    version: str = "1.0"
    saved_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    model: BaseForecastConfig = field(default_factory=BaseForecastConfig)
    calculate: ForecastCalculateConfig | None = None
    train: ForecastTrainConfig | None = None
    predict: ForecastPredictConfig | None = None
    evaluate: ForecastEvaluateConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert the full config to a nested plain dictionary.

        Sections that are ``None`` are omitted so the serialized output only
        contains stages that were actually executed.

        Returns:
            dict[str, Any]: Nested dictionary ready for YAML/JSON
                serialisation.
        """
        data: dict[str, Any] = {
            "version": self.version,
            "saved_at": self.saved_at,
            "model": self.model.to_dict(),
        }
        for section_name in ("calculate", "train", "predict", "evaluate"):
            section = getattr(self, section_name)
            if section is not None:
                data[section_name] = section.to_dict()
        return data

    def save(self, path: str, fmt: Literal["yaml", "json"] = "yaml") -> str:
        """Save the forecast configuration to *path*.

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
                f.write("# eruption-forecast ForecastModel configuration\n")
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
        """Load a forecast configuration from *path*.

        The format (YAML or JSON) is detected from the file extension.

        Args:
            path (str): Source file path (``.yaml``/``.yml`` or ``.json``).

        Returns:
            ForecastConfig: A fully populated config instance.

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
            model=BaseForecastConfig.from_dict(data.get("model", {})),
        )

        section_map: dict[str, Any] = {
            "calculate": ForecastCalculateConfig,
            "train": ForecastTrainConfig,
            "predict": ForecastPredictConfig,
            "evaluate": ForecastEvaluateConfig,
        }
        for section_name, section_cls in section_map.items():
            if section_name in data:
                setattr(config, section_name, section_cls.from_dict(data[section_name]))

        return config
