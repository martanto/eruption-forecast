import os
from typing import Any, Self, Literal
from datetime import datetime, timedelta

import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import setup_nslc_directories
from eruption_forecast.utils.date_utils import to_datetime
from eruption_forecast.utils.formatting import slugify
from eruption_forecast.tremor.tremor_data import TremorData
from eruption_forecast.model.training_model import TrainingModel
from eruption_forecast.config.forecast_config import (
    ForecastConfig,
    BaseForecastConfig,
    ForecastTrainConfig,
    ForecastExplainConfig,
    ForecastPredictConfig,
    ForecastEvaluateConfig,
    ForecastCalculateConfig,
)
from eruption_forecast.model.evaluation_model import EvaluationModel
from eruption_forecast.model.prediction_model import PredictionModel
from eruption_forecast.model.explanation_model import ExplanationModel
from eruption_forecast.tremor.calculate_tremor import CalculateTremor
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class ForecastModel:
    """Orchestrate the complete volcanic eruption forecasting pipeline.

    Provides a high-level interface to coordinate all pipeline stages from
    raw seismic data to trained models and predictions. Uses method chaining
    for fluent API design.
    """

    def __init__(
        self,
        station: str,
        channel: str,
        network: str,
        location: str = "",
        day_to_forecast: int = 2,
        output_dir: str | None = None,
        root_dir: str | None = None,
        overwrite: bool = False,
        prefix_config: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialize the ForecastModel pipeline orchestrator.

        Args:
            station (str): Station code. Uppercased on assignment.
            channel (str): Channel code (e.g. ``"EHZ"``). Uppercased
                on assignment.
            network (str): Network code. Uppercased on assignment.
            location (str): Location code. Defaults to ``""``.
            day_to_forecast (int): Forecast lead time in days. Threaded
                into ``TrainingModel`` / ``PredictionModel`` as
                ``window_size`` and used to back-shift the tremor
                calculation start date so the first label window has
                full lead-in coverage. Defaults to ``2``.
            output_dir (str | None): Root output directory. ``None``
                resolves via :func:`setup_nslc_directories`. Defaults
                to ``None``.
            root_dir (str | None): Project root used when resolving a
                relative ``output_dir``. ``None`` keeps the resolved
                ``output_dir`` as-is. Defaults to ``None``.
            overwrite (bool): Default overwrite flag inherited by every
                stage method when its own ``overwrite`` kwarg is
                ``None``. Defaults to ``False``.
            prefix_config (str | None): Discriminator slugified into
                every stage ``save_config()`` filename, inserted before
                ``.config`` (e.g. ``"scenario 1"`` →
                ``forecast.scenario-1.config.yaml``). Threaded into
                ``TrainingModel``, ``PredictionModel``,
                ``EvaluationModel``, and ``ExplanationModel`` so all
                five per-stage YAMLs pick up the same discriminator when
                multiple scenarios share an ``output_dir``. ``None``
                keeps today's filenames. Defaults to ``None``.
            n_jobs (int): Default parallel-worker count inherited by
                every stage method when its own ``n_jobs`` kwarg is
                ``None``. Defaults to ``1``.
            verbose (bool): Default verbose flag inherited by every
                stage method when its own ``verbose`` kwarg is
                ``None``. Defaults to ``False``.

        Example:
            >>> fm = ForecastModel(
            ...     station="OJN", channel="EHZ", network="VG",
            ...     location="00", day_to_forecast=2,
            ...     root_dir="/path/to/project", n_jobs=4,
            ... )
        """
        _config_output_dir = output_dir
        _config_root_dir = root_dir

        root_dir = os.path.abspath(root_dir) if root_dir is not None else None
        nslc, output_dir, station_dir = setup_nslc_directories(
            network, station, location, channel, output_dir, root_dir
        )

        self.station = station.upper()
        self.channel = channel.upper()
        self.network = network.upper()
        self.location = location.upper()
        self.day_to_forecast = day_to_forecast
        self.output_dir = output_dir
        self.root_dir = root_dir
        self.overwrite = overwrite
        self.prefix_config = prefix_config
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.nslc = nslc
        self.station_dir = station_dir

        # Will be set after calculate() run
        self.CalculateTremor: CalculateTremor | None = None
        self.tremor_df: pd.DataFrame = pd.DataFrame()
        self.tremor_start_date: datetime | None = None
        self.tremor_end_date: datetime | None = None

        # Will be set after train() run
        self.TrainingModel: TrainingModel | None = None
        self.ClassifierEnsemble: ClassifierEnsemble | None = None
        self.select_tremor_columns: list[str] | None = None
        self.save_tremor_matrix_per_method: bool = False
        self.exclude_features: list[str] | None = None
        self._eruption_dates: list[str] | None = None
        self._training_cache_hash: str | None = None

        # Will be set after predict() run
        self.PredictionModel: PredictionModel | None = None
        self.results: pd.DataFrame = pd.DataFrame()

        # Will be set after evaluate() run
        self.EvaluationModel: EvaluationModel | None = None
        self.evaluation_results: dict[str, pd.DataFrame] = {}

        # Will be set after explain() run
        self.ExplanationModel: ExplanationModel | None = None

        # Pipeline configuration: populated incrementally as each stage runs.
        # ``save_config()`` serialises whatever stages have executed so far.
        self._config: ForecastConfig = ForecastConfig(
            model=BaseForecastConfig(
                station=station,
                channel=channel,
                network=network,
                location=location,
                day_to_forecast=day_to_forecast,
                output_dir=_config_output_dir,
                root_dir=_config_root_dir,
                overwrite=overwrite,
                prefix_config=prefix_config,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )

    def calculate(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        source: Literal["sds", "fdsn"] = "sds",
        methods: str | list[str] | None = None,
        remove_outlier_method: Literal["all", "maximum"] = "maximum",
        remove_tremor_anomalies: bool = False,
        interpolate: bool = True,
        value_multiplier: float | None = None,
        cleanup_daily_dir: bool = False,
        plot_daily: bool = False,
        save_plot: bool = False,
        sds_dir: str | None = None,
        client_url: str = "https://service.iris.edu",
        minimum_completion_ratio: float = 0.3,
        plot_eruption_dates: list[str] | None = None,
        plot_rsam_as_log: bool = False,
        plot_rolling_window: str | None = None,
        plot_filter_dsar_value: float | None = None,
        plot_overwrite: bool = False,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Run tremor calculation for the configured station.

        Reads seismic data from SDS or FDSN and computes RSAM, DSAR,
        and Shannon entropy across the default frequency bands. The
        effective start date is shifted backward by ``day_to_forecast``
        days so the first downstream label window has full lead-in
        coverage.

        Args:
            start_date (str | datetime): Start of the calculation
                window in ``"YYYY-MM-DD"`` format or as a
                ``datetime``.
            end_date (str | datetime): End of the calculation window
                in ``"YYYY-MM-DD"`` format or as a ``datetime``.
            source (Literal["sds", "fdsn"]): Seismic data source.
                Defaults to ``"sds"``.
            methods (str | list[str] | None): Tremor metrics to
                compute (``"rsam"``, ``"dsar"``, ``"entropy"``).
                ``None`` computes all three. Defaults to ``None``.
            remove_outlier_method (Literal["all", "maximum"]): Outlier
                removal strategy applied per band. Defaults to
                ``"maximum"``.
            remove_tremor_anomalies (bool): Drop z-score anomalies
                from the merged tremor frame. Defaults to ``False``.
            interpolate (bool): Interpolate gaps in the daily output.
                Defaults to ``True``.
            value_multiplier (float | None): Optional scalar applied
                to every tremor sample. Defaults to ``None``.
            cleanup_daily_dir (bool): Remove the per-day CSV directory
                after merging. Defaults to ``False``.
            plot_daily (bool): Render a per-day diagnostic plot.
                Defaults to ``False``.
            save_plot (bool): Persist the daily plot to disk.
                Defaults to ``False``.
            plot_overwrite (bool): Overwrite an existing daily plot.
                Defaults to ``False``.
            sds_dir (str | None): Root directory of the SDS archive.
                Required when ``source="sds"``. Defaults to ``None``.
            client_url (str): FDSN base URL used when
                ``source="fdsn"``. Defaults to
                ``"https://service.iris.edu"``.
            minimum_completion_ratio (float): Minimum per-day
                completion ratio required for a daily file to be kept.
                Defaults to ``0.3``.
            plot_eruption_dates (list[str] | None): Eruption dates
                (``"YYYY-MM-DD"``) overlaid as vertical markers on the
                merged tremor summary figure. Forwarded to
                ``CalculateTremor.run()`` and ultimately to
                :func:`~eruption_forecast.plots.tremor_plots.plot_tremor`.
                Only takes effect when ``save_plot=True``. Defaults to
                ``None``.
            plot_rsam_as_log (bool): Render the RSAM subplot of the
                merged tremor summary figure on a log y-axis. Forwarded
                to ``CalculateTremor.run()`` and only takes effect when
                ``save_plot=True``. Defaults to ``False``.
            plot_rolling_window (str | None): Pandas offset alias
                (e.g. ``"2D"``, ``"12H"``) used as the rolling-window
                size for the merged tremor summary figure. ``None``
                plots the raw series. Only takes effect when
                ``save_plot=True``. Defaults to ``None``.
            plot_filter_dsar_value (float | None): Upper bound applied
                to every DSAR series before plotting on the merged
                tremor summary figure — samples at or above this value
                are masked with ``NaN`` so a few spikes do not flatten
                the visible band. RSAM and entropy series are
                unaffected. Only takes effect when ``save_plot=True``.
                Defaults to ``None``.
            overwrite (bool | None): Overwrite cached daily outputs.
                ``None`` inherits from ``self.overwrite``. Defaults to
                ``None``.
            n_jobs (int | None): Parallel workers. ``None`` inherits
                from ``self.n_jobs``. Defaults to ``None``.
            verbose (bool | None): Verbose logging. ``None`` inherits
                from ``self.verbose``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If ``source="sds"`` is passed without
                ``sds_dir``, or if ``source`` is not ``"sds"`` or
                ``"fdsn"``.

        Example:
            >>> fm.calculate(
            ...     start_date="2020-01-01", end_date="2020-12-31",
            ...     source="sds", sds_dir="/path/to/sds",
            ...     methods=["rsam", "dsar", "entropy"],
            ... )
        """
        # Snapshot the user's original kwargs before any normalization so the
        # captured config round-trips cleanly through YAML.
        _cfg_start_date = (
            start_date if isinstance(start_date, str) else start_date.isoformat()
        )
        _cfg_end_date = end_date if isinstance(end_date, str) else end_date.isoformat()
        _cfg_methods = methods
        _cfg_overwrite = overwrite
        _cfg_n_jobs = n_jobs
        _cfg_verbose = verbose

        start_date = to_datetime(start_date) - timedelta(days=self.day_to_forecast)
        methods = [methods] if isinstance(methods, str) else methods
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

        if verbose:
            logger.info(
                f"Adjusting tremor calculation start date by reducing to "
                f"day_to_forecast={self.day_to_forecast}: {start_date}"
            )

        calculate = CalculateTremor(
            start_date=start_date,
            end_date=end_date,
            station=self.station,
            channel=self.channel,
            network=self.network,
            location=self.location,
            methods=methods,
            output_dir=self.output_dir,
            remove_outlier_method=remove_outlier_method,
            remove_tremor_anomalies=remove_tremor_anomalies,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_daily_dir=cleanup_daily_dir,
            plot_daily=plot_daily,
            save_plot=save_plot,
            plot_overwrite=plot_overwrite,
            overwrite=overwrite,
            minimum_completion_ratio=minimum_completion_ratio,
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
            verbose=verbose,
        )

        if source.upper() == "SDS":
            if sds_dir is None:
                raise ValueError(
                    "You chose 'sds' as source, please provide 'sds_dir' parameter. "
                    "Example: calculate(source='sds', sds_dir='converted')"
                )
            calculate = calculate.from_sds(sds_dir=sds_dir)
        elif source.upper() == "FDSN":
            calculate = calculate.from_fdsn(client_url=client_url)
        else:
            raise ValueError(f"Unknown source {source!r}. Expected 'sds' or 'fdsn'.")

        self.CalculateTremor = calculate.run(
            plot_eruption_dates=plot_eruption_dates,
            plot_rsam_as_log=plot_rsam_as_log,
            plot_rolling_window=plot_rolling_window,
            plot_filter_dsar_value=plot_filter_dsar_value,
        )

        if verbose:
            start_date_str = start_date.strftime("%Y-%m-%d")
            logger.info(f"Calculate Tremor from {start_date_str} to {end_date}")

        tremor_data = TremorData(calculate.df)
        self.tremor_df = tremor_data.df
        self.tremor_start_date = tremor_data.start_date
        self.tremor_end_date = tremor_data.end_date

        self._config.calculate = ForecastCalculateConfig(
            start_date=_cfg_start_date,
            end_date=_cfg_end_date,
            source=source,
            methods=_cfg_methods,
            remove_outlier_method=remove_outlier_method,
            remove_tremor_anomalies=remove_tremor_anomalies,
            interpolate=interpolate,
            value_multiplier=value_multiplier,
            cleanup_daily_dir=cleanup_daily_dir,
            plot_daily=plot_daily,
            save_plot=save_plot,
            sds_dir=sds_dir,
            client_url=client_url,
            minimum_completion_ratio=minimum_completion_ratio,
            plot_eruption_dates=(
                list(plot_eruption_dates) if plot_eruption_dates is not None else None
            ),
            plot_rsam_as_log=plot_rsam_as_log,
            plot_rolling_window=plot_rolling_window,
            plot_filter_dsar_value=plot_filter_dsar_value,
            plot_overwrite=plot_overwrite,
            overwrite=_cfg_overwrite,
            n_jobs=_cfg_n_jobs,
            verbose=_cfg_verbose,
        )

        return self

    def train(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        eruption_dates: list[str],
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        label_builder: Literal["standard", "dynamic"] = "standard",
        days_before_eruption: int | None = None,
        classifiers: str | list[str] = "rf",
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified"
        ] = "shuffle-stratified",
        cv_splits: int = 5,
        scoring: str = "balanced_accuracy",
        top_n_features: int = 20,
        include_eruption_date: bool = True,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = True,
        exclude_features: list[str] | None = None,
        select_features: str | list[str] | None = None,
        minimum_completion: float = 1.0,
        seeds: int = 10,
        resample_method: Literal["under", "over", "auto"] | None = "auto",
        minority_threshold: float = 0.15,
        sampling_strategy: str | float = 0.75,
        plot_features: bool = True,
        output_dir: str | None = None,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        n_grids: int = 1,
        use_cache: bool = True,
        verbose: bool | None = None,
    ) -> Self:
        """Build labels, extract features and fit the classifier ensemble.

        Wraps a full ``TrainingModel`` pipeline
        (``build_label → extract_features → fit``) over the tremor
        frame produced by :meth:`calculate`. Results are cached by a
        content-addressable identity, so a re-run with identical
        kwargs and tremor data is a cache hit.

        Args:
            start_date (str | datetime): Start of the training window
                in ``"YYYY-MM-DD"`` format or as a ``datetime``.
            end_date (str | datetime): End of the training window in
                ``"YYYY-MM-DD"`` format or as a ``datetime``.
            eruption_dates (list[str]): Ground-truth eruption dates
                in ``"YYYY-MM-DD"`` format.
            window_step (int): Sliding window step size.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.
            label_builder (Literal["standard", "dynamic"]): Label
                construction strategy. ``"standard"`` builds a single
                global sliding window grid; ``"dynamic"`` builds
                per-eruption windows and deduplicates. Defaults to
                ``"standard"``.
            days_before_eruption (int | None): Number of days marked
                positive ahead of each eruption when
                ``label_builder="dynamic"``. Defaults to ``None``.
            classifiers (str | list[str]): Classifier slugs to fit
                (``"rf"``, ``"gb"``, ``"xgb"``, ``"svm"``, ``"lr"``,
                ``"nn"``, ``"dt"``, ``"knn"``, ``"nb"``, ``"voting"``,
                ``"lite-rf"``). Defaults to ``"rf"``.
            cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified"]):
                Cross-validation strategy. Defaults to
                ``"shuffle-stratified"``.
            cv_splits (int): Number of CV splits. Defaults to ``5``.
            scoring (str): sklearn scoring metric for ``GridSearchCV``.
                Defaults to ``"balanced_accuracy"``.
            top_n_features (int): Number of features retained by
                ``FeatureSelector``. Defaults to ``20``.
            include_eruption_date (bool): Include the eruption day in
                the positive label window. Defaults to ``True``.
            select_tremor_columns (list[str] | None): Subset of
                tremor columns used for feature extraction. ``None``
                uses every available column. Defaults to ``None``.
            save_tremor_matrix_per_method (bool): Persist per-column
                tremor matrices under ``per_method/``. Defaults to
                ``True``.
            exclude_features (list[str] | None): Feature substrings
                to drop after tsfresh extraction. Defaults to
                ``None``.
            select_features (str | list[str] | None): Feature
                selection method (``"tsfresh"`` or
                ``"random_forest"``). Defaults to ``None``.
            minimum_completion (float): Minimum per-window
                completion ratio required for a window to be kept.
                Defaults to ``1.0``.
            seeds (int): Number of seed models per classifier.
                Defaults to ``10``.
            resample_method (Literal["under", "over", "auto"] | None):
                Imbalance handling. ``"auto"`` picks ``"under"`` when
                minority share is below ``minority_threshold``.
                Defaults to ``"auto"``.
            minority_threshold (float): Minority-class share below
                which ``"auto"`` triggers undersampling. Defaults to
                ``0.15``.
            sampling_strategy (str | float): Forwarded to the
                imbalanced-learn sampler. Defaults to ``0.75``.
            plot_features (bool): Render per-seed feature importance
                plots. Defaults to ``True``.
            output_dir (str | None): Root output directory for
                training artefacts. Defaults to the station directory.
            overwrite (bool | None): Overwrite cached artefacts.
                ``None`` inherits from ``self.overwrite``. Defaults
                to ``None``.
            n_jobs (int | None): Outer seed workers. ``None`` inherits
                from ``self.n_jobs``. Defaults to ``None``.
            n_grids (int): Inner ``GridSearchCV`` /
                ``FeatureSelector`` workers. Defaults to ``1``.
            use_cache (bool): Short-circuit re-fit when a cache hit
                exists. Defaults to ``True``.
            verbose (bool | None): Verbose logging. ``None`` inherits
                from ``self.verbose``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If :meth:`calculate` has not produced tremor
                data yet.

        Example:
            >>> fm.train(
            ...     start_date="2020-01-01", end_date="2020-12-31",
            ...     eruption_dates=["2020-06-15"],
            ...     window_step=6, window_step_unit="hours",
            ...     classifiers=["rf", "xgb"], seeds=500,
            ... )
        """
        if self.CalculateTremor is None:
            raise ValueError("Tremor data not found. Please run calculate() first.")

        # Snapshot originals for the captured config, must happen before the
        # n_jobs/verbose/overwrite fallback so ``None`` keeps the "inherit at
        # replay" semantics.
        self._config.train = ForecastTrainConfig(
            start_date=(
                start_date if isinstance(start_date, str) else start_date.isoformat()
            ),
            end_date=(end_date if isinstance(end_date, str) else end_date.isoformat()),
            eruption_dates=list(eruption_dates),
            window_step=window_step,
            window_step_unit=window_step_unit,
            label_builder=label_builder,
            days_before_eruption=days_before_eruption,
            classifiers=classifiers,
            cv_strategy=cv_strategy,
            cv_splits=cv_splits,
            scoring=scoring,
            top_n_features=top_n_features,
            include_eruption_date=include_eruption_date,
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=save_tremor_matrix_per_method,
            exclude_features=exclude_features,
            select_features=select_features,
            minimum_completion=minimum_completion,
            seeds=seeds,
            resample_method=resample_method,
            minority_threshold=minority_threshold,
            sampling_strategy=sampling_strategy,
            plot_features=plot_features,
            output_dir=output_dir,
            overwrite=overwrite,
            n_jobs=n_jobs,
            n_grids=n_grids,
            use_cache=use_cache,
            verbose=verbose,
        )

        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

        resolved_output_dir = output_dir or self.station_dir

        identity = TrainingModel.build_identity(
            nslc=self.nslc,
            tremor_df=self.tremor_df,
            start_date=start_date,
            end_date=end_date,
            classifiers=classifiers,
            eruption_dates=eruption_dates,
            window_size=self.day_to_forecast,
            cv_strategy=cv_strategy,
            cv_splits=cv_splits,
            scoring=scoring,
            top_n_features=top_n_features,
            include_eruption_date=include_eruption_date,
            build_label_params={
                "window_step": window_step,
                "window_step_unit": window_step_unit,
                "builder": label_builder,
                "days_before_eruption": days_before_eruption,
            },
            extract_features_params={
                "select_tremor_columns": select_tremor_columns,
                "save_tremor_matrix_per_method": save_tremor_matrix_per_method,
                "exclude_features": exclude_features,
                "select_features": select_features,
                "minimum_completion": minimum_completion,
            },
            fit_params={
                "seeds": seeds,
                "resample_method": resample_method,
                "minority_threshold": minority_threshold,
                "sampling_strategy": sampling_strategy,
            },
        )

        training_stage_dir = os.path.join(resolved_output_dir, "training")

        if use_cache:
            cached = TrainingModel.load(training_stage_dir, identity)
            if cached is not None:
                if self.verbose:
                    logger.warning("Loading cached training data...")

                self.TrainingModel = cached
                self.ClassifierEnsemble = cached.ClassifierEnsemble
                self.select_tremor_columns = select_tremor_columns
                self.save_tremor_matrix_per_method = save_tremor_matrix_per_method
                self.exclude_features = exclude_features
                self._eruption_dates = eruption_dates
                self._training_cache_hash = TrainingModel.compute_hash(identity)
                return self

        training_model = (
            TrainingModel(
                tremor_data=self.tremor_df,
                start_date=start_date,
                end_date=end_date,
                classifiers=classifiers,
                eruption_dates=eruption_dates,
                window_size=self.day_to_forecast,
                cv_strategy=cv_strategy,
                cv_splits=cv_splits,
                top_n_features=top_n_features,
                include_eruption_date=include_eruption_date,
                nslc=self.nslc,
                output_dir=resolved_output_dir,
                overwrite=overwrite,
                prefix_config=self.prefix_config,
                n_jobs=n_jobs,
                n_grids=n_grids,
                verbose=verbose,
            )
            .build_label(
                window_step=window_step,
                window_step_unit=window_step_unit,
                builder=label_builder,
                days_before_eruption=days_before_eruption,
                verbose=verbose,
            )
            .extract_features(
                select_tremor_columns=select_tremor_columns,
                save_tremor_matrix_per_method=save_tremor_matrix_per_method,
                exclude_features=exclude_features,
                select_features=select_features,
                minimum_completion=minimum_completion,
                overwrite=overwrite,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            .fit(
                seeds=seeds,
                resample_method=resample_method,
                minority_threshold=minority_threshold,
                sampling_strategy=sampling_strategy,
                plot_features=plot_features,
            )
        )

        self.TrainingModel = training_model
        self.ClassifierEnsemble = training_model.ClassifierEnsemble
        self.select_tremor_columns = select_tremor_columns
        self.save_tremor_matrix_per_method = save_tremor_matrix_per_method
        self.exclude_features = exclude_features

        self._eruption_dates = eruption_dates

        # ``fit()`` has already persisted the cache via ``self.save(identity)``.
        # The hash is needed downstream by :meth:`predict` for its own identity.
        self._training_cache_hash = (
            training_model.training_hash or TrainingModel.compute_hash(identity)
        )

        return self

    def predict(
        self,
        start_date: str | datetime,
        end_date: str | datetime,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        save_seed_result: bool = True,
        plot_threshold: float = 0.5,
        plot_title: str | None = None,
        plot_pdf: bool = True,
        use_features_from: Literal["all", "files", "training"] = "all",
        features_matrix_path: str | None = None,
        label_features_csv: str | None = None,
        enable_segments_plot: bool = False,
        output_dir: str | None = None,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        use_cache: bool = True,
        verbose: bool | None = None,
        **plot_kwargs: Any,
    ) -> Self:
        """Forecast over an unlabelled window grid.

        Builds a fresh sliding window grid over ``[start_date,
        end_date]``, runs feature extraction with the columns and
        exclusions captured during :meth:`train`, and dispatches the
        cached ``ClassifierEnsemble`` to produce per-seed and
        consensus eruption probabilities.

        Feature scoping is controlled by ``use_features_from``:
        ``"all"`` (default) extracts every tsfresh feature, ``"files"``
        skips tsfresh entirely and loads a pre-built
        ``features_matrix_path`` + ``label_features_csv`` pair, and
        ``"training"`` narrows tsfresh to the union of features picked
        by any seed during :meth:`train` — pulled from
        ``self.TrainingModel.features_selected_df.index`` and forwarded
        into ``PredictionModel.extract_features`` as ``select_features``
        so tsfresh only computes the trained-on set.

        Args:
            start_date (str | datetime): Start of the forecast
                window in ``"YYYY-MM-DD"`` format or as a
                ``datetime``.
            end_date (str | datetime): End of the forecast window in
                ``"YYYY-MM-DD"`` format or as a ``datetime``.
            window_step (int): Sliding window step size.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.
            save_seed_result (bool): Persist per-seed probability
                CSVs under ``prediction/results/``. Defaults to
                ``True``.
            plot_threshold (float): Decision threshold drawn on the
                forecast plot. Defaults to ``0.5``.
            plot_title (str | None): Forecast plot title. Defaults to
                ``None``.
            plot_pdf (bool): Also render the forecast plot as PDF.
                Defaults to ``True``.
            use_features_from (Literal["all", "files", "training"]):
                Feature scoping mode. ``"all"`` extracts every tsfresh
                feature (``select_features=None``). ``"files"`` skips
                tsfresh entirely and loads the pair
                ``features_matrix_path`` + ``label_features_csv`` via
                :meth:`PredictionModel.load_features`; both paths must
                be supplied and must exist (raises otherwise) and
                ``use_cache`` is forced to ``False``. ``"training"``
                narrows tsfresh to the union of features selected
                during :meth:`train`. Defaults to ``"all"``.
            features_matrix_path (str | None): Path to a pre-built
                ``features-matrix_*.parquet`` produced by an earlier
                :meth:`PredictionModel.extract_features` run. Only
                honoured when ``use_features_from="files"``; then it
                replaces the ``build_label → extract_features`` prefix
                with :meth:`PredictionModel.load_features`, skipping
                tsfresh entirely. Both paths must be provided together
                (raises ``ValueError`` otherwise). Also forces
                ``use_cache`` to ``False`` because ``load_features``
                does not populate the extract-features kwargs the
                prediction cache identity depends on. Defaults to
                ``None``.
            label_features_csv (str | None): Companion
                ``features-label_*.csv`` for ``features_matrix_path``.
                Only honoured when ``use_features_from="files"``; both
                paths must be provided together. Defaults to ``None``.
            enable_segments_plot (bool): When ``True``, forward the
                training and prediction date ranges to
                :func:`~eruption_forecast.plots.forecast_plots.plot_forecast`
                so it renders the top segment strip above the forecast
                panels. When ``False``, the four date kwargs are passed
                as ``None`` and the strip is omitted. Defaults to
                ``False``.
            output_dir (str | None): Root output directory for
                prediction artefacts. Defaults to the station
                directory.
            overwrite (bool | None): Overwrite cached artefacts.
                ``None`` inherits from ``self.overwrite``. Defaults
                to ``None``.
            n_jobs (int | None): Parallel workers. ``None`` inherits
                from ``self.n_jobs``. Defaults to ``None``.
            use_cache (bool): Short-circuit re-run when a cache hit
                exists. Ignored (forced to ``False``) when
                ``features_matrix_path`` / ``label_features_csv`` are
                supplied. Defaults to ``True``.
            verbose (bool | None): Verbose logging. ``None`` inherits
                from ``self.verbose``. Defaults to ``None``.
            **plot_kwargs (Any): Extra keyword arguments forwarded to
                the forecast plotter. Intentionally excluded from the
                captured config since they may carry non-serialisable
                matplotlib objects.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If :meth:`train` has not produced a trained
                ensemble yet, if only one of ``features_matrix_path`` /
                ``label_features_csv`` is supplied, or if
                ``use_features_from="files"`` is passed without both
                paths.
            FileNotFoundError: If ``use_features_from="files"`` is
                passed with a ``features_matrix_path`` or
                ``label_features_csv`` that does not exist on disk.

        Example:
            >>> fm.predict(
            ...     start_date="2020-07-01", end_date="2020-07-31",
            ...     window_step=10, window_step_unit="minutes",
            ...     plot_threshold=0.5,
            ... )
        """
        if self.TrainingModel is None or self.ClassifierEnsemble is None:
            raise ValueError("Training model not found. Please run train() first.")

        if (features_matrix_path is None) != (label_features_csv is None):
            raise ValueError(
                "features_matrix_path and label_features_csv must be provided together."
            )

        select_features: list[str] | None
        feature_shortcut_paths: tuple[str, str] | None = None

        if use_features_from == "files":
            if features_matrix_path is None or label_features_csv is None:
                raise ValueError(
                    "use_features_from='files' requires both features_matrix_path "
                    "and label_features_csv to be provided."
                )
            if not os.path.exists(features_matrix_path):
                raise FileNotFoundError(
                    f"features_matrix_path does not exist: {features_matrix_path}"
                )
            if not os.path.exists(label_features_csv):
                raise FileNotFoundError(
                    f"label_features_csv does not exist: {label_features_csv}"
                )
            select_features = None
            feature_shortcut_paths = (features_matrix_path, label_features_csv)
            use_cache = False
        elif use_features_from == "training":
            select_features = (
                self.TrainingModel.features_selected_df.index.tolist()
                if not self.TrainingModel.features_selected_df.empty
                else None
            )
        else:
            select_features = None

        # ``plot_kwargs`` are intentionally excluded from the captured config
        # because they may carry non-serialisable matplotlib objects.
        self._config.predict = ForecastPredictConfig(
            start_date=(
                start_date if isinstance(start_date, str) else start_date.isoformat()
            ),
            end_date=(end_date if isinstance(end_date, str) else end_date.isoformat()),
            window_step=window_step,
            window_step_unit=window_step_unit,
            save_seed_result=save_seed_result,
            plot_threshold=plot_threshold,
            plot_title=plot_title,
            plot_pdf=plot_pdf,
            use_features_from=use_features_from,
            features_matrix_path=features_matrix_path,
            label_features_csv=label_features_csv,
            enable_segments_plot=enable_segments_plot,
            output_dir=output_dir,
            overwrite=overwrite,
            n_jobs=n_jobs,
            use_cache=use_cache,
            verbose=verbose,
        )

        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

        resolved_output_dir = output_dir or self.station_dir

        identity = PredictionModel.build_identity(
            nslc=self.nslc,
            tremor_df=self.tremor_df,
            training_hash=self._training_cache_hash,
            start_date=start_date,
            end_date=end_date,
            window_size=self.day_to_forecast,
            build_label_params={
                "window_step": window_step,
                "window_step_unit": window_step_unit,
            },
            extract_features_params={
                "select_tremor_columns": self.select_tremor_columns,
                "save_tremor_matrix_per_method": self.save_tremor_matrix_per_method,
                "exclude_features": self.exclude_features,
                "select_features": select_features,
            },
        )

        prediction_stage_dir = os.path.join(resolved_output_dir, "prediction")

        if use_cache:
            cached = PredictionModel.load(prediction_stage_dir, identity)
            if cached is not None:
                if self.verbose:
                    logger.warning("Loading cached prediction data...")

                self.PredictionModel = cached
                self.results = cached.results
                return self

        prediction_model = PredictionModel(
            model=self.ClassifierEnsemble,
            tremor_data=self.tremor_df,
            start_date=start_date,
            end_date=end_date,
            window_size=self.day_to_forecast,
            nslc=self.nslc,
            training_hash=self._training_cache_hash,
            output_dir=resolved_output_dir,
            overwrite=overwrite,
            prefix_config=self.prefix_config,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        if feature_shortcut_paths is not None:
            matrix_path, label_csv = feature_shortcut_paths
            prediction_model = prediction_model.load_features(
                features_matrix_path=matrix_path,
                label_features_csv=label_csv,
                window_step=window_step,
                window_step_unit=window_step_unit,
            )

        else:
            prediction_model = prediction_model.build_label(
                window_step=window_step,
                window_step_unit=window_step_unit,
            ).extract_features(
                select_tremor_columns=self.select_tremor_columns,
                save_tremor_matrix_per_method=self.save_tremor_matrix_per_method,
                exclude_features=self.exclude_features,
                select_features=select_features,
                overwrite=overwrite,
                n_jobs=n_jobs,
                verbose=verbose,
            )

        self.PredictionModel = prediction_model
        self.results = prediction_model.forecast(
            save_seed_result=save_seed_result,
            plot_threshold=plot_threshold,
            plot_title=plot_title,
            plot_pdf=plot_pdf,
            training_start_date=(
                self.TrainingModel.start_date_str if enable_segments_plot else None
            ),
            training_end_date=(
                self.TrainingModel.end_date_str if enable_segments_plot else None
            ),
            prediction_start_date=(
                prediction_model.start_date_str if enable_segments_plot else None
            ),
            prediction_end_date=(
                prediction_model.end_date_str if enable_segments_plot else None
            ),
            **plot_kwargs,
        )

        return self

    def evaluate(
        self,
        model: Literal["training", "prediction"] = "prediction",
        eruption_dates: list[str] | None = None,
        plot_per_seed: bool = False,
        plot_aggregate: bool = True,
        output_dir: str | None = None,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Evaluate a previously trained ensemble against ground-truth labels.

        Reuses the ``TrainingModel`` or ``PredictionModel`` already produced
        in the current session.  No tsfresh re-run or model re-fit is
        performed, the window grid and extracted features are taken
        directly from the chosen reuse source.

        Args:
            model (Literal["training", "prediction"]): Which model in the
                current pipeline to evaluate.  ``"training"`` performs an
                in-sample / training-window evaluation against the
                ``TrainingModel`` from the last ``train()`` call.
                ``"prediction"`` performs a forecast-window evaluation
                against the ``PredictionModel`` from the last ``predict()``
                call. Defaults to ``"prediction"``.
            eruption_dates (list[str] | None): Ground-truth eruption dates in
                ``YYYY-MM-DD`` format.  Falls back to the dates captured
                during ``train()`` when ``None``. Defaults to ``None``.
            plot_per_seed (bool): Render per-seed evaluation plots. Expensive
                across many seeds. Defaults to ``False``.
            plot_aggregate (bool): Render aggregate plots per classifier.
                Defaults to ``True``.
            output_dir (str | None): Root output directory for evaluation
                artefacts. Defaults to the station directory.
            overwrite (bool | None): Overwrite cached results. Falls back to
                ``self.overwrite`` when ``None``. Defaults to ``None``.
            n_jobs (int | None): Parallel workers. Falls back to
                ``self.n_jobs`` when ``None``. Defaults to ``None``.
            verbose (bool | None): Verbose logging. Falls back to
                ``self.verbose`` when ``None``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If the required model for the selected ``model``
                mode has not been produced yet.
        """
        if model == "training" and self.TrainingModel is None:
            raise ValueError(
                "TrainingModel is required for model='training'. "
                "Please run train() first."
            )
        if model == "prediction" and self.PredictionModel is None:
            raise ValueError(
                "PredictionModel is required for model='prediction'. "
                "Please run train() then predict()."
            )

        self._config.evaluate = ForecastEvaluateConfig(
            model=model,
            eruption_dates=list(eruption_dates) if eruption_dates is not None else None,
            plot_per_seed=plot_per_seed,
            plot_aggregate=plot_aggregate,
            output_dir=output_dir,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

        # Caller-supplied eruption_dates win over the values captured from
        # train(); without either, EvaluationModel runs label-free.
        eruption_dates = (
            eruption_dates if eruption_dates is not None else self._eruption_dates
        )

        model_object: TrainingModel | PredictionModel
        if model == "prediction" and self.PredictionModel is not None:
            model_object = self.PredictionModel
        elif self.TrainingModel is not None:
            model_object = self.TrainingModel
        else:
            raise ValueError(
                f"Model {model} is not supported. Choose between "
                f"'prediction' and 'training'"
            )

        evaluation_model = EvaluationModel(
            model=model_object,
            eruption_dates=eruption_dates,
            output_dir=output_dir or self.station_dir,
            overwrite=overwrite,
            prefix_config=self.prefix_config,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.EvaluationModel = evaluation_model
        self.evaluation_results = evaluation_model.evaluate(
            plot_per_seed=plot_per_seed,
            plot_aggregate=plot_aggregate,
            compare_classifiers=True,
        )

        self.save_config()

        return self

    def explain(
        self,
        model: Literal["training", "prediction"] = "prediction",
        eruption_dates: list[str] | None = None,
        save_per_seed: bool = True,
        plot_per_seed: bool = True,
        plot_aggregate: bool = True,
        figsize: tuple[float, float] | None = None,
        max_display: int = 20,
        group_remaining_features: bool = False,
        dpi: int = 150,
        check_additivity: bool = False,
        overwrite_classifier_explanation: bool = False,
        output_dir: str | None = None,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Explain a previously trained ensemble via SHAP TreeExplainer.

        Reuses the ``TrainingModel`` or ``PredictionModel`` already produced
        in the current session.  No tsfresh re-run or model re-fit is
        performed, the window grid and extracted features are taken
        directly from the chosen reuse source.

        Args:
            model (Literal["training", "prediction"]): Which model in the
                current pipeline to explain. Defaults to ``"prediction"``.
            eruption_dates (list[str] | None): Ground-truth eruption dates
                in ``"YYYY-MM-DD"`` format. Required for prediction-mode
                explanation when the upstream ``PredictionModel`` does not
                carry them. Falls back to the dates captured during
                ``train()`` when ``None``. Defaults to ``None``.
            save_per_seed (bool): Persist each per-seed
                ``shap.Explanation`` to disk so a subsequent run can
                short-circuit recomputation. Defaults to ``True``.
            plot_per_seed (bool): Render per-seed bar and beeswarm plots.
                Defaults to ``True``.
            plot_aggregate (bool): Render per-classifier aggregate bar
                and beeswarm plots over the NaN-padded union feature
                space. Defaults to ``True``.
            figsize (tuple[float, float] | None): Figure size in inches
                for SHAP plots. ``None`` auto-sizes from ``max_display``.
                Defaults to ``None``.
            max_display (int): Maximum number of features to display in
                SHAP plots. Defaults to ``20``.
            group_remaining_features (bool): Forwarded to
                ``shap.plots.beeswarm`` to group features beyond
                ``max_display``. Defaults to ``False``.
            dpi (int): Figure resolution in dots per inch. Defaults to
                ``150``.
            check_additivity (bool): Forwarded to ``shap.TreeExplainer``
                to verify SHAP additivity against the model output.
                Defaults to ``False``.
            overwrite_classifier_explanation (bool): Overwrite the cached
                per-classifier ``ClassifierExplanation.pkl`` artefact.
                Defaults to ``False``.
            output_dir (str | None): Root output directory for explanation
                artefacts. Defaults to the station directory.
            overwrite (bool | None): Overwrite existing files. ``None``
                inherits from ``self.overwrite``. Defaults to ``None``.
            n_jobs (int | None): Parallel workers. ``None`` inherits from
                ``self.n_jobs``. Defaults to ``None``.
            verbose (bool | None): Verbose logging. ``None`` inherits from
                ``self.verbose``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If the required model for the selected ``model``
                mode has not been produced yet.
        """
        if model == "training" and self.TrainingModel is None:
            raise ValueError(
                "TrainingModel is required for model='training'. "
                "Please run train() first."
            )
        if model == "prediction" and self.PredictionModel is None:
            raise ValueError(
                "PredictionModel is required for model='prediction'. "
                "Please run train() then predict()."
            )

        self._config.explain = ForecastExplainConfig(
            model=model,
            eruption_dates=list(eruption_dates) if eruption_dates is not None else None,
            save_per_seed=save_per_seed,
            plot_per_seed=plot_per_seed,
            plot_aggregate=plot_aggregate,
            figsize=figsize,
            max_display=max_display,
            group_remaining_features=group_remaining_features,
            dpi=dpi,
            check_additivity=check_additivity,
            overwrite_classifier_explanation=overwrite_classifier_explanation,
            output_dir=output_dir,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

        eruption_dates = (
            eruption_dates if eruption_dates is not None else self._eruption_dates
        )

        model_object: TrainingModel | PredictionModel
        if model == "prediction" and self.PredictionModel is not None:
            model_object = self.PredictionModel
        elif self.TrainingModel is not None:
            model_object = self.TrainingModel
        else:
            raise ValueError(
                f"Model {model} is not supported. Choose between "
                f"'prediction' and 'training'"
            )

        explanation_model = (
            ExplanationModel(
                model=model_object,
                eruption_dates=eruption_dates,
                output_dir=output_dir or self.station_dir,
                overwrite=overwrite,
                prefix_config=self.prefix_config,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            .explain(
                save_per_seed=save_per_seed,
                check_additivity=check_additivity,
                overwrite_classifier_explanation=overwrite_classifier_explanation,
            )
            .plot(
                figsize=figsize,
                max_display=max_display,
                group_remaining_features=group_remaining_features,
                dpi=dpi,
                plot_per_seed=plot_per_seed,
                plot_aggregate=plot_aggregate,
            )
        )

        self.ExplanationModel = explanation_model

        self.save_config()

        return self

    def save_config(
        self,
        path: str | None = None,
        fmt: Literal["yaml", "json"] = "yaml",
    ) -> str:
        """Persist the captured pipeline configuration to disk.

        Each stage method (``calculate``, ``train``, ``predict``,
        ``evaluate``, ``explain``) auto-captures its kwargs into
        ``self._config`` as it runs, so calling ``save_config()`` at any
        point writes whatever has run so far. A partial pipeline produces a
        partial config that ``run()`` can resume.

        ``evaluate()`` and ``explain()`` both call ``save_config()`` at the
        end so the config persists from whichever terminal stage the caller
        stops at. When both stages run the write fires twice with identical
        content, intentional and idempotent.

        Args:
            path (str | None): Destination file path.  ``None`` resolves to
                ``{station_dir}/forecast.config.{fmt}``, a sibling of the
                per-stage cache pickles written next to each stage's outputs
                by :meth:`BaseModel.save`. When ``self.prefix_config`` is
                set, its slugified form is inserted before ``.config`` —
                e.g. ``forecast.scenario-1.config.yaml`` — so multiple
                scenarios sharing the same ``station_dir`` do not clobber
                each other. Defaults to ``None``.
            fmt (Literal["yaml", "json"]): Output format.  Defaults to
                ``"yaml"``.

        Returns:
            str: The absolute path the configuration was written to.
        """
        if path is None:
            suffix = (
                f".{slug}"
                if self.prefix_config and (slug := slugify(self.prefix_config))
                else ""
            )
            path = os.path.join(self.station_dir, f"forecast{suffix}.config.{fmt}")
        return self._config.save(path, fmt)

    @classmethod
    def from_config(cls, path: str) -> Self:
        """Reconstruct a :class:`ForecastModel` from a saved configuration.

        Loads the YAML/JSON file at ``path``, instantiates a fresh
        ``ForecastModel`` from the ``model`` section, and attaches the loaded
        stage sections so :meth:`run` can replay them.

        Args:
            path (str): Path to a configuration file previously written by
                :meth:`save_config`.

        Returns:
            ForecastModel: A new instance ready for :meth:`run`.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
        """
        config = ForecastConfig.load(path)
        instance = cls(**config.model.to_init_kwargs())
        instance._config = config
        return instance

    def run(self) -> Self:
        """Replay every captured stage in pipeline order.

        Iterates over ``calculate``, ``train``, ``predict``, ``evaluate`` and
        calls the corresponding method for each non-``None`` section.  Each
        stage's auto-capture overwrites its own slot, so the operation is
        idempotent.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        if self._config.calculate is not None:
            self.calculate(**self._config.calculate.to_init_kwargs())
        if self._config.train is not None:
            self.train(**self._config.train.to_init_kwargs())
        if self._config.predict is not None:
            self.predict(**self._config.predict.to_init_kwargs())
        if self._config.evaluate is not None:
            self.evaluate(**self._config.evaluate.to_init_kwargs())
        if self._config.explain is not None:
            self.explain(**self._config.explain.to_init_kwargs())
        return self
