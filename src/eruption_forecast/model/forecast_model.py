import os
from typing import Any, Self, Literal
from datetime import datetime, timedelta

import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import setup_nslc_directories
from eruption_forecast.utils.date_utils import to_datetime
from eruption_forecast.model.cache_model import CacheModel
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
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialize the ForecastModel pipeline orchestrator."""
        # Capture the user-supplied output_dir / root_dir before
        # setup_nslc_directories() and os.path.abspath() rewrite them — the
        # saved config should round-trip to the user's original intent
        # (often a relative path or ``None``).
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

        # Eruption dates from train() forwarded to evaluate() as a fallback
        # when the caller does not pass them explicitly.
        self._eruption_dates: list[str] | None = None

        # Hash of the TrainingModel identity produced by the last train() call.
        # Threaded into PredictionModel.build_cache_identity so the prediction
        # cache invalidates whenever the upstream training model changes.
        self._training_cache_hash: str | None = None

        # Will be set after predict() run
        self.PredictionModel: PredictionModel | None = None
        self.results: pd.DataFrame = pd.DataFrame()

        # Will be set after evaluate() run
        self.EvaluationModel: EvaluationModel | None = None
        self.evaluation_results: dict[str, pd.DataFrame] = {}

        # Will be set after explain() run
        self.ExplanationModel: ExplanationModel | None = None

        # Pipeline configuration — populated incrementally as each stage runs.
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
        overwrite_plot: bool = False,
        sds_dir: str | None = None,
        client_url: str = "https://service.iris.edu",
        minimum_completion_ratio: float = 0.3,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
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
            overwrite_plot=overwrite_plot,
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
            calculate = calculate.from_sds(sds_dir=sds_dir).run()
        elif source.upper() == "FDSN":
            calculate = calculate.from_fdsn(client_url=client_url).run()
        else:
            raise ValueError(f"Unknown source {source!r}. Expected 'sds' or 'fdsn'.")

        self.CalculateTremor = calculate

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
            overwrite_plot=overwrite_plot,
            sds_dir=sds_dir,
            client_url=client_url,
            minimum_completion_ratio=minimum_completion_ratio,
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
        if self.CalculateTremor is None:
            raise ValueError("Tremor data not found. Please run calculate() first.")

        # Snapshot originals for the captured config — must happen before the
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

        identity = TrainingModel.build_cache_identity(
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

        if use_cache:
            cached = TrainingModel.load_from_cache(resolved_output_dir, identity)
            if cached is not None:
                if self.verbose:
                    logger.warning("Loading cached training data...")

                self.TrainingModel = cached
                self.ClassifierEnsemble = cached.ClassifierEnsemble
                self.select_tremor_columns = select_tremor_columns
                self.save_tremor_matrix_per_method = save_tremor_matrix_per_method
                self.exclude_features = exclude_features
                self._eruption_dates = eruption_dates
                self._training_cache_hash = CacheModel.compute_hash(identity)
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
                output_dir=resolved_output_dir,
                overwrite=overwrite,
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

        training_model.save_to_cache(identity)
        self._training_cache_hash = CacheModel.compute_hash(identity)

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
        output_dir: str | None = None,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        use_cache: bool = True,
        verbose: bool | None = None,
        **plot_kwargs: Any,
    ) -> Self:
        if self.TrainingModel is None or self.ClassifierEnsemble is None:
            raise ValueError("Training model not found. Please run train() first.")

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

        identity = PredictionModel.build_cache_identity(
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
            },
        )

        if use_cache:
            cached = PredictionModel.load_from_cache(resolved_output_dir, identity)
            if cached is not None:
                if self.verbose:
                    logger.warning("Loading cached prediction data...")

                self.PredictionModel = cached
                self.results = cached.results
                return self

        prediction_model = (
            PredictionModel(
                model=self.ClassifierEnsemble,
                tremor_data=self.tremor_df,
                start_date=start_date,
                end_date=end_date,
                window_size=self.day_to_forecast,
                output_dir=resolved_output_dir,
                overwrite=overwrite,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            .build_label(
                window_step=window_step,
                window_step_unit=window_step_unit,
            )
            .extract_features(
                select_tremor_columns=self.select_tremor_columns,
                save_tremor_matrix_per_method=self.save_tremor_matrix_per_method,
                exclude_features=self.exclude_features,
                overwrite=overwrite,
                n_jobs=n_jobs,
                verbose=verbose,
            )
        )

        self.PredictionModel = prediction_model
        self.results = prediction_model.forecast(
            save_seed_result=save_seed_result,
            plot_threshold=plot_threshold,
            plot_title=plot_title,
            plot_pdf=plot_pdf,
            **plot_kwargs,
        )

        prediction_model.save_to_cache(identity)

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
        performed — the window grid and extracted features are taken
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
        n_observations_to_explain: int = 10,
        method: Literal["shap"] = "shap",
        feature_perturbation: Literal[
            "tree_path_dependent", "interventional"
        ] = "tree_path_dependent",
        model_output: Literal["raw", "probability", "log_loss"] = "raw",
        background_size: int = 100,
        check_additivity: bool = True,
        selection: Literal["top_proba", "near_threshold"] = "top_proba",
        plot_aggregate: bool = True,
        plot_per_seed: bool = False,
        plot_waterfall: bool = True,
        output_dir: str | None = None,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Explain a previously trained ensemble via SHAP TreeExplainer.

        Reuses the ``TrainingModel`` or ``PredictionModel`` already produced
        in the current session.  No tsfresh re-run or model re-fit is
        performed — the window grid and extracted features are taken
        directly from the chosen reuse source.

        Args:
            model (Literal["training", "prediction"]): Which model in the
                current pipeline to explain. Defaults to ``"prediction"``.
            n_observations_to_explain (int): Top-N observations per
                classifier forwarded to ``ExplainerEnsemble``. Defaults to
                ``10``.
            method (Literal["shap"]): Explanation method. Reserved for
                future additions. Defaults to ``"shap"``.
            feature_perturbation (Literal["tree_path_dependent",
                "interventional"]): SHAP perturbation mode. Defaults to
                ``"tree_path_dependent"``.
            model_output (Literal["raw", "probability", "log_loss"]):
                SHAP output unit. ``"probability"`` and ``"log_loss"``
                require ``feature_perturbation="interventional"``.
                Defaults to ``"raw"``.
            background_size (int): Background sample size for the
                interventional path. Defaults to ``100``.
            check_additivity (bool): Forwarded to the inner
                ``explainer(X, ...)`` call. Defaults to ``True``.
            selection (Literal["top_proba", "near_threshold"]):
                Observation-ranking strategy. Defaults to ``"top_proba"``.
            plot_aggregate (bool): Render aggregate plots per classifier.
                Defaults to ``True``.
            plot_per_seed (bool): Render per-seed bar / beeswarm plots.
                Defaults to ``False``.
            plot_waterfall (bool): When ``plot_per_seed=True``, also
                render per-(seed, observation) waterfall plots. Defaults
                to ``True``.
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
            n_observations_to_explain=n_observations_to_explain,
            method=method,
            feature_perturbation=feature_perturbation,
            model_output=model_output,
            background_size=background_size,
            check_additivity=check_additivity,
            selection=selection,
            plot_aggregate=plot_aggregate,
            plot_per_seed=plot_per_seed,
            plot_waterfall=plot_waterfall,
            output_dir=output_dir,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        verbose = verbose if verbose is not None else self.verbose
        overwrite = overwrite if overwrite is not None else self.overwrite

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

        explanation_model = ExplanationModel(
            model=model_object,
            n_observations_to_explain=n_observations_to_explain,
            method=method,
            feature_perturbation=feature_perturbation,
            model_output=model_output,
            background_size=background_size,
            check_additivity=check_additivity,
            selection=selection,
            output_dir=output_dir or self.station_dir,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        ).explain(
            plot_aggregate=plot_aggregate,
            plot_per_seed=plot_per_seed,
            plot_waterfall=plot_waterfall,
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
        ``evaluate``) auto-captures its kwargs into ``self._config`` as it
        runs, so calling ``save_config()`` at any point writes whatever has
        run so far.  A partial pipeline produces a partial config that
        ``run()`` can resume.

        Args:
            path (str | None): Destination file path.  ``None`` resolves to
                ``{station_dir}/forecast.config.{fmt}`` — a sibling of
                the per-stage ``cache/`` directories written by
                :class:`~eruption_forecast.model.cache_model.CacheModel`.
                Defaults to ``None``.
            fmt (Literal["yaml", "json"]): Output format.  Defaults to
                ``"yaml"``.

        Returns:
            str: The absolute path the configuration was written to.
        """
        if path is None:
            path = os.path.join(self.station_dir, f"forecast.config.{fmt}")
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
        instance = cls(**config.model.to_dict())
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
            self.calculate(**self._config.calculate.to_dict())
        if self._config.train is not None:
            self.train(**self._config.train.to_dict())
        if self._config.predict is not None:
            self.predict(**self._config.predict.to_dict())
        if self._config.evaluate is not None:
            self.evaluate(**self._config.evaluate.to_dict())
        if self._config.explain is not None:
            self.explain(**self._config.explain.to_dict())
        return self
