import os
import json
import socket
from typing import Any, Self, Literal, TypedDict, NotRequired
from datetime import datetime

from eruption_forecast import ForecastModel, TelegramNotification
from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import setup_nslc_directories
from eruption_forecast.utils.formatting import slugify


class Scenario(TypedDict):
    """Specification for a single scenario in a ``ForecastModelScenario`` sweep.

    A ``Scenario`` bundles the training window, prediction window, and
    prediction-grid parameters that vary between scenarios in a sweep. The
    training params (classifiers, seeds, etc.) and prediction params
    (threshold, features source, etc.) that stay constant across scenarios
    are supplied once via :meth:`ForecastModelScenario.shared_training` and
    :meth:`ForecastModelScenario.shared_prediction`.

    Keys:
        name (str): Human-readable scenario name. Slugified into the
            on-disk directory (``"Scenario 1"`` → ``scenarios/scenario-1/``)
            and must be unique across the sweep.
        description (str): Short free-text label logged at the start of
            each scenario and used as the caption on the per-scenario
            Telegram notification.
        train_start_date (str): Training window start
            (``"YYYY-MM-DD"``).
        train_end_date (str): Training window end (``"YYYY-MM-DD"``).
        prediction_start_date (str): Prediction window start
            (``"YYYY-MM-DD"``).
        prediction_end_date (str): Prediction window end
            (``"YYYY-MM-DD"``).
        prediction_window_step (int): Sliding step for the prediction
            grid.
        prediction_window_step_unit (Literal["minutes", "hours"]): Unit
            for ``prediction_window_step``.
        prediction_plot_kwargs (NotRequired[dict[str, Any]]): Optional
            forecast-plot kwargs (e.g. ``rolling_window``,
            ``x_days_interval``, ``legend_n_cols``). Merged into the
            ``**plot_kwargs`` forwarded to
            ``ForecastModel.predict(..., **plot_kwargs)`` at run time.
    """

    name: str
    description: str
    train_start_date: str
    train_end_date: str
    prediction_start_date: str
    prediction_end_date: str
    prediction_window_step: int
    prediction_window_step_unit: Literal["minutes", "hours"]
    prediction_plot_kwargs: NotRequired[dict[str, Any]]


class ForecastModelScenario:
    """Multi-scenario orchestrator that packages a ``ForecastModel`` sweep.

    Wraps one :class:`ForecastModel` instance and loops
    ``train → predict → evaluate → explain`` over N :class:`Scenario`
    dicts. A single :meth:`calculate` pass produces the shared tremor
    frame reused by every scenario; shared per-stage parameter dicts
    (populated by :meth:`shared_training`, :meth:`shared_prediction`,
    :meth:`shared_evaluation`, and :meth:`shared_explanation`) are
    forwarded verbatim to the wrapped ``ForecastModel`` on each iteration.

    The class is a packaged alternative to the hand-rolled scenario loop
    that lived in ``scenarios.py``; it enforces the loop invariants, saves
    a ``scenarios.json`` manifest for auditability, and ships a Telegram
    document per scenario when a forecast plot is produced.
    """

    def __init__(
        self,
        station: str,
        channel: str,
        network: str,
        eruption_dates: list[str],
        scenarios: list[Scenario],
        location: str = "",
        day_to_forecast: int = 2,
        output_dir: str | None = None,
        scenarios_dir: str = "scenarios",
        root_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Initialize the multi-scenario orchestrator.

        Args:
            station (str): Station code. Uppercased on assignment.
            channel (str): Channel code (e.g. ``"EHZ"``). Uppercased on
                assignment.
            network (str): Network code. Uppercased on assignment.
            eruption_dates (list[str]): Ground-truth eruption dates in
                ``"YYYY-MM-DD"`` format. Reused by every scenario for
                labelling, evaluation, and forecast-plot markers.
            scenarios (list[Scenario]): Scenario specs (see
                :class:`Scenario`). Names must be unique after
                slugification.
            location (str): Location code. Uppercased on assignment.
                Defaults to ``""``.
            day_to_forecast (int): Forecast lead time in days. Threaded
                into the wrapped ``ForecastModel``. Defaults to ``2``.
            output_dir (str | None): Root output directory. ``None``
                resolves via :func:`setup_nslc_directories`. Defaults
                to ``None``.
            scenarios_dir (str): Child directory name (under the
                resolved station directory) that holds per-scenario
                artefacts and the ``scenarios.json`` manifest. Defaults
                to ``"scenarios"``.
            root_dir (str | None): Project root used when resolving a
                relative ``output_dir``. Defaults to ``None``.
            overwrite (bool): Default overwrite flag inherited by every
                ``shared_*`` wrapper when its own ``overwrite`` kwarg is
                ``None``. Defaults to ``False``.
            n_jobs (int): Default parallel-worker count forwarded to the
                wrapped ``ForecastModel``. Also inherited by
                :meth:`calculate` when its own ``n_jobs`` kwarg is
                ``None``. Defaults to ``1``.
            verbose (bool): Default verbose flag inherited by every
                ``shared_*`` wrapper when its own ``verbose`` kwarg is
                ``None``. Defaults to ``False``.

        Example:
            >>> scenarios: list[Scenario] = [
            ...     {
            ...         "name": "Scenario 1",
            ...         "description": "Train on 1 eruption, forecast 2+",
            ...         "train_start_date": "2025-01-01",
            ...         "train_end_date": "2025-03-31",
            ...         "prediction_start_date": "2025-04-01",
            ...         "prediction_end_date": "2025-08-22",
            ...         "prediction_window_step": 10,
            ...         "prediction_window_step_unit": "minutes",
            ...     },
            ... ]
            >>> fms = ForecastModelScenario(
            ...     station="OJN", channel="EHZ", network="VG",
            ...     location="00", day_to_forecast=2,
            ...     eruption_dates=["2025-03-20", "2025-04-10"],
            ...     scenarios=scenarios, n_jobs=8,
            ... )
        """
        root_dir = os.path.abspath(root_dir) if root_dir is not None else None
        nslc, output_dir, station_dir = setup_nslc_directories(
            network, station, location, channel, output_dir, root_dir
        )

        self.station = station.upper()
        self.channel = channel.upper()
        self.network = network.upper()
        self.location = location.upper()
        self.eruption_dates = eruption_dates
        self.day_to_forecast = day_to_forecast
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.nslc = nslc
        self.ForecastModel = ForecastModel(
            network=network,
            station=station,
            location=location,
            channel=channel,
            day_to_forecast=day_to_forecast,
            n_jobs=n_jobs,
            output_dir=output_dir,
            verbose=verbose,
        )
        self.scenarios: list[Scenario] = scenarios
        self.scenarios_dir = os.path.join(station_dir, scenarios_dir)

        self.shared_training_params: dict[str, Any] = {}
        self.shared_prediction_params: dict[str, Any] = {}
        self.shared_evaluation_params: dict[str, Any] = {}
        self.shared_explanation_params: dict[str, Any] = {}

        if verbose:
            logger.info(f"Init Forecast Model: {self.ForecastModel}")
            logger.info(f"Scenarios output directory: {self.scenarios_dir}")

    @staticmethod
    def build_plot_kwargs(
        scenario: Scenario, eruption_dates: list[str]
    ) -> dict[str, Any]:
        """Assemble the ``**plot_kwargs`` dict passed to ``ForecastModel.predict``.

        Copies the scenario's optional ``prediction_plot_kwargs`` (if any)
        and always injects ``eruption_dates`` so per-scenario forecast
        plots carry eruption-day markers.

        Args:
            scenario (Scenario): Scenario spec to source
                ``prediction_plot_kwargs`` from.
            eruption_dates (list[str]): Eruption dates in
                ``"YYYY-MM-DD"`` format, injected under the
                ``eruption_dates`` key.

        Returns:
            dict[str, Any]: Kwargs to splat into
            ``ForecastModel.predict(..., **plot_kwargs)``.
        """
        plot_kwargs = dict(scenario.get("prediction_plot_kwargs", {}))
        plot_kwargs["eruption_dates"] = eruption_dates
        return plot_kwargs

    def validate(self) -> Self:
        """Enforce always-required preconditions before :meth:`run` begins.

        Checks that (a) every scenario slug is unique, (b) :meth:`calculate`
        has produced a tremor frame on the wrapped ``ForecastModel``,
        (c) :meth:`shared_training` has populated
        ``self.shared_training_params``, and (d) :meth:`shared_prediction`
        has populated ``self.shared_prediction_params``. Creates
        ``self.scenarios_dir`` on disk when validation passes.

        :meth:`shared_evaluation` and :meth:`shared_explanation` are
        checked separately inside :meth:`run`, gated on the
        ``evaluate_model`` / ``explain_model`` flags, so a sweep that
        opts out of both stages does not need those calls.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If two scenarios slugify to the same name.
            RuntimeError: If :meth:`calculate`, :meth:`shared_training`,
                or :meth:`shared_prediction` have not been called yet.
        """
        scenario_names = []
        for scenario in self.scenarios:
            scenario_name = slugify(scenario["name"])
            if len(scenario_names) == 0:
                scenario_names.append(scenario_name)
            elif scenario_name in scenario_names:
                raise ValueError(
                    f"Scenario name: {scenario['name']} already exists. "
                    f"Please choose a different one."
                )
            else:
                scenario_names.append(scenario_name)

        if self.ForecastModel.CalculateTremor is None:
            raise RuntimeError("Please run `calculate(...)` first.")

        if not self.shared_training_params:
            raise RuntimeError("Please run `shared_training(...)` first.")

        if not self.shared_prediction_params:
            raise RuntimeError("Please run `shared_prediction(...)` first.")

        os.makedirs(self.scenarios_dir, exist_ok=True)

        return self

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
        plot_rsam_as_log: bool = False,
        plot_rolling_window: str | None = None,
        plot_filter_dsar_value: float | None = None,
        plot_overwrite: bool = False,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Run the wrapped ``ForecastModel.calculate`` once for every scenario.

        Delegates to :meth:`ForecastModel.calculate` and keeps the produced
        tremor frame on the wrapped instance so every scenario in
        :meth:`run` can reuse it without recomputation.
        ``plot_eruption_dates`` is threaded in from ``self.eruption_dates``
        so daily band plots carry eruption-day markers.

        Args:
            start_date (str | datetime): Start of the tremor window in
                ``"YYYY-MM-DD"`` format or as a ``datetime``. Internally
                back-shifted by ``day_to_forecast`` days.
            end_date (str | datetime): End of the tremor window.
            source (Literal["sds", "fdsn"]): Data source. Defaults to
                ``"sds"``.
            methods (str | list[str] | None): Tremor metrics to compute
                (``"rsam"``, ``"dsar"``, ``"entropy"``). ``None`` runs
                every metric. Defaults to ``None``.
            remove_outlier_method (Literal["all", "maximum"]): Outlier
                strategy. Defaults to ``"maximum"``.
            remove_tremor_anomalies (bool): Drop z-score anomalies from
                the merged tremor frame. Defaults to ``False``.
            interpolate (bool): Fill miniSEED gaps before computing
                metrics so tsfresh receives a continuous signal.
                Defaults to ``True``.
            value_multiplier (float | None): Scalar multiplier applied
                to every tremor value. Defaults to ``None``.
            cleanup_daily_dir (bool): Delete the per-day CSVs after
                merging. Defaults to ``False``.
            plot_daily (bool): Render per-day band plots under
                ``tremor/figures/``. Defaults to ``False``.
            save_plot (bool): Save the daily plots to disk. Defaults to
                ``False``.
            sds_dir (str | None): SDS archive root (required when
                ``source="sds"``). Defaults to ``None``.
            client_url (str): FDSN endpoint (used when
                ``source="fdsn"``). Defaults to
                ``"https://service.iris.edu"``.
            minimum_completion_ratio (float): Minimum daily completion
                ratio required to keep a day. Defaults to ``0.3``.
            plot_rsam_as_log (bool): Plot RSAM on a log axis. Defaults
                to ``False``.
            plot_rolling_window (str | None): Rolling-window smoothing
                applied to plotted tremor. Defaults to ``None``.
            plot_filter_dsar_value (float | None): Upper-bound filter
                applied to plotted DSAR. Defaults to ``None``.
            plot_overwrite (bool): Overwrite existing plot files.
                Defaults to ``False``.
            overwrite (bool | None): Overwrite tremor CSV cache.
                ``None`` defers to ``ForecastModel.calculate``'s own
                default. Defaults to ``None``.
            n_jobs (int | None): Parallel workers. ``None`` inherits
                from ``self.n_jobs``. Defaults to ``None``.
            verbose (bool | None): Verbose logging. ``None`` inherits
                from ``self.verbose``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        self.ForecastModel.calculate(
            start_date=start_date,
            end_date=end_date,
            source=source,
            methods=methods,
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
            plot_eruption_dates=self.eruption_dates,
            plot_rsam_as_log=plot_rsam_as_log,
            plot_rolling_window=plot_rolling_window,
            plot_filter_dsar_value=plot_filter_dsar_value,
            plot_overwrite=plot_overwrite,
            overwrite=overwrite,
            n_jobs=n_jobs if n_jobs is not None else self.n_jobs,
            verbose=verbose if verbose is not None else self.verbose,
        )

        return self

    def shared_training(
        self,
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
        resample_method: Literal["under", "over", "auto"] | None = "under",
        minority_threshold: float = 0.15,
        sampling_strategy: str | float = 0.75,
        plot_features: bool = True,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        n_grids: int = 1,
        use_cache: bool = True,
        verbose: bool | None = None,
    ) -> Self:
        """Capture the training kwargs shared across every scenario.

        Populates ``self.shared_training_params`` with every kwarg
        forwarded to :meth:`ForecastModel.train` at :meth:`run` time. The
        per-scenario ``start_date`` / ``end_date`` / ``output_dir`` are
        injected inside :meth:`run` and are deliberately absent from this
        method's signature; ``eruption_dates`` is threaded in from
        ``self.eruption_dates`` set on the constructor.

        Args:
            window_step (int): Sliding window step size for the label
                grid.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.
            label_builder (Literal["standard", "dynamic"]): Label
                construction strategy. Defaults to ``"standard"``.
            days_before_eruption (int | None): Positive-window width
                ahead of each eruption when ``label_builder="dynamic"``.
                Defaults to ``None``.
            classifiers (str | list[str]): Classifier slugs to fit.
                Defaults to ``"rf"``.
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
            select_tremor_columns (list[str] | None): Subset of tremor
                columns fed to tsfresh. Defaults to ``None``.
            save_tremor_matrix_per_method (bool): Persist per-column
                tremor matrices under ``per_method/``. Defaults to
                ``True``.
            exclude_features (list[str] | None): Feature substrings to
                drop after tsfresh extraction. Defaults to ``None``.
            select_features (str | list[str] | None): Feature selection
                method. Defaults to ``None``.
            minimum_completion (float): Minimum per-window completion
                ratio required to keep a window. Defaults to ``1.0``.
            seeds (int): Number of seed models per classifier. Defaults
                to ``10``.
            resample_method (Literal["under", "over", "auto"] | None):
                Imbalance handling. Defaults to ``"under"``.
            minority_threshold (float): Minority-class share below
                which ``"auto"`` triggers undersampling. Defaults to
                ``0.15``.
            sampling_strategy (str | float): Forwarded to the
                imbalanced-learn sampler. Defaults to ``0.75``.
            plot_features (bool): Render per-seed feature importance
                plots. Defaults to ``True``.
            overwrite (bool | None): Overwrite cached training
                artefacts. ``None`` inherits from ``self.overwrite``.
                Defaults to ``None``.
            n_jobs (int | None): Outer seed workers. ``None`` defers to
                ``ForecastModel.train``'s own default. Defaults to
                ``None``.
            n_grids (int): Inner ``GridSearchCV`` /
                ``FeatureSelector`` workers. Defaults to ``1``.
            use_cache (bool): Short-circuit re-fit on a cache hit.
                Defaults to ``True``.
            verbose (bool | None): Verbose logging. ``None`` inherits
                from ``self.verbose``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        shared_training_params = {
            "eruption_dates": self.eruption_dates,
            "window_step": window_step,
            "window_step_unit": window_step_unit,
            "label_builder": label_builder,
            "days_before_eruption": days_before_eruption,
            "classifiers": classifiers,
            "cv_strategy": cv_strategy,
            "cv_splits": cv_splits,
            "scoring": scoring,
            "top_n_features": top_n_features,
            "include_eruption_date": include_eruption_date,
            "select_tremor_columns": select_tremor_columns,
            "save_tremor_matrix_per_method": save_tremor_matrix_per_method,
            "exclude_features": exclude_features,
            "select_features": select_features,
            "minimum_completion": minimum_completion,
            "seeds": seeds,
            "resample_method": resample_method,
            "minority_threshold": minority_threshold,
            "sampling_strategy": sampling_strategy,
            "plot_features": plot_features,
            "overwrite": overwrite if overwrite is not None else self.overwrite,
            "n_jobs": n_jobs,
            "n_grids": n_grids,
            "use_cache": use_cache,
            "verbose": verbose if verbose is not None else self.verbose,
        }

        self.shared_training_params = shared_training_params

        return self

    def shared_prediction(
        self,
        save_seed_result: bool = True,
        plot_threshold: float = 0.5,
        plot_title: str | None = None,
        plot_pdf: bool = True,
        use_features_from: Literal["all", "files", "training"] = "all",
        enable_segments_plot: bool = False,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        use_cache: bool = True,
        verbose: bool | None = None,
    ) -> Self:
        """Capture the prediction kwargs shared across every scenario.

        Populates ``self.shared_prediction_params`` with the kwargs
        forwarded to :meth:`ForecastModel.predict` at :meth:`run` time.
        The per-scenario ``start_date`` / ``end_date`` / ``window_step`` /
        ``window_step_unit`` / ``output_dir`` are injected inside
        :meth:`run`.

        Note:
            When ``use_features_from="files"`` :meth:`run` coerces the
            first scenario to ``"all"`` (so features are extracted fresh
            over the first scenario's prediction window) and then reuses
            the first scenario's ``features_matrix_path`` /
            ``label_features_csv`` for every subsequent scenario. This is
            a substantial speedup for long sweeps, **but** it requires
            that the first scenario's prediction window
            (``prediction_start_date`` → ``prediction_end_date``) span
            every datetime any later scenario predicts over. If a later
            scenario predicts over datetimes the first scenario did not
            cover, the reused features matrix will be missing those rows
            and the run will fail. Order scenarios so the widest
            prediction window comes first, or leave ``use_features_from``
            at ``"all"`` and let every scenario re-extract.

        Args:
            save_seed_result (bool): Persist per-seed probability CSVs.
                Defaults to ``True``.
            plot_threshold (float): Probability threshold used to draw
                the forecast decision line. Defaults to ``0.5``.
            plot_title (str | None): Optional forecast plot title.
                Defaults to ``None``.
            plot_pdf (bool): Emit a PDF alongside the PNG forecast plot.
                Defaults to ``True``.
            use_features_from (Literal["all", "files", "training"]):
                Source of the features matrix. ``"all"`` re-extracts
                over the prediction window; ``"files"`` reuses the
                first scenario's on-disk matrix (see Note above);
                ``"training"`` reuses the training window's matrix.
                Defaults to ``"all"``.
            enable_segments_plot (bool): Render the segments plot.
                Defaults to ``False``.
            overwrite (bool | None): Overwrite cached prediction
                artefacts. ``None`` inherits from ``self.overwrite``.
                Defaults to ``None``.
            n_jobs (int | None): Parallel workers. ``None`` defers to
                ``ForecastModel.predict``'s own default. Defaults to
                ``None``.
            use_cache (bool): Short-circuit re-forecast on a cache hit.
                Defaults to ``True``.
            verbose (bool | None): Verbose logging. ``None`` inherits
                from ``self.verbose``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        shared_prediction_params = {
            "save_seed_result": save_seed_result,
            "plot_threshold": plot_threshold,
            "plot_title": plot_title,
            "plot_pdf": plot_pdf,
            "use_features_from": use_features_from,
            "enable_segments_plot": enable_segments_plot,
            "overwrite": overwrite if overwrite is not None else self.overwrite,
            "n_jobs": n_jobs,
            "use_cache": use_cache,
            "verbose": verbose if verbose is not None else self.verbose,
        }

        self.shared_prediction_params = shared_prediction_params

        return self

    def shared_evaluation(
        self,
        model: Literal["training", "prediction"] = "prediction",
        plot_per_seed: bool = False,
        plot_aggregate: bool = True,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        use_cache: bool = True,
        verbose: bool | None = None,
    ) -> Self:
        """Capture the evaluation kwargs shared across every scenario.

        Populates ``self.shared_evaluation_params`` with the kwargs
        forwarded to :meth:`ForecastModel.evaluate` at :meth:`run` time.
        ``eruption_dates`` is threaded in from ``self.eruption_dates``.
        Required only when :meth:`run` is called with
        ``evaluate_model=True`` (the default) — :meth:`run` raises
        ``RuntimeError`` in that case if this call has not been made.
        Sweeps that pass ``evaluate_model=False`` to :meth:`run` do
        not need to call this method at all.

        Args:
            model (Literal["training", "prediction"]): Which pipeline
                stage to evaluate against. Defaults to ``"prediction"``.
            plot_per_seed (bool): Render per-seed evaluation plots.
                Defaults to ``False``.
            plot_aggregate (bool): Render aggregate cross-seed
                evaluation plots. Defaults to ``True``.
            overwrite (bool | None): Overwrite cached evaluation
                artefacts. ``None`` inherits from ``self.overwrite``.
                Defaults to ``None``.
            n_jobs (int | None): Parallel workers. ``None`` defers to
                ``ForecastModel.evaluate``'s own default. Defaults to
                ``None``.
            use_cache (bool): Short-circuit re-evaluate on a cache hit.
                Defaults to ``True``.
            verbose (bool | None): Verbose logging. ``None`` inherits
                from ``self.verbose``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        shared_evaluation_params = {
            "model": model,
            "eruption_dates": self.eruption_dates,
            "plot_per_seed": plot_per_seed,
            "plot_aggregate": plot_aggregate,
            "overwrite": overwrite if overwrite is not None else self.overwrite,
            "n_jobs": n_jobs,
            "use_cache": use_cache,
            "verbose": verbose if verbose is not None else self.verbose,
        }

        self.shared_evaluation_params = shared_evaluation_params

        return self

    def shared_explanation(
        self,
        model: Literal["training", "prediction"] = "prediction",
        save_per_seed: bool = True,
        plot_per_seed: bool = True,
        plot_aggregate: bool = True,
        figsize: tuple[float, float] | None = None,
        max_display: int = 20,
        group_remaining_features: bool = False,
        dpi: int = 150,
        check_additivity: bool = False,
        overwrite_classifier_explanation: bool = False,
        overwrite: bool | None = None,
        n_jobs: int | None = None,
        use_cache: bool = True,
        verbose: bool | None = None,
    ) -> Self:
        """Capture the explanation kwargs shared across every scenario.

        Populates ``self.shared_explanation_params`` with the kwargs
        forwarded to :meth:`ForecastModel.explain` at :meth:`run` time.
        ``eruption_dates`` is threaded in from ``self.eruption_dates``.
        Required only when :meth:`run` is called with
        ``explain_model=True`` (the default) — :meth:`run` raises
        ``RuntimeError`` in that case if this call has not been made.
        Sweeps that pass ``explain_model=False`` to :meth:`run` do
        not need to call this method at all.

        Args:
            model (Literal["training", "prediction"]): Which upstream
                stage to explain. Defaults to ``"prediction"``.
            save_per_seed (bool): Persist per-seed SHAP payloads under
                ``shap_values/{seed:05d}.pkl``. Defaults to ``True``.
            plot_per_seed (bool): Render per-seed bar + beeswarm plots.
                Defaults to ``True``.
            plot_aggregate (bool): Render aggregate cross-seed bar +
                beeswarm plots. Defaults to ``True``.
            figsize (tuple[float, float] | None): Optional figure size
                for SHAP plots. Defaults to ``None``.
            max_display (int): Maximum features shown per SHAP plot.
                Defaults to ``20``.
            group_remaining_features (bool): Group the tail of features
                past ``max_display``. Defaults to ``False``.
            dpi (int): Figure DPI. Defaults to ``150``.
            check_additivity (bool): Toggle SHAP's additivity check.
                Defaults to ``False``.
            overwrite_classifier_explanation (bool): Overwrite the
                bundled ``ClassifierExplanation_*.pkl``. Defaults to
                ``False``.
            overwrite (bool | None): Overwrite cached explanation
                artefacts. ``None`` inherits from ``self.overwrite``.
                Defaults to ``None``.
            n_jobs (int | None): Parallel workers. ``None`` defers to
                ``ForecastModel.explain``'s own default. Defaults to
                ``None``.
            use_cache (bool): Short-circuit re-explain on a cache hit.
                Defaults to ``True``.
            verbose (bool | None): Verbose logging. ``None`` inherits
                from ``self.verbose``. Defaults to ``None``.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        shared_explanation_params = {
            "model": model,
            "eruption_dates": self.eruption_dates,
            "save_per_seed": save_per_seed,
            "plot_per_seed": plot_per_seed,
            "plot_aggregate": plot_aggregate,
            "figsize": figsize,
            "max_display": max_display,
            "group_remaining_features": group_remaining_features,
            "dpi": dpi,
            "check_additivity": check_additivity,
            "overwrite_classifier_explanation": overwrite_classifier_explanation,
            "overwrite": overwrite if overwrite is not None else self.overwrite,
            "n_jobs": n_jobs,
            "use_cache": use_cache,
            "verbose": verbose if verbose is not None else self.verbose,
        }

        self.shared_explanation_params = shared_explanation_params

        return self

    def run(
        self,
        evaluate_model: bool = True,
        explain_model: bool = True,
        notification_message: str | None = None,
    ) -> None:
        """Execute every scenario in sequence.

        Runs :meth:`validate` (which enforces preconditions and creates
        ``self.scenarios_dir``), writes the ``scenarios.json`` manifest,
        then for each scenario in ``self.scenarios``:

        1. Builds the per-scenario output directory
           ``{scenarios_dir}/{slugify(name)}/``.
        2. Calls :meth:`ForecastModel.train` with the per-scenario
           training window and ``self.shared_training_params``.
        3. Calls :meth:`ForecastModel.predict` with the per-scenario
           prediction window and ``self.shared_prediction_params``
           merged with :meth:`build_plot_kwargs`. When the caller
           requested ``use_features_from="files"`` the first scenario
           is coerced to ``"all"`` and its resulting features matrix +
           labels CSV are captured for reuse by every later scenario.
        4. Ships the forecast plot to Telegram via
           :class:`TelegramNotification` when the plot file is present.
        5. Calls :meth:`ForecastModel.evaluate` with
           ``self.shared_evaluation_params`` when ``evaluate_model=True``.
        6. Calls :meth:`ForecastModel.explain` with
           ``self.shared_explanation_params`` when ``explain_model=True``.

        Conditional preconditions: when ``evaluate_model=True``,
        :meth:`shared_evaluation` must have been called; when
        ``explain_model=True``, :meth:`shared_explanation` must have
        been called. Either flag set to ``False`` waives the matching
        precondition, so a sweep that opts out of a stage does not have
        to configure it. These two gates fire before :meth:`validate`
        runs; the always-required preconditions
        (:meth:`calculate` / :meth:`shared_training` /
        :meth:`shared_prediction`) live inside :meth:`validate`.

        Note:
            When the caller passed ``use_features_from="files"`` to
            :meth:`shared_prediction`, the first scenario's prediction
            window must span every datetime any later scenario predicts
            over. See the Note on :meth:`shared_prediction` for the full
            caveat.

        Args:
            evaluate_model (bool): Run the per-scenario evaluate step.
                When ``True``, :meth:`shared_evaluation` must already
                have been called; :meth:`run` raises ``RuntimeError``
                otherwise. When ``False``, the evaluate call is skipped
                for this sweep and :meth:`shared_evaluation` does not
                need to have been configured. Defaults to ``True``.
            explain_model (bool): Run the per-scenario explain step.
                When ``True``, :meth:`shared_explanation` must already
                have been called; :meth:`run` raises ``RuntimeError``
                otherwise. When ``False``, the explain call is skipped
                for this sweep and :meth:`shared_explanation` does not
                need to have been configured. Defaults to ``True``.
            notification_message (str | None): Caption prefix for the
                per-scenario Telegram forecast plot. ``None`` falls back
                to :func:`socket.gethostname`. Defaults to ``None``.

        Returns:
            None

        Example:
            >>> (
            ...     fms.calculate(...)
            ...        .shared_training(...)
            ...        .shared_prediction(...)
            ...        .shared_evaluation(...)
            ...        .shared_explanation(...)
            ... )
            >>> fms.run()
        """
        if evaluate_model and not self.shared_evaluation_params:
            raise RuntimeError(
                "Please run `shared_evaluation(...)` first, "
                "or pass `evaluate_model=False` to `run(...)`."
            )

        if explain_model and not self.shared_explanation_params:
            raise RuntimeError(
                "Please run `shared_explanation(...)` first, "
                "or pass `explain_model=False` to `run(...)`."
            )

        self.validate()._save_scenarios()

        fm = self.ForecastModel
        use_features_from = self.shared_prediction_params["use_features_from"]
        features_matrix_path = None
        label_features_csv = None

        for index, scenario in enumerate(self.scenarios):
            name = scenario["name"]
            description = scenario["description"]
            plot_kwargs = ForecastModelScenario.build_plot_kwargs(
                scenario, self.eruption_dates
            )

            logger.info(f"Running {name}: {description}")

            output_dir = os.path.join(self.scenarios_dir, slugify(name))

            fm.prefix_config = slugify(name)

            fm.train(
                start_date=scenario["train_start_date"],
                end_date=scenario["train_end_date"],
                output_dir=output_dir,
                **self.shared_training_params,
            )

            if index == 0:
                self.shared_prediction_params["use_features_from"] = "all"
            else:
                self.shared_prediction_params["use_features_from"] = use_features_from

            fm.predict(
                start_date=scenario["prediction_start_date"],
                end_date=scenario["prediction_end_date"],
                window_step=scenario["prediction_window_step"],
                window_step_unit=scenario["prediction_window_step_unit"],
                features_matrix_path=features_matrix_path,
                label_features_csv=label_features_csv,
                output_dir=output_dir,
                **self.shared_prediction_params,
                **plot_kwargs,
            )

            #  Capture the first scenario's features matrix + labels CSV so
            #  every later scenario reuses them (compare against the original
            #  ``use_features_from`` captured above, not the dict entry that
            #  was force-set to ``"all"`` for index 0 a few lines up).
            if (
                use_features_from == "files"
                and index == 0
                and features_matrix_path is None
                and label_features_csv is None
            ):
                features_matrix_path = (
                    None if not fm.PredictionModel else fm.PredictionModel.features_path
                )
                label_features_csv = (
                    None if not fm.PredictionModel else fm.PredictionModel.labels_csv
                )

            if fm.PredictionModel and fm.PredictionModel.forecast_plot_path:
                caption = (
                    socket.gethostname()
                    if notification_message is None
                    else notification_message
                )
                tn = TelegramNotification(verbose=True)
                tn.send_document(
                    fm.PredictionModel.forecast_plot_path,
                    caption=f"[{caption.upper()}] {name}: {description}",
                )

            if evaluate_model:
                fm.evaluate(
                    output_dir=output_dir,
                    **self.shared_evaluation_params,
                )

            if explain_model:
                fm.explain(
                    output_dir=output_dir,
                    **self.shared_explanation_params,
                )

        return None

    def _save_scenarios(self) -> Self:
        """Persist the scenarios list to ``{scenarios_dir}/scenarios.json``.

        Written once by :meth:`run` after :meth:`validate` succeeds so
        every downstream artefact has an auditable manifest of the sweep
        parameters.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        filepath = os.path.join(self.scenarios_dir, "scenarios.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.scenarios, f, indent=2, default=str)

        return self
