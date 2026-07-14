import os
from typing import Any, Self, Literal
from datetime import datetime, timedelta

import pandas as pd

from eruption_forecast.plots import plot_forecast
from eruption_forecast.logger import logger
from eruption_forecast.utils.window import construct_windows
from eruption_forecast.utils.dataframe import load_select_features
from eruption_forecast.utils.pathutils import ensure_dir, save_figure
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.utils.date_utils import to_datetime_index
from eruption_forecast.utils.formatting import slugify
from eruption_forecast.utils.validation import check_sampling_consistency
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.config.prediction_config import PredictionConfig
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class PredictionModel(BaseModel):
    """Forecast volcanic eruption probabilities from a trained ensemble.

    Loads a trained ensemble produced by ``TrainingModel.fit()`` and runs
    forecast inference over an unlabelled window grid. Builds the grid via
    ``build_label()``, extracts tsfresh features via ``extract_features()``,
    then aggregates per-seed and per-classifier probabilities into a consensus
    forecast via ``forecast()``. Use :meth:`forecast` for the full forecast +
    plotting flow.

    Args:
        model (str | ClassifierEnsemble | SeedEnsemble): Trained model source.
            Accepted forms:

            - ``ClassifierEnsemble`` object — used directly.
            - ``SeedEnsemble`` object — wrapped into a ``ClassifierEnsemble``.
            - Path to ``ClassifierEnsemble.json`` (from ``TrainingModel``).
            - Path to ``ClassifierEnsemble.pkl`` or ``SeedEnsemble_*.pkl``.
            - Path to a trained-model registry ``*.json`` (new, written by
              :func:`~eruption_forecast.utils.ml.save_model_json`; one value
              from ``TrainingModel.results``) or legacy ``*.csv``.
        tremor_data (str | pd.DataFrame): Path to a tremor CSV file or a
            pre-loaded tremor DataFrame.
        start_date (str | datetime): Inclusive start of the forecast period.
        end_date (str | datetime): Inclusive end of the forecast period.
        window_size (int, optional): Sliding window size in days. Defaults to
            ``2``.
        overwrite (bool, optional): Overwrite existing output files when
            ``True``. Defaults to ``False``.
        output_dir (str | None, optional): Root output directory. Resolved
            relative to ``root_dir`` when omitted. Defaults to ``None``.
        root_dir (str | None, optional): Project root directory used to resolve
            ``output_dir``. Defaults to ``None``.
        prefix_config (str | None, optional): Discriminator slugified into the
            ``save_config()`` filename, inserted before ``.config`` (e.g.
            ``"scenario 1"`` → ``prediction.scenario-1.config.yaml``). ``None``
            keeps the default filename. Defaults to ``None``.
        n_jobs (int, optional): Number of parallel workers. Defaults to ``1``.
        verbose (bool, optional): Emit verbose log messages when ``True``.
            Defaults to ``False``.

    Attributes:
        ClassifierEnsemble (ClassifierEnsemble): Trained ensemble used for
            inference. Always populated after construction.
        kind (Literal["training", "prediction"]): Stage marker consumed by
            ``EvaluationModel`` to dispatch on reuse type. Always
            ``"prediction"``.
        basename (str): ``"{start_date_str}_{end_date_str}"`` — shared
            filename stem for label, feature, plot, and result artefacts.
        prediction_dir (str): Root prediction output directory.
        features_dir (str): Forecast-grid features directory.
        result_dir (str): Per-seed probability output directory.
        labels (pd.Series): Forecast-window index. ``pd.DatetimeIndex`` with
            integer ``id`` values; carries no truth labels. Empty until
            ``build_label()`` is called. May be re-narrowed by
            ``extract_features()`` to drop ids whose tremor windows failed
            ``minimum_completion``.
        labels_csv (str | None): Path to the forecast-grid CSV written by
            ``build_label()`` (and overwritten by ``extract_features()`` when
            the grid is narrowed). ``None`` until ``build_label()`` is called.
        forecast_plot_path (str): Path to the saved forecast plot (PDF when
            ``plot_pdf=True``, otherwise PNG). Set by ``forecast()``.
        results (pd.DataFrame): Forecast results indexed by datetime, with
            one ``{classifier}_{probability|uncertainty|prediction|confidence}``
            column per registered classifier plus the four ``consensus_*``
            columns. Empty until ``forecast()`` is called.

    Example:
        >>> prediction = (
        ...     PredictionModel(
        ...         model="output/.../ClassifierEnsemble.pkl",
        ...         tremor_data="output/.../tremor.csv",
        ...         start_date="2025-03-16",
        ...         end_date="2025-03-22",
        ...     )
        ...     .build_label(window_step=10, window_step_unit="minutes")
        ...     .extract_features()
        ... )
        >>> df_forecast = prediction.forecast()
    """

    def __init__(
        self,
        model: str | ClassifierEnsemble | SeedEnsemble,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int = 2,
        nslc: str | None = None,
        training_hash: str | None = None,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        prefix_config: str | None = None,
        n_jobs: int = 1,
        save_model: bool = False,
        verbose: bool = False,
    ):
        """Initialise the prediction model.

        Resolves the trained ensemble from ``model``, delegates shared
        configuration to ``BaseModel.__init__``, computes the output
        subdirectories, and initialises empty placeholders for the forecast
        window grid. Heavy work (CSV reads, window construction, feature
        extraction, inference) is deferred to the corresponding method call.

        Args:
            model (str | ClassifierEnsemble | SeedEnsemble): Trained model
                source. See the class docstring for accepted forms.
            tremor_data (str | pd.DataFrame): Path to a tremor CSV file or a
                pre-loaded tremor DataFrame.
            start_date (str | datetime): Inclusive start of the forecast period.
            end_date (str | datetime): Inclusive end of the forecast period.
            window_size (int, optional): Sliding window size in days. Defaults
                to ``2``.
            overwrite (bool, optional): Overwrite existing output files when
                ``True``. Defaults to ``False``.
            output_dir (str | None, optional): Output directory path. Defaults
                to ``None``.
            root_dir (str | None, optional): Project root directory. Defaults
                to ``None``.
            prefix_config (str | None, optional): Discriminator slugified into
                the ``save_config()`` filename, inserted before ``.config``.
                Defaults to ``None``.
            n_jobs (int, optional): Number of parallel workers. Defaults to ``1``.
            verbose (bool, optional): Emit verbose log messages when ``True``.
                Defaults to ``False``.
        """
        self._config: PredictionConfig = self._init_config(
            model=model,
            tremor_data=tremor_data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            nslc=nslc,
            training_hash=training_hash,
            overwrite=overwrite,
            output_dir=output_dir,
            root_dir=root_dir,
            prefix_config=prefix_config,
            n_jobs=n_jobs,
            save_model=save_model,
            verbose=verbose,
        )

        super().__init__(
            tremor_data=tremor_data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            eruption_dates=None,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            save_model=save_model,
            verbose=verbose,
        )

        self.ClassifierEnsemble: ClassifierEnsemble = (
            model
            if isinstance(model, ClassifierEnsemble)
            else ClassifierEnsemble.from_any(model, verbose)
        )

        self.kind: Literal["training", "prediction"] = "prediction"
        self.nslc: str | None = nslc
        self.training_hash: str | None = training_hash
        self.overwrite = overwrite
        self.prefix_config: str | None = prefix_config
        self.basename = (
            f"{self.start_date_str}_{self.end_date_str}_ws-{self.window_size}"
        )

        # Captured from extract_features() kwargs so forecast() can rebuild
        # the cache identity at save time without the caller passing them
        # again. ``None`` until extract_features() has run.
        self._extract_features_kwargs: dict | None = None

        (
            self.prediction_dir,
            self.features_dir,
            self.result_dir,
        ) = self.set_directories()

        # ``self.labels`` exposes the ``id`` Series for downstream callers;
        # ``self._labels`` keeps the full forecast grid (``id`` + ``is_erupted``
        # all-zero) so prediction labels match the training schema produced by
        # ``construct_windows``.
        self.labels: pd.Series = pd.Series()
        self._labels: pd.DataFrame = pd.DataFrame()
        self.labels_csv: str | None = None

        # Will be set after forecast() called
        self.forecast_plot_path: str | None = None
        self.results: pd.DataFrame = pd.DataFrame()

    @property
    def stage_dir(self) -> str:
        """Stage directory where the prediction cache artefact is written.

        Returns:
            str: ``self.prediction_dir`` — the ``prediction/`` subtree under
                ``output_dir`` that already hosts the forecast outputs.
        """
        return self.prediction_dir

    @classmethod
    def build_identity(  # ty:ignore[invalid-method-override]
        cls,
        *,
        nslc: str | None,
        tremor_df: pd.DataFrame,
        training_hash: str | None,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int,
        build_label_params: dict,
        extract_features_params: dict,
    ) -> dict:
        """Return the canonical identity dict that defines this prediction cache entry.

        Bundles the station-channel id, the tremor data fingerprint, the
        upstream training-model hash, and all prediction-specific params into
        a canonical dict. Including ``training_hash`` ensures the prediction
        cache invalidates automatically whenever the trained ensemble it
        depends on changes — even when the prediction inputs themselves are
        unchanged.

        Args:
            nslc (str | None): ``Network.Station.Location.Channel`` identifier,
                or ``None`` for standalone runs.
            tremor_df (pd.DataFrame): The tremor DataFrame used for the
                forecast. Reduced to a fingerprint via
                :meth:`BaseModel.tremor_fingerprint`.
            training_hash (str | None): Hash of the upstream
                :class:`TrainingModel` identity that produced the loaded
                ensemble. ``None`` only when the ensemble was loaded from a
                source outside the current pipeline run.
            start_date (str | datetime): Forecast period start.
            end_date (str | datetime): Forecast period end.
            window_size (int): Sliding window size in days.
            build_label_params (dict): Kwargs passed to ``build_label()``.
            extract_features_params (dict): Kwargs passed to
                ``extract_features()`` excluding ``overwrite``, ``n_jobs``,
                ``verbose``.

        Returns:
            dict: Canonical identity dict ready for hashing.
        """
        return {
            "class": cls.__name__,
            "nslc": nslc,
            "training_hash": training_hash,
            "tremor": cls.tremor_fingerprint(tremor_df),
            "constructor": {
                "start_date": start_date,
                "end_date": end_date,
                "window_size": window_size,
            },
            "build_label": build_label_params,
            "extract_features": extract_features_params,
        }

    def set_directories(self) -> tuple[str, str, str]:
        """Build and return the prediction output directory paths.

        Computes stage-specific paths from ``self.output_dir`` without
        materialising them on disk. Used by ``create_directories()`` to lay
        out the prediction tree.

        Returns:
            tuple[str, str, str]: A three-element tuple
                ``(prediction_dir, features_dir, result_dir)`` rooted under
                ``self.output_dir``.
        """
        prediction_dir = os.path.join(self.output_dir, "prediction")
        features_dir = os.path.join(prediction_dir, "features")
        result_dir = os.path.join(prediction_dir, "results")

        return prediction_dir, features_dir, result_dir

    def create_directories(self) -> None:
        """Create the prediction output directories on disk.

        Materialises ``prediction_dir``, ``features_dir``, and ``result_dir``
        so downstream steps can write into them without additional setup.
        """
        ensure_dir(self.prediction_dir)
        ensure_dir(self.features_dir)
        ensure_dir(self.result_dir)

    def validate(self) -> Self:
        """No-op validation hook required by :class:`BaseModel`.

        All required checks are already enforced by ``BaseModel._validate()``
        (date ordering, ``n_jobs`` clamping). The trained ensemble is
        resolved during ``__init__``, so no additional stage-specific
        validation is needed.

        Returns:
            Self: The current instance, for method chaining.
        """
        return self

    def describe(self) -> str:
        """Return a one-line prose summary of the prediction configuration.

        Builds a compact human-readable string covering the forecast period,
        sliding window size, and registered classifiers. Suitable for CLI
        output and human-facing reports. For raw data use ``to_dict()``; for
        an MCP/LLM-friendly block use ``to_prompt()``.

        Returns:
            str: One-line description such as
                ``"PredictionModel(period=2025-03-16 → 2025-03-22,
                window_size=2d, classifiers=[RandomForestClassifier, XGBClassifier])"``.

        Example:
            >>> print(prediction.describe())
            PredictionModel(period=2025-03-16 → 2025-03-22, window_size=2d, classifiers=[RandomForestClassifier, XGBClassifier])
        """
        classifier_names = ", ".join(self.ClassifierEnsemble.classifiers)
        return (
            f"PredictionModel("
            f"period={self.start_date_str} → {self.end_date_str}, "
            f"window_size={self.window_size}d, "
            f"classifiers=[{classifier_names}]"
            f")"
        )

    def to_dict(self) -> dict:
        """Serialise core prediction parameters to a JSON-friendly dictionary.

        Produces a stable schema for MCP tool responses and downstream LLM
        consumption. Always includes the construction-time configuration
        (``start_date``, ``end_date``, ``window_size``, ``classifiers``,
        ``overwrite``, ``output_dir``, ``root_dir``, ``n_jobs``, ``verbose``,
        and ``basename``). Stage-specific keys (``window_step``,
        ``window_step_unit``, ``features_path``, ``forecast_plot_path``) are
        added once the corresponding method has been called.

        Returns:
            dict: Mapping of parameter name to its serialised value.

        Example:
            >>> d = prediction.to_dict()
            >>> d["window_size"]
            2
            >>> "classifiers" in d
            True
        """
        result: dict = {
            "model": self._config.model,
            "tremor_data": self._config.tremor_data,
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "window_size": self.window_size,
            "classifiers": self.ClassifierEnsemble.classifiers,
            "overwrite": self.overwrite,
            "output_dir": self.output_dir,
            "root_dir": self.root_dir,
            "n_jobs": self.n_jobs,
            "verbose": self.verbose,
            "basename": self.basename,
        }

        if self.window_step is not None:
            result["window_step"] = self.window_step

        if self.window_step_unit is not None:
            result["window_step_unit"] = self.window_step_unit

        if self.features_path is not None:
            result["features_path"] = self.features_path

        if self.forecast_plot_path is not None:
            result["forecast_plot_path"] = self.forecast_plot_path

        return result

    def to_prompt(self) -> str:
        """Return a structured bullet-list block suitable for LLM and MCP prompts.

        Builds a deterministic, template-driven prose block from
        :meth:`to_dict`. Paths are reduced to filenames only to avoid leaking
        absolute system paths into prompts. For human-readable prose use
        :meth:`describe`. For raw data use :meth:`to_dict`.

        Returns:
            str: Prompt-ready block ending with the build state (window grid,
                feature extraction, and forecast plot).

        Example:
            >>> for part in prediction.to_prompt().split(". "):
            ...     print(part)
            Forecast period: 2025-03-16 to 2025-03-22
            Window size: 2 day(s)
            Classifiers: RandomForestClassifier, XGBClassifier
            Window step: 10 minutes
            Features: features-matrix_2025-03-16_2025-03-22_step-10-minutes.parquet
            Forecast plot: forecast_2025-03-16_2025-03-22.pdf
            Overwrite: False
            Output dir: output/VG.OJN.00.EHZ
            Root dir: None
            n_jobs: 4
            Verbose: False
            Basename: 2025-03-16_2025-03-22.
        """
        classifier_names = ", ".join(self.ClassifierEnsemble.classifiers)

        window_step_str = (
            f"{self.window_step} {self.window_step_unit}"
            if self.window_step is not None and self.window_step_unit is not None
            else "not built"
        )
        features_path_str = (
            os.path.basename(self.features_path)
            if self.features_path is not None
            else "not extracted"
        )
        forecast_plot_str = (
            os.path.basename(self.forecast_plot_path)
            if self.forecast_plot_path is not None
            else "not rendered"
        )

        return (
            f"Forecast period: {self.start_date_str} to {self.end_date_str}. "
            f"Window size: {self.window_size} day(s). "
            f"Classifiers: {classifier_names}. "
            f"Window step: {window_step_str}. "
            f"Features: {features_path_str}. "
            f"Forecast plot: {forecast_plot_str}. "
            f"Overwrite: {self.overwrite}. "
            f"Output dir: {self.output_dir}. "
            f"Root dir: {self.root_dir}. "
            f"n_jobs: {self.n_jobs}. "
            f"Verbose: {self.verbose}. "
            f"Basename: {self.basename}."
        )

    def save_config(
        self,
        path: str | None = None,
        fmt: Literal["yaml", "json"] = "yaml",
    ) -> str:
        """Persist the captured ``PredictionModel`` init configuration to disk.

        Writes the parameter snapshot captured during ``__init__`` so a
        standalone prediction run can save its constructor surface without
        going through :class:`~eruption_forecast.model.forecast_model.ForecastModel`.

        Args:
            path (str | None): Destination file path. ``None`` resolves to
                ``{prediction_dir}/prediction.config.{fmt}`` so the config
                sits next to the artefacts produced by ``forecast()``. When
                ``self.prefix_config`` is set, its slugified form is inserted
                before ``.config`` — e.g.
                ``prediction.scenario-1.config.yaml`` — so multiple scenarios
                sharing the same ``prediction_dir`` do not overwrite each
                other. Defaults to ``None``.
            fmt (Literal["yaml", "json"]): Output format. Defaults to
                ``"yaml"``.

        Returns:
            str: The absolute path the configuration was written to.

        Example:
            >>> path = prediction.save_config()
            >>> path  # doctest: +SKIP
            'output/VG.OJN.00.EHZ/prediction/prediction.config.yaml'
        """
        if path is None:
            suffix = (
                f".{slug}"
                if self.prefix_config and (slug := slugify(self.prefix_config))
                else ""
            )
            path = os.path.join(self.prediction_dir, f"prediction{suffix}.config.{fmt}")
        return self._config.save(path, fmt)

    def build_label(
        self, window_step: int, window_step_unit: Literal["minutes", "hours"]
    ) -> Self:
        """Build the unlabelled forecast window grid.

        Constructs evenly spaced windows between ``start_date`` and
        ``end_date`` via ``construct_windows()`` and assigns each row an
        integer ``id``. The result is cached as a CSV under ``features_dir``
        and stored on ``self.labels`` as a ``pd.Series`` with a
        ``pd.DatetimeIndex``. When the CSV already exists and ``overwrite``
        is ``False``, the cached grid is loaded instead of rebuilt.

        Args:
            window_step (int): Step size between consecutive windows. Must be
                strictly greater than ``0``.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.

        Returns:
            Self: The current instance, for method chaining.

        Raises:
            ValueError: If ``window_step`` is not strictly greater than ``0``.
            ValueError: If a cached CSV is loaded but lacks an ``id`` column.
            ValueError: If the constructed window grid is empty (typically
                caused by an inverted date range or a step larger than the
                forecast period).
        """
        if window_step <= 0:
            raise ValueError("window_step must be > 0.")

        self.window_step = window_step
        self.window_step_unit = window_step_unit

        filename = (
            f"features-label_{self.basename}_step-{window_step}-{window_step_unit}"
        )
        label_csv = os.path.join(self.features_dir, f"{filename}.csv")
        ensure_dir(self.features_dir)

        if os.path.exists(label_csv) and not self.overwrite:
            label_df = pd.read_csv(label_csv, index_col=0, parse_dates=True)

            # Backward compat: legacy cached forecast grids predate the
            # ``is_erupted`` placeholder column.
            if "is_erupted" not in label_df.columns:
                label_df["is_erupted"] = 0

            self._labels = label_df
            self.labels = label_df["id"]
            self.labels_csv = label_csv

            return self

        label_df = construct_windows(
            start_date=self.start_date,
            end_date=self.end_date,
            window_step=window_step,
            window_step_unit=window_step_unit,
        )

        if label_df.empty:
            raise ValueError(
                f"Labels is empty. Check your start date {self.start_date} "
                f"and end date {self.end_date}, window step {window_step} and "
                f"window step unit {window_step_unit}."
            )

        label_df.to_csv(label_csv, index=True)

        # Label with ``is_erupted`` values are 0.
        self._labels = label_df
        self.labels = label_df["id"]
        self.labels_csv = label_csv

        if self.verbose:
            logger.info(f"Label for prediction: {label_csv}")

        return self

    def extract_features(
        self,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = False,
        exclude_features: list[str] | None = None,
        select_features: str | list[str] | None = None,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Extract tsfresh features over the forecast window grid.

        Builds the windowed tremor matrix aligned to the labels produced by
        :meth:`build_label`, then runs ``FeaturesBuilder`` in prediction mode
        (no relevance filtering). Populates ``self.features_df`` and
        ``self.features_path``. When features were already extracted on this
        instance, the method early-returns without re-running tsfresh.

        Passing a curated ``select_features`` list narrows tsfresh to the
        supplied feature names via ``kind_to_fc_parameters``, so extraction
        computes only those features per tremor column instead of the full
        ``ComprehensiveFCParameters`` set — the standard fast path for
        forecasting with a previously-trained ensemble.

        When ``TremorMatrixBuilder`` drops windows that fail
        ``minimum_completion``, the surviving feature ids become a strict
        subset of the label grid built by :meth:`build_label`. The method
        detects the mismatch and narrows ``self._labels`` / ``self.labels``
        to the surviving ids, rewriting ``self.labels_csv`` on disk so the
        invariant *"``self._labels`` and ``self.features_df`` share length
        and id space"* holds for every downstream consumer.

        Args:
            select_tremor_columns (list[str] | None, optional): Tremor column
                names to include. All columns are used when ``None``. Defaults
                to ``None``.
            save_tremor_matrix_per_method (bool, optional): Save per-column
                tremor matrices under ``per_method/`` when ``True``. Defaults
                to ``False``.
            exclude_features (list[str] | None, optional): tsfresh feature
                names to drop after extraction. Defaults to ``None``.
            select_features (str | list[str] | None, optional): Curated
                feature list used to narrow tsfresh extraction. Accepts either
                a path to a ``top_{N}_features.csv`` / ``top_features.csv`` /
                ``significant_features.csv`` (resolved via
                :func:`load_select_features`) or an explicit ``list[str]`` of
                fully-qualified tsfresh feature names (e.g.
                ``"rsam_f2__mean"``). ``None`` runs the full
                ``ComprehensiveFCParameters`` extraction. Typically populated
                by :meth:`ForecastModel.predict` with the union of features
                selected during training. Defaults to ``None``.
            overwrite (bool, optional): Overwrite existing matrix and feature
                files when ``True``. Defaults to ``False``.
            n_jobs (int | None, optional): Worker count for tsfresh extraction.
                Falls back to ``self.n_jobs`` when ``None``. Defaults to
                ``None``.
            verbose (bool | None, optional): Override ``self.verbose`` for
                this call only. Defaults to ``None``.

        Returns:
            Self: The current instance, for method chaining.

        Raises:
            ValueError: If ``build_label()`` has not been called first.
        """
        resolved_select_features = (
            load_select_features(select_features, number_of_features=0)
            if select_features is not None
            else None
        )

        new_kwargs = {
            "select_tremor_columns": select_tremor_columns,
            "save_tremor_matrix_per_method": save_tremor_matrix_per_method,
            "exclude_features": exclude_features,
            "select_features": resolved_select_features,
        }

        features_already_populated = (
            not self.features_df.empty and self.features_path is not None
        )

        if features_already_populated and not overwrite:
            if (
                self._extract_features_kwargs is not None
                and new_kwargs != self._extract_features_kwargs
            ):
                logger.warning(
                    "extract_features: incoming kwargs differ from the "
                    "previously-extracted matrix; keeping the existing matrix "
                    "and its snapshot. Pass overwrite=True to force a rebuild. "
                    f"incoming={new_kwargs} snapshot={self._extract_features_kwargs}"
                )
            if self.verbose:
                logger.info(f"Features already extracted: {self.features_path}")
            return self

        if self._labels.empty:
            raise ValueError("Please run build_label() first.")

        self._extract_features_kwargs = new_kwargs

        features_builder = self._build_features(
            label_df=self._labels,
            output_dir=self.prediction_dir,
            features_dir=self.features_dir,
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=save_tremor_matrix_per_method,
            save_tremor_matrix_per_id=False,
            select_features=resolved_select_features,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.features_df = features_builder.extract_features(
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        self.features_path = features_builder.path

        # ``TremorMatrixBuilder`` drops windows that fail ``minimum_completion``,
        # so narrow the label grid to the surviving feature ids here — this
        # invariant lets every downstream consumer (forecast, evaluation,
        # explanation) treat ``self._labels`` and ``self.features_df`` as
        # aligned without a separate re-check.
        if len(self._labels) != len(self.features_df):
            self._labels = self._labels[self._labels["id"].isin(self.features_df.index)]
            self._labels.to_csv(self.labels_csv, index=True)
            self.labels = self._labels["id"]

        return self

    def forecast(
        self,
        save_seed_result: bool = True,
        plot_threshold: float = 0.5,
        plot_title: str | None = None,
        plot_pdf: bool = True,
        **plot_kwargs,
    ) -> pd.DataFrame:
        """Run forecast inference and render the forecast plot.

        Calls :meth:`ClassifierEnsemble.predict_with_uncertainty` over the
        extracted features, assembles per-classifier and consensus
        probabilities into a single DataFrame indexed by forecast datetime,
        writes the result CSV under ``self.output_dir``, and renders the
        forecast plot via :meth:`_plot_forecast`.

        Args:
            save_seed_result (bool, optional): Persist per-seed probability
                CSVs under ``result_dir/{classifier_name}/`` when ``True``.
                Defaults to ``True``.
            plot_threshold (float, optional): Decision threshold drawn as a
                horizontal reference line on the forecast plot. Defaults to
                ``0.5``.
            plot_title (str | None, optional): Title to render on the forecast
                plot. Defaults to ``None``.
            plot_pdf (bool, optional): Save a PDF copy of the forecast plot
                with embedded TrueType fonts in addition to the PNG. Defaults
                to ``True``.
            **plot_kwargs: Extra keyword arguments forwarded to
                :func:`plot_forecast`.

        Returns:
            pd.DataFrame: Forecast results indexed by datetime with one column
                per ``{classifier}_{probability|uncertainty|prediction|confidence}``
                plus the four ``consensus_*`` columns.

        Raises:
            ValueError: If ``build_label()`` has not been called first.
            ValueError: If ``extract_features()`` has not been called first.
        """
        if self.labels.empty or self._labels.empty or self.labels_csv is None:
            raise ValueError("Please run build_label() first.")

        if self.features_df.empty and self.features_path is None:
            raise ValueError(
                "Features (matrix) dataframe (features_df) is empty. "
                "Please run extract_features() first."
            )

        self.create_directories()

        results = self.ClassifierEnsemble.predict_with_uncertainty(
            X=self.features_df,
            save=save_seed_result,
            output_dir=self.result_dir,
            overwrite=self.overwrite,
            verbose=self.verbose,
        )

        df_forecast = pd.DataFrame(results, index=self.features_df.index)
        csv_path = os.path.join(
            self.result_dir, f"forecast-results_{self.basename}.csv"
        )

        df_forecast = to_datetime_index(self._labels, df_forecast)

        df_forecast.to_csv(csv_path)
        logger.info(f"Predictions saved to: {csv_path}")

        self._plot_forecast(
            df_forecast,
            plot_threshold,
            title=plot_title,
            plot_pdf=plot_pdf,
            **plot_kwargs,
        )

        self.results = df_forecast

        identity = type(self).build_identity(
            nslc=self.nslc,
            tremor_df=self.tremor_df,
            training_hash=self.training_hash,
            start_date=self.start_date,
            end_date=self.end_date,
            window_size=self.window_size,
            build_label_params={
                "window_step": self.window_step,
                "window_step_unit": self.window_step_unit,
            },
            extract_features_params=dict(self._extract_features_kwargs or {}),
        )
        self.id = type(self).compute_hash(identity)

        if self.save_model:
            self.save(identity)

        try:
            self.save_config()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to save prediction config: {exc}")

        return df_forecast

    def load_features(
        self,
        features_matrix_path: str,
        label_features_csv: str,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
    ) -> Self:
        """Reuse a feature matrix written by a previous prediction run.

        Skips tsfresh extraction entirely by reading the persisted
        ``features-matrix_*.parquet`` and ``features-label_*.csv`` produced by
        an earlier :meth:`extract_features` call. Designed for repeat
        forecasting where the windowing and tremor data have not changed — for
        example replaying :meth:`forecast` with a different trained ensemble
        against an existing feature matrix.

        Drops in as a replacement for the
        ``build_label() → extract_features()`` prefix before :meth:`forecast`;
        the loaded label CSV populates both ``self._labels`` (full datetime
        indexed frame with ``id`` + ``is_erupted`` columns) and
        ``self.labels`` (the ``id`` Series) so downstream consumers see the
        same shape they would after ``build_label()``.

        Both paths are required — no auto-resolve fallback — so the user
        always states their intent explicitly and there is no risk of
        silently picking up a stale ``features-*_*`` artefact left in
        ``self.features_dir`` from a previous run.

        On success, writes a plain-text
        ``{features_dir}/features-loaded-from.txt`` marker so the otherwise
        empty ``prediction/features/`` directory records which external
        paths supplied the matrix and label CSV. The marker is overwritten
        on every :meth:`load_features` call so it always reflects the most
        recent load.

        Args:
            features_matrix_path (str): Path to the
                ``features-matrix_*.parquet`` written under
                ``{output_dir}/prediction/features/`` by
                :meth:`extract_features`.
            label_features_csv (str): Path to the matching
                ``features-label_*.csv``.
            window_step (int): Step size between consecutive windows. Must be
                strictly greater than ``0``. Should have the same interval with the
                label_features_csv and features_matrix_path
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``. Should have the same interval with the
                label_features_csv and features_matrix_path

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            FileNotFoundError: If either path does not exist.
            ValueError: If the loaded label CSV's sampling — checked via
                :func:`~eruption_forecast.utils.validation.check_sampling_consistency`
                — does not match the caller-supplied ``window_step`` /
                ``window_step_unit``, or if its ``datetime`` span does not
                fully cover the configured ``[start_date, end_date]``
                forecast range (compared at day granularity).

        Example:
            >>> (
            ...     prediction
            ...     .load_features(
            ...         features_matrix_path="output/.../features-matrix_2025-03-16_2025-03-22_step-10-minutes.parquet",
            ...         label_features_csv="output/.../features-label_2025-03-16_2025-03-22_step-10-minutes.csv",
            ...     )
            ...     .forecast()
            ... )
            >>> # No build_label() / extract_features() needed.
        """
        if not os.path.isfile(features_matrix_path):
            raise FileNotFoundError(
                f"features_matrix_path not found: {features_matrix_path}"
            )
        if not os.path.isfile(label_features_csv):
            raise FileNotFoundError(
                f"label_features_csv not found: {label_features_csv}"
            )

        if self.verbose:
            logger.info(
                f"[Prediction Model]: Load features matrix from: {features_matrix_path}"
                f"[Prediction Model]: Load label features from: {label_features_csv}"
            )

        # Ensure ``self.start_date`` and ``self.end_date`` is in label date range
        label_datetimes = pd.read_csv(
            label_features_csv, usecols=["datetime"], parse_dates=["datetime"]
        )["datetime"]

        # Validate sampling rate
        unit_alias = {"minutes": "min", "hours": "h"}[window_step_unit]
        expected_freq = f"{window_step}{unit_alias}"
        is_consistent, _, inconsistent_data, _ = check_sampling_consistency(
            pd.DataFrame(index=pd.DatetimeIndex(label_datetimes)),
            expected_freq=expected_freq,
            tolerance="0s",
        )
        if not is_consistent:
            raise ValueError(
                f"Sampling-rate mismatch in {label_features_csv!r}: caller "
                f"passed window_step={window_step} {window_step_unit} "
                f"(={expected_freq}) but {len(inconsistent_data)} row(s) in "
                "the loaded label CSV disagree with that step."
            )

        loaded_start: pd.Timestamp = label_datetimes.min()
        loaded_end: pd.Timestamp = label_datetimes.max()
        if (
            self.start_date.date() < loaded_start.date()
            or self.end_date.date() > loaded_end.date()
        ):
            if (
                self.start_date.date()
                == (loaded_start.date() - timedelta(days=self.window_size))
                and self.end_date.date() == loaded_end.date()
            ):
                logger.info(
                    f"Adjusting start_date and end_date: {loaded_start:%Y-%m-%d} -> {loaded_end:%Y-%m-%d}"
                )
                self.start_date = loaded_start.to_pydatetime()
                self.end_date = loaded_end.to_pydatetime()
                self.start_date_str = loaded_start.strftime("%Y-%m-%d")
                self.end_date_str = loaded_end.strftime("%Y-%m-%d")

            else:
                raise ValueError(
                    f"Loaded feature range [{loaded_start:%Y-%m-%d} → "
                    f"{loaded_end:%Y-%m-%d}] does not fully cover the configured "
                    f"forecast range [{self.start_date_str} → {self.end_date_str}]. "
                    f"Re-run extract_features() or point to a features run that "
                    f"spans the requested range."
                )

        label_df = pd.read_csv(label_features_csv, index_col=0, parse_dates=True)

        # Backward compat: legacy cached forecast grids predate the
        # ``is_erupted`` placeholder column. Mirrors ``build_label()``.
        if "is_erupted" not in label_df.columns:
            label_df["is_erupted"] = 0

        self._labels = label_df
        self.labels = label_df["id"]
        self.labels_csv = label_features_csv

        self.features_df = pd.read_parquet(features_matrix_path)
        self.features_path = features_matrix_path
        self.window_step = window_step
        self.window_step_unit = window_step_unit

        if self.verbose:
            logger.info(
                f"[Prediction Model]: Loaded features_df={self.features_df.shape}, "
                f"labels={self.labels.shape}"
            )

        self._write_load_features_note(
            features_matrix_path=features_matrix_path,
            label_features_csv=label_features_csv,
            window_step=window_step,
            window_step_unit=window_step_unit,
        )

        return self

    @staticmethod
    def _init_config(
        *,
        model: str | ClassifierEnsemble | SeedEnsemble,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        window_size: int,
        nslc: str | None,
        training_hash: str | None,
        overwrite: bool,
        output_dir: str | None,
        root_dir: str | None,
        prefix_config: str | None,
        n_jobs: int,
        save_model: bool,
        verbose: bool,
    ) -> PredictionConfig:
        """Snapshot the ``__init__`` surface into a :class:`PredictionConfig`.

        Normalises non-serializable inputs to string handles so the saved
        YAML/JSON round-trips the user's original intent: ``model`` and
        ``tremor_data`` become ``None`` when a live in-memory object was
        passed, ``start_date`` / ``end_date`` are emitted in ISO-8601 form.

        Args:
            model (str | ClassifierEnsemble | SeedEnsemble): Trained model
                source supplied by the caller.
            tremor_data (str | pd.DataFrame): Tremor source supplied by the caller.
            start_date (str | datetime): Forecast period start.
            end_date (str | datetime): Forecast period end.
            window_size (int): Sliding window size in days.
            overwrite (bool): Overwrite cached artefacts.
            output_dir (str | None): Root output directory.
            root_dir (str | None): Project root.
            prefix_config (str | None): Slugified discriminator inserted into
                the ``save_config()`` filename before ``.config``.
            n_jobs (int): Parallel workers.
            verbose (bool): Emit verbose logs.

        Returns:
            PredictionConfig: Snapshot ready for ``save_config()``.
        """
        return PredictionConfig(
            model=model if isinstance(model, str) else None,
            tremor_data=tremor_data if isinstance(tremor_data, str) else None,
            start_date=start_date
            if isinstance(start_date, str)
            else start_date.isoformat(),
            end_date=end_date if isinstance(end_date, str) else end_date.isoformat(),
            window_size=window_size,
            nslc=nslc,
            training_hash=training_hash,
            overwrite=overwrite,
            output_dir=output_dir,
            root_dir=root_dir,
            prefix_config=prefix_config,
            n_jobs=n_jobs,
            save_model=save_model,
            verbose=verbose,
        )

    def _write_load_features_note(
        self,
        features_matrix_path: str,
        label_features_csv: str,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
    ) -> str:
        """Write a plain-text marker recording where load_features sourced its inputs.

        Materialises ``self.features_dir`` if needed, then writes
        ``{features_dir}/features-loaded-from.txt`` (overwriting any prior
        note) so the otherwise empty ``prediction/features/`` directory
        records which external paths supplied the matrix and label CSV.
        Called from :meth:`load_features` after the frames have been
        loaded onto ``self``.

        Args:
            features_matrix_path (str): Absolute path to the loaded
                ``features-matrix_*.parquet``.
            label_features_csv (str): Absolute path to the loaded
                ``features-label_*.csv``.
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit for
                ``window_step``.

        Returns:
            str: Absolute path to the written marker file.
        """
        ensure_dir(self.features_dir)
        note_path = os.path.join(self.features_dir, "features-loaded-from.txt")
        with open(note_path, "w", encoding="utf-8") as f:
            f.write(
                "This prediction/features/ directory holds no features matrix — "
                "PredictionModel.load_features() sourced them from external paths, "
                "so no tsfresh extraction ran for this forecast.\n\n"
                f"Features matrix : {features_matrix_path}\n"
                f"Label CSV       : {label_features_csv}\n"
                f"Window step     : {window_step} {window_step_unit}\n"
                f"Forecast period : {self.start_date_str} to {self.end_date_str}\n"
                f"Written at      : {datetime.now().isoformat(timespec='seconds')}\n"
            )

        if self.verbose:
            logger.info(f"[Prediction Model]: Wrote load-features note: {note_path}")

        return note_path

    def _plot_forecast(
        self,
        df: pd.DataFrame,
        threshold: float,
        title: str | None = None,
        plot_pdf: bool = False,
        **plot_kwargs: Any,
    ) -> None:
        """Render the forecast plot to PNG and optionally PDF.

        Calls :func:`plot_forecast` to produce the figure, saves it as PNG
        under ``prediction_dir/figures/``, and optionally writes a PDF copy
        with ``pdf.fonttype=42`` so text remains selectable in vector
        editors. Stores the resulting artefact path on
        ``self.forecast_plot_path``.

        Args:
            df (pd.DataFrame): Forecast results with one row per forecast
                datetime and the per-classifier plus consensus probability
                columns produced by :meth:`forecast`.
            threshold (float): Decision threshold drawn as a horizontal
                reference line.
            title (str | None, optional): Title rendered on the plot. Defaults
                to ``None``.
            plot_pdf (bool, optional): Save a PDF copy in addition to the PNG
                when ``True``. Defaults to ``False``.
            **plot_kwargs: Extra keyword arguments forwarded to
                :func:`plot_forecast`.
        """
        fig = plot_forecast(
            df=df,
            label_df=self._labels,
            title=title,
            threshold=threshold,
            **plot_kwargs,
        )

        figure_dir = os.path.join(self.prediction_dir, "figures")
        png_path = os.path.join(figure_dir, f"forecast_{self.basename}.png")

        save_figure(
            fig=fig,
            filepath=png_path,
            dpi=300,
            save_as_pdf=plot_pdf,
            pdf_title=f"Eruption Forecast: {self.start_date_str} to {self.end_date_str}",
            verbose=self.verbose,
        )

        self.forecast_plot_path = (
            png_path.replace(".png", ".pdf") if plot_pdf else png_path
        )
