import os
import glob
import json
from typing import Self, Literal
from datetime import datetime
from functools import partial
from collections.abc import Callable

import numpy as np
import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import learning_curve

from eruption_forecast.plots import plot_significant_features
from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import (
    resample,
    grid_search_cv,
    save_model_json,
    get_classifier_models,
    load_features_resampled,
)
from eruption_forecast.utils.dataframe import (
    load_label_csv,
    load_select_features,
    concat_significant_features,
)
from eruption_forecast.utils.pathutils import ensure_dir, generate_features_filepaths
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.utils.date_utils import to_datetime
from eruption_forecast.utils.formatting import slugify
from eruption_forecast.label.label_builder import LabelBuilder
from eruption_forecast.config.training_config import TrainingConfig
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.features.feature_selector import FeatureSelector
from eruption_forecast.label.dynamic_label_builder import DynamicLabelBuilder
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class TrainingModel(BaseModel):
    """Train classifier models across multiple random seeds on tremor feature data.

    Orchestrates the full training pipeline: label building, tremor matrix
    construction, tsfresh feature extraction, per-seed resampling and feature
    selection, and parallel GridSearchCV model fitting. Trained models and a
    per-classifier trained-model JSON registry are written to the configured
    output directory.

    Args:
        tremor_data (str | pd.DataFrame): Path to a tremor CSV file or a
            pre-loaded tremor DataFrame.
        start_date (str | datetime): Start of the training period.
        end_date (str | datetime): End of the training period.
        classifiers (str | list[str]): One or more classifier keys (e.g.
            ``"rf"``, ``["rf", "xgb"]``).
        eruption_dates (list[str]): ISO-format eruption dates used for labelling.
        window_size (int): Look-ahead window in days for eruption forecasting.
            Defaults to 2.
        cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified"]):
            Cross-validation strategy passed to each ``ClassifierModel``.
            Defaults to ``"shuffle-stratified"``.
        cv_splits (int): Number of CV folds. Defaults to 5.
        top_n_features (int): Top-N features retained after feature
            selection. Defaults to 20.
        include_eruption_date (bool): Whether to include the eruption day
            itself as a positive label. Defaults to False.
        output_dir (str | None): Root output directory. Resolved automatically
            when None. Defaults to None.
        root_dir (str | None): Project root used for path resolution. Defaults
            to None.
        overwrite (bool): Re-run and overwrite cached feature and model files.
            Defaults to False.
        prefix_config (str | None): Discriminator slugified into the
            ``save_config()`` filename, inserted before ``.config``
            (e.g. ``"scenario 1"`` → ``training.scenario-1.config.yaml``).
            ``None`` keeps the default filename. Defaults to ``None``.
        n_jobs (int): Number of parallel outer workers for seed-level
            parallelism. Defaults to 1.
        n_grids (int): Parallel workers used inside ``GridSearchCV`` and
            ``FeatureSelector``. Defaults to 1.
        verbose (bool): Emit detailed progress logs. Defaults to False.

    Example:
        >>> model = TrainingModel(
        ...     tremor_data="output/tremor.csv",
        ...     start_date="2025-01-01",
        ...     end_date="2025-12-31",
        ...     classifiers=["rf", "xgb"],
        ...     eruption_dates=["2025-06-15"],
        ...     window_size=2,
        ...     n_jobs=4,
        ... )
        >>> model.build_label(window_step=6, window_step_unit="hours")
        >>> model.extract_features()
        >>> model.fit(seeds=25)
    """

    def __init__(
        self,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        classifiers: str | list[str],
        eruption_dates: list[str],
        window_size: int = 2,
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified"
        ] = "shuffle-stratified",
        cv_splits: int = 5,
        top_n_features: int = 20,
        include_eruption_date: bool = False,
        nslc: str | None = None,
        output_dir: str | None = None,
        root_dir: str | None = None,
        overwrite: bool = False,
        prefix_config: str | None = None,
        n_jobs: int = 1,
        n_grids: int = 1,
        verbose: bool = False,
    ) -> None:
        self._config: TrainingConfig = self._init_config(
            tremor_data=tremor_data,
            start_date=start_date,
            end_date=end_date,
            classifiers=classifiers,
            eruption_dates=eruption_dates,
            window_size=window_size,
            cv_strategy=cv_strategy,
            cv_splits=cv_splits,
            top_n_features=top_n_features,
            include_eruption_date=include_eruption_date,
            nslc=nslc,
            output_dir=output_dir,
            root_dir=root_dir,
            overwrite=overwrite,
            prefix_config=prefix_config,
            n_jobs=n_jobs,
            n_grids=n_grids,
            verbose=verbose,
        )

        # Set properties
        super().__init__(
            tremor_data=tremor_data,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            eruption_dates=eruption_dates,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        classifiers = [classifiers] if isinstance(classifiers, str) else classifiers

        # Set default properties
        self.kind: Literal["training", "prediction"] = "training"
        self.classifiers = classifiers
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.top_n_features = top_n_features
        self.include_eruption_date: bool = include_eruption_date
        self.nslc: str | None = nslc
        self.overwrite: bool = overwrite
        self.prefix_config: str | None = prefix_config
        self.n_grids: int = n_grids

        # Captured from extract_features() / fit() kwargs so fit() can rebuild
        # the cache identity at save time without the caller passing them
        # again. ``None`` until the corresponding method has run.
        self._extract_features_kwargs: dict | None = None
        self._fit_kwargs: dict | None = None
        self.training_hash: str | None = None

        # Set additional properties
        self.classifier_models: list[ClassifierModel] = get_classifier_models(
            classifiers,
            cv_strategy=self.cv_strategy,
            cv_splits=self.cv_splits,
            verbose=verbose,
        )
        self.cv_name = self.classifier_models[0].slug_cv_name
        self.FeatureSelector = FeatureSelector(
            method="tsfresh",
            n_jobs=n_grids,
            verbose=verbose,
        )

        (
            self.training_dir,
            self.classifier_dir,
            self.features_dir,
            self.features_seed_dir,
            self.features_resampled_dir,
            self.figures_seed_dir,
            self.classifier_dirs,
            self.models_dir,
            self.learning_curve_dirs,
        ) = self.set_directories()
        self.features_csvs: list[str] = []
        self.features_selected_df: pd.DataFrame = pd.DataFrame()

        # Will be set after build_label() called
        self.LabelBuilder: LabelBuilder | None = None
        self.labels: pd.Series = pd.Series()
        self.builder: Literal["standard", "dynamic"] = "standard"
        self.days_before_eruption: int | None = None

        # Will be set after fit() called
        self.results: dict[str, str] = {}
        self.results_json: str | None = None
        self.seed_ensembles: dict[str, str] = {}
        self.classifier_ensemble_path: str | None = None
        self._scoring: str = "balanced_accuracy"

        if verbose:
            logger.info("[Training Model]: Starting Prediction...")

        self.validate()

    @staticmethod
    def _init_config(
        *,
        tremor_data: str | pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        classifiers: str | list[str],
        eruption_dates: list[str],
        window_size: int,
        cv_strategy: Literal["shuffle", "stratified", "shuffle-stratified"],
        cv_splits: int,
        top_n_features: int,
        include_eruption_date: bool,
        nslc: str | None,
        output_dir: str | None,
        root_dir: str | None,
        overwrite: bool,
        prefix_config: str | None,
        n_jobs: int,
        n_grids: int,
        verbose: bool,
    ) -> TrainingConfig:
        """Snapshot the ``__init__`` surface into a :class:`TrainingConfig`.

        Normalises non-serializable inputs to string handles so the saved
        YAML/JSON round-trips the user's original intent: ``tremor_data``
        becomes ``None`` when a pre-loaded ``pd.DataFrame`` was passed,
        ``start_date`` / ``end_date`` are emitted in ISO-8601 form.

        Args:
            tremor_data (str | pd.DataFrame): Tremor source supplied by the caller.
            start_date (str | datetime): Training period start.
            end_date (str | datetime): Training period end.
            classifiers (str | list[str]): Classifier key(s) to train.
            eruption_dates (list[str]): Known eruption dates.
            window_size (int): Sliding window size in days.
            cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified"]):
                Cross-validation strategy.
            cv_splits (int): Number of CV folds.
            top_n_features (int): Top-N features retained after selection.
            include_eruption_date (bool): Whether to label the eruption day positive.
            output_dir (str | None): Root output directory.
            root_dir (str | None): Project root.
            overwrite (bool): Overwrite cached artefacts.
            prefix_config (str | None): Slugified discriminator inserted into
                the ``save_config()`` filename before ``.config``.
            n_jobs (int): Outer parallel workers.
            n_grids (int): Inner grid-search workers.
            verbose (bool): Emit verbose logs.

        Returns:
            TrainingConfig: Snapshot ready for ``save_config()``.
        """
        return TrainingConfig(
            tremor_data=tremor_data if isinstance(tremor_data, str) else None,
            start_date=start_date
            if isinstance(start_date, str)
            else start_date.isoformat(),
            end_date=end_date if isinstance(end_date, str) else end_date.isoformat(),
            classifiers=[classifiers] if isinstance(classifiers, str) else classifiers,
            eruption_dates=list(eruption_dates),
            window_size=window_size,
            cv_strategy=cv_strategy,
            cv_splits=cv_splits,
            top_n_features=top_n_features,
            include_eruption_date=include_eruption_date,
            nslc=nslc,
            output_dir=output_dir,
            root_dir=root_dir,
            overwrite=overwrite,
            prefix_config=prefix_config,
            n_jobs=n_jobs,
            n_grids=n_grids,
            verbose=verbose,
        )

    def set_directories(
        self,
    ) -> tuple[
        str, str, str, str, str, str, dict[str, str], dict[str, str], dict[str, str]
    ]:
        """Build and return all output directory paths used during training.

        Creates classifier-level subdirectories immediately so downstream
        steps can write files without additional setup calls.

        Returns:
            tuple: A nine-element tuple containing:
                - training_dir (str): Root training output path.
                - classifier_dir (str): Classifier training output path.
                - features_dir (str): CV-scoped features directory.
                - features_seed_dir (str): Per-seed selected-feature CSVs.
                - features_resampled_dir (str): Per-seed resampled-data CSVs.
                - figures_seed_dir (str): Per-seed feature importance figures.
                - classifier_dirs (dict[str, str]): Classifier-slug → directory.
                - models_dir (dict[str, str]): Classifier-slug → model directory.
                - learning_curve_dirs (dict[str, str]): Classifier-slug →
                  per-seed learning-curve JSON directory.
        """
        training_dir = os.path.join(self.output_dir, "training")
        classifier_dir = os.path.join(training_dir, "classifiers")
        features_dir = os.path.join(training_dir, "features", self.cv_name)
        features_seed_dir = os.path.join(features_dir, "seed")
        features_resampled_dir = os.path.join(features_dir, "resampled")
        figures_seed_dir = os.path.join(features_seed_dir, "figures")

        classifier_dirs: dict[str, str] = {}
        models_dir: dict[str, str] = {}
        learning_curve_dirs: dict[str, str] = {}

        for classifier_model in self.classifier_models:
            classifier_name = classifier_model.name
            classifier_dirs[classifier_name] = os.path.join(
                classifier_dir, classifier_name, self.cv_name
            )
            model_dir = os.path.join(classifier_dirs[classifier_name], "models")
            models_dir[classifier_name] = model_dir
            learning_curve_dirs[classifier_name] = os.path.join(
                classifier_dirs[classifier_name], "learning_curves"
            )

        return (
            training_dir,
            classifier_dir,
            features_dir,
            features_seed_dir,
            features_resampled_dir,
            figures_seed_dir,
            classifier_dirs,
            models_dir,
            learning_curve_dirs,
        )

    def create_directories(
        self,
        plot_features: bool = False,
    ) -> None:
        """Create all output directories required before training begins.

        Args:
            plot_features (bool): Also create the per-seed figures directory
                when True. Defaults to False.

        Example:
            >>> model.create_directories(plot_features=True)
        """
        ensure_dir(self.training_dir)
        ensure_dir(self.features_dir)
        ensure_dir(self.features_seed_dir)
        ensure_dir(self.features_resampled_dir)

        for model_name in self.models_dir.keys():
            ensure_dir(self.models_dir[model_name])

        if plot_features:
            ensure_dir(self.figures_seed_dir)

    def validate(self) -> Self:
        """Validate and reconcile model parameters against system and data constraints.

        Clamps ``n_grids`` so that the product ``n_jobs × n_grids`` never
        exceeds the available CPU count. When the caller leaves both
        ``n_jobs`` and ``n_grids`` at their default of ``1``, ``n_grids`` is
        boosted to ``total_cpu - 2`` so the grid search can use the available
        cores. Creates the root training directory as a side effect. Date
        range clamping against the tremor data bounds is deferred to
        ``_sync_dates_to_tremor()``, which is called lazily from
        ``build_label()`` to avoid loading the tremor CSV during construction.

        Returns:
            Self: The current instance, enabling method chaining.
        """
        # Ensure total grid not over than total CPU
        total_grid = self.n_jobs * self.n_grids
        if total_grid > self.total_cpu:
            self.n_grids = np.clip(self.total_cpu // self.n_jobs, 1, self.total_cpu)

        # Optimize n_grids search to utitlize all available CPU
        if self.n_jobs == 1 and self.n_grids == 1:
            self.n_grids = max(1, self.total_cpu - 2)

        ensure_dir(self.training_dir)

        return self

    def describe(self) -> str:
        """Return a human-readable summary of the training configuration.

        Returns:
            str: Descriptive string for this training model instance.

        Example:
            >>> print(model.describe())
            TrainingModel(period=2025-01-01 → 2025-12-31, window_size=2d, ...)
        """
        classifier_names = ", ".join(m.name for m in self.classifier_models)
        return (
            f"TrainingModel("
            f"period={self.start_date_str} → {self.end_date_str}, "
            f"window_size={self.window_size}d, "
            f"classifiers=[{classifier_names}], "
            f"cv={self.cv_strategy}/{self.cv_splits}-fold, "
            f"top_n_features={self.top_n_features}, "
            f"eruptions={len(self.eruption_dates) if self.eruption_dates is not None else 0}"
            f")"
        )

    def to_dict(self) -> dict:
        """Serialise core training parameters to a plain dictionary.

        Returns:
            dict: Mapping of parameter names to their current values. Always
                includes ``start_date``, ``end_date``, ``classifiers``,
                ``eruption_dates``, ``window_size``, ``cv_strategy``,
                ``cv_splits``, ``top_n_features``,
                ``include_eruption_date``, ``overwrite``, ``output_dir``,
                ``root_dir``, ``n_jobs``, ``n_grids``, and ``verbose``. Also
                includes ``basename`` once ``build_label()`` has been called.

        Example:
            >>> d = model.to_dict()
            >>> d["window_size"]
            2
            >>> "eruption_dates" in d
            True
        """
        result: dict = {
            "start_date": self.start_date_str,
            "end_date": self.end_date_str,
            "classifiers": self.classifiers,
            "eruption_dates": self.eruption_dates,
            "window_size": self.window_size,
            "cv_strategy": self.cv_strategy,
            "cv_splits": self.cv_splits,
            "top_n_features": self.top_n_features,
            "include_eruption_date": self.include_eruption_date,
            "overwrite": self.overwrite,
            "output_dir": self.output_dir,
            "root_dir": self.root_dir,
            "n_jobs": self.n_jobs,
            "n_grids": self.n_grids,
            "verbose": self.verbose,
        }

        if self.basename is not None:
            result["basename"] = self.basename

        return result

    def to_prompt(self) -> str:
        """Return a prompt-ready string representation of the training model.

        Returns:
            str: Prompt string for this training model instance.

        Example:
            >>> prompt = model.to_prompt()
            >>> "Training period" in prompt
            True
        """
        classifier_names = ", ".join(m.name for m in self.classifier_models)
        eruption_list = (
            ", ".join(self.eruption_dates)
            if self.eruption_dates is not None
            else "none"
        )
        basename_str = (
            f" Basename: {self.basename}." if self.basename is not None else ""
        )
        return (
            f"Training period: {self.start_date_str} to {self.end_date_str}. "
            f"Window size: {self.window_size} day(s). "
            f"Classifiers: {classifier_names}. "
            f"CV strategy: {self.cv_strategy} with {self.cv_splits} folds. "
            f"Top features retained: {self.top_n_features}. "
            f"Include eruption date: {self.include_eruption_date}. "
            f"Overwrite: {self.overwrite}. "
            f"Output dir: {self.output_dir}. "
            f"Root dir: {self.root_dir}. "
            f"n_jobs: {self.n_jobs}. "
            f"n_grids: {self.n_grids}. "
            f"Verbose: {self.verbose}. "
            f"Eruption dates: {eruption_list}."
            f"{basename_str}"
        )

    def save_config(
        self,
        path: str | None = None,
        fmt: Literal["yaml", "json"] = "yaml",
    ) -> str:
        """Persist the captured ``TrainingModel`` init configuration to disk.

        Writes the parameter snapshot captured during ``__init__`` so a
        standalone training run can save its constructor surface without
        going through :class:`~eruption_forecast.model.forecast_model.ForecastModel`.

        Args:
            path (str | None): Destination file path. ``None`` resolves to
                ``{training_dir}/training.config.{fmt}`` so the config sits
                next to the artefacts produced by ``fit()``. When
                ``self.prefix_config`` is set, its slugified form is inserted
                before ``.config`` — e.g.
                ``training.scenario-1.config.yaml`` — so multiple scenarios
                sharing the same ``training_dir`` do not overwrite each other.
                Defaults to ``None``.
            fmt (Literal["yaml", "json"]): Output format. Defaults to
                ``"yaml"``.

        Returns:
            str: The absolute path the configuration was written to.

        Example:
            >>> path = model.save_config()
            >>> path  # doctest: +SKIP
            'output/VG.OJN.00.EHZ/training/training.config.yaml'
        """
        if path is None:
            suffix = (
                f".{slug}"
                if self.prefix_config and (slug := slugify(self.prefix_config))
                else ""
            )
            path = os.path.join(self.training_dir, f"training{suffix}.config.{fmt}")
        return self._config.save(path, fmt)

    @property
    def stage_dir(self) -> str:
        """Stage directory where the training cache artefact is written.

        Returns:
            str: ``self.training_dir`` — the ``training/`` subtree under
                ``output_dir`` that already hosts every training artefact.
        """
        return self.training_dir

    @classmethod
    def build_identity(  # ty:ignore[invalid-method-override]
        cls,
        *,
        nslc: str | None,
        tremor_df: pd.DataFrame,
        start_date: str | datetime,
        end_date: str | datetime,
        classifiers: str | list[str],
        eruption_dates: list[str],
        window_size: int,
        cv_strategy: str,
        cv_splits: int,
        scoring: str,
        top_n_features: int,
        include_eruption_date: bool,
        build_label_params: dict,
        extract_features_params: dict,
        fit_params: dict,
    ) -> dict:
        """Return the canonical identity dict that defines this training cache entry.

        Builds the parameter bundle that uniquely identifies a trained
        ``TrainingModel``: the station-channel id, the tremor data
        fingerprint, all constructor params that affect the output, plus the
        params passed to ``build_label()``, ``extract_features()``, and
        ``fit()``. Runtime knobs (``n_jobs``, ``n_grids``, ``verbose``,
        ``overwrite``) are excluded because they do not change the produced
        artefact.

        Args:
            nslc (str | None): ``Network.Station.Location.Channel`` identifier,
                or ``None`` for standalone runs.
            tremor_df (pd.DataFrame): The tremor DataFrame that will be used
                for training. Reduced to a small fingerprint via
                :meth:`BaseModel.tremor_fingerprint`.
            start_date (str | datetime): Training period start.
            end_date (str | datetime): Training period end.
            classifiers (str | list[str]): Classifier keys.
            eruption_dates (list[str]): Eruption dates used for labelling.
            window_size (int): Sliding window size in days.
            cv_strategy (str): Cross-validation strategy name.
            cv_splits (int): Number of CV folds.
            scoring (str): GridSearchCV scoring name.
            top_n_features (int): Top-N features retained.
            include_eruption_date (bool): Whether the eruption day itself is
                labelled positive.
            build_label_params (dict): Kwargs passed to ``build_label()``
                excluding ``verbose``.
            extract_features_params (dict): Kwargs passed to
                ``extract_features()`` excluding ``overwrite``, ``n_jobs``,
                ``verbose``.
            fit_params (dict): Kwargs passed to ``fit()`` excluding
                ``plot_features``.

        Returns:
            dict: Canonical identity dict ready for hashing.
        """
        classifiers_list = (
            [classifiers] if isinstance(classifiers, str) else list(classifiers)
        )

        # Normalize ``select_features`` (CSV path or raw list) into a sorted
        # list so two callers that point at the same curated feature set
        # produce the same cache hash, regardless of how they spelled the
        # input. Done here so callers do not have to resolve the value
        # themselves before building the identity.
        extract_features_params = dict(extract_features_params)
        raw_select_features = extract_features_params.get("select_features")
        if raw_select_features is not None:
            extract_features_params["select_features"] = sorted(
                load_select_features(
                    raw_select_features, number_of_features=top_n_features
                )
            )

        return {
            "class": cls.__name__,
            "nslc": nslc,
            "tremor": cls.tremor_fingerprint(tremor_df),
            "constructor": {
                "start_date": start_date,
                "end_date": end_date,
                "classifiers": sorted(classifiers_list),
                "eruption_dates": sorted(eruption_dates),
                "window_size": window_size,
                "cv_strategy": cv_strategy,
                "cv_splits": cv_splits,
                "scoring": scoring,
                "top_n_features": top_n_features,
                "include_eruption_date": include_eruption_date,
            },
            "build_label": build_label_params,
            "extract_features": extract_features_params,
            "fit": fit_params,
        }

    def build_label(
        self,
        window_step: int,
        window_step_unit: Literal["minutes", "hours"],
        builder: Literal["standard", "dynamic"] = "standard",
        days_before_eruption: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Instantiate and build a label builder of the requested type.

        Constructs either a global sliding-window ``LabelBuilder`` or a
        per-eruption ``DynamicLabelBuilder``, runs its ``build()`` method, and
        stores the result on ``self.LabelBuilder``. Must be called before
        ``extract_features()``.

        Args:
            window_step (int): Step size between consecutive windows.
            window_step_unit (Literal["minutes", "hours"]): Unit of
                ``window_step``.
            builder (Literal["standard", "dynamic"]): Label builder variant.
                ``"standard"`` uses a single global window; ``"dynamic"``
                generates one window per eruption event. Defaults to
                ``"standard"``.
            days_before_eruption (int | None): Days before each eruption to
                start its positive window. Required when
                ``builder="dynamic"``. Defaults to None.
            verbose (bool | None): Override ``self.verbose`` for this call
                only. Defaults to None.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If ``window_step`` is not greater than zero.
            ValueError: If ``builder="dynamic"`` and ``days_before_eruption``
                is None.

        Example:
            >>> model.build_label(window_step=6, window_step_unit="hours")
            >>> model.build_label(
            ...     window_step=10,
            ...     window_step_unit="minutes",
            ...     builder="dynamic",
            ...     days_before_eruption=3,
            ... )
        """
        if window_step <= 0:
            raise ValueError("window_step must be > 0.")

        self.window_step = window_step
        self.window_step_unit = window_step_unit
        self.builder = builder
        self.days_before_eruption = days_before_eruption

        self._sync_dates_to_tremor()
        verbose = verbose if verbose is not None else self.verbose

        filtered_eruption_dates = (
            [
                eruption_date_str
                for eruption_date_str in self.eruption_dates
                if self.start_date <= to_datetime(eruption_date_str) <= self.end_date
            ]
            if self.eruption_dates is not None
            else None
        )

        if (
            self.verbose
            and filtered_eruption_dates is not None
            and self.eruption_dates is not None
            and len(filtered_eruption_dates) != len(self.eruption_dates)
        ):
            logger.info(f"Eruption dates updated to: {filtered_eruption_dates}")

        if builder == "dynamic":
            if days_before_eruption is None:
                raise ValueError(
                    "days_before_eruption is required when builder='dynamic'."
                )

            label_builder = DynamicLabelBuilder(
                days_before_eruption=days_before_eruption,
                window_step=window_step,
                window_step_unit=window_step_unit,
                day_to_forecast=self.window_size,
                eruption_dates=filtered_eruption_dates,  # ty:ignore[invalid-argument-type]
                output_dir=self.training_dir,
                root_dir=self.root_dir,
                verbose=verbose,
            ).build()
        else:
            if days_before_eruption:
                logger.info(
                    "Using standard label builder, ``days_before_eruption`` will be ignored."
                )

            label_builder = LabelBuilder(
                start_date=self.start_date,
                end_date=self.end_date,
                window_step=window_step,
                window_step_unit=window_step_unit,
                day_to_forecast=self.window_size,
                eruption_dates=filtered_eruption_dates,  # ty:ignore[invalid-argument-type]
                output_dir=self.training_dir,
                root_dir=self.root_dir,
                verbose=verbose,
            ).build()

        self.LabelBuilder = label_builder
        self.basename = (
            os.path.basename(label_builder.csv).split("label_")[1].split(".csv")[0]
        )

        return self

    def extract_features(
        self,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = False,
        exclude_features: list[str] | None = None,
        select_features: str | list[str] | None = None,
        save_tremor_matrix_per_id: bool = False,
        minimum_completion: float = 1.0,
        overwrite: bool = False,
        n_jobs: int | None = None,
        verbose: bool | None = None,
    ) -> Self:
        """Build the tremor matrix and extract tsfresh features from it.

        Slices tremor data into label-aligned windows via
        ``TremorMatrixBuilder``, then runs ``FeaturesBuilder`` to extract and
        filter relevant tsfresh features. Stores the result in
        ``self.features_df`` and the aligned labels in
        ``self.labels``. Must be called after ``build_label()``.

        Args:
            select_tremor_columns (list[str] | None): Subset of tremor columns
                to include. Uses all columns when None. Defaults to None.
            save_tremor_matrix_per_method (bool): Write one CSV per tremor
                column under the ``per_method/`` subdirectory. Defaults to
                False.
            exclude_features (list[str] | None): tsfresh feature names to
                drop before saving. Defaults to None.
            select_features (str | list[str] | None): Pre-filter tsfresh to a
                curated set of fully-qualified feature names — accepts either
                the path to a ``top_{N}_features.csv`` (top-N subset) /
                ``top_features.csv`` (full ranked list) /
                ``significant_features.csv`` (raw per-seed rows) written by a
                prior training run, or an explicit ``list[str]`` of feature
                names. When supplied, tsfresh computes only those features per
                tremor column instead of the full ``ComprehensiveFCParameters``
                set; downstream per-seed feature selection in :meth:`fit` is
                unchanged. Defaults to ``None``.
            save_tremor_matrix_per_id (bool): Write one CSV per window ID.
                Defaults to False.
            minimum_completion (float, optional): Minimum data-completeness
                ratio in the range 0.0–1.0. Tremor windows whose sample count
                falls below this fraction of the expected count are skipped
                before feature extraction. Defaults to 1.0 (no gaps
                tolerated).
            overwrite (bool): Re-extract even when cached files exist.
                Defaults to False.
            n_jobs (int | None): Worker count for tsfresh extraction. Falls
                back to ``self.n_jobs`` when None. Defaults to None.
            verbose (bool | None): Override ``self.verbose`` for this call
                only. Defaults to None.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If ``build_label()`` has not been called first.

        Example:
            >>> model.build_label(window_step=6, window_step_unit="hours")
            >>> model.extract_features(select_tremor_columns=["rsam_f2", "entropy"])
            >>> model.features_df.shape
            (n_windows, n_features)
            >>> # Reuse a prior run's top-N feature CSV to short-circuit tsfresh
            >>> model.extract_features(
            ...     select_features="output/.../top_20_features.csv",
            ... )
        """
        new_kwargs = {
            "select_tremor_columns": select_tremor_columns,
            "save_tremor_matrix_per_method": save_tremor_matrix_per_method,
            "exclude_features": exclude_features,
            "select_features": select_features,
            "minimum_completion": minimum_completion,
        }

        features_already_populated = (
            not self.features_df.empty
            and not self.labels.empty
            and self.features_path is not None
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

        if self.LabelBuilder is None:
            raise ValueError("Please run build_label() first.")

        self._extract_features_kwargs = new_kwargs

        resolved_select_features = (
            load_select_features(
                select_features, number_of_features=self.top_n_features
            )
            if select_features is not None
            else None
        )

        features_builder = self._build_features(
            label_df=self.LabelBuilder.df,
            output_dir=self.training_dir,
            features_dir=self.features_dir,
            features_label=self.LabelBuilder.df,
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=save_tremor_matrix_per_method,
            save_tremor_matrix_per_id=save_tremor_matrix_per_id,
            minimum_completion=minimum_completion,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
            select_features=resolved_select_features,
        )

        self.features_df = features_builder.extract_features(
            use_relevant_features=True,
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        self.features_path = features_builder.path

        labels: pd.DataFrame = features_builder.label_df

        if "id" in labels.columns:
            labels = labels.set_index("id")
        if "datetime" in labels.columns:
            labels = labels.drop("datetime", axis=1)

        # Label with ``pd.RangeIndex`` and column ``is_erupted`` only
        self.labels = labels["is_erupted"]

        return self

    def fit(
        self,
        seeds: int = 25,
        resample_method: Literal["under", "over", "auto"] | None = "auto",
        minority_threshold: float = 0.15,
        sampling_strategy: str | float = 0.75,
        plot_features: bool = False,
        scoring: str = "balanced_accuracy",
        compute_learning_curve: bool = False,
    ) -> Self:
        """Train classifier models on the full dataset across multiple random seeds.

        For each seed, resamples the extracted features, selects the top-N
        features, and fits every configured classifier via ``GridSearchCV``.
        Existing feature and model files are reused unless ``overwrite=True``.
        Populates ``self.results`` with the trained-model JSON registry path
        for each classifier (written by
        :func:`~eruption_forecast.utils.ml.save_model_json`) and
        ``self.features_selected_df`` with the aggregated top-N feature
        importance DataFrame.

        Args:
            seeds (int): Number of random seeds to train over. Defaults to 25.
            resample_method (Literal["under", "over", "auto"] | None):
                Resampling strategy applied before feature selection.
                ``"auto"`` chooses ``"under"`` when the minority class share
                is below ``minority_threshold``, otherwise skips resampling.
                Defaults to ``"auto"``.
            minority_threshold (float): Minority-class share threshold used
                when ``resample_method="auto"``. Defaults to 0.15.
            sampling_strategy (str | float): Target class ratio passed to the
                resampler. Defaults to 0.75.
            plot_features (bool): Save per-seed feature importance figures.
                Defaults to False.
            scoring (str, optional): Scoring GridSearchCV. Defaults to ``"balanced_accuracy"``.
                See here: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
            compute_learning_curve (bool): If ``True``, run
                :meth:`compute_learning_curve` after the ensemble is built and
                saved, generating per-seed sklearn learning-curve JSON files
                under ``training/classifiers/{slug}/{cv}/learning_curves/``.
                Defaults to ``False``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If neither ``extract_features()`` nor
                ``load_features()`` has populated ``features_df`` / ``labels``.
            ValueError: If no seed ensembles are produced (every classifier
                yielded zero successful seeds).

        Example:
            >>> model.build_label(window_step=6, window_step_unit="hours")
            >>> model.extract_features()
            >>> model.fit(seeds=25, resample_method="auto", plot_features=True)
            >>> model.results  # {"RandomForestClassifier": "path/to/trained-model__*.json"}
        """
        if self.features_df.empty and self.features_path is None:
            raise ValueError(
                "Features (matrix) dataframe (features_df) is empty. "
                "Please run extract_features() or load_features() first."
            )
        if self.labels.empty:
            raise ValueError(
                "Labels are empty. Please run extract_features() or "
                "load_features() first."
            )

        self.create_directories(plot_features=plot_features)
        self._scoring = scoring

        # Capture the input ``fit`` kwargs (pre-"auto" resolution for
        # ``resample_method``) so the save-time identity dict produces the
        # same cache hash whether the user runs through ``ForecastModel`` or
        # standalone.
        self._fit_kwargs = {
            "seeds": seeds,
            "resample_method": resample_method,
            "minority_threshold": minority_threshold,
            "sampling_strategy": sampling_strategy,
        }

        if resample_method == "auto":
            # Prefer ``LabelBuilder`` when present so the share matches the
            # full labelled window grid; fall back to ``self.labels`` when
            # the user came in through ``load_features()`` and never built a
            # label builder.
            label_series = (
                self.LabelBuilder.df["is_erupted"]
                if self.LabelBuilder is not None
                else self.labels
            )
            minority_share = label_series.value_counts(normalize=True).min()
            if minority_share <= minority_threshold:
                resample_method = "under"
                logger.info(
                    f"resample_method='auto': minority class is {minority_share:.1%} "
                    f"(<{minority_threshold * 100}%) — using 'under' (RandomUnderSampler)."
                )
            else:
                resample_method = None
                logger.info(
                    f"resample_method='auto': minority class is {minority_share:.1%} "
                    f"(>{minority_threshold * 100}%) — skipping resampling."
                )

        random_states: list[int] = list(range(seeds))
        (
            pending_feature_selection_jobs,
            pending_training_model_jobs,
            records_per_classifier,
            existing_feature_paths,
        ) = self._collect_pending_train_jobs(
            random_states=random_states,
            resample_method=resample_method,
            sampling_strategy=sampling_strategy,
            plot_features=plot_features,
        )

        if self.verbose:
            logger.info(
                f"Pending Feature Selection: Found {len(pending_feature_selection_jobs)} job(s)"
            )

        feature_selection_results: list[str | None] = self._run_jobs(
            self._features_selection,
            pending_feature_selection_jobs,
            job_name=f"Running {len(pending_feature_selection_jobs)} Pending Feature Selection",
        )

        new_training_model_jobs: list[tuple] = []

        for feature_selection_result, pending_feature_selection_job in zip(
            feature_selection_results, pending_feature_selection_jobs, strict=True
        ):
            if feature_selection_result is None:
                continue
            _random_state = pending_feature_selection_job[0]
            for classifier_model in self.classifier_models:
                new_training_model_jobs.append((_random_state, classifier_model.name))

        all_training_model_jobs = pending_training_model_jobs + new_training_model_jobs

        if self.verbose:
            logger.info(
                f"Pending Training Model: Found {len(all_training_model_jobs)} job(s)"
            )

        training_model_results: list[str | None] = self._run_jobs(
            self._train,
            all_training_model_jobs,
            job_name=f"Running {len(all_training_model_jobs)} Training Model",
        )

        if self.verbose:
            logger.info(f"Pending Training: Found {len(training_model_results)} job(s)")

        for result in training_model_results:
            if result is None:
                continue

            (
                classifier_name,
                _random_state,
                features_seed_path,
                model_seed_path,
                top_n_features,
            ) = result

            if classifier_name not in records_per_classifier:
                records_per_classifier[classifier_name] = []

            if features_seed_path not in self.features_csvs:
                self.features_csvs.append(features_seed_path)

            records_per_classifier[classifier_name].append(
                {
                    "random_state": _random_state,
                    "features": top_n_features,
                    "model_filepath": model_seed_path,
                }
            )

        for path in existing_feature_paths:
            if path not in self.features_csvs:
                self.features_csvs.append(path)

        if self.verbose:
            logger.info("Training: Concatenating significant features")

        self.features_selected_df = concat_significant_features(
            features_csvs=self.features_csvs,
            features_dir=self.features_dir,
            number_of_features=self.top_n_features,
        )

        if plot_features and not self.features_selected_df.empty:
            plot_significant_features(
                df=self.features_selected_df.reset_index(),
                filepath=os.path.join(
                    self.features_dir, f"top_{self.top_n_features}_features"
                ),
                top_features=self.top_n_features,
                overwrite=self.overwrite,
                legend_loc="lower right",
                values_column="score",
            )

        # Save SeedEnsemble
        seed_ensembles: dict[str, SeedEnsemble] = {}
        cv_name = self.classifier_models[0].cv_name
        for classifier_model in self.classifier_models:
            if self.verbose:
                logger.info(
                    f"Training: Saving SeedEnsemble for {classifier_model.name} model ..."
                )

            classifier_name = classifier_model.name
            if not records_per_classifier[classifier_name]:
                continue

            trained_model_json = save_model_json(
                seeds=seeds,
                records=records_per_classifier[classifier_name],
                classifier_dir=self.classifier_dirs[classifier_name],
                classifier_model=classifier_model,
                number_of_features=self.top_n_features,
                prefix_filename="trained-model",
                verbose=self.verbose,
            )

            seed_ensemble_path, seed_ensemble = self.build_seed_ensemble(
                output_dir=self.classifier_dirs[classifier_name],
                classifier_name=classifier_model.name,
                registry_path=trained_model_json,
                verbose=self.verbose,
            )

            seed_ensembles[classifier_model.name] = seed_ensemble
            self.seed_ensembles[classifier_model.name] = seed_ensemble_path
            self.results[classifier_model.name] = trained_model_json

        filename = f"ClassifierEnsemble_{cv_name}"
        if self.results:
            results_json_path = os.path.join(self.classifier_dir, f"{filename}.json")
            with open(results_json_path, "w") as f:
                json.dump(self.results, f, indent=2)
            self.results_json = results_json_path

        if seed_ensembles:
            self.classifier_ensemble_path, self.ClassifierEnsemble = (
                self.build_classifier_ensemble(
                    output_dir=self.classifier_dir,
                    seed_ensembles=seed_ensembles,
                    filename=filename,
                )
            )
        else:
            raise ValueError("No seed ensembles found")

        # Save model
        identity = type(self).build_identity(
            nslc=self.nslc,
            tremor_df=self.tremor_df,
            start_date=self.start_date,
            end_date=self.end_date,
            classifiers=self.classifiers,
            eruption_dates=self.eruption_dates or [],
            window_size=self.window_size,
            cv_strategy=self.cv_strategy,
            cv_splits=self.cv_splits,
            scoring=self._scoring,
            top_n_features=self.top_n_features,
            include_eruption_date=self.include_eruption_date,
            build_label_params={
                "window_step": self.window_step,
                "window_step_unit": self.window_step_unit,
                "builder": self.builder,
                "days_before_eruption": self.days_before_eruption,
            },
            extract_features_params=dict(self._extract_features_kwargs or {}),
            fit_params=dict(self._fit_kwargs or {}),
        )
        self.save(identity)
        self.training_hash = type(self).compute_hash(identity)

        try:
            self.save_config()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to save training config: {exc}")

        if compute_learning_curve:
            self.compute_learning_curve(overwrite=self.overwrite)

        return self

    def load_features(
        self,
        features_matrix_path: str | None = None,
        label_features_csv: str | None = None,
        select_features: str | list[str] | None = None,
    ) -> Self:
        """Reuse the feature matrix written by a previous training run.

        Skips tsfresh extraction entirely by reading the persisted
        ``features-matrix_*.parquet`` and ``features-label_*.csv`` produced by
        an earlier :meth:`extract_features` call, then optionally projecting
        the matrix to a curated column subset. Designed for repeat training
        where the windowing and tremor data have not changed but the model
        needs to be refit — e.g. iterating on hyperparameters with the curated
        feature list from a prior ``top_{N}_features.csv``.

        Drops in as a replacement for the
        ``build_label() → extract_features()`` prefix before :meth:`fit`;
        ``self.LabelBuilder`` is left untouched (may be ``None``), and
        :meth:`fit` derives the resampling minority-share check from
        ``self.labels`` when ``LabelBuilder`` is absent.

        Args:
            features_matrix_path (str | None, optional): Path to the
                ``features-matrix_*.parquet`` written under
                ``{output_dir}/training/features/{cv}/``. When ``None``, the
                method globs ``self.features_dir`` for exactly one match.
                Defaults to ``None``.
            label_features_csv (str | None, optional): Path to the matching
                ``features-label_*.csv``. Same auto-resolve behaviour as
                ``features_matrix_path``. Defaults to ``None``.
            select_features (str | list[str] | None, optional): Curated
                feature list — accepts either the path to a
                ``top_{N}_features.csv`` / ``top_features.csv`` /
                ``significant_features.csv`` or an explicit ``list[str]``.
                Resolved via :func:`load_select_features` and intersected
                with the matrix columns; missing names log a warning and are
                skipped. ``None`` loads the full matrix. Defaults to
                ``None``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If auto-resolve finds zero or more than one matching
                artefact under ``self.features_dir``.
            ValueError: If the loaded label CSV's ``datetime`` span does not
                fully cover the configured ``[start_date, end_date]`` training
                range (compared at day granularity).
            ValueError: If ``select_features`` is provided but its intersection
                with the matrix columns is empty.
            FileNotFoundError: If an explicit path argument does not exist.

        Example:
            >>> # Reuse the artefacts from a prior run with the curated top-N
            >>> top_n_csv = "output/.../training/features/.../top_20_features.csv"
            >>> model.load_features(select_features=top_n_csv).fit(seeds=25)
            >>> # No build_label() / extract_features() needed.
        """
        if features_matrix_path is None:
            features_matrix_path = self._resolve_single_artefact(
                pattern="features-matrix_*.parquet", label="feature matrix"
            )
        elif not os.path.isfile(features_matrix_path):
            raise FileNotFoundError(
                f"features_matrix_path not found: {features_matrix_path}"
            )

        if label_features_csv is None:
            label_features_csv = self._resolve_single_artefact(
                pattern="features-label_*.csv", label="feature label"
            )
        elif not os.path.isfile(label_features_csv):
            raise FileNotFoundError(
                f"label_features_csv not found: {label_features_csv}"
            )

        if self.verbose:
            logger.info(
                f"[Training Model]: Loading feature matrix {features_matrix_path}"
            )

        # Ensure ``self.start_date`` and ``self.end_date`` is in label date range
        label_datetimes = pd.read_csv(
            label_features_csv, usecols=["datetime"], parse_dates=["datetime"]
        )["datetime"]
        loaded_start: pd.Timestamp = label_datetimes.min()
        loaded_end: pd.Timestamp = label_datetimes.max()
        if (
            self.start_date.date() < loaded_start.date()
            or self.end_date.date() > loaded_end.date()
        ):
            raise ValueError(
                f"Loaded feature range [{loaded_start:%Y-%m-%d} → "
                f"{loaded_end:%Y-%m-%d}] does not fully cover the configured "
                f"training range [{self.start_date_str} → {self.end_date_str}]. "
                f"Re-run extract_features() or point to a features run that "
                f"spans the requested range."
            )

        self.features_df = pd.read_parquet(features_matrix_path)
        self.labels = load_label_csv(label_features_csv)
        self.features_path = features_matrix_path

        if select_features is not None:
            resolved = load_select_features(
                select_features, number_of_features=self.top_n_features
            )
            matrix_columns = set(self.features_df.columns)
            present = [name for name in resolved if name in matrix_columns]
            missing = [name for name in resolved if name not in matrix_columns]
            if missing:
                logger.warning(
                    f"load_features: {len(missing)} of {len(resolved)} curated "
                    f"features absent from matrix; skipping {missing[:5]}"
                    + ("..." if len(missing) > 5 else "")
                )
            if not present:
                raise ValueError(
                    "select_features has no overlap with the loaded feature matrix."
                )
            self.features_df = self.features_df[present]

        if self.verbose:
            logger.info(
                f"[Training Model]: Loaded features_df={self.features_df.shape}, "
                f"labels={self.labels.shape}"
            )

        return self

    def _resolve_single_artefact(self, pattern: str, label: str) -> str:
        """Glob ``self.features_dir`` for exactly one file matching ``pattern``.

        Args:
            pattern (str): Glob pattern relative to ``self.features_dir``
                (e.g. ``"features-matrix_*.parquet"``,
                ``"features-label_*.csv"``).
            label (str): Human-readable name used in error messages.

        Returns:
            str: Absolute path to the single matching file.

        Raises:
            ValueError: If zero or more than one match is found — the caller
                must then pass an explicit path.
        """
        matches = sorted(glob.glob(os.path.join(self.features_dir, pattern)))
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"No {label} matching '{pattern}' under {self.features_dir}; "
                "pass an explicit path."
            )
        raise ValueError(
            f"Multiple {label} files matching '{pattern}' under "
            f"{self.features_dir}: {matches}; pass an explicit path."
        )

    @staticmethod
    def build_classifier_ensemble(
        output_dir: str,
        seed_ensembles: dict[str, SeedEnsemble],
        filename: str = "ClassifierEnsemble",
        verbose: bool = False,
    ) -> tuple[str, ClassifierEnsemble]:
        """Build and save a ClassifierEnsemble from in-memory SeedEnsemble objects.

        Assembles all per-classifier ``SeedEnsemble`` objects into a single
        ``ClassifierEnsemble`` via :meth:`ClassifierEnsemble.from_seed_ensembles`
        and persists it to ``{output_dir}/ClassifierEnsemble.pkl``.`.

        Returns:
            str: Absolute path to the saved ``ClassifierEnsemble.pkl`` file.

        Example:
            >>> path = TrainingModel.build_classifier_ensemble(
            ...     output_dir="training/classifiers",
            ...     seed_ensembles={"RandomForestClassifier": rf_ensemble, "XGBClassifier": xgb_ensemble},
            ... )
        """
        classifier_ensemble_path = os.path.join(output_dir, f"{filename}.pkl")
        classifier_ensemble = ClassifierEnsemble.from_seed_ensembles(
            seed_ensembles, verbose
        )
        classifier_ensemble.save(classifier_ensemble_path)
        return classifier_ensemble_path, classifier_ensemble

    @staticmethod
    def build_seed_ensemble(
        output_dir: str,
        classifier_name: str,
        registry_path: str,
        verbose: bool = False,
    ) -> tuple[str, SeedEnsemble]:
        """Build and save a SeedEnsemble from a trained-model registry.

        Loads all per-seed model paths from ``registry_path`` via
        :meth:`SeedEnsemble.from_any` (dispatches on extension — ``.json`` for
        the new inline-features registry, ``.csv`` for the legacy registry),
        saves the ensemble to
        ``{output_dir}/SeedEnsemble_{classifier_name}.pkl``, and returns both
        the path and the in-memory object so the caller can pass it directly
        to :meth:`build_classifier_ensemble` without a disk reload.

        Args:
            output_dir (str): Directory where ``SeedEnsemble_{classifier_name}.pkl``
                is written.
            classifier_name (str): Human-readable classifier name used as the
                filename suffix (e.g. ``"RandomForestClassifier"``).
            registry_path (str): Path to the trained-model registry produced by
                :func:`~eruption_forecast.utils.ml.save_model_json` (``.json``)
                or the legacy CSV writer (``.csv``).
            verbose (bool): Whether to emit load progress logs. Defaults to ``False``.

        Returns:
            tuple[str, SeedEnsemble]: A two-element tuple ``(filepath, seed_ensemble)``
            where ``filepath`` is the absolute path to the saved ``.pkl`` file and
            ``seed_ensemble`` is the constructed :class:`SeedEnsemble` instance.

        Example:
            >>> path, ensemble = TrainingModel.build_seed_ensemble(
            ...     output_dir="training/classifiers",
            ...     classifier_name="RandomForestClassifier",
            ...     registry_path="training/classifiers/trained-model__rf.json",
            ... )
        """
        # Strip the extension and split on the ``__`` separator to recover the
        # ``{Classifier}_{CV}_seeds-{N}_features-{K}`` suffix used by the
        # ensemble filename.
        suffix = os.path.splitext(os.path.basename(registry_path))[0].split("__")[-1]

        ensure_dir(output_dir)
        filepath = os.path.join(output_dir, f"SeedEnsemble_{suffix}.pkl")
        seed_ensemble = SeedEnsemble.from_any(
            registry_path, classifier_name=classifier_name, verbose=verbose
        )
        seed_ensemble.save(filepath)
        return filepath, seed_ensemble

    def compute_learning_curve(
        self,
        scoring: list[str] | None = None,
        train_sizes: list[float] | np.ndarray | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Compute sklearn learning curves for every (classifier, seed) pair.

        For each previously trained ``(classifier, seed)``, loads the tuned
        estimator pickle and the per-seed resampled training data, then calls
        :func:`sklearn.model_selection.learning_curve` once per scoring metric.
        Hyperparameters are reused from the existing ``best_estimator_`` — no
        ``GridSearchCV`` re-tuning happens. Outputs JSON files matching the
        schema consumed by
        :func:`eruption_forecast.plots.evaluation_plots.plot_learning_curve`.

        Must be called after :meth:`fit`. Existing JSON files are kept unless
        ``overwrite=True``.

        Args:
            scoring (list[str] | None): sklearn scoring keys to evaluate. Each
                triggers an independent ``learning_curve`` call sharing the
                same CV splits. Defaults to
                ``["balanced_accuracy", "roc_auc"]``.
            train_sizes (list[float] | np.ndarray | None): Fractions of the
                training set used to seed the curve. Defaults to
                ``np.linspace(0.1, 1.0, 5)``.
            overwrite (bool): Re-compute and overwrite existing per-seed JSON.
                Defaults to False.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If :meth:`fit` has not been called first.

        Example:
            >>> model.fit(seeds=25)
            >>> model.compute_learning_curve()
            >>> # → training/classifiers/{slug}/{cv}/learning_curves/{seed:05d}.json
        """
        ensemble = self.ClassifierEnsemble
        if ensemble is None:
            raise ValueError("Please run fit() first.")

        if self.features_df.empty:
            raise ValueError(
                "compute_learning_curve() requires self.features_df to be populated. "
                "Run extract_features() or load_features() before fit()."
            )

        scoring = (
            list(scoring) if scoring is not None else ["balanced_accuracy", "roc_auc"]
        )
        train_sizes = (
            np.asarray(train_sizes)
            if train_sizes is not None
            else np.linspace(0.1, 1.0, 5)
        )

        # Lazy directory creation — only when this opt-in step actually runs.
        for lc_dir in self.learning_curve_dirs.values():
            ensure_dir(lc_dir)

        jobs: list[tuple] = []
        for classifier_model in self.classifier_models:
            classifier_name = classifier_model.name
            seed_ensemble = ensemble.ensembles.get(classifier_model.name)
            if seed_ensemble is None:
                continue
            lc_dir = self.learning_curve_dirs[classifier_name]
            for seed_record in seed_ensemble.seeds:
                random_state = seed_record["random_state"]
                lc_path = os.path.join(lc_dir, f"{random_state:05d}.json")
                if not overwrite and os.path.isfile(lc_path):
                    if self.verbose:
                        logger.info(
                            f"Learning curve {random_state:05d} / {classifier_name}: "
                            f"exists, skipping."
                        )
                    continue
                jobs.append((random_state, classifier_name))

        if not jobs:
            logger.info("Learning Curve: nothing to compute (all JSONs present).")
            return self

        logger.info(f"Running learning-curve computation across {len(jobs)} job(s)..")

        worker = partial(
            self._run_compute_learning_curve,
            classifier_ensemble=ensemble,
            scoring=scoring,
            train_sizes=train_sizes,
        )

        results: list[str | None] = self._run_jobs(
            worker,
            jobs,
            job_name="Learning Curve",
        )

        if self.verbose:
            written = sum(1 for result in results if result is not None)
            logger.info(f"Learning Curve: wrote {written}/{len(results)} JSON(s)")

        return self

    def _train(
        self,
        random_state: int,
        classifier_name: str,
    ) -> tuple | None:
        """Fit a single classifier for one random seed and persist the model.

        Reads the pre-computed top-N feature list and resampled training data
        for the given seed, then runs ``grid_search_cv`` to find the best
        estimator. Skips fitting if the model file already exists and
        ``overwrite`` is False.

        Args:
            random_state (int): Seed index identifying the feature and
                resampled data files to use.
            classifier_name (str): Slug name of the classifier to train,
                e.g. ``"random-forest-classifier"``.

        Returns:
            tuple | None: A five-element tuple
                ``(classifier_name, random_state, features_seed_path,
                model_seed_path, top_n_features)`` on success, or ``None``
                when the feature file is missing or contains no features.
                ``top_n_features`` is the inline list of selected column names
                used at fit time, embedded in the trained-model JSON registry
                by ``save_model_json``.
        """
        filename = f"{random_state:05d}"
        features_seed_path = os.path.join(self.features_seed_dir, f"{filename}.csv")
        features_resampled_path = os.path.join(
            self.features_resampled_dir, f"{filename}.csv"
        )

        if not os.path.isfile(features_seed_path):
            logger.warning(
                f"Features {random_state:05d}: {features_seed_path} missing, skipping."
            )
            return None

        top_n_features = pd.read_csv(features_seed_path, index_col=0).index.tolist()
        if not top_n_features:
            return None

        classifier_model = next(
            m for m in self.classifier_models if m.name == classifier_name
        )

        model_seed_path = os.path.join(
            self.models_dir[classifier_name], f"{filename}.pkl"
        )

        # Return cached result without re-training if the model already exists.
        if not self.overwrite and os.path.isfile(model_seed_path):
            logger.info(
                f"Seed {random_state:05d} / {classifier_name}: model exists, skipping."
            )
            return (
                classifier_name,
                random_state,
                features_seed_path,
                model_seed_path,
                top_n_features,
            )

        features_resampled, labels_resampled = load_features_resampled(
            features=self.features_df,
            resampled=features_resampled_path,
            columns=top_n_features,
        )

        if self.verbose:
            logger.info(f"Fitting {random_state:05d}/{classifier_name}...")

        _classifier_model, _grid_search, best_model = grid_search_cv(
            random_state,
            features_resampled,
            labels_resampled,
            top_n_features,
            classifier_model=classifier_model,
            scoring=self._scoring,
        )

        joblib.dump(best_model, model_seed_path)

        if self.verbose:
            logger.info(
                f"Fitted {random_state:05d}/{classifier_name}: {model_seed_path}"
            )

        return (
            classifier_name,
            random_state,
            features_seed_path,
            model_seed_path,
            top_n_features,
        )

    def _features_selection(
        self,
        random_state: int,
        features_seed_path: str,
        features_resampled_path: str,
        figures_seed_path: str | None,
        resample_method: Literal["under", "over"] | None,
        sampling_strategy: str | float,
    ) -> str | None:
        """Resample the dataset and select the top-N features for one seed.

        Applies the configured resampler, writes the resampled id list and
        ``is_erupted`` label per seed to ``features_resampled_path``, then
        delegates to ``_select_features`` to run tsfresh feature selection
        and persist the top-N list. The matching feature rows are recovered
        downstream via ``load_features_resampled`` slicing
        ``self.features_df`` by the resampled id index, so the feature
        matrix is never duplicated on disk per seed.

        Args:
            random_state (int): Seed used for reproducible resampling and
                feature selection.
            features_seed_path (str): Destination CSV path for the top-N
                selected feature list.
            features_resampled_path (str): Destination CSV path for the
                per-seed resampled label series (``id`` index + ``is_erupted``
                column). The matching feature rows are recovered downstream
                via ``self.features_df.loc[...]``.
            figures_seed_path (str | None): Destination path for the feature
                importance figure, or None to skip plotting.
            resample_method (Literal["under", "over"] | None): Resampling
                strategy; None skips resampling.
            sampling_strategy (str | float): Target minority-to-majority ratio
                passed to the resampler.

        Returns:
            str | None: ``features_seed_path`` on success, or ``None`` when
                feature selection yields zero features.
        """
        features_resampled, labels_resampled = resample(
            features=self.features_df,
            labels=self.labels,
            method=resample_method,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            verbose=self.verbose,
        )

        # Persist only the resampled id index + ``is_erupted`` per seed; the
        # matching feature rows are reconstructed downstream by slicing
        # ``self.features_df`` via ``load_features_resampled``. Avoids a
        # per-seed copy of the full tsfresh matrix on disk.
        labels_resampled.to_frame(name="is_erupted").to_csv(features_resampled_path)

        result = self._select_features(
            features=features_resampled,
            labels=labels_resampled,
            random_state=random_state,
            number_of_features=self.top_n_features,
            features_seed_path=features_seed_path,
            figures_seed_path=figures_seed_path,
            overwrite=self.overwrite,
        )

        if result is None:
            return None

        return features_seed_path

    def _run_jobs(
        self,
        method: Callable,
        jobs: list[tuple],
        job_name: str = "Training Model",
    ) -> list:
        """Execute a list of jobs either sequentially or via joblib Parallel.

        Uses ``Parallel(backend="loky")`` when ``self.n_jobs != 1``, otherwise
        iterates sequentially to avoid unnecessary process-pool overhead.

        Args:
            method (Callable): The method to call for each job, accepting
                unpacked tuple arguments.
            jobs (list[tuple]): Each tuple is unpacked as positional arguments
                to ``method``.
            job_name (str): Label used in progress log messages. Defaults to
                ``"Training Model"``.

        Returns:
            list: Results returned by each ``method`` call, in the same order
                as ``jobs``.
        """
        if self.n_jobs != 1:
            logger.info(
                f"[{job_name}]:On {self.n_jobs} job(s) with {self.n_grids} grid(s) search..."
            )
            return Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(method)(*job) for job in jobs
            )
        return [method(*job) for job in jobs]

    def _collect_pending_train_jobs(
        self,
        random_states: list[int],
        resample_method: Literal["under", "over", "auto"] | None,
        sampling_strategy: str | float,
        plot_features: bool,
    ) -> tuple[list[tuple], list[tuple], dict[str, list[dict]], list[str]]:
        """Determine which seeds still require feature selection or model training.

        Iterates over every random state and checks whether cached feature and
        model files already exist. Seeds with complete feature files skip
        feature selection; seeds with complete model files skip training.
        Existing model records are collected into ``records_per_classifier``
        so the registry can be rebuilt even when no new training occurs.

        Args:
            random_states (list[int]): List of seed indices to evaluate.
            resample_method (Literal["under", "over", "auto"] | None):
                Resampling strategy forwarded to each feature-selection job.
            sampling_strategy (str | float): Target class ratio forwarded to
                each feature-selection job.
            plot_features (bool): Whether to include a figure output path in
                feature-selection job tuples.

        Returns:
            tuple[list[tuple], list[tuple], dict[str, list[dict]], list[str]]: A
                four-element tuple containing:

                - pending_feature_selection_jobs: Jobs that still need feature
                  selection, each a tuple of
                  ``(random_state, features_seed_path, features_resampled_path,
                  figures_seed_path, resample_method, sampling_strategy)``.
                - pending_training_model_jobs: Jobs that still need model
                  training for seeds whose feature files already exist on disk.
                  Each entry is a tuple of ``(random_state, classifier_name)``.
                - records_per_classifier: Classifier-slug → list of record
                  dicts for seeds whose models already exist on disk.
                - existing_feature_paths: Feature CSV paths that already exist
                  on disk (can_skip seeds), collected here to avoid a post-hoc
                  filesystem rescan in ``fit()``.
        """
        pending_feature_selection_jobs: list[tuple] = []
        pending_training_model_jobs: list[tuple] = []
        existing_feature_paths: list[str] = []

        # Init records per classifier
        records_per_classifier: dict[str, list[dict]] = {
            classifier_model.name: [] for classifier_model in self.classifier_models
        }

        for random_state in random_states:
            (
                can_skip,
                features_seed_path,
                features_resampled_path,
                figures_seed_path,
            ) = generate_features_filepaths(
                random_state=random_state,
                features_seed_dir=self.features_seed_dir,
                features_resampled_dir=self.features_resampled_dir,
                figures_seed_dir=self.figures_seed_dir,
                plot_features=plot_features,
                overwrite=self.overwrite,
            )

            if can_skip:
                existing_feature_paths.append(features_seed_path)
                filename = f"{random_state:05d}"
                cached_top_n_features = pd.read_csv(
                    features_seed_path, index_col=0
                ).index.tolist()
                for classifier_model in self.classifier_models:
                    classifier_name = classifier_model.name
                    model_seed_path = os.path.join(
                        self.models_dir[classifier_name], f"{filename}.pkl"
                    )

                    # Check if seed model per classifier exists
                    # Should be exists in: <self.models_dir[classifier_name]>/models
                    # Example: training\classifiers\lite-random-forest-classifier\stratified-shuffle-split\models
                    if os.path.isfile(model_seed_path):
                        records_per_classifier[classifier_name].append(
                            {
                                "random_state": random_state,
                                "features": cached_top_n_features,
                                "model_filepath": model_seed_path,
                            }
                        )
                    else:
                        pending_training_model_jobs.append(
                            (random_state, classifier_name)
                        )
                continue

            pending_feature_selection_jobs.append(
                (
                    random_state,
                    features_seed_path,
                    features_resampled_path,
                    figures_seed_path,
                    resample_method,
                    sampling_strategy,
                )
            )

        return (
            pending_feature_selection_jobs,
            pending_training_model_jobs,
            records_per_classifier,
            existing_feature_paths,
        )

    def _select_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        random_state: int,
        features_seed_path: str,
        number_of_features: int,
        figures_seed_path: str | None,
        overwrite: bool = False,
    ) -> tuple | None:
        """Run feature selection and persist the top-N feature list to disk.

        Fits ``self.FeatureSelector`` on the provided features and labels,
        then saves the top-N selected features as a CSV. Optionally writes a
        feature importance figure. Returns None when selection yields zero
        features, signalling the caller to skip model training for this seed.

        Args:
            features (pd.DataFrame): Resampled feature matrix used to fit the
                selector.
            labels (pd.Series): Resampled binary labels aligned with
                ``features``.
            random_state (int): Seed passed to the feature selector for
                reproducibility.
            features_seed_path (str): Destination CSV path for the top-N
                selected feature scores.
            number_of_features (int): Maximum number of features to retain.
                Reduced automatically if fewer features are available.
            figures_seed_path (str | None): Destination path for the feature
                importance figure, or None to skip plotting.
            overwrite (bool): Overwrite an existing figure file. Defaults to
                False.

        Returns:
            tuple | None: A four-element tuple
                ``(df_selected_features, top_selected_features,
                selected_features, number_of_features)`` on success, or
                ``None`` when selection reduces features to zero.
        """
        # Reduced features/columns
        features_selector = self.FeatureSelector.set_random_state(random_state)

        df_selected_features = features_selector.fit_transform(
            features, labels, top_n=number_of_features
        )

        if features_selector.n_features == 0:
            logger.warning(
                f"{random_state:05d}: Features reduced to 0. Skip training model."
            )
            return None

        selected_features = features_selector.selected_features_

        # Handle if columns in df_selected_features has less than number_of_significant_features
        len_features_columns = len(df_selected_features.columns)
        if len_features_columns < number_of_features:
            logger.warning(
                f"{random_state:05d}: Number of features after extracted ({len_features_columns}) "
                f"are less than {number_of_features} features."
            )
            number_of_features = len_features_columns

        top_selected_features = selected_features.head(number_of_features)

        # Save TOP-N significant features
        top_selected_features.to_csv(features_seed_path, index=True)

        if figures_seed_path is not None:
            plot_significant_features(
                df=pd.DataFrame(selected_features).reset_index(),
                filepath=figures_seed_path,
                top_features=number_of_features,
                values_column="score",
                overwrite=overwrite,
                dpi=150,
            )

        return (
            df_selected_features,
            top_selected_features,
            selected_features,
            number_of_features,
        )

    def _run_compute_learning_curve(
        self,
        classifier_ensemble: ClassifierEnsemble,
        random_state: int,
        classifier_name: str,
        scoring: list[str],
        train_sizes: np.ndarray,
    ) -> str | None:
        """Compute and persist the learning-curve JSON for one (seed, classifier).

        Reads the per-seed resampled CSV for the training data, then runs
        :func:`learning_curve` once per scoring metric. The tuned estimator
        and its feature list are pulled from ``classifier_ensemble`` (passed
        in by :meth:`compute_learning_curve` after it has validated the
        in-memory ensemble is populated) — no pickle or feature-list CSV is
        read from disk. The CV splitter is rebuilt reproducibly via
        :meth:`ClassifierModel.set_random_state` ``+`` :meth:`get_cv_splitter`,
        so train/validation splits are identical across the metric loop.

        Args:
            classifier_ensemble (ClassifierEnsemble): The in-memory
                ``ClassifierEnsemble`` produced by :meth:`fit`. Caller
                guarantees this is non-None.
            random_state (int): Seed identifying the resampled CSV and
                in-memory seed record.
            classifier_name (str): Slug name of the classifier to evaluate.
            scoring (list[str]): sklearn scoring keys to evaluate.
            train_sizes (np.ndarray): Fractions of the training set to evaluate.

        Returns:
            str | None: Absolute path of the written JSON, or ``None`` when
                the resampled CSV is missing or the seed was skipped during
                ``fit()``.
        """
        filename = f"{random_state:05d}"
        features_resampled_path = os.path.join(
            self.features_resampled_dir, f"{filename}.csv"
        )

        if not os.path.isfile(features_resampled_path):
            logger.warning(
                f"Learning curve {filename} / {classifier_name}: "
                f"missing {features_resampled_path}, skipping."
            )
            return None

        classifier_model = next(
            m for m in self.classifier_models if m.name == classifier_name
        )
        seed_ensemble = classifier_ensemble.ensembles.get(classifier_model.name)
        if seed_ensemble is None:
            logger.warning(
                f"Learning curve {filename} / {classifier_name}: "
                f"{classifier_model.name} missing from ClassifierEnsemble, skipping."
            )
            return None

        seed_record = next(
            (s for s in seed_ensemble.seeds if s["random_state"] == random_state),
            None,
        )
        if seed_record is None:
            logger.warning(
                f"Learning curve {filename} / {classifier_name}: "
                f"seed not found in SeedEnsemble (likely skipped during fit), skipping."
            )
            return None

        estimator = seed_record["model"]
        top_n_features = list(seed_record["feature_names"])
        if not top_n_features:
            return None

        X, y = load_features_resampled(
            features=self.features_df,
            resampled=features_resampled_path,
            columns=top_n_features,
        )

        cv_splitter = classifier_model.set_random_state(random_state).get_cv_splitter()

        metrics_payload: dict[str, dict[str, list[float]]] = {}
        train_sizes_abs: np.ndarray | None = None

        for metric in scoring:
            sizes_abs, train_scores, test_scores = learning_curve(
                estimator=estimator,
                X=X,
                y=y,
                cv=cv_splitter,
                scoring=metric,
                train_sizes=train_sizes,
                n_jobs=1,
                shuffle=False,
                random_state=random_state,
            )
            train_sizes_abs = sizes_abs
            metrics_payload[metric] = {
                "train_scores_mean": train_scores.mean(axis=1).tolist(),
                "train_scores_std": train_scores.std(axis=1).tolist(),
                "test_scores_mean": test_scores.mean(axis=1).tolist(),
                "test_scores_std": test_scores.std(axis=1).tolist(),
            }

        payload = {
            "random_state": random_state,
            "classifier_name": classifier_name,
            "train_sizes": train_sizes_abs.tolist()
            if train_sizes_abs is not None
            else [],
            "metrics": metrics_payload,
        }

        lc_path = os.path.join(
            self.learning_curve_dirs[classifier_name], f"{filename}.json"
        )
        with open(lc_path, "w") as f:
            json.dump(payload, f, indent=2)

        if self.verbose:
            logger.info(
                f"Learning curve {filename} / {classifier_name}: wrote {lc_path}"
            )

        return lc_path
