import os
from typing import Self, Literal
from datetime import datetime
from collections.abc import Callable

import numpy as np
import joblib
import pandas as pd
from joblib import Parallel, delayed

from eruption_forecast import LabelBuilder, DynamicLabelBuilder
from eruption_forecast.plots import plot_significant_features
from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import (
    resample,
    grid_search_cv,
    save_model_registry,
    get_classifier_models,
)
from eruption_forecast.utils.dataframe import concat_significant_features
from eruption_forecast.utils.pathutils import ensure_dir, generate_features_filepaths
from eruption_forecast.model.base_model import BaseModel
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.features.feature_selector import FeatureSelector


class TrainingModel(BaseModel):
    """Train classifier models across multiple random seeds on tremor feature data.

    Orchestrates the full training pipeline: label building, tremor matrix
    construction, tsfresh feature extraction, per-seed resampling and feature
    selection, and parallel GridSearchCV model fitting. Trained models and a
    model registry CSV are written to the configured output directory.

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
        number_of_features (int): Top-N features retained after feature
            selection. Defaults to 20.
        include_eruption_date (bool): Whether to include the eruption day
            itself as a positive label. Defaults to False.
        overwrite (bool): Re-run and overwrite cached feature and model files.
            Defaults to False.
        output_dir (str | None): Root output directory. Resolved automatically
            when None. Defaults to None.
        root_dir (str | None): Project root used for path resolution. Defaults
            to None.
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
        number_of_features: int = 20,
        include_eruption_date: bool = False,
        overwrite: bool = False,
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        n_grids: int = 1,
        verbose: bool = False,
    ) -> None:
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
        self.classifiers = classifiers
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.number_of_features = number_of_features
        self.include_eruption_date: bool = include_eruption_date
        self.overwrite: bool = overwrite
        self.n_grids: int = n_grids

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
            self.features_dir,
            self.features_seed_dir,
            self.features_resampled_dir,
            self.figures_seed_dir,
            self.classifier_dirs,
            self.models_dir,
        ) = self.set_directories()
        self.features_csvs: list[str] = []
        self.features_selected_df: pd.DataFrame = pd.DataFrame()

        # Will be set after build_label() called
        self.LabelBuilder: LabelBuilder | None = None
        self.labels: pd.Series = pd.Series()
        self.basename: str | None = None

        # Will be set after fit() called
        self.csv: dict[str, str] = {}
        self.model = None

        self.validate()

    def set_directories(
        self,
    ) -> tuple[str, str, str, str, str, dict[str, str], dict[str, str]]:
        """Build and return all output directory paths used during training.

        Creates classifier-level subdirectories immediately so downstream
        steps can write files without additional setup calls.

        Returns:
            tuple: A seven-element tuple containing:
                - training_dir (str): Root training output path.
                - features_dir (str): CV-scoped features directory.
                - features_seed_dir (str): Per-seed selected-feature CSVs.
                - features_resampled_dir (str): Per-seed resampled-data CSVs.
                - figures_seed_dir (str): Per-seed feature importance figures.
                - classifier_dirs (dict[str, str]): Classifier-slug → directory.
                - models_dir (dict[str, str]): Classifier-slug → model directory.
        """
        training_dir = os.path.join(self.output_dir, "training")
        features_dir = os.path.join(training_dir, "features", self.cv_name)
        features_seed_dir = os.path.join(features_dir, "seed")
        features_resampled_dir = os.path.join(features_dir, "resampled")
        figures_seed_dir = os.path.join(features_seed_dir, "figures")

        classifier_dirs: dict[str, str] = {}
        models_dir: dict[str, str] = {}

        for classifier_model in self.classifier_models:
            classifier_slug_name = classifier_model.slug_name
            classifier_dir = os.path.join(
                training_dir,
                "classifiers",
                classifier_slug_name,
                self.cv_name,
            )
            classifier_dirs[classifier_slug_name] = classifier_dir
            model_dir = os.path.join(classifier_dir, "models")
            ensure_dir(classifier_dir)
            models_dir[classifier_slug_name] = model_dir

        return (
            training_dir,
            features_dir,
            features_seed_dir,
            features_resampled_dir,
            figures_seed_dir,
            classifier_dirs,
            models_dir,
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
        exceeds the available CPU count. Creates the root training directory
        as a side effect. Date range clamping against the tremor data bounds
        is deferred to ``_clamp_dates_to_tremor()``, which is called lazily
        from ``build_label()`` to avoid loading the tremor CSV during
        construction.

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
        classifier_names = ", ".join(m.slug_name for m in self.classifier_models)
        return (
            f"TrainingModel("
            f"period={self.start_date_str} → {self.end_date_str}, "
            f"window_size={self.window_size}d, "
            f"classifiers=[{classifier_names}], "
            f"cv={self.cv_strategy}/{self.cv_splits}-fold, "
            f"top_features={self.number_of_features}, "
            f"eruptions={len(self.eruption_dates) if self.eruption_dates is not None else 0}"
            f")"
        )

    def to_dict(self) -> dict:
        """Serialise core training parameters to a plain dictionary.

        Returns:
            dict: Mapping of parameter names to their current values, including
                ``start_date``, ``end_date``, ``window_size``,
                ``eruption_dates``, and ``n_jobs``.

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
            "number_of_features": self.number_of_features,
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
        classifier_names = ", ".join(m.slug_name for m in self.classifier_models)
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
            f"Top features retained: {self.number_of_features}. "
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

        self._sync_dates_to_tremor()
        verbose = verbose if verbose is not None else self.verbose

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
                eruption_dates=self.eruption_dates,  # ty:ignore[invalid-argument-type]
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
                eruption_dates=self.eruption_dates,  # ty:ignore[invalid-argument-type]
                output_dir=self.training_dir,
                root_dir=self.root_dir,
                verbose=verbose,
            ).build()

        self.LabelBuilder = label_builder
        self.basename = os.path.basename(label_builder.csv).split(".csv")[0]

        return self

    def extract_features(
        self,
        select_tremor_columns: list[str] | None = None,
        save_tremor_matrix_per_method: bool = False,
        exclude_features: list[str] | None = None,
        save_tremor_matrix_per_id: bool = False,
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
            save_tremor_matrix_per_id (bool): Write one CSV per window ID.
                Defaults to False.
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
            >>> model.extracted_features_df.shape
            (n_windows, n_features)
        """
        if self.LabelBuilder is None:
            raise ValueError("Please run build_label() first.")

        features_builder = self._build_features(
            label_df=self.LabelBuilder.df,
            output_dir=self.training_dir,
            features_dir=self.features_dir,
            select_tremor_columns=select_tremor_columns,
            save_tremor_matrix_per_method=save_tremor_matrix_per_method,
            save_tremor_matrix_per_id=save_tremor_matrix_per_id,
            overwrite=overwrite,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        self.features_df = features_builder.extract_features(
            use_relevant_features=True,
            select_tremor_columns=select_tremor_columns,
            exclude_features=exclude_features,
        )

        self.features_csv = features_builder.csv

        labels = features_builder.label_df.copy()

        if "id" in labels.columns:
            labels = labels.set_index("id")
        if "datetime" in labels.columns:
            labels = labels.drop("datetime", axis=1)
        self.labels = labels["is_erupted"]

        return self

    def fit(
        self,
        seeds: int = 25,
        resample_method: Literal["under", "over", "auto"] | None = "auto",
        minority_threshold: float = 0.15,
        sampling_strategy: str | float = 0.75,
        plot_features: bool = False,
    ) -> Self:
        """Train classifier models on the full dataset across multiple random seeds.

        For each seed, resamples the extracted features, selects the top-N
        features, and fits every configured classifier via ``GridSearchCV``.
        Existing feature and model files are reused unless ``overwrite=True``.
        Populates ``self.csv`` with the registry CSV path for each classifier
        and ``self.features_selected_df`` with the aggregated top-N feature
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

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If ``build_label()`` has not been called first.
            ValueError: If ``extracted_features_df`` is empty, meaning
                ``extract_features()`` has not been called.

        Example:
            >>> model.build_label(window_step=6, window_step_unit="hours")
            >>> model.extract_features()
            >>> model.fit(seeds=25, resample_method="auto", plot_features=True)
            >>> model.csv  # {"random-forest-classifier": "path/to/trained_model_*.csv"}
        """
        if self.LabelBuilder is None:
            raise ValueError("Please run build_label() first.")

        if self.features_df.empty and self.features_csv is None:
            raise ValueError(
                "Features (matrix) dataframe (features_df) is empty. "
                "Please run extract_features() first."
            )

        self.create_directories(plot_features=plot_features)

        if resample_method == "auto":
            minority_share = (
                self.LabelBuilder.df["is_erupted"].value_counts(normalize=True).min()
            )
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

        logger.info(f"Running parallel feature selection across {seeds} seeds..")

        feature_selection_results: list[str | None] = self._run_jobs(
            self._features_selection,
            pending_feature_selection_jobs,
            job_name="Pending Feature Selection",
        )

        new_training_model_jobs: list[tuple] = []

        for feature_selection_result, pending_feature_selection_job in zip(
            feature_selection_results, pending_feature_selection_jobs, strict=True
        ):
            if feature_selection_result is None:
                continue
            _random_state = pending_feature_selection_job[0]
            for classifier_model in self.classifier_models:
                new_training_model_jobs.append(
                    (_random_state, classifier_model.slug_name)
                )

        all_training_model_jobs = pending_training_model_jobs + new_training_model_jobs
        training_model_results: list[str | None] = self._run_jobs(
            self._train,
            all_training_model_jobs,
            job_name="Pending Training",
        )

        for result in training_model_results:
            if result is None:
                continue

            classifier_slug, _random_state, features_seed_path, model_seed_path = result

            if classifier_slug not in records_per_classifier:
                records_per_classifier[classifier_slug] = []

            if features_seed_path not in self.features_csvs:
                self.features_csvs.append(features_seed_path)

            records_per_classifier[classifier_slug].append(
                {
                    "random_state": _random_state,
                    "features_csv": features_seed_path,
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
            number_of_features=self.number_of_features,
        )

        if plot_features and not self.features_selected_df.empty:
            plot_significant_features(
                df=self.features_selected_df.reset_index(),
                filepath=os.path.join(
                    self.features_dir, f"top_{self.number_of_features}_features"
                ),
                overwrite=True,
                values_column="score",
            )

        # Save registry per classifier
        for classifier_model in self.classifier_models:
            if self.verbose:
                logger.info(
                    f"Training: Saving {classifier_model.name} model registry..."
                )

            classifier_slug = classifier_model.slug_name
            if not records_per_classifier[classifier_slug]:
                continue
            self.csv[classifier_slug] = save_model_registry(
                seeds=seeds,
                records=records_per_classifier[classifier_slug],
                classifier_dir=self.classifier_dirs[classifier_slug],
                classifier_model=classifier_model,
                number_of_features=self.number_of_features,
            )

        if self.verbose:
            logger.info(f"Training: Models saved to: {self.csv}")

        return self

    def _train(
        self,
        random_state: int,
        classifier_slug: str,
    ) -> tuple | None:
        """Fit a single classifier for one random seed and persist the model.

        Reads the pre-computed top-N feature list and resampled training data
        for the given seed, then runs ``grid_search_cv`` to find the best
        estimator. Skips fitting if the model file already exists and
        ``overwrite`` is False.

        Args:
            random_state (int): Seed index identifying the feature and
                resampled data files to use.
            classifier_slug (str): Slug name of the classifier to train,
                e.g. ``"random-forest-classifier"``.

        Returns:
            tuple | None: A four-element tuple
                ``(classifier_slug, random_state, features_seed_path,
                model_seed_path)`` on success, or ``None`` when the feature
                file is missing or contains no features.
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
            m for m in self.classifier_models if m.slug_name == classifier_slug
        )

        model_seed_path = os.path.join(
            self.models_dir[classifier_slug], f"{filename}.pkl"
        )

        # Return cached result without re-training if the model already exists.
        if not self.overwrite and os.path.isfile(model_seed_path):
            logger.info(
                f"Seed {random_state:05d} / {classifier_slug}: model exists, skipping."
            )
            return classifier_slug, random_state, features_seed_path, model_seed_path

        _resampled_df = pd.read_csv(features_resampled_path, index_col=0)
        labels_resampled = _resampled_df["is_erupted"]
        features_resampled = _resampled_df.drop(columns=["is_erupted"])

        if self.verbose:
            logger.info(f"Fitting Seed: {random_state:05d} / {classifier_slug}...")

        _, _, best_model = grid_search_cv(
            random_state,
            features_resampled,
            labels_resampled,
            top_n_features,
            classifier_model=classifier_model,
        )

        joblib.dump(best_model, model_seed_path)

        if self.verbose:
            logger.info(
                f"Fitted Model {random_state:05d} / {classifier_slug} : {model_seed_path}"
            )

        return classifier_slug, random_state, features_seed_path, model_seed_path

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

        Applies the configured resampler, writes the balanced dataset to
        ``features_resampled_path``, then delegates to ``_select_features``
        to run tsfresh feature selection and persist the top-N list.

        Args:
            random_state (int): Seed used for reproducible resampling and
                feature selection.
            features_seed_path (str): Destination CSV path for the top-N
                selected feature list.
            features_resampled_path (str): Destination CSV path for the
                resampled feature-and-label DataFrame.
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

        pd.concat([features_resampled, labels_resampled], axis=1).to_csv(
            features_resampled_path
        )

        result = self._select_features(
            features=features_resampled,
            labels=labels_resampled,
            random_state=random_state,
            number_of_features=self.number_of_features,
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
                f"[{job_name}]: Running on {self.n_jobs} job(s). Grid search jobs {self.n_grids}..."
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
                  Each entry is a tuple of ``(random_state, classifier_slug)``.
                - records_per_classifier: Classifier-slug → list of record
                  dicts for seeds whose models already exist on disk.
                - existing_feature_paths: Feature CSV paths that already exist
                  on disk (can_skip seeds), collected here to avoid a post-hoc
                  filesystem rescan in ``fit()``.
        """
        pending_feature_selection_jobs: list[tuple] = []
        pending_training_model_jobs: list[tuple] = []
        existing_feature_paths: list[str] = []

        records_per_classifier: dict[str, list[dict]] = {
            classifier_model.slug_name: []
            for classifier_model in self.classifier_models
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
                for classifier_model in self.classifier_models:
                    classifier_slug = classifier_model.slug_name
                    model_seed_path = os.path.join(
                        self.models_dir[classifier_slug], f"{filename}.pkl"
                    )
                    if os.path.isfile(model_seed_path):
                        records_per_classifier[classifier_slug].append(
                            {
                                "random_state": random_state,
                                "features_csv": features_seed_path,
                                "model_filepath": model_seed_path,
                            }
                        )
                    else:
                        pending_training_model_jobs.append(
                            (random_state, classifier_slug)
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
