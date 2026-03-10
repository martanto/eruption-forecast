import os
import json
import warnings
from typing import Any, Self, Literal
from collections.abc import Callable

import numpy as np
import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, make_scorer, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import (
    merge_seed_models,
    load_labels_from_csv,
    random_under_sampler,
    merge_all_classifiers,
)
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.config.constants import (
    TRAIN_TEST_SPLIT,
    DEFAULT_CV_SPLITS,
    LEARNING_CURVE_SCORINGS,
    DEFAULT_N_SIGNIFICANT_FEATURES,
)
from eruption_forecast.features.constants import (
    SIGNIFICANT_FEATURES_FILENAME,
)
from eruption_forecast.plots.feature_plots import (
    plot_significant_features as _plot_significant_features,
)
from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.features.feature_selector import FeatureSelector


def _safe_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute recall, returning 0.0 when class 1 is absent from y_true.

    Guards against sklearn's pos_label validation error that fires before
    zero_division is considered — common in small CV folds with class imbalance.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: Recall score, or 0.0 if class 1 is not present in y_true.
    """
    if 1 not in y_true:
        return 0.0
    return recall_score(y_true, y_pred, zero_division=0)


def _safe_f1_weighted(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute weighted F1, returning 0.0 when only one class is present in y_true.

    Guards against degenerate CV folds where the training slice contains only
    one class after random under-sampling at small training sizes.

    Args:
        y_true (np.ndarray): True binary labels.
        y_pred (np.ndarray): Predicted binary labels.

    Returns:
        float: Weighted F1 score, or 0.0 if fewer than two classes are present.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


# Module-level callables are picklable by loky workers (unlike lambdas or
# make_scorer objects built inside __init__).
_LEARNING_CURVE_SCORER_MAP: dict[str, str | Any] = {
    "balanced_accuracy": "balanced_accuracy",
    "recall": make_scorer(_safe_recall),
    "f1_weighted": make_scorer(_safe_f1_weighted),
}


class ModelTrainer:
    """Train feature-selection and classifier models over multiple random seeds.

    Loads pre-extracted features and labels, then for each random seed performs:

    1. Train/test split (80/20, stratified) to prevent data leakage
    2. Random under-sampling on training set only to balance classes
    3. Feature selection on training set using tsfresh relevance filtering (ONCE per seed)
    4. For each classifier: GridSearchCV training and cross-validation
    5. Evaluation on held-out test set (when using train_and_evaluate)

    Under-sampling and feature selection are shared across all classifiers for
    each seed, eliminating redundant computation when training multiple classifiers.

    Attributes:
        df_features (pd.DataFrame): Loaded features DataFrame (from extracted_features_csv).
        df_labels (pd.Series): Loaded labels Series (from label_features_csv).
        output_dir (str): Root directory for training outputs.
        n_jobs (int): Number of parallel seed workers (outer loop).
        grid_search_n_jobs (int): Number of parallel jobs inside each GridSearchCV call.
        prefix_filename (str | None): Optional prefix for output filenames.
        classifiers (list[str]): Normalised list of classifier keys passed at construction.
        cv_strategy (str): Cross-validation strategy.
        cv_splits (int): Number of CV splits.
        number_of_significant_features (int): Number of top features to save separately.
        feature_selection_method (str): Feature selection method ("tsfresh", "random_forest", "combined").
        overwrite (bool): Whether to overwrite existing output files.
        verbose (bool): Enable verbose logging.
        debug (bool): Enable debug mode.
        FeatureSelector (FeatureSelector): Feature selection component instance.
        classifier_models (list[ClassifierModel]): One ClassifierModel per classifier key.
        shared_features_dir (str): Shared directory for feature outputs (all classifiers).
        shared_significant_dir (str): Shared directory for significant features.
        shared_all_features_dir (str): Shared directory for all features.
        shared_figures_dir (str): Shared directory for feature importance plots.
        shared_tests_dir (str): Shared directory for per-seed held-out test splits.
        classifier_dirs (dict[str, str]): Per-classifier output directories keyed by slug.
        models_dirs (dict[str, str]): Per-classifier model directories keyed by slug.
        metrics_dirs (dict[str, str]): Per-classifier metrics directories keyed by slug.
        significant_features_csvs (list[str]): Paths to per-seed significant features CSVs.
        df_significant_features (pd.DataFrame): Aggregated significant features across seeds.
        df (dict[str, pd.DataFrame]): Model registry DataFrames keyed by classifier slug.
        csv (dict[str, str]): Paths to saved model registry CSVs keyed by classifier slug.

    Args:
        extracted_features_csv (str): Path to the extracted features CSV file.
        label_features_csv (str): Path to the label CSV file built by LabelBuilder.
        output_dir (str | None, optional): Directory for output files. Defaults to None.
        root_dir (str | None, optional): Anchor directory for resolving relative paths.
            Defaults to None.
        prefix_filename (str | None, optional): Prefix for output filenames.
            Defaults to None.
        classifiers (str | list[str], optional): Classifier key or list of classifier
            keys to train. Defaults to ``"rf"``.
        cv_strategy (Literal[...], optional): Cross-validation strategy.
            Defaults to "shuffle-stratified".
        cv_splits (int, optional): Number of CV splits. Defaults to DEFAULT_CV_SPLITS.
        number_of_significant_features (int, optional): Number of top features to save.
            Defaults to DEFAULT_N_SIGNIFICANT_FEATURES.
        feature_selection_method (Literal[...], optional): Feature selection method.
            Defaults to "tsfresh".
        overwrite (bool, optional): Overwrite existing output files. Defaults to False.
        plot_shap (bool, optional): Compute and save SHAP summary plots after training.
            Defaults to False.
        n_jobs (int, optional): Number of parallel seed workers. Defaults to 1.
        grid_search_n_jobs (int, optional): Parallel jobs inside GridSearchCV.
            Defaults to 1.
        use_gpu (bool, optional): Enable GPU acceleration for XGBoost. Defaults to False.
        gpu_id (int, optional): GPU device index when use_gpu is True. Defaults to 0.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        debug (bool, optional): Enable debug mode. Defaults to False.

    Raises:
        ValueError: If features or labels are empty, or their lengths do not match.
        ValueError: If n_jobs is <= 0.

    Examples:
        >>> # Train with shared feature selection across rf and xgb
        >>> trainer = ModelTrainer(
        ...     extracted_features_csv="output/features/extracted_features.csv",
        ...     label_features_csv="output/features/label_features.csv",
        ...     classifiers=["rf", "xgb"],
        ...     output_dir="output/trainings",
        ...     n_jobs=4,
        ... )
        >>> trainer.fit(with_evaluation=True, random_state=0, total_seed=100)
    """

    def __init__(
        self,
        extracted_features_csv: str,
        label_features_csv: str,
        output_dir: str | None = None,
        root_dir: str | None = None,
        prefix_filename: str | None = None,
        classifiers: str | list[str] = "rf",
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified", "timeseries"
        ] = "shuffle-stratified",
        cv_splits: int = DEFAULT_CV_SPLITS,
        number_of_significant_features: int = DEFAULT_N_SIGNIFICANT_FEATURES,
        feature_selection_method: Literal[
            "tsfresh", "random_forest", "combined"
        ] = "tsfresh",
        overwrite: bool = False,
        plot_shap: bool = False,
        n_jobs: int = 1,
        grid_search_n_jobs: int = 1,
        use_gpu: bool = False,
        gpu_id: int = 0,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize ModelTrainer with feature data, classifier settings, and output paths.

        Reads the extracted features CSV and label features CSV into DataFrames,
        constructs ClassifierModel instances for each classifier key, resolves the
        output directory hierarchy (shared and per-classifier), and stores all
        training configuration. No training occurs until fit() is called.

        Args:
            extracted_features_csv (str): Path to the tsfresh-extracted features CSV
                (index column is the window ID).
            label_features_csv (str): Path to the aligned label CSV produced by
                FeaturesBuilder. Must contain 'id' and 'is_erupted' columns.
            output_dir (str | None, optional): Base output directory for training
                artefacts. Defaults to ``root_dir/output/trainings/evaluations``.
                Defaults to None.
            root_dir (str | None, optional): Anchor directory for relative path
                resolution. Defaults to None (uses os.getcwd()).
            prefix_filename (str | None, optional): Custom prefix for output filenames.
                Defaults to None.
            classifiers (str | list[str], optional): Classifier key or list of
                classifier keys to train. Defaults to ``"rf"``.
            cv_strategy (Literal["shuffle", "stratified", "shuffle-stratified", "timeseries"], optional):
                Cross-validation strategy. Defaults to "shuffle-stratified".
            cv_splits (int, optional): Number of cross-validation folds.
                Defaults to DEFAULT_CV_SPLITS.
            number_of_significant_features (int, optional): Number of top features
                to retain after selection. Defaults to DEFAULT_N_SIGNIFICANT_FEATURES.
            feature_selection_method (Literal["tsfresh", "random_forest", "combined"],
                optional): Feature selection method. Defaults to "tsfresh".
            overwrite (bool, optional): Overwrite existing output files. Defaults to False.
            n_jobs (int, optional): Number of parallel seed workers (outer loop).
                Defaults to 1.
            grid_search_n_jobs (int, optional): Number of parallel jobs inside each
                GridSearchCV call (inner loop). Defaults to 1.
            verbose (bool, optional): Emit progress log messages. Defaults to False.
            debug (bool, optional): Emit debug log messages. Defaults to False.
            use_gpu (bool, optional): Enable GPU acceleration for XGBoost. Defaults to False.
            gpu_id (int, optional): GPU device index to use when use_gpu is True.
                Defaults to 0.
        """
        # Normalise to list[str] regardless of whether a single string was passed
        _classifiers: list[str] = (
            [classifiers] if isinstance(classifiers, str) else classifiers
        )

        _xgb_classifiers = {"xgb", "voting"}
        _has_xgb = any(c in _xgb_classifiers for c in _classifiers)
        if use_gpu and _has_xgb and n_jobs != 1:
            logger.warning(
                "use_gpu=True forces n_jobs=1 to avoid GPU memory contention "
                "from parallel seed workers sharing the same device."
            )
            n_jobs = 1

        df_features = pd.read_csv(extracted_features_csv, index_col=0)
        df_labels = load_labels_from_csv(label_features_csv)

        # Build one ClassifierModel per classifier key
        classifier_models: list[ClassifierModel] = [
            ClassifierModel(
                classifier=classifier,  # ty:ignore[invalid-argument-type]
                cv_strategy=cv_strategy,
                n_splits=cv_splits,
                verbose=verbose,
                use_gpu=use_gpu and classifier in _xgb_classifiers,
                gpu_id=gpu_id,
            )
            for classifier in _classifiers
        ]

        # Output training dir: ``<root_dir>/output/trainings``
        output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output"),
        )

        # Base evaluations dir: ``<output_dir>/trainings/evaluations``
        _output_dir = os.path.join(output_dir, "trainings", "evaluations")

        # CV slug is shared across all classifiers (same CV strategy)
        _cv_slug = classifier_models[0].slug_cv_name

        # Shared feature directories (under the CV slug, not per classifier)
        shared_features_dir = os.path.join(_output_dir, "features", _cv_slug)
        shared_significant_dir = os.path.join(
            shared_features_dir, "significant_features"
        )
        shared_all_features_dir = os.path.join(shared_features_dir, "all_features")
        shared_figures_dir = os.path.join(shared_features_dir, "figures", "significant")
        shared_tests_dir = os.path.join(shared_features_dir, "tests")

        # Per-classifier directories keyed by classifier slug
        classifier_dirs: dict[str, str] = {}
        models_dirs: dict[str, str] = {}
        metrics_dirs: dict[str, str] = {}
        for classifier_model in classifier_models:
            slug = classifier_model.slug_name
            classifier_dir = os.path.join(_output_dir, "classifiers", slug, _cv_slug)
            classifier_dirs[slug] = classifier_dir
            models_dirs[slug] = os.path.join(classifier_dir, "models")
            metrics_dirs[slug] = os.path.join(classifier_dir, "metrics")

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.df_features = df_features
        self.df_labels = df_labels
        self.output_dir = _output_dir
        self.n_jobs = n_jobs
        self.grid_search_n_jobs = grid_search_n_jobs
        self.prefix_filename = prefix_filename
        self.classifiers = _classifiers
        self.classifier_models = classifier_models
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.number_of_significant_features: int = number_of_significant_features
        self.feature_selection_method = feature_selection_method
        self.plot_shap = plot_shap
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self.FeatureSelector = FeatureSelector(
            method=feature_selection_method, verbose=verbose, n_jobs=grid_search_n_jobs
        )

        # Shared directories
        self.shared_features_dir = shared_features_dir
        self.shared_significant_dir = shared_significant_dir
        self.shared_all_features_dir = shared_all_features_dir
        self.shared_figures_dir = shared_figures_dir
        self.shared_tests_dir = shared_tests_dir

        # Per-classifier directories
        self.classifier_dirs = classifier_dirs
        self.models_dirs = models_dirs
        self.metrics_dirs = metrics_dirs

        # ------------------------------------------------------------------
        # Will be set after train_and_evaluate() method called
        # ------------------------------------------------------------------
        self.significant_features_csvs: list[str] = []
        self.df_significant_features: pd.DataFrame = pd.DataFrame()

        # Will be set after train_and_evaluate() or train() called
        # Keyed by classifier slug
        self.df: dict[str, pd.DataFrame] = {}
        self.csv: dict[str, str] = {}

        # ------------------------------------------------------------------
        # Validate and create directories
        # ------------------------------------------------------------------
        self.validate()

        # ------------------------------------------------------------------
        # Verbose and logging
        # ------------------------------------------------------------------
        if verbose:
            logger.info(
                f"Train model using {n_jobs} jobs with {', '.join(self.classifiers)} classifier(s) "
                f"and {cv_strategy} CV strategy ({cv_splits} splits)"
            )

    def validate(self) -> None:
        """Validate that features and labels are non-empty and aligned.

        Checks that both features and labels DataFrames contain data
        and that they have matching row counts (same number of windows).

        Raises:
            ValueError: If features DataFrame is empty (0 rows).
            ValueError: If labels DataFrame is empty (0 rows).
            ValueError: If the number of rows in features and labels do not match.
            ValueError: If n_jobs or grid_search_n_jobs is zero or negative (excluding -1).
            ValueError: If n_jobs × grid_search_n_jobs exceeds the available CPU cores.

        Example:
            >>> trainer = ModelTrainer(
            ...     extracted_features_csv="features.csv",
            ...     label_features_csv="labels.csv",
            ... )
            >>> trainer.validate()  # Called automatically in __init__
        """
        len_features = self.df_features.shape[0]
        len_labels = self.df_labels.shape[0]
        logger.debug(
            f"Validating: {len_features} feature rows, {len_labels} label rows"
        )

        if len_features == 0:
            raise ValueError("Features cannot be empty. Check your features CSV file.")
        if len_labels == 0:
            raise ValueError("Labels cannot be empty. Check your labels CSV file.")
        if len_features != len_labels:
            raise ValueError(
                f"Length of features and labels do not match. "
                f"Length of features: {len_features}, labels: {len_labels}. "
                f"Features CSV should be located under output/_nslc_/features, "
                f"with filename starting with extracted_features_(start_date)_(end_date).csv or "
                f"extracted_relevant_(start_date)_(end_date).csv"
            )
        total_cores: int = os.cpu_count() or 1

        # Resolve -1 to actual core count (joblib convention)
        effective_n_jobs = total_cores if self.n_jobs == -1 else self.n_jobs
        effective_gs_jobs = (
            total_cores if self.grid_search_n_jobs == -1 else self.grid_search_n_jobs
        )

        if effective_n_jobs <= 0:
            raise ValueError(
                f"n_jobs must be -1 or a positive integer. Got {self.n_jobs}."
            )
        if effective_gs_jobs <= 0:
            raise ValueError(
                f"grid_search_n_jobs must be -1 or a positive integer. Got {self.grid_search_n_jobs}."
            )
        if effective_n_jobs * effective_gs_jobs > total_cores:
            raise ValueError(
                f"n_jobs ({effective_n_jobs}) × grid_search_n_jobs ({effective_gs_jobs}) = "
                f"{effective_n_jobs * effective_gs_jobs} exceeds available cores ({total_cores}). "
                f"Reduce n_jobs or grid_search_n_jobs so their product is ≤ {total_cores}."
            )

    @staticmethod
    def get_classifier_properties(
        classifier_model: ClassifierModel,
    ) -> tuple[str, str, str, str]:
        """Extract name, slug, and identifier strings from a ClassifierModel.

        Convenience method to extract various string representations of the
        classifier and CV strategy for use in filenames and directory paths.

        Args:
            classifier_model (ClassifierModel): ClassifierModel instance to inspect.

        Returns:
            tuple[str, str, str, str]: A 4-tuple containing:

                - **classifier_name** (str): Class name (e.g., "RandomForestClassifier")
                - **classifier_slug_name** (str): Slugified classifier name (e.g., "random-forest-classifier")
                - **classifier_slug_cv_name** (str): Slugified CV name (e.g., "stratified-k-fold")
                - **classifier_id** (str): Combined identifier (e.g., "RandomForestClassifier-StratifiedKFold")

        Examples:
            >>> name, slug, cv_slug, id_ = trainer.get_classifier_properties(classifier_model)
            >>> print(name)  # "RandomForestClassifier"
            >>> print(slug)  # "random-forest-classifier"
        """
        classifier_name: str = classifier_model.name
        classifier_slug_name: str = classifier_model.slug_name
        classifier_cv_name: str = classifier_model.cv_name

        classifier_slug_cv_name: str = classifier_model.slug_cv_name
        classifier_id = f"{classifier_name}-{classifier_cv_name}"

        return (
            classifier_name,
            classifier_slug_name,
            classifier_slug_cv_name,
            classifier_id,
        )

    def update_directories(
        self, output_dir: str | None = None, root_dir: str | None = None
    ) -> Self:
        """Recompute and update all subdirectory paths from a new output directory.

        Resolves the new output directory using the same rules as ``__init__``,
        then rebuilds every derived path (classifier, models, metrics, features,
        figures). Calls ``create_directories()`` to ensure the paths exist.

        Args:
            output_dir (str | None, optional): New base output directory. Resolved
                against ``root_dir`` (or ``os.getcwd()`` when None). Defaults to None.
            root_dir (str | None, optional): Anchor directory for resolving relative
                ``output_dir`` values. Defaults to None.

        Returns:
            Self: The ModelTrainer instance for method chaining.

        Examples:
            >>> trainer = ModelTrainer(...)
            >>> trainer.update_directories(output_dir="new_output", root_dir="/project")
        """
        self.output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output", "trainings", "evaluations"),
        )

        _cv_slug = self.classifier_models[0].slug_cv_name

        # Rebuild shared feature directories
        self.shared_features_dir = os.path.join(self.output_dir, "features", _cv_slug)
        self.shared_significant_dir = os.path.join(
            self.shared_features_dir, "significant_features"
        )
        self.shared_all_features_dir = os.path.join(
            self.shared_features_dir, "all_features"
        )
        self.shared_figures_dir = os.path.join(
            self.shared_features_dir, "figures", "significant"
        )
        self.shared_tests_dir = os.path.join(self.shared_features_dir, "tests")

        # Rebuild per-classifier directories
        for classifier_model in self.classifier_models:
            slug = classifier_model.slug_name
            classifier_dir = os.path.join(
                self.output_dir, "classifiers", slug, _cv_slug
            )
            self.classifier_dirs[slug] = classifier_dir
            self.models_dirs[slug] = os.path.join(classifier_dir, "models")
            self.metrics_dirs[slug] = os.path.join(classifier_dir, "metrics")

        self.create_directories()

        return self

    def update_grid_params(
        self, classifier: ClassifierModel, grid_params: dict[str, Any]
    ) -> ClassifierModel:
        """Update the hyperparameter grid for a classifier.

        Replaces the current grid with new parameters and logs the change
        if verbose mode is enabled.

        Args:
            classifier (ClassifierModel): Classifier model instance to update.
            grid_params (dict[str, Any]): New grid search parameter dictionary.

        Returns:
            ClassifierModel: The updated classifier instance (same object, modified in-place).

        Examples:
            >>> new_grid = {"n_estimators": [100, 200], "max_depth": [10, 20]}
            >>> trainer.update_grid_params(trainer.classifier_models[0], new_grid)
        """
        current_grid = classifier.grid
        classifier.grid = grid_params
        if self.verbose:
            logger.info(
                f"Grid parameters updated from {current_grid} to {grid_params}."
            )

        return classifier

    def create_directories(self) -> None:
        """Create required output directories for training results.

        Creates shared feature directories and per-classifier model/metric
        directories. Called at the start of ``train_and_evaluate()``, ``train()``,
        and ``update_directories()``.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> trainer.create_directories()
        """
        ensure_dir(self.output_dir)
        ensure_dir(self.shared_significant_dir)
        ensure_dir(self.shared_tests_dir)
        ensure_dir(self.shared_figures_dir)
        for slug in self.classifier_dirs:
            ensure_dir(self.models_dirs[slug])

    def concat_significant_features(self, plot: bool = False) -> pd.DataFrame:
        """Concatenate significant features from all training seeds.

        Merges significant feature CSVs from all random seeds, aggregates
        by feature name to count occurrences across seeds, and saves the
        top-N most frequently selected features.

        Args:
            plot (bool, optional): If True, generates a plot of the top
                features. Defaults to False.

        Returns:
            pd.DataFrame: Concatenated significant features with counts of
                how many times each feature appeared across all seeds.

        Raises:
            ValueError: If no CSV files are found in self.significant_features_csvs.
            ValueError: If concatenated DataFrame is empty.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> trainer.train_and_evaluate(total_seed=100)
            >>> _df = trainer.concat_significant_features(plot=True)
            >>> print(_df.head())
        """
        number_of_significant_features = self.number_of_significant_features

        if len(self.significant_features_csvs) == 0:
            raise ValueError(
                f"No significant features CSV file inside directory {self.shared_significant_dir}"
            )

        combined_features_df = pd.concat(
            [pd.read_csv(file) for file in self.significant_features_csvs],
            ignore_index=True,
        )

        if combined_features_df.empty:
            raise ValueError("No data found inside csv files.")

        filename = (
            f"{self.prefix_filename}_{SIGNIFICANT_FEATURES_FILENAME}"
            if self.prefix_filename
            else SIGNIFICANT_FEATURES_FILENAME
        )
        combined_features_df.to_csv(
            os.path.join(self.shared_features_dir, f"{filename}.csv"), index=False
        )

        # Save number_of_significant_features
        if (
            number_of_significant_features is not None
            and number_of_significant_features > 0
        ):
            filename = (
                f"{self.prefix_filename}_top_{number_of_significant_features}_{SIGNIFICANT_FEATURES_FILENAME}"
                if self.prefix_filename
                else f"top_{number_of_significant_features}_{SIGNIFICANT_FEATURES_FILENAME}"
            )

            combined_features_df = (
                combined_features_df.groupby(by="features")
                .count()
                .sort_values(by="p_values", ascending=False)
            )
            combined_features_df.index.name = "features"
            combined_features_df.to_csv(
                os.path.join(self.shared_features_dir, f"{filename}.csv"),
                index=True,
            )

            if plot:
                _plot_significant_features(
                    df=combined_features_df.reset_index(),
                    filepath=os.path.join(self.shared_features_dir, filename),
                    overwrite=True,
                )

        return combined_features_df

    @logger.catch(level="ERROR")
    def select_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        significant_filepath: str,
        random_state: int = 42,
        all_features_filepath: str | None = None,
        all_figures_filepath: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, int] | None:
        """Select the most predictive features from a resampled training set.

        Uses tsfresh statistical significance testing with FDR control
        or permutation importance analysis, depending on ``self.feature_selection_method``.

        Args:
            features (pd.DataFrame): Resampled features to select.
            labels (pd.Series): Target labels.
            significant_filepath (str): Save path for significant features.
            random_state (int, optional): Random seed for feature selection. Defaults to 42.
            all_features_filepath (str, optional): Save path for all features.
            all_figures_filepath (str, optional): Save path for all features figures.

        Returns:
            tuple[pd.DataFrame, pd.Series, pd.Series, int]: Selected features dataframe,
                top (n) features, all selected features, (n) features
            None: If features reduced to zero after selection.
        """
        selector = self.FeatureSelector.set_random_state(random_state)
        number_of_significant_features = self.number_of_significant_features

        # Reduced features/columns
        df_selected_features = selector.fit_transform(
            features, labels, top_n=number_of_significant_features
        )

        if selector.n_features == 0:
            logger.warning(
                f"{random_state:05d}: Features reduced to 0. Skip training model."
            )
            return None

        # Series indexed by feature name; values are p-values or importance score sorted by it's value.
        all_selected_features = selector.selected_features_

        # Handle if columns in df_selected_features has less than number_of_significant_features
        len_features_columns = len(df_selected_features.columns)
        if len_features_columns < number_of_significant_features:
            logger.warning(
                f"{random_state:05d}: Number of features after extracted ({len_features_columns}) "
                f"are less than {number_of_significant_features} features."
            )
            number_of_significant_features = len_features_columns

        top_selected_features = all_selected_features.head(
            number_of_significant_features
        )

        # Save TOP-N significant features
        # significant_filepath Will be used in ModelEvaluator.from_files(...) method
        # as selected_features_path paramater
        top_selected_features.to_csv(significant_filepath, index=True)

        # Save all SELECTED features if requested
        if all_features_filepath:
            all_selected_features.to_csv(all_features_filepath, index=True)
            if all_figures_filepath:
                self._plot_all_significant_features(
                    all_selected_features=pd.DataFrame(
                        all_selected_features
                    ).reset_index(),
                    all_figures_filepath=all_figures_filepath,
                    top_features=number_of_significant_features,
                )

        return (
            df_selected_features,
            top_selected_features,
            all_selected_features,
            number_of_significant_features,
        )

    def _plot_all_significant_features(
        self,
        all_selected_features: pd.DataFrame,
        all_figures_filepath: str,
        top_features: int | None = None,
    ):
        """Plot the significant feature importances accumulated across all seeds.

        Delegates to the module-level _plot_significant_features helper with the
        configured number_of_significant_features as the default top_features limit.

        Args:
            all_selected_features (pd.DataFrame): DataFrame of feature importance
                scores aggregated across all random seeds.
            all_figures_filepath (str): Full file path where the plot image is saved.
            top_features (int | None, optional): Number of top features to display.
                If None, uses self.number_of_significant_features. Defaults to None.

        Returns:
            None
        """
        _plot_significant_features(
            df=all_selected_features,
            filepath=all_figures_filepath,
            top_features=top_features or self.number_of_significant_features,
            overwrite=self.overwrite,
            dpi=150,
        )

        return None

    def _generate_shared_filepaths(
        self,
        random_state: int,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> tuple[str, str, str | None, str | None, str, str, bool]:
        """Generate shared filepaths that are the same for all classifiers in a seed.

        These paths cover feature selection outputs, test data, and learning curve prefix.
        All classifiers in the same seed share the same under-sampling and feature
        selection results, so these paths are written once.

        Args:
            random_state (int): Random seed for this training run.
            save_features (bool, optional): Whether all-features output is requested.
                Defaults to False.
            plot_features (bool, optional): Whether feature plots are requested.
                Defaults to False.

        Returns:
            tuple: A 7-tuple of:

                - **filename** (str): Base filename stem for this seed.
                - **significant_filepath** (str): Path to significant features CSV.
                - **all_features_filepath** (str | None): Path to all features CSV, or None.
                - **all_figures_filepath** (str | None): Path to feature plot PNG, or None.
                - **X_test_filepath** (str): Path to held-out X_test CSV.
                - **y_test_filepath** (str): Path to held-out y_test CSV.
                - **can_skip_shared** (bool): True if all shared outputs already exist.
        """
        filename = (
            f"{self.prefix_filename}_{random_state:05d}"
            if self.prefix_filename
            else f"{random_state:05d}"
        )

        significant_filepath = os.path.join(
            self.shared_significant_dir, f"{filename}.csv"
        )
        all_features_filepath = (
            os.path.join(self.shared_all_features_dir, f"{filename}.csv")
            if save_features
            else None
        )
        all_figures_filepath = (
            os.path.join(self.shared_figures_dir, f"{filename}.png")
            if plot_features
            else None
        )
        X_test_filepath = os.path.join(self.shared_tests_dir, f"{filename}_X_test.csv")
        y_test_filepath = os.path.join(self.shared_tests_dir, f"{filename}_y_test.csv")

        shared_required = [significant_filepath]
        can_skip_shared = not self.overwrite and all(
            os.path.isfile(p) for p in shared_required
        )

        return (
            filename,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            X_test_filepath,
            y_test_filepath,
            can_skip_shared,
        )

    def _generate_classifier_filepaths(
        self,
        random_state: int,
        classifier_slug: str,
    ) -> tuple[str, str, str, str]:
        """Generate per-classifier filepaths for a given seed and classifier.

        These paths are unique to each (seed, classifier) combination and cover
        the trained model pickle, per-seed metrics JSON, SHAP explanation cache,
        and learning curve JSON.

        Args:
            random_state (int): Random seed for this training run.
            classifier_slug (str): Slugified classifier name used to locate the
                correct output subdirectory.

        Returns:
            tuple: A 4-tuple of:

                - **model_filepath** (str): Path to the trained model pickle.
                - **metrics_filepath** (str): Path to per-seed metrics JSON.
                - **shap_explanation_filepath** (str): Path to SHAP cache pickle.
                - **learning_curve_path** (str): Path to learning curve JSON.
        """
        filename = (
            f"{self.prefix_filename}_{random_state:05d}"
            if self.prefix_filename
            else f"{random_state:05d}"
        )
        models_dir = self.models_dirs[classifier_slug]
        metrics_dir = self.metrics_dirs[classifier_slug]

        model_filepath = os.path.join(models_dir, f"{filename}.pkl")
        metrics_filepath = os.path.join(metrics_dir, f"{filename}.json")
        shap_explanation_filepath = os.path.join(
            metrics_dir, "shap", f"{random_state:05d}.pkl"
        )
        learning_curve_path = os.path.join(
            metrics_dir, "learning_curve", f"{filename}.json"
        )

        return (
            model_filepath,
            metrics_filepath,
            shap_explanation_filepath,
            learning_curve_path,
        )

    def _setup_grid_search(
        self,
        random_state: int,
        features: pd.DataFrame,
        labels: pd.Series,
        top_n_features: list[str],
        classifier_model: ClassifierModel,
    ) -> tuple[ClassifierModel, GridSearchCV, Any]:
        """Set up, fit, and return a GridSearchCV with the configured classifier.

        Args:
            random_state (int): Random seed for the classifier.
            features (pd.DataFrame): Training features (unsliced).
            labels (pd.Series): Training labels.
            top_n_features (list[str]): Column names to select from features.
            classifier_model (ClassifierModel): ClassifierModel instance to use.

        Returns:
            tuple: Configured classifier, fitted GridSearchCV, best estimator.
        """
        classifier: ClassifierModel = classifier_model.set_random_state(
            random_state=random_state
        )

        # Force n_jobs=1 for GPU classifiers: parallel CV fold workers would
        # each try to use the same GPU device simultaneously, causing VRAM contention.
        # FeatureSelector (CPU-only) is unaffected and keeps self.grid_search_n_jobs.
        _gs_n_jobs = 1 if classifier.use_gpu else self.grid_search_n_jobs

        grid_search = GridSearchCV(
            estimator=classifier.model,
            param_grid=classifier.grid,
            cv=classifier.get_cv_splitter(),
            scoring="balanced_accuracy",
            n_jobs=_gs_n_jobs,
            verbose=0,
        )

        # Force loky backend to avoid the threading backend that Intel's
        # scikit-learn extension (sklearnex) does not support.
        with joblib.parallel_backend("loky"):
            grid_search.fit(features[top_n_features], labels)

        return classifier, grid_search, grid_search.best_estimator_

    def _run_jobs(self, method: Callable, jobs: list[tuple]) -> list:
        """Dispatch jobs sequentially or in parallel depending on n_jobs.

        Uses joblib's loky backend so that nested parallelism inside each worker
        (e.g. GridSearchCV with grid_search_n_jobs > 1) is safe and free of
        deadlocks that would occur with multiprocessing.Pool.

        Args:
            method (Callable): The function to call for each job.
            jobs (list[tuple]): List of argument tuples, one per job.

        Returns:
            list: Collected return values in submission order.
        """
        if self.n_jobs != 1:
            logger.info(f"Running on {self.n_jobs} job(s)")
            return Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(method)(*job) for job in jobs
            )
        return [method(*job) for job in jobs]

    def _save_models_registry(
        self,
        records: list[dict],
        random_state: int,
        total_seed: int,
        classifier_slug: str,
    ) -> str:
        """Build and save the trained-models registry CSV for one classifier.

        Save model registry DataFrame with filename format:
        trained_model_{classifier_name}_rs-{random_state}_ts-{total_seed}_top-{number_of_significant_features}.csv

        Example Filename:
            trained_model_RandomForestClassifier-StratifiedShuffleSplit_rs-0_ts-500_top-20.csv

        Args:
            records (list[dict]): One dict per seed with keys
                ``random_state``, ``significant_features_csv``,
                ``trained_model_filepath``, and (for ``train_and_evaluate`` only)
                ``X_test_filepath`` and ``y_test_filepath``.
            random_state (int): Initial random state used for this run.
            total_seed (int): Total number of seeds used for this run.
            classifier_slug (str): Slugified classifier name used to locate the correct
                output directory and build the registry filename.

        Returns:
            str: The suffix string used in the output filename.

        Raises:
            ValueError: If no records were produced (no models trained).
        """
        classifier_dir = self.classifier_dirs[classifier_slug]
        # Build the classifier_id from the matching ClassifierModel
        _classifier_model = next(
            m for m in self.classifier_models if m.slug_name == classifier_slug
        )
        classifier_id = f"{_classifier_model.name}-{_classifier_model.cv_name}"

        suffix = (
            f"{classifier_id}_rs-{random_state}_ts-{total_seed}"
            f"_top-{self.number_of_significant_features}"
        )
        filename = f"trained_model_{suffix}.csv"

        registry_df = pd.DataFrame(records).set_index("random_state")
        if registry_df.empty:
            raise ValueError("No significant features or trained models found.")

        csv = os.path.join(classifier_dir, filename)
        registry_df.to_csv(csv, index=True)

        self.df[classifier_slug] = registry_df
        self.csv[classifier_slug] = csv

        return suffix

    @staticmethod
    def _split_and_resample(
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int,
        sampling_strategy: str | float = 0.75,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Steps 1-2: stratified train/test split then RandomUnderSampler on train.

        Saves the held-out test split to disk for later aggregate evaluation.

        Args:
            X (pd.DataFrame): Full feature matrix.
            y (pd.Series): Full label series.
            random_state (int): Random seed for both split and sampler.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A 4-tuple of
                (X_train_resampled, X_test, y_train_resampled, y_test) where
                X_train_resampled and y_train_resampled are under-sampled training
                data and X_test / y_test are the held-out test split.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TRAIN_TEST_SPLIT,
            random_state=random_state,
            stratify=y,
        )

        X_train_resampled, y_train_resampled = random_under_sampler(
            features=X_train,
            labels=y_train,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
        return X_train_resampled, X_test, y_train_resampled, y_test

    def _select_features_for_seed(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
        significant_filepath: str,
        all_features_filepath: str | None = None,
        all_figures_filepath: str | None = None,
    ) -> tuple | None:
        """Step 3: feature selection on the resampled training data.

        Delegates to :meth:`select_features` and returns its result unchanged.

        Args:
            X_train (pd.DataFrame): Resampled training features.
            y_train (pd.Series): Resampled training labels.
            random_state (int): Random seed for feature selector.
            significant_filepath (str): Path to save top-N features CSV.
            all_features_filepath (str | None, optional): Path to save all features.
            all_figures_filepath (str | None, optional): Path to save feature plots.

        Returns:
            tuple | None: Result from :meth:`select_features`, or None if no
                features survived selection.
        """
        return self.select_features(
            features=X_train,
            labels=y_train,
            random_state=random_state,
            significant_filepath=significant_filepath,
            all_features_filepath=all_features_filepath,
            all_figures_filepath=all_figures_filepath,
        )

    def compute_learning_curve(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
        scoring: str = "balanced_accuracy",
    ) -> dict:
        """Compute a learning curve for a single scoring metric.

        Evaluates ``estimator`` at ten linearly spaced training-set fractions
        (0.1 → 1.0) using the configured CV splitter and the given ``scoring``
        string.  Mean and standard deviation of train and validation scores are
        computed across CV folds and returned as a dict (no file I/O).

        Args:
            estimator (Any): Fitted or unfitted sklearn-compatible estimator.
            X_train (pd.DataFrame): Training features (post-selection).
            y_train (pd.Series): Training labels.
            random_state (int): Random seed used to configure the CV splitter.
            scoring (str, optional): Sklearn scoring string. Defaults to
                ``"balanced_accuracy"``.

        Returns:
            dict: Dict with keys ``train_sizes``, ``train_scores_mean``,
                ``train_scores_std``, ``test_scores_mean``, ``test_scores_std``.
        """
        # StratifiedKFold ensures every fold contains both classes, preventing
        # the single-label warning that occurs when small train_sizes + class
        # imbalance leave only one class in a fold.
        cv = StratifiedKFold(
            n_splits=self.cv_splits, shuffle=True, random_state=random_state
        )

        # Force n_jobs=1: already inside a loky parallel worker (outer seed loop).
        # Suppress the single-label UserWarning: at very small training-size
        # fractions (≤20%) the model may predict only one class, causing
        # balanced_accuracy_score to warn about a single label in y_pred.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A single label was found",
                category=UserWarning,
            )
            train_sizes_abs, train_scores, test_scores = learning_curve(
                estimator=estimator,
                X=X_train,
                y=y_train,
                cv=cv,
                scoring=_LEARNING_CURVE_SCORER_MAP.get(scoring, scoring),
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=1,
            )

        return {
            "train_sizes": train_sizes_abs.tolist(),
            "train_scores_mean": train_scores.mean(axis=1).tolist(),
            "train_scores_std": train_scores.std(axis=1).tolist(),
            "test_scores_mean": test_scores.mean(axis=1).tolist(),
            "test_scores_std": test_scores.std(axis=1).tolist(),
        }

    def _compute_all_learning_curves(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
        filepath: str,
    ) -> dict:
        """Compute learning curves for all configured scoring metrics and save as JSON.

        Iterates over ``LEARNING_CURVE_SCORINGS``, calls
        :meth:`compute_learning_curve` for each scoring string, collects
        per-metric results under a ``"metrics"`` key, and writes a single JSON
        file.  ``train_sizes`` is shared across all metrics.

        Args:
            estimator (Any): Fitted or unfitted sklearn-compatible estimator.
            X_train (pd.DataFrame): Training features (post-selection).
            y_train (pd.Series): Training labels.
            random_state (int): Random seed used to configure the CV splitter.
            filepath (str): Destination path for the JSON output file.

        Returns:
            dict: Dict with keys ``random_state``, ``train_sizes``, and
                ``metrics`` (a nested dict keyed by scoring name).
        """
        metrics: dict[str, dict] = {}
        train_sizes = None
        for scoring in LEARNING_CURVE_SCORINGS:
            result = self.compute_learning_curve(
                estimator, X_train, y_train, random_state, scoring=scoring
            )
            train_sizes = result["train_sizes"]
            metrics[scoring] = {k: v for k, v in result.items() if k != "train_sizes"}

        data = {
            "random_state": random_state,
            "train_sizes": train_sizes,
            "metrics": metrics,
        }

        ensure_dir(os.path.dirname(filepath))
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)
        return data

    def _cv_train_evaluate(
        self,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        selected_features: tuple,
        random_state: int,
        model_filepath: str,
        metrics_filepath: str,
        learning_curve_path: str,
        classifier_model: ClassifierModel,
        shap_explanation_filepath: str | None = None,
    ) -> dict[str, Any]:
        """Steps 4-5: GridSearchCV training then evaluation on the held-out test set.

        Fits the configured classifier via GridSearchCV on the selected training
        features, saves the best model to disk, evaluates it on the held-out test
        set, and writes per-seed metrics JSON.

        Args:
            y_train (pd.Series): Resampled training labels.
            X_test (pd.DataFrame): Held-out test features (pre-selection).
            y_test (pd.Series): Held-out test labels.
            selected_features (tuple): Return value of :meth:`select_features`
                containing (df_selected, top_features, all_features, n_features).
            random_state (int): Random seed used for the classifier.
            model_filepath (str): Path to save the trained model pickle.
            metrics_filepath (str): Path to save the per-seed metrics JSON.
            learning_curve_path (str): Path to save the per-seed learning curve JSON.
            classifier_model (ClassifierModel): ClassifierModel instance to use.
            shap_explanation_filepath (str | None, optional): Path for the SHAP
                explanation cache. Defaults to None.

        Returns:
            dict[str, Any]: Metrics dictionary produced by ModelEvaluator.
        """
        classifier_model = classifier_model
        features_train_resampled_selected, top_selected_features, _, _ = (
            selected_features
        )
        top_n_features = top_selected_features.index.tolist()
        X_test_selected = X_test[top_n_features]

        classifier, grid_search, best_model = self._setup_grid_search(
            random_state,
            features_train_resampled_selected,
            y_train,
            top_n_features,
            classifier_model=classifier_model,
        )

        joblib.dump(best_model, model_filepath)
        if self.verbose:
            logger.info(f"{random_state:05d}: Model at {model_filepath}")

        try:
            self._compute_all_learning_curves(
                estimator=best_model,
                X_train=features_train_resampled_selected[top_n_features],
                y_train=y_train,
                random_state=random_state,
                filepath=learning_curve_path,
            )
        except Exception as e:
            logger.warning(
                f"Seed {random_state:05d}: learning curve computation failed. Reason: {e}"
            )
            learning_curve_path = None  # type: ignore[assignment]

        # Evaluating model
        model_evaluator = ModelEvaluator(
            random_state=random_state,
            model=grid_search,
            X_test=X_test_selected,
            y_test=y_test,
            model_name=classifier_model.name,
            output_dir=self.classifier_dirs[classifier_model.slug_name],
            plot_shap=self.plot_shap,
            selected_features=top_n_features,
            shap_explanation_filepath=shap_explanation_filepath,
            learning_curve_path=learning_curve_path,
        )

        metrics: dict[str, Any] = model_evaluator.get_metrics()

        grid_params = grid_search.best_params_
        best_params = {f"best_params_{k}": v for k, v in grid_params.items()}
        metrics.update(
            {
                "cv_strategy": classifier.cv_strategy,
                "random_state": random_state,
                "best_cv_score": grid_search.best_score_,
                **best_params,
            }
        )

        with open(metrics_filepath, "w") as f:
            json.dump(metrics, f, indent=4)

        try:
            model_evaluator.plot_all(dpi=150)
        except Exception as e:
            logger.warning(
                f"Seed {random_state:05d}: plot_all() failed and plots were skipped. "
                f"Reason: {e}"
            )

        if self.verbose:
            logger.info(f"{random_state:05d}: Metrics at {metrics_filepath}")
            logger.info(
                f"Seed {random_state:05d} - Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}"
            )

        return metrics

    def _run_train_and_evaluate(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> dict[str, tuple[int, str, str, dict, str, str, str]] | None:
        """Train feature selection and all classifiers for a single random seed (eval mode).

        Orchestrates the per-seed pipeline:

        1–2. Under-sampling and train/test split (ONCE per seed).
        3.   Feature selection on training set (ONCE per seed, shared across classifiers).
        4–5. For each classifier: GridSearchCV training and evaluation on held-out test set.

        Args:
            random_state (int): Base random state value.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.
            save_features (bool, optional): Save all ranked features. Defaults to False.
            plot_features (bool, optional): Generate feature plots. Defaults to False.

        Returns:
            dict[str, tuple]: Keyed by classifier slug. Each value is a 7-tuple of
                (random_state, significant_filepath, model_filepath, metrics,
                X_test_filepath, y_test_filepath, shap_explanation_filepath).
            None: When features reduced to zero.
        """
        if self.debug:
            logger.debug(f"_run_train_and_evaluate: seed={random_state}")

        # ========== STEP 0: Shared filepath preparation ==========
        (
            _,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            X_test_filepath,
            y_test_filepath,
            _,
        ) = self._generate_shared_filepaths(
            random_state=random_state,
            save_features=save_features,
            plot_features=plot_features,
        )

        logger.info(f"Training Seed: {random_state:05d}")

        # ========== STEPS 1-2: Train/Test Split + Resample (ONCE) ==========
        X_train, X_test, y_train, y_test = self._split_and_resample(
            X=self.df_features,
            y=self.df_labels,
            random_state=random_state,
            sampling_strategy=sampling_strategy,
        )

        # ========== STEP 3: Feature Selection (ONCE, shared) ==========
        result = self._select_features_for_seed(
            X_train=X_train,
            y_train=y_train,
            random_state=random_state,
            significant_filepath=significant_filepath,
            all_features_filepath=all_features_filepath,
            all_figures_filepath=all_figures_filepath,
        )

        if result is None:
            return None

        # Save held-out test data ONCE (shared across all classifiers)
        _, top_selected_features, _, _ = result
        X_test[top_selected_features.index.tolist()].to_csv(X_test_filepath)
        y_test.to_csv(y_test_filepath)

        # ========== STEPS 4-5: Per-classifier GridSearchCV + Evaluation ==========
        seed_results: dict[str, tuple] = {}
        for classifier_model in self.classifier_models:
            classifier_slug = classifier_model.slug_name
            (
                model_filepath,
                metrics_filepath,
                shap_explanation_filepath,
                learning_curve_path,
            ) = self._generate_classifier_filepaths(random_state, classifier_slug)

            metrics = self._cv_train_evaluate(
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                selected_features=result,
                random_state=random_state,
                model_filepath=model_filepath,
                metrics_filepath=metrics_filepath,
                learning_curve_path=learning_curve_path,
                classifier_model=classifier_model,
                shap_explanation_filepath=shap_explanation_filepath,
            )

            seed_results[classifier_slug] = (
                random_state,
                significant_filepath,
                model_filepath,
                metrics,
                X_test_filepath,
                y_test_filepath,
                shap_explanation_filepath,
            )

        return seed_results

    def train_and_evaluate(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
    ) -> None:
        """Train feature selection and classifier models across multiple random seeds.

        For each seed, performs:

        1. Train/test split (80/20, stratified)
        2. Random under-sampling on training set only
        3. Feature selection on training set only (shared across all classifiers)
        4. For each classifier: training with GridSearchCV
        5. For each classifier: evaluation on held-out test set

        Under-sampling and feature selection are run ONCE per seed and reused
        across all classifiers, which significantly reduces redundant computation
        when training multiple classifiers simultaneously.

        Args:
            random_state (int, optional): Initial random state seed. Defaults to 0.
            total_seed (int, optional): Total number of seeds to run. Defaults to 500.
            sampling_strategy (str | float, optional): Under-sampling ratio for
                balancing classes. Defaults to 0.75.
            save_all_features (bool, optional): Save all features per seed,
                not just top-N. Defaults to False.
            plot_significant_features (bool, optional): Generate feature importance
                plots. Defaults to False.

        Example:
            >>> ModelTrainer(
            ...     extracted_features_csv="...",
            ...     label_features_csv="...",
            ...     classifiers=["rf", "xgb"],
            ... ).train_and_evaluate(random_state=42, total_seed=100)
        """
        self.create_directories()
        for slug in self.classifier_dirs:
            ensure_dir(self.metrics_dirs[slug])

        if save_all_features:
            ensure_dir(self.shared_all_features_dir)

        if plot_significant_features:
            ensure_dir(self.shared_figures_dir)

        # Accumulate results across all seeds per classifier before aggregating
        all_metrics: dict[str, list[dict]] = {
            classifier_model.slug_name: []
            for classifier_model in self.classifier_models
        }
        records_per_clf: dict[str, list[dict]] = {
            classifier_model.slug_name: []
            for classifier_model in self.classifier_models
        }

        # Pre-filter: collect already-completed seeds without re-running them
        random_states: list[int] = [random_state + seed for seed in range(total_seed)]
        pending_jobs: list[tuple[int, str | float, bool, bool]] = []

        for _rs in random_states:
            (
                _,
                _significant_filepath,
                _,
                _,
                _X_test_filepath,
                _y_test_filepath,
                _,
            ) = self._generate_shared_filepaths(_rs)

            # Check if all classifiers have already been trained for this seed
            _all_classifier_done = True
            for classifier_model in self.classifier_models:
                classifier_slug = classifier_model.slug_name
                (
                    _model_filepath,
                    _metrics_filepath,
                    _shap_filepath,
                    _,
                ) = self._generate_classifier_filepaths(_rs, classifier_slug)
                _done = (
                    not self.overwrite
                    and os.path.isfile(_significant_filepath)
                    and os.path.isfile(_model_filepath)
                    and os.path.isfile(_metrics_filepath)
                )
                if not _done:
                    _all_classifier_done = False
                    break

            if _all_classifier_done:
                logger.info(f"Seed {_rs:05d} already trained.")
                self.significant_features_csvs.append(_significant_filepath)
                for classifier_model in self.classifier_models:
                    classifier_slug = classifier_model.slug_name
                    (
                        _model_filepath,
                        _metrics_filepath,
                        _shap_filepath,
                        _,
                    ) = self._generate_classifier_filepaths(_rs, classifier_slug)
                    with open(_metrics_filepath) as f:
                        metrics = json.load(f)
                    records_per_clf[classifier_slug].append(
                        {
                            "random_state": _rs,
                            "significant_features_csv": _significant_filepath,
                            "trained_model_filepath": _model_filepath,
                            "X_test_filepath": _X_test_filepath,
                            "y_test_filepath": _y_test_filepath,
                            "shap_explanation_filepath": _shap_filepath,
                        }
                    )
                    all_metrics[classifier_slug].append(metrics)
            else:
                pending_jobs.append(
                    (
                        _rs,
                        sampling_strategy,
                        save_all_features,
                        plot_significant_features,
                    )
                )

        for seed_results in self._run_jobs(self._run_train_and_evaluate, pending_jobs):
            if seed_results is None:  # Feature selection returned nothing
                continue
            # seed_results is dict[classifier_slug -> 7-tuple]
            for classifier_slug, (
                _random_state,
                significant_features_csv,
                trained_model_filepath,
                metrics,
                X_test_filepath,
                y_test_filepath,
                shap_explanation_filepath,
            ) in seed_results.items():
                if classifier_slug not in records_per_clf:
                    records_per_clf[classifier_slug] = []
                    all_metrics[classifier_slug] = []
                # Only append significant_features_csv once (same for all classifiers)
                if significant_features_csv not in self.significant_features_csvs:
                    self.significant_features_csvs.append(significant_features_csv)
                records_per_clf[classifier_slug].append(
                    {
                        "random_state": _random_state,
                        "significant_features_csv": significant_features_csv,
                        "trained_model_filepath": trained_model_filepath,
                        "X_test_filepath": X_test_filepath,
                        "y_test_filepath": y_test_filepath,
                        "shap_explanation_filepath": shap_explanation_filepath,
                    }
                )
                all_metrics[classifier_slug].append(metrics)

        # Aggregate feature selection results (shared across classifiers)
        self.df_significant_features = self.concat_significant_features(
            plot=plot_significant_features,
        )

        # Save registry and metrics per classifier
        for classifier_model in self.classifier_models:
            classifier_slug = classifier_model.slug_name
            if not records_per_clf[classifier_slug]:
                continue
            suffix_filename = self._save_models_registry(
                records_per_clf[classifier_slug],
                random_state,
                total_seed,
                classifier_slug=classifier_slug,
            )
            self._aggregate_metrics(
                all_metrics[classifier_slug],
                suffix_filename=suffix_filename,
                classifier_slug=classifier_slug,
            )

        if self.verbose:
            logger.info(f"Models saved to: {self.csv}")

        return None

    def _run_train(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> dict[str, tuple[int, str, str]] | None:
        """Train on the full dataset (no train/test split) for a single random seed.

        Performs:

        1. Random under-sampling on full dataset (ONCE per seed).
        2. Feature selection on resampled data (ONCE per seed, shared).
        3. For each classifier: GridSearchCV training.
        4. For each classifier: save trained model.

        Args:
            random_state (int): Random seed for reproducibility.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.
            save_features (bool, optional): Save all ranked features. Defaults to False.
            plot_features (bool, optional): Generate feature plots. Defaults to False.

        Returns:
            dict[str, tuple[int, str, str]]: Keyed by classifier slug. Each value is
                a 3-tuple of (random_state, significant_filepath, model_filepath).
            None: When features reduced to zero.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> seed_results = trainer._run_train(random_state=42)
        """
        if self.debug:
            logger.debug(f"_run_train: seed={random_state}")

        # ========== STEP 0: Shared filepath preparation ==========
        (
            _,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            _,
            _,
            _,
        ) = self._generate_shared_filepaths(
            random_state=random_state,
            save_features=save_features,
            plot_features=plot_features,
        )

        # ========== STEP 1: Resample Full Dataset (ONCE) ==========
        features_resampled, labels_resampled = random_under_sampler(
            features=self.df_features,
            labels=self.df_labels,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

        # ========== STEP 2: Feature Selection (ONCE, shared) ==========
        result = self.select_features(
            features=features_resampled,
            labels=labels_resampled,
            random_state=random_state,
            significant_filepath=significant_filepath,
            all_features_filepath=all_features_filepath,
            all_figures_filepath=all_figures_filepath,
        )

        if result is None:
            return None

        (
            features_resampled_selected,
            top_n_features_series,
            _all_selected_features,
            _n_features,
        ) = result

        top_n_features = top_n_features_series.index.tolist()

        # ========== STEP 3-4: Per-classifier GridSearchCV + Save ==========
        logger.info(f"Fitting Seed: {random_state:05d}")

        seed_results: dict[str, tuple] = {}
        for classifier_model in self.classifier_models:
            classifier_slug = classifier_model.slug_name
            (
                model_filepath,
                _,
                _,
                _,
            ) = self._generate_classifier_filepaths(random_state, classifier_slug)

            _, _, best_model = self._setup_grid_search(
                random_state,
                features_resampled_selected,
                labels_resampled,
                top_n_features,
                classifier_model=classifier_model,
            )

            joblib.dump(best_model, model_filepath)

            if self.verbose:
                logger.info(f"Model {random_state:05d}: {model_filepath}")

            seed_results[classifier_slug] = (
                random_state,
                significant_filepath,
                model_filepath,
            )

        return seed_results

    def train(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
    ) -> None:
        """Train on the full dataset across multiple seeds (no train/test split).

        For each seed, performs:

        1. Random under-sampling on full dataset (ONCE per seed)
        2. Feature selection on resampled data (ONCE per seed, shared across classifiers)
        3. For each classifier: training with GridSearchCV
        4. For each classifier: save trained model

        Unlike ``train_and_evaluate()``, this method does NOT perform a train/test
        split and does NOT compute evaluation metrics. It is intended for final model
        training when a separate "future" dataset will be used for evaluation
        via ``ModelPredictor``.

        Args:
            random_state (int, optional): Initial random state seed. Defaults to 0.
            total_seed (int, optional): Total number of seeds to run. Defaults to 500.
            sampling_strategy (str | float, optional): Under-sampling ratio for
                balancing classes. Defaults to 0.75.
            save_all_features (bool, optional): Save all features per seed,
                not just top-N. Defaults to False.
            plot_significant_features (bool, optional): Generate feature importance
                plots. Defaults to False.

        Example:
            >>> trainer = ModelTrainer(
            ...     extracted_features_csv="output/features/extracted_features.csv",
            ...     label_features_csv="output/features/label_features.csv",
            ...     classifiers=["rf", "xgb"],
            ... )
            >>> trainer.train(random_state=0, total_seed=5)
        """
        # Since we are not using evaluation, change the folder name from
        # ``evaluations`` to ``predictions``
        output_dir = self.output_dir.replace("evaluations", "predictions")

        # Update current directories with new output directory
        self.update_directories(output_dir=output_dir)

        if save_all_features:
            ensure_dir(self.shared_all_features_dir)

        if plot_significant_features:
            ensure_dir(self.shared_figures_dir)

        records_per_clf: dict[str, list[dict]] = {
            classifier_model.slug_name: []
            for classifier_model in self.classifier_models
        }

        # Pre-filter: collect already-completed seeds without re-running them
        random_states: list[int] = [random_state + seed for seed in range(total_seed)]
        pending_jobs: list[tuple[int, str | float, bool, bool]] = []

        for _rs in random_states:
            (
                _,
                _significant_filepath,
                _,
                _,
                _,
                _,
                _,
            ) = self._generate_shared_filepaths(_rs)

            _all_classifier_done = True
            for classifier_model in self.classifier_models:
                classifier_slug = classifier_model.slug_name
                (
                    _model_filepath,
                    _,
                    _,
                    _,
                ) = self._generate_classifier_filepaths(_rs, classifier_slug)
                _done = (
                    not self.overwrite
                    and os.path.isfile(_significant_filepath)
                    and os.path.isfile(_model_filepath)
                )
                if not _done:
                    _all_classifier_done = False
                    break

            if _all_classifier_done:
                logger.info(f"Seed {_rs:05d} already fitted.")
                self.significant_features_csvs.append(_significant_filepath)
                for classifier_model in self.classifier_models:
                    classifier_slug = classifier_model.slug_name
                    (
                        _model_filepath,
                        _,
                        _,
                        _,
                    ) = self._generate_classifier_filepaths(_rs, classifier_slug)
                    records_per_clf[classifier_slug].append(
                        {
                            "random_state": _rs,
                            "significant_features_csv": _significant_filepath,
                            "trained_model_filepath": _model_filepath,
                        }
                    )
            else:
                pending_jobs.append(
                    (
                        _rs,
                        sampling_strategy,
                        save_all_features,
                        plot_significant_features,
                    )
                )

        for seed_results in self._run_jobs(self._run_train, pending_jobs):
            if seed_results is None:  # Feature selection returned nothing
                continue
            # seed_results is dict[classifier_slug -> 3-tuple]
            for classifier_slug, (
                _random_state,
                significant_features_csv,
                trained_model_filepath,
            ) in seed_results.items():
                if classifier_slug not in records_per_clf:
                    records_per_clf[classifier_slug] = []
                if significant_features_csv not in self.significant_features_csvs:
                    self.significant_features_csvs.append(significant_features_csv)
                records_per_clf[classifier_slug].append(
                    {
                        "random_state": _random_state,
                        "significant_features_csv": significant_features_csv,
                        "trained_model_filepath": trained_model_filepath,
                    }
                )

        # Aggregate feature selection results (shared)
        self.df_significant_features = self.concat_significant_features(
            plot=plot_significant_features,
        )

        # Save registry per classifier
        for classifier_model in self.classifier_models:
            classifier_slug = classifier_model.slug_name
            if not records_per_clf[classifier_slug]:
                continue
            self._save_models_registry(
                records_per_clf[classifier_slug],
                random_state,
                total_seed,
                classifier_slug=classifier_slug,
            )

        if self.verbose:
            logger.info(f"Models saved to: {self.csv}")

        return None

    def fit(self, with_evaluation: bool = True, **kwargs) -> None:
        """Dispatch to ``train_and_evaluate()`` or ``train()`` based on ``with_evaluation``.

        Args:
            with_evaluation (bool, optional): If True, calls ``train_and_evaluate()``
                (80/20 split + metrics). If False, calls ``train()`` (full dataset,
                no metrics). Defaults to True.
            **kwargs: Additional keyword arguments forwarded to the chosen method.
        """
        if with_evaluation:
            return self.train_and_evaluate(**kwargs)
        return self.train(**kwargs)

    def _aggregate_metrics(
        self,
        all_metrics: list[dict],
        suffix_filename: str = "",
        classifier_slug: str = "",
    ) -> None:
        """Aggregate metrics across all seeds for one classifier.

        Computes mean and std for each metric and saves to CSV.
        Also saves all individual metrics for detailed analysis.

        Args:
            all_metrics (list[dict]): List of metric dictionaries, one per seed.
            suffix_filename (str, optional): Suffix appended to output filenames.
                Defaults to ``""``.
            classifier_slug (str): Slugified classifier name used to resolve the correct
                output directory from ``self.classifier_dirs`` and ``self.metrics_dirs``.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> trainer.train_and_evaluate(total_seed=100)
            >>> # Creates: all_metrics_{suffix}.csv and metrics_summary_{suffix}.csv
        """
        classifier_dir = self.classifier_dirs[classifier_slug]
        metrics_dir = self.metrics_dirs[classifier_slug]
        ensure_dir(metrics_dir)

        df_metrics = pd.DataFrame(all_metrics)

        # Calculate summary statistics
        summary = df_metrics.describe().T
        summary_filepath = os.path.join(
            classifier_dir, f"metrics_summary_{suffix_filename}.csv"
        )
        summary.to_csv(summary_filepath)

        # Save all individual metrics
        all_metrics_filepath = os.path.join(
            classifier_dir, f"all_metrics_{suffix_filename}.csv"
        )
        df_metrics = df_metrics.set_index("random_state")
        df_metrics.to_csv(all_metrics_filepath, index=True)

        logger.info("=" * 60)
        logger.info("Metrics Summary (mean ± std across seeds)")
        logger.info("=" * 60)
        for metric in [
            "accuracy",
            "balanced_accuracy",
            "f1_score",
            "precision",
            "recall",
        ]:
            mean = df_metrics[metric].mean()
            std = df_metrics[metric].std()
            logger.info(f"{metric:20s}: {mean:.4f} ± {std:.4f}")
        logger.info("=" * 60)

        if self.verbose:
            logger.info(f"Summary metrics saved to: {summary_filepath}")
            logger.info(f"All metrics saved to: {all_metrics_filepath}")

    def merge_models(self, output_path: str | None = None) -> str:
        """Merge all seed models for the first classifier into a single SeedEnsemble pkl.

        Convenience wrapper around
        :func:`eruption_forecast.utils.ml.merge_seed_models` that uses
        the first classifier's registry CSV from ``self.csv``.  The registry CSV
        must exist (i.e. ``train()`` or ``train_and_evaluate()`` must have been
        called first).

        Args:
            output_path (str | None, optional): Destination path for the merged
                ``.pkl`` file.  If ``None``, the file is placed alongside the
                registry CSV as ``merged_model_{suffix}.pkl``.
                Defaults to ``None``.

        Returns:
            str: Absolute path to the saved merged ``.pkl`` file.

        Raises:
            RuntimeError: If no registry CSV is available (training not yet run).
        """
        if not self.csv:
            raise RuntimeError(
                "No model registry CSV found. Run train() or "
                "train_and_evaluate() before calling merge_models()."
            )

        # Use first available classifier CSV
        first_csv = next(iter(self.csv.values()))
        output_path = output_path or self.output_dir

        return merge_seed_models(first_csv, output_path=output_path)

    @staticmethod
    def merge_classifier_models(
        trained_models: dict[str, str],
        output_path: str | None = None,
    ) -> str:
        """Merge multiple classifier registry CSVs into one combined pkl.

        Convenience wrapper around
        :func:`eruption_forecast.utils.ml.merge_all_classifiers`.  Use this
        after training multiple classifiers to bundle their ensembles into a
        single file for ``ModelPredictor``.

        Args:
            trained_models (dict[str, str]): Mapping of classifier name to
                the path of its trained-model registry CSV.
            output_path (str | None, optional): Destination path for the
                combined ``.pkl`` file.  If ``None``, a default path is derived
                from the first registry CSV.  Defaults to ``None``.

        Returns:
            str: Absolute path to the saved combined ``.pkl`` file.
        """
        return merge_all_classifiers(trained_models, output_path=output_path)
