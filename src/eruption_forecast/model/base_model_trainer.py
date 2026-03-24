"""Base infrastructure shared by all model-trainer subclasses.

Provides the constructor, validation, directory management, feature selection,
grid-search setup, and model-registry utilities that are reused by both
:class:`EvaluationTrainer` and :class:`ModelTrainer`.
"""

import os
from typing import Any, Literal
from collections.abc import Callable

import joblib
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GridSearchCV, train_test_split

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import (
    merge_seed_models,
    load_labels_from_csv,
    random_under_sampler,
    get_classifier_models,
    merge_all_classifiers,
)
from eruption_forecast.model.constants import GPU_CLASSIFIERS
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.config.constants import (
    TRAIN_TEST_SPLIT,
    DEFAULT_CV_SPLITS,
    DEFAULT_N_SIGNIFICANT_FEATURES,
)
from eruption_forecast.features.constants import (
    SIGNIFICANT_FEATURES_FILENAME,
)
from eruption_forecast.plots.feature_plots import (
    plot_significant_features as _plot_significant_features,
)
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.features.feature_selector import FeatureSelector


class BaseModelTrainer:
    """Shared infrastructure for feature-selection and classifier training.

    Provides the constructor, validation, directory management, feature
    selection, grid-search setup, and model-registry utilities that are
    reused by both :class:`EvaluationTrainer` and :class:`ModelTrainer`.

    Attributes:
        df_features (pd.DataFrame): Loaded features DataFrame.
        df_labels (pd.Series): Loaded labels Series.
        output_dir (str): Root directory for training outputs.
        n_jobs (int): Number of parallel seed workers (outer loop).
        grid_search_n_jobs (int): Number of parallel jobs inside each GridSearchCV call.
        prefix_filename (str | None): Optional prefix for output filenames.
        classifiers (list[str]): Normalised list of classifier keys passed at construction.
        cv_strategy (str): Cross-validation strategy.
        cv_splits (int): Number of CV splits.
        number_of_significant_features (int): Number of top features to save separately.
        feature_selection_method (str): Feature selection method.
        overwrite (bool): Whether to overwrite existing output files.
        verbose (bool): Enable verbose logging.
        debug (bool): Enable debug mode.
        FeatureSelector (FeatureSelector): Feature selection component instance.
        classifier_models (list[ClassifierModel]): One ClassifierModel per classifier key.
        shared_features_dir (str): Shared directory for feature outputs.
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
    """

    def __init__(
        self,
        extracted_features_csv: str,
        label_features_csv: str,
        classifiers: str | list[str] = "rf",
        output_dir: str | None = None,
        root_dir: str | None = None,
        cv_strategy: Literal[
            "shuffle", "stratified", "shuffle-stratified", "timeseries"
        ] = "shuffle-stratified",
        cv_splits: int = DEFAULT_CV_SPLITS,
        number_of_significant_features: int = DEFAULT_N_SIGNIFICANT_FEATURES,
        feature_selection_method: Literal[
            "tsfresh", "random_forest", "combined"
        ] = "tsfresh",
        prefix_filename: str | None = None,
        overwrite: bool = False,
        plot_shap: bool = False,
        n_jobs: int = 1,
        grid_search_n_jobs: int = 1,
        use_gpu: bool = False,
        gpu_id: int = 0,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        """Initialize BaseModelTrainer with feature data, classifier settings, and output paths.

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
            plot_shap (bool, optional): Compute and save SHAP summary plots. Defaults to False.
        """
        df_features = pd.read_csv(extracted_features_csv, index_col=0)
        df_labels = load_labels_from_csv(label_features_csv)

        classifiers: list[str] = (
            [classifiers] if isinstance(classifiers, str) else classifiers
        )

        classifier_models: list[ClassifierModel] = get_classifier_models(
            classifiers,
            cv_strategy=cv_strategy,
            cv_splits=cv_splits,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            verbose=verbose,
        )

        # Output training dir: ``<root_dir>/output/trainings``
        output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output"),
        )
        output_dir = os.path.join(output_dir, "trainings")

        classifier_use_gpu = any(
            classifier in GPU_CLASSIFIERS for classifier in classifiers
        )
        if use_gpu and classifier_use_gpu and n_jobs != 1:
            logger.warning(
                "use_gpu=True forces n_jobs=1 to avoid GPU memory contention "
                "from parallel seed workers sharing the same device."
            )
            n_jobs = 1

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.df_features = df_features
        self.df_labels = df_labels
        self.classifiers = classifiers
        self.root_dir = root_dir
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.number_of_significant_features = number_of_significant_features
        self.feature_selection_method: str = feature_selection_method
        self.prefix_filename = prefix_filename
        self.overwrite = overwrite
        self.plot_shap = plot_shap
        self.n_jobs = n_jobs
        self.grid_search_n_jobs = grid_search_n_jobs
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self.cv_slug_name = classifier_models[0].slug_cv_name
        self.classifier_models = classifier_models
        self.FeatureSelector = FeatureSelector(
            method=feature_selection_method, verbose=verbose, n_jobs=grid_search_n_jobs
        )

        # Must be initialised before set_directories() because that method
        # reads this flag to choose the "evaluations" vs "predictions" sub-path.
        self.with_evaluation = False

        # Set directories
        (
            self.output_dir,
            self.shared_features_dir,
            self.shared_significant_dir,
            self.shared_all_features_dir,
            self.shared_figures_dir,
            self.shared_tests_dir,
            self.classifier_dirs,
            self.models_dirs,
            self.metrics_dirs,
        ) = self.set_directories(output_dir)
        self.significant_features_csvs: list[str] = []
        self.df_significant_features: pd.DataFrame = pd.DataFrame()

        # Will be set after evaluate() or train() called
        # Keyed by classifier slug
        self.df: dict[str, pd.DataFrame] = {}
        self.csv: dict[str, str] = {}

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

    def set_directories(self, output_dir: str):
        """Build and store the full directory hierarchy for training outputs.

        Derives all shared (feature-selection) and per-classifier output paths
        from ``output_dir``, appending ``evaluations/`` or ``predictions/``
        depending on ``self.with_evaluation``.  The returned paths are assigned
        to the corresponding instance attributes by the caller.

        Args:
            output_dir (str): Base trainings directory (e.g. ``output/trainings``).

        Returns:
            tuple: A 9-tuple of
                ``(output_dir, shared_features_dir, shared_significant_dir,
                shared_all_features_dir, shared_figures_dir, shared_tests_dir,
                classifier_dirs, models_dirs, metrics_dirs)``.
        """
        sub_output_dir = "evaluations" if self.with_evaluation else "predictions"
        output_dir = os.path.join(output_dir, sub_output_dir)

        shared_features_dir = os.path.join(output_dir, "features", self.cv_slug_name)
        shared_significant_dir = os.path.join(
            shared_features_dir, "significant_features"
        )
        shared_all_features_dir = os.path.join(shared_features_dir, "all_features")
        shared_figures_dir = os.path.join(shared_features_dir, "figures", "significant")
        shared_tests_dir = os.path.join(shared_features_dir, "tests")

        classifier_dirs: dict[str, str] = {}
        models_dirs: dict[str, str] = {}
        metrics_dirs: dict[str, str] = {}
        for classifier_model in self.classifier_models:
            slug = classifier_model.slug_name
            classifier_dir = os.path.join(
                output_dir, "classifiers", slug, self.cv_slug_name
            )
            classifier_dirs[slug] = classifier_dir
            models_dirs[slug] = os.path.join(classifier_dir, "models")
            metrics_dirs[slug] = os.path.join(classifier_dir, "metrics")

        return (
            output_dir,
            shared_features_dir,
            shared_significant_dir,
            shared_all_features_dir,
            shared_figures_dir,
            shared_tests_dir,
            classifier_dirs,
            models_dirs,
            metrics_dirs,
        )

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

    def create_directories(
        self,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
        with_evaluation: bool = False,
    ) -> None:
        """Create all required output directories for the current training run.

        Always creates the base output and significant-features directories.
        Optionally creates the all-features, figures, test-split, and per-classifier
        metrics directories when the corresponding flags are set.

        Args:
            save_all_features (bool, optional): Create the all-features directory.
                Defaults to False.
            plot_significant_features (bool, optional): Create the figures directory.
                Defaults to False.
            with_evaluation (bool, optional): Create the held-out test-split and
                per-classifier metrics directories needed by the evaluation pipeline.
                Defaults to False.

        Returns:
            None
        """
        ensure_dir(self.output_dir)
        ensure_dir(self.shared_significant_dir)

        # Always ensure per-classifier output and models directories so that
        # later writes (e.g., joblib.dump and model registry CSVs) do not fail
        # with FileNotFoundError.
        for classifier_model in self.classifier_models:
            ensure_dir(self.classifier_dirs[classifier_model.slug_name])
            ensure_dir(self.models_dirs[classifier_model.slug_name])

        if save_all_features:
            ensure_dir(self.shared_all_features_dir)

        if plot_significant_features:
            ensure_dir(self.shared_figures_dir)

        if with_evaluation:
            ensure_dir(self.shared_tests_dir)
            for classifier_model in self.classifier_models:
                ensure_dir(self.metrics_dirs[classifier_model.slug_name])

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
            >>> trainer.evaluate(total_seed=100)
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

            # The score column is always "score" for aggregated per-seed files.
            # Sort by whatever column is present for backward-compat with old files.
            score_cols = [c for c in combined_features_df.columns if c != "features"]
            if not score_cols:
                raise ValueError(
                    "Significant features CSV must contain a score column "
                    "(e.g. 'score') besides the 'features' index."
                )
            score_col = score_cols[0]
            combined_features_df = (
                combined_features_df.groupby(by="features")
                .count()
                .sort_values(by=score_col, ascending=False)
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
                    values_column=score_col,
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
            values_column="score",
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

    def _get_model_filepath(self, random_state: int, classifier_slug: str) -> str:
        """Return the trained model pickle path for a given seed and classifier.

        Convenience accessor that computes only the model filepath without
        unpacking the full 4-tuple returned by :meth:`_generate_classifier_filepaths`.
        Use this when only the model path is needed (e.g., in pre-filter checks
        inside :meth:`train`).

        Args:
            random_state (int): Random seed for this training run.
            classifier_slug (str): Slugified classifier name used to locate the
                correct models directory.

        Returns:
            str: Absolute path to the trained model pickle file.
        """
        filename = (
            f"{self.prefix_filename}_{random_state:05d}"
            if self.prefix_filename
            else f"{random_state:05d}"
        )
        return os.path.join(self.models_dirs[classifier_slug], f"{filename}.pkl")

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
            logger.info(
                f"Running on {self.n_jobs} job(s). Grid search jobs {self.grid_search_n_jobs}."
            )
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
                ``trained_model_filepath``, and (for ``evaluate`` only)
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

    def merge_models(self, output_path: str | None = None) -> dict[str, str]:
        """Merge all seed models for each classifier into a single SeedEnsemble pkl.

        Convenience wrapper around
        :func:`eruption_forecast.utils.ml.merge_seed_models` that iterates over
        all classifier registry CSVs from ``self.csv``.  The registry CSVs
        must exist (i.e. ``train()`` or ``evaluate()`` must have been
        called first).

        Args:
            output_path (str | None, optional): Destination path for the merged
                ``.pkl`` files.  If ``None``, each file is placed alongside its
                registry CSV as ``merged_model_{suffix}.pkl``.
                Defaults to ``None``.

        Returns:
            dict[str, str]: Mapping of classifier class name (e.g.,
                ``"RandomForestClassifier"``) — not the slug — to the absolute
                path of the saved merged ``.pkl`` file for each classifier.

        Raises:
            RuntimeError: If no registry CSV is available (training not yet run).
        """
        if not self.csv:
            raise RuntimeError(
                "No model registry CSV found. Run train() or "
                "evaluate() before calling merge_models()."
            )

        _output_dir = output_path or os.path.join(self.output_dir, "classifiers")

        # Build slug class name mapping so callers receive consistent class-name keys
        slug_to_name: dict[str, str] = {
            clf_model.slug_name: clf_model.name for clf_model in self.classifier_models
        }

        return {
            slug_to_name.get(slug, slug): merge_seed_models(csv, output_dir=_output_dir)
            for slug, csv in self.csv.items()
        }

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
