import os
import json
from typing import Any, Self, Literal
from collections.abc import Callable
from multiprocessing import Pool

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import random_under_sampler
from eruption_forecast.utils.pathutils import resolve_output_dir
from eruption_forecast.config.constants import (
    TRAIN_TEST_SPLIT,
    DEFAULT_CV_SPLITS,
    DEFAULT_N_SIGNIFICANT_FEATURES,
)
from eruption_forecast.features.constants import (
    ID_COLUMN,
    ERUPTED_COLUMN,
    DATETIME_COLUMN,
    SIGNIFICANT_FEATURES_FILENAME,
)
from eruption_forecast.plots.feature_plots import (
    plot_significant_features as _plot_significant_features,
)
from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.features.feature_selector import FeatureSelector


class ModelTrainer:
    """Train feature-selection and classifier models over multiple random seeds.

    Loads pre-extracted features and labels, then for each random seed performs:

    1. Train/test split (80/20, stratified) to prevent data leakage
    2. Random under-sampling on training set only to balance classes
    3. Feature selection on training set using tsfresh relevance filtering
    4. Classifier training with GridSearchCV and cross-validation
    5. Evaluation on held-out test set (when using train_and_evaluate)

    This multi-seed approach provides robust feature selection and model
    evaluation by averaging results across many random data splits, reducing
    the risk of overfitting to a particular train/test configuration.

    Attributes:
        df_features (pd.DataFrame): Loaded features DataFrame (from extracted_features_csv).
        df_labels (pd.Series): Loaded labels Series (from label_features_csv).
        output_dir (str): Root directory for training outputs.
        n_jobs (int): Number of parallel workers for training.
        prefix_filename (str | None): Optional prefix for output filenames.
        classifier (str): Classifier type ("rf", "gb", "xgb", etc.).
        cv_strategy (str): Cross-validation strategy ("shuffle", "stratified", "timeseries").
        cv_splits (int): Number of CV splits.
        number_of_significant_features (int): Number of top features to save separately.
        feature_selection_method (str): Feature selection method ("tsfresh", "random_forest", "combined").
        overwrite (bool): Whether to overwrite existing output files.
        verbose (bool): Enable verbose logging.
        debug (bool): Enable debug mode.
        FeatureSelector: Feature selection component instance.
        ClassifierModel: Classifier configuration and model instance.
        classifier_name (str): Human-readable classifier name.
        classifier_slug_name (str): Slugified classifier name.
        classifier_slug_cv_name (str): Slugified CV strategy name.
        classifier_id (str): Unique identifier combining classifier and CV.
        features_dir (str): Directory for feature outputs.
        significant_features_dir (str): Directory for significant features.
        all_features_dir (str): Directory for all features.
        figures_dir (str): Directory for plots.
        significant_figures_dir (str): Directory for feature importance plots.
        classifier_dir (str): Directory for classifier outputs.
        models_dir (str): Directory for saved model files.
        metrics_dir (str): Directory for metrics JSON files.
        tests_dir (str): Directory for per-seed held-out test splits (X_test/y_test CSVs).
        figures_dir (str): Directory for aggregate evaluation figures and data CSVs.
        significant_features_csvs (list[str]): Paths to per-seed significant features.
        df_significant_features (pd.DataFrame): Aggregated significant features across seeds.
        df (pd.DataFrame): Model registry DataFrame.
        csv (str | None): Path to saved model registry CSV.

    Args:
        extracted_features_csv (str): Path to the extracted features CSV file.
            Located at ``output/{nslc}/features/all_extracted_features_{dates}.csv``
            or ``output/{nslc}/features/relevant_features_{dates}.csv``.
        label_features_csv (str): Path to the label CSV file built by LabelBuilder.
            Located at ``output/{nslc}/features/label_features_{dates}.csv``.
        output_dir (str | None, optional): Directory for output files. If None,
            defaults to ``root_dir/output/trainings``. Relative paths are resolved
            against ``root_dir`` (or ``os.getcwd()`` when ``root_dir`` is None).
            Absolute paths are used as-is. Defaults to None.
        root_dir (str | None, optional): Anchor directory for resolving relative
            ``output_dir`` values. Defaults to None (uses ``os.getcwd()``).
        prefix_filename (str | None, optional): Prefix for output filenames.
            Defaults to None.
        classifier (Literal[ "svm", "knn", "dt", "rf", "gb", "xgb", "nn", "nb", "lr", "voting", "lite-rf"], optional):
            Classifier type. Defaults to "rf".
        cv_strategy (Literal["shuffle", "stratified", "timeseries"], optional):
            Cross-validation strategy. Defaults to "shuffle".
        cv_splits (int, optional): Number of CV splits. Defaults to 5.
        number_of_significant_features (int, optional): Number of top features
            to save separately. Defaults to 20.
        feature_selection_method (Literal["tsfresh", "random_forest", "combined"], optional):
            Feature selection method. Defaults to "tsfresh".
        overwrite (bool, optional): Overwrite existing output files. Defaults to False.
        n_jobs (int, optional): Number of parallel workers. Defaults to 1.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        debug (bool, optional): Enable debug mode. Defaults to False.

    Raises:
        ValueError: If features or labels are empty, or their lengths do not match.
        ValueError: If n_jobs is <= 0.

    Examples:
        >>> # Train with Random Forest (default)
        >>> trainer = ModelTrainer(
        ...     extracted_features_csv="output/features/extracted_features.csv",
        ...     label_features_csv="output/features/label_features.csv",
        ...     output_dir="output/trainings",
        ...     n_jobs=4,
        ... )
        >>> trainer.train_and_evaluate(
        ...     random_state=0,
        ...     total_seed=100,
        ...     number_of_significant_features=20,
        ...     sampling_strategy=0.75,
        ... )

        >>> # Train with Gradient Boosting and TimeSeriesSplit
        >>> trainer = ModelTrainer(
        ...     extracted_features_csv="output/features/extracted_features.csv",
        ...     label_features_csv="output/features/label_features.csv",
        ...     output_dir="output/trainings",
        ...     classifier="gb",
        ...     cv_strategy="timeseries",
        ...     n_jobs=4,
        ... )
        >>> trainer.fit(with_evaluation=True, random_state=0, total_seed=500)

        >>> # Train VotingClassifier ensemble with combined feature selection
        >>> trainer = ModelTrainer(
        ...     extracted_features_csv="output/features/extracted_features.csv",
        ...     label_features_csv="output/features/label_features.csv",
        ...     classifier="voting",
        ...     cv_strategy="shuffle",
        ...     feature_selection_method="combined",
        ...     n_jobs=4,
        ... )
        >>> trainer.train(random_state=0, total_seed=500)

        >>> # Results saved to:
        >>> # - output/trainings/model-with-evaluation/{classifier}/{cv}/features/significant_features.csv
        >>> # - output/trainings/model-with-evaluation/{classifier}/{cv}/models/ (trained models .pkl)
        >>> # - output/trainings/model-with-evaluation/{classifier}/{cv}/metrics/ (per-seed metrics .json)
        >>> # - output/trainings/model-with-evaluation/{classifier}/{cv}/all_metrics_{suffix}.csv
        >>> # - output/trainings/model-with-evaluation/{classifier}/{cv}/metrics_summary_{suffix}.csv
    """

    def __init__(
        self,
        extracted_features_csv: str,
        label_features_csv: str,
        output_dir: str | None = None,
        root_dir: str | None = None,
        prefix_filename: str | None = None,
        classifier: Literal[
            "svm", "knn", "dt", "rf", "gb", "xgb", "nn", "nb", "lr", "voting", "lite-rf"
        ] = "rf",
        cv_strategy: Literal["shuffle", "stratified", "timeseries"] = "shuffle",
        cv_splits: int = DEFAULT_CV_SPLITS,
        number_of_significant_features: int = DEFAULT_N_SIGNIFICANT_FEATURES,
        feature_selection_method: Literal[
            "tsfresh", "random_forest", "combined"
        ] = "tsfresh",
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        df_features = pd.read_csv(extracted_features_csv, index_col=0)

        df_labels = pd.read_csv(label_features_csv)
        if ID_COLUMN in df_labels.columns:
            df_labels = df_labels.set_index(ID_COLUMN)
        if DATETIME_COLUMN in df_labels.columns:
            df_labels = df_labels.drop(DATETIME_COLUMN, axis=1)
        df_labels = df_labels[ERUPTED_COLUMN]

        classifier_model: ClassifierModel = ClassifierModel(
            classifier=classifier,
            cv_strategy=cv_strategy,
            n_splits=cv_splits,
            verbose=verbose,
        )

        (
            self.classifier_name,
            self.classifier_slug_name,
            self.classifier_slug_cv_name,
            self.classifier_id,
        ) = self.get_classifier_properties(classifier_model)

        # Output training dir: ``<root_dir>/output/trainings``
        output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output"),
        )
        output_dir = os.path.join(output_dir, "trainings", "model-with-evaluation")

        # Classifier training dir: ``<output_dir>/<classifier_slug_name>/<classifier_slug_cv_name>``
        classifier_dir = os.path.join(
            output_dir, self.classifier_slug_name, self.classifier_slug_cv_name
        )

        # Classifier training model dir: ``<classifier_dir>/models``
        models_dir = os.path.join(classifier_dir, "models")

        # Classifier metrics dir: ``<classifier_dir>/metrics``
        metrics_dir = os.path.join(classifier_dir, "metrics")

        # Per-seed test data dir: ``<classifier_dir>/tests``
        tests_dir = os.path.join(classifier_dir, "tests")

        # Aggregate plots and data dir: ``<classifier_dir>/plots``
        figures_dir = os.path.join(classifier_dir, "figures")

        # Filtered features dir: ``<classifier_dir>/features``
        features_dir = os.path.join(classifier_dir, "features")

        # All features dir: ``<features_dir>/all_features``
        all_features_dir = os.path.join(features_dir, "all_features")

        # Significant features dir: ``<features_dir>/significant_features``
        significant_features_dir = os.path.join(features_dir, "significant_features")

        # Plot significant features dir: ``<features_dir>/figures/significant``
        figures_dir = os.path.join(features_dir, "figures")
        significant_figures_dir = os.path.join(figures_dir, "significant")

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.df_features = df_features
        self.df_labels = df_labels
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.prefix_filename = prefix_filename
        self.classifier = classifier
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.number_of_significant_features: int = number_of_significant_features
        self.feature_selection_method = feature_selection_method
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self.FeatureSelector = FeatureSelector(
            method=feature_selection_method, verbose=verbose
        )
        self.ClassifierModel = classifier_model
        self.features_dir = features_dir
        self.significant_features_dir = significant_features_dir
        self.all_features_dir = all_features_dir
        self.figures_dir = figures_dir
        self.significant_figures_dir = significant_figures_dir
        self.classifier_dir = classifier_dir
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.tests_dir = tests_dir
        self.figures_dir = figures_dir

        # ------------------------------------------------------------------
        # Will be set after train_and_evaluate() method called
        # ------------------------------------------------------------------
        self.significant_features_csvs: list[str] = []
        self.df_significant_features: pd.DataFrame = pd.DataFrame()

        # Will be set after train_and_evaluate() or train() called
        self.df: pd.DataFrame = pd.DataFrame()
        self.csv: str | None = None

        # ------------------------------------------------------------------
        # Validate and create directories
        # ------------------------------------------------------------------
        self.validate()

        # ------------------------------------------------------------------
        # Verbose and logging
        # ------------------------------------------------------------------
        if verbose:
            logger.info(
                f"Train model using {n_jobs} jobs with {self.classifier_name} classifier "
                f"and {cv_strategy} CV strategy ({cv_splits} splits)"
            )

    def validate(self) -> None:
        """Validate that features and labels are non-empty and aligned.

        Checks that both features and labels DataFrames contain data
        and that they have matching row counts (same number of windows).

        Raises:
            ValueError: If features DataFrame is empty (0 rows).
            ValueError: If labels DataFrame is empty (0 rows).
            ValueError: If the number of rows in features and labels
                do not match.

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
        if self.n_jobs <= 0:
            raise ValueError(
                "n_jobs cannot be negative or equals to 0. Check your n_jobs parameter."
            )

    def get_classifier_properties(
        self,
        classifier_model: ClassifierModel | None = None,
    ) -> tuple[str, str, str, str]:
        """Extract name, slug, and identifier strings from a ClassifierModel.

        Convenience method to extract various string representations of the
        classifier and CV strategy for use in filenames and directory paths.

        Args:
            classifier_model (ClassifierModel | None, optional): ClassifierModel
                instance to inspect. If None, uses ``self.ClassifierModel``.
                Defaults to None.

        Returns:
            tuple[str, str, str, str]: A 4-tuple containing:

                - **classifier_name** (str): Class name (e.g., "RandomForestClassifier")
                - **classifier_slug_name** (str): Slugified classifier name (e.g., "random-forest-classifier")
                - **classifier_slug_cv_name** (str): Slugified CV name (e.g., "stratified-k-fold")
                - **classifier_id** (str): Combined identifier (e.g., "RandomForestClassifier-StratifiedKFold")

        Examples:
            >>> name, slug, cv_slug, id_ = trainer.get_classifier_properties()
            >>> print(name)  # "RandomForestClassifier"
            >>> print(slug)  # "random-forest-classifier"
        """
        classifier_model = classifier_model or self.ClassifierModel

        classifier_name = classifier_model.name
        classifier_slug_name = classifier_model.slug_name
        classifier_cv_name = classifier_model.cv_name

        classifier_slug_cv_name = classifier_model.slug_cv_name
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
        # Output training dir: ``<root_dir>/output/trainings``
        self.output_dir = resolve_output_dir(
            output_dir,
            root_dir,
            os.path.join("output", "trainings", "model-with-evaluation"),
        )

        # Classifier training dir: ``<output_dir>/<classifier_slug_name>/<classifier_slug_cv_name>``
        self.classifier_dir = os.path.join(
            self.output_dir, self.classifier_slug_name, self.classifier_slug_cv_name
        )

        # Classifier training model dir: ``<classifier_dir>/models``
        self.models_dir = os.path.join(self.classifier_dir, "models")

        # Classifier metrics dir: ``<classifier_dir>/metrics``
        self.metrics_dir = os.path.join(self.classifier_dir, "metrics")

        # Per-seed test data dir: ``<classifier_dir>/tests``
        self.tests_dir = os.path.join(self.classifier_dir, "tests")

        # Aggregate plots and data dir: ``<classifier_dir>/plots``
        self.figures_dir = os.path.join(self.classifier_dir, "plots")

        # Filtered features dir: ``<classifier_dir>/features``
        self.features_dir = os.path.join(self.classifier_dir, "features")

        # All features dir: ``<features_dir>/all_features``
        self.all_features_dir = os.path.join(self.features_dir, "all_features")

        # Significant features dir: ``<features_dir>/significant_features``
        self.significant_features_dir = os.path.join(
            self.features_dir, "significant_features"
        )

        # Plot significant features dir: ``<features_dir>/figures/significant``
        figures_dir = os.path.join(self.features_dir, "figures")
        self.significant_figures_dir = os.path.join(figures_dir, "significant")

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
            >>> trainer.update_grid_params(trainer.ClassifierModel, new_grid)
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

        Creates the main output directory and subdirectories for storing
        significant features CSVs and trained models. Called at the start
        of ``train_and_evaluate()`` and ``update_directories()``.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> trainer.create_directories()
            >>> # Creates: output_dir/, significant_features/, models/
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.significant_features_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.tests_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

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
            >>> _df = trainer.concat_significant_features(
            ...     plot=True
            ... )
            >>> print(_df.head())
            # Shows features and their occurrence counts across seeds
        """
        number_of_significant_features = self.number_of_significant_features

        if len(self.significant_features_csvs) == 0:
            raise ValueError(
                f"No significant features CSV file inside directory {self.significant_features_dir}"
            )

        df = pd.concat(
            [pd.read_csv(file) for file in self.significant_features_csvs],
            ignore_index=True,
        )

        if df.empty:
            raise ValueError("No data found inside csv files.")

        filename = (
            f"{self.prefix_filename}_{SIGNIFICANT_FEATURES_FILENAME}"
            if self.prefix_filename
            else SIGNIFICANT_FEATURES_FILENAME
        )
        df.to_csv(os.path.join(self.features_dir, f"{filename}.csv"), index=False)

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

            df = (
                df.groupby(by="features")
                .count()
                .sort_values(by="p_values", ascending=False)
            )
            df.index.name = "features"
            df.to_csv(
                os.path.join(self.features_dir, f"{filename}.csv"),
                index=True,
            )

            if plot:
                _plot_significant_features(
                    df=df.reset_index(),
                    filepath=os.path.join(self.features_dir, filename),
                    overwrite=True,
                )

        return df

    @logger.catch(level="ERROR")
    def select_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        significant_filepath: str,
        random_state: int = 42,
        all_features_filepath: str | None = None,
        all_figures_filepath: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series, int]:
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
        """
        selector = self.FeatureSelector.set_random_state(random_state)
        number_of_significant_features = self.number_of_significant_features

        # Reduced features/columns
        df_selected_features = selector.fit_transform(
            features, labels, top_n=number_of_significant_features
        )

        # Series indexed by feature name; values are p-values sorted by significance.
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
        _plot_significant_features(
            df=all_selected_features,
            filepath=all_figures_filepath,
            top_features=top_features or self.number_of_significant_features,
            overwrite=self.overwrite,
            dpi=150,
        )

        return None

    def _generate_filepaths(
        self,
        random_state: int,
    ) -> tuple[str, str, str, str, str, str, str, str, bool]:
        """Generate filepaths based on random seed.

        Args:
            random_state (int): Random seed for feature selection.

        Returns:
            str: Base filename
            str: Significant filepath
            str: All features filepath
            str: All figures filepath
            str: Model filepath
            str: Metrics filepath
            str: X_test filepath (held-out features for this seed)
            str: y_test filepath (held-out labels for this seed)
            bool: Whether all required output files already exist and can be skipped
        """
        filename = (
            f"{self.prefix_filename}_{random_state:05d}"
            if self.prefix_filename
            else f"{random_state:05d}"
        )
        significant_filepath = os.path.join(
            self.significant_features_dir, f"{filename}.csv"
        )
        all_features_filepath = os.path.join(self.all_features_dir, f"{filename}.csv")
        all_figures_filepath = os.path.join(
            self.significant_figures_dir, f"{filename}.png"
        )
        model_filepath = os.path.join(self.models_dir, f"{filename}.pkl")
        metrics_filepath = os.path.join(self.metrics_dir, f"{filename}.json")
        X_test_filepath = os.path.join(self.tests_dir, f"{filename}_X_test.csv")
        y_test_filepath = os.path.join(self.tests_dir, f"{filename}_y_test.csv")

        can_skip = not self.overwrite and all(
            os.path.isfile(p)
            for p in [
                significant_filepath,
                model_filepath,
                metrics_filepath,
                all_features_filepath,
                all_figures_filepath,
                X_test_filepath,
                y_test_filepath,
            ]
        )

        return (
            filename,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            model_filepath,
            metrics_filepath,
            X_test_filepath,
            y_test_filepath,
            can_skip,
        )

    def _setup_grid_search(
        self,
        random_state: int,
        features: pd.DataFrame,
        labels: pd.Series,
        top_n_features: list[str],
    ) -> tuple[ClassifierModel, GridSearchCV, Any]:
        """Set up, fit, and return a GridSearchCV with the configured classifier.

        Args:
            random_state (int): Random seed for the classifier.
            features (pd.DataFrame): Training features (unsliced).
            labels (pd.Series): Training labels.
            top_n_features (list[str]): Column names to select from features.

        Returns:
            tuple: Configured classifier, fitted GridSearchCV, best estimator.
        """
        clf = self.ClassifierModel.set_random_state(random_state=random_state)
        grid_search = GridSearchCV(
            estimator=clf.model,
            param_grid=clf.grid,
            cv=clf.get_cv_splitter(),
            scoring="balanced_accuracy",
            n_jobs=1,  # Nested parallelism would deadlock.
            verbose=0,
        )
        grid_search.fit(features[top_n_features], labels)
        return clf, grid_search, grid_search.best_estimator_

    def _run_jobs(self, method: Callable, jobs: list[tuple]) -> list[tuple]:
        """Dispatch jobs sequentially or in parallel depending on n_jobs."""
        if self.n_jobs > 1:
            logger.info(f"Running on {self.n_jobs} job(s)")
            with Pool(self.n_jobs) as pool:
                return pool.starmap(method, jobs)
        return [method(*job) for job in jobs]

    def _save_models_registry(
        self,
        records: list[dict],
        random_state: int,
        total_seed: int,
    ) -> str:
        """Build and save the trained-models registry CSV.

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

        Returns:
            str: The suffix string used in the output filename.

        Raises:
            ValueError: If no records were produced (no models trained).
        """
        suffix = (
            f"{self.classifier_id}_rs-{random_state}_ts-{total_seed}"
            f"_top-{self.number_of_significant_features}"
        )
        filename = f"trained_model_{suffix}.csv"

        df = pd.DataFrame(records).set_index("random_state")
        if df.empty:
            raise ValueError("No significant features or trained models found.")
        csv = os.path.join(self.classifier_dir, filename)
        df.to_csv(csv, index=True)
        self.df = df
        self.csv = csv

        return suffix

    def _run_train_and_evaluate(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> tuple[int, str, str, dict, str, str]:
        """Train feature selection and classifier model for a single random seed.

        Performs:
        1. Train/test split (80/20, stratified)
        2. Random under-sampling on training set only
        3. Feature selection on training set only
        4. Classifier training with GridSearchCV (using configured classifier type)
        5. Evaluation on held-out test set

        Args:
            random_state (int): Base random state value.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.
            save_features (bool, optional): Save all ranked features. Defaults to False.
            plot_features (bool, optional): Generate feature plots. Defaults to False.

        Returns:
            tuple[int, str, str, dict, str, str]: Random state value, path to significant
                features CSV, trained model filepath, metrics dictionary, path to held-out
                X_test CSV, and path to held-out y_test CSV.
        """
        if self.debug:
            logger.debug(
                f"_run_train_and_evaluate: seed={random_state}, random_state={random_state}, state={random_state}"
            )

        # ========== STEP 0: Preparation ==========
        (
            _,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            model_filepath,
            metrics_filepath,
            X_test_filepath,
            y_test_filepath,
            can_skip,
        ) = self._generate_filepaths(random_state=random_state)

        # Skip if files already exist
        if can_skip:
            logger.info(f"Seed {random_state:05d} already trained.")
            with open(metrics_filepath) as f:
                metrics = json.load(f)
            return (
                random_state,
                significant_filepath,
                model_filepath,
                metrics,
                X_test_filepath,
                y_test_filepath,
            )

        logger.info(f"Training Seed: {random_state:05d}")

        # ========== STEP 1: Train/Test Split ==========
        # X_train, X_test, y_train, y_test
        features_train, features_test, labels_train, labels_test = train_test_split(
            self.df_features,
            self.df_labels,
            test_size=TRAIN_TEST_SPLIT,
            random_state=random_state,
            stratify=self.df_labels,
        )

        # Save held-out test splits for aggregate evaluation
        features_test.to_csv(X_test_filepath)
        labels_test.to_csv(y_test_filepath)

        # ========== STEP 2: Resample ONLY Training Data ==========
        features_train_resampled, labels_train_resampled = random_under_sampler(
            features=features_train,
            labels=labels_train,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

        # ========== STEP 3: Feature Selection ONLY on Training Data ==========
        (
            features_train_resampled_selected,
            top_selected_features,
            _selected_features,
            _n_features,
        ) = self.select_features(
            features=features_train_resampled,
            labels=labels_train_resampled,
            random_state=random_state,
            significant_filepath=significant_filepath,
            all_features_filepath=all_features_filepath if save_features else None,
            all_figures_filepath=all_figures_filepath if plot_features else None,
        )

        # ========== STEP 4: Cross-Validation with Dynamic Classifier ==========
        top_n_features = top_selected_features.index.tolist()
        features_test_selected = features_test[top_n_features]

        clf, grid_search, best_model = self._setup_grid_search(
            random_state,
            features_train_resampled_selected,
            labels_train_resampled,
            top_n_features,
        )

        # ========== STEP 5: Evaluate on Test Set ==========
        labels_pred = best_model.predict(features_test_selected)

        # ========== STEP 6: Save Outputs ==========
        # Save model
        joblib.dump(best_model, model_filepath)

        if self.verbose:
            logger.info(f"{random_state:05d}: Model at {model_filepath}")

        # Get and save metrics
        model_evaluator = ModelEvaluator(
            model=grid_search,
            X_test=features_test_selected,
            y_test=labels_test,
            model_name=self.classifier_name,
            output_dir=self.classifier_dir,
            selected_features=top_n_features,
        )

        grid_params = grid_search.best_params_
        with open(metrics_filepath, "w") as f:
            metrics = model_evaluator.get_metrics()
            metrics.update(
                {
                    "cv_strategy": clf.cv_strategy,
                    "random_state": random_state,
                    "best_params_criterion": grid_params["criterion"],
                    "best_params_max_depth": grid_params["max_depth"],
                    "best_params_max_features": grid_params["max_features"],
                    "best_params_n_estimators": grid_params["n_estimators"],
                    "best_cv_score": grid_search.best_score_,
                }
            )
            json.dump(metrics, f, indent=4)

        if self.verbose:
            logger.info(f"{random_state:05d}: Metrics at {metrics_filepath}")
            logger.info(
                f"Seed {random_state:05d} - Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}"
            )

        return (
            random_state,
            significant_filepath,
            model_filepath,
            metrics,
            X_test_filepath,
            y_test_filepath,
        )

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
        3. Feature selection on training set only
        4. Classifier training with GridSearchCV (using configured classifier type)
        5. Evaluation on held-out test set

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
            >>> ModelTrainer(...).train_and_evaluate(
            ...     random_state=42,
            ...     total_seed=100,
            ... )
        """
        self.create_directories()
        os.makedirs(self.metrics_dir, exist_ok=True)

        if save_all_features:
            os.makedirs(self.all_features_dir, exist_ok=True)

        if plot_significant_features:
            os.makedirs(self.significant_figures_dir, exist_ok=True)

        # Set job parameters
        random_states: list[int] = [random_state + seed for seed in range(total_seed)]
        jobs: list[tuple[int, str | float, bool, bool]] = [
            (
                _random_state,
                sampling_strategy,
                save_all_features,
                plot_significant_features,
            )
            for _random_state in random_states
        ]

        # Accumulate results across all seeds before aggregating
        all_metrics = []
        significant_features_and_trained_models = []

        for (
            _random_state,
            significant_features_csv,
            trained_model_filepath,
            metrics,
            X_test_filepath,
            y_test_filepath,
        ) in self._run_jobs(self._run_train_and_evaluate, jobs):
            self.significant_features_csvs.append(significant_features_csv)
            significant_features_and_trained_models.append(
                {
                    "random_state": _random_state,
                    "significant_features_csv": significant_features_csv,
                    "trained_model_filepath": trained_model_filepath,
                    "X_test_filepath": X_test_filepath,
                    "y_test_filepath": y_test_filepath,
                }
            )
            all_metrics.append(metrics)

        # Aggregate feature selection results
        self.df_significant_features = self.concat_significant_features(
            plot=plot_significant_features,
        )

        suffix_filename = self._save_models_registry(
            significant_features_and_trained_models, random_state, total_seed
        )

        # Aggregate and save metrics
        self._aggregate_metrics(all_metrics, suffix_filename=suffix_filename)

        if self.verbose:
            logger.info(f"Models saved to: {self.csv}")

        return None

    def _run_train(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> tuple[int, str, str]:
        """Train on the full dataset (no train/test split) for a single random seed.

        Performs:
        1. Random under-sampling on full dataset
        2. Feature selection on resampled data
        3. Classifier training with GridSearchCV (using configured classifier type)
        4. Save trained model

        Args:
            random_state (int): Random seed for reproducibility.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.
            save_features (bool, optional): Save all ranked features. Defaults to False.
            plot_features (bool, optional): Generate feature plots. Defaults to False.

        Returns:
            tuple[int, str, str]: Random state value, path to significant features CSV,
                and trained model filepath.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> random_state, sig_csv, model_path = trainer._run_train(random_state=42)
        """
        if self.debug:
            logger.debug(f"_run_train: seed={random_state}")

        # ========== STEP 0: Preparation ==========
        (
            _filename,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            model_filepath,
            _metrics_filepath,
            _X_test_filepath,  # not used — train() has no test split
            _y_test_filepath,  # not used — train() has no test split
            _can_skip,  # ignored; _run_train() uses its own reduced skip check below
        ) = self._generate_filepaths(random_state=random_state)

        # For train(), skip only based on significant + model files (no metrics,
        # no figures — figures are optional and controlled by plot_significant_features).
        can_skip = (
            not self.overwrite
            and os.path.isfile(significant_filepath)
            and os.path.isfile(model_filepath)
        )

        if can_skip:
            logger.info(f"Seed {random_state:05d} already fitted.")
            return random_state, significant_filepath, model_filepath

        # ========== STEP 1: Resample Full Dataset ==========
        features_resampled, labels_resampled = random_under_sampler(
            features=self.df_features,
            labels=self.df_labels,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

        # ========== STEP 2: Feature Selection on Resampled Data ==========
        (
            features_resampled_selected,
            top_selected_features,
            _selected_features,
            _n_features,
        ) = self.select_features(
            features=features_resampled,
            labels=labels_resampled,
            random_state=random_state,
            significant_filepath=significant_filepath,
            all_features_filepath=all_features_filepath if save_features else None,
            all_figures_filepath=all_figures_filepath if plot_features else None,
        )

        # ========== STEP 3: Cross-Validation with Dynamic Classifier ==========
        logger.info(f"Fitting Seed: {random_state:05d}")

        top_n_features = top_selected_features.index.tolist()

        _, _, best_model = self._setup_grid_search(
            random_state,
            features_resampled_selected,
            labels_resampled,
            top_n_features,
        )

        # ========== STEP 4: Save Model ==========
        joblib.dump(best_model, model_filepath)

        if self.verbose:
            logger.info(f"Model {random_state:05d}: {model_filepath}")

        return random_state, significant_filepath, model_filepath

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
        1. Random under-sampling on full dataset
        2. Feature selection on resampled data
        3. Classifier training with GridSearchCV
        4. Save trained model

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
            ... )
            >>> trainer.train(random_state=0, total_seed=5)
        """

        # Since we are not using evaluation, we change the folder name from
        # ``model-with-evaluation`` to ``model-only``
        output_dir = self.output_dir.replace("model-with-evaluation", "model-only")

        # Update current directories with new output directory
        self.update_directories(output_dir=output_dir)

        if save_all_features:
            os.makedirs(self.all_features_dir, exist_ok=True)

        if plot_significant_features:
            os.makedirs(self.significant_figures_dir, exist_ok=True)

        random_states: list[int] = [random_state + seed for seed in range(total_seed)]
        jobs: list[tuple[int, str | float, bool, bool]] = [
            (
                _random_state,
                sampling_strategy,
                save_all_features,
                plot_significant_features,
            )
            for _random_state in random_states
        ]

        significant_features_and_trained_models = []

        for (
            _random_state,
            significant_features_csv,
            trained_model_filepath,
        ) in self._run_jobs(self._run_train, jobs):
            self.significant_features_csvs.append(significant_features_csv)
            significant_features_and_trained_models.append(
                {
                    "random_state": _random_state,
                    "significant_features_csv": significant_features_csv,
                    "trained_model_filepath": trained_model_filepath,
                }
            )

        # Aggregate feature selection results
        self.df_significant_features = self.concat_significant_features(
            plot=plot_significant_features,
        )

        self._save_models_registry(
            significant_features_and_trained_models, random_state, total_seed
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
        self, all_metrics: list[dict], suffix_filename: str = ""
    ) -> None:
        """Aggregate metrics across all seeds.

        Computes mean and std for each metric and saves to CSV.
        Also saves all individual metrics for detailed analysis.

        Args:
            all_metrics (list[dict]): List of metric dictionaries, one per seed.
            suffix_filename (str, optional): Suffix appended to output filenames.
                Defaults to ``""``.

        Example:
            >>> trainer = ModelTrainer(...)
            >>> trainer.train_and_evaluate(total_seed=100)
            >>> # Creates: all_metrics_{suffix}.csv and metrics_summary_{suffix}.csv
        """
        os.makedirs(self.metrics_dir, exist_ok=True)

        df_metrics = pd.DataFrame(all_metrics)

        # Calculate summary statistics
        summary = df_metrics.describe().T
        summary_filepath = os.path.join(
            self.classifier_dir, f"metrics_summary_{suffix_filename}.csv"
        )
        summary.to_csv(summary_filepath)

        # Save all individual metrics
        all_metrics_filepath = os.path.join(
            self.classifier_dir, f"all_metrics_{suffix_filename}.csv"
        )
        df_metrics.to_csv(
            all_metrics_filepath,
            index=False,
        )

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
