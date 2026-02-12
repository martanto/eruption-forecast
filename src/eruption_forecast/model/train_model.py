import os
import json
from typing import Any, Literal
from multiprocessing import Pool

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from eruption_forecast.plot import plot_significant_features as plot_significant
from eruption_forecast.utils import get_metrics, random_under_sampler
from eruption_forecast.logger import logger
from eruption_forecast.features.constants import (
    ID_COLUMN,
    ERUPTED_COLUMN,
    DATETIME_COLUMN,
    SIGNIFICANT_FEATURES_FILENAME,
)
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.features.feature_selector import FeatureSelector


class TrainModel:
    """Train feature-selection and classifier models over multiple random seeds.

    Loads pre-extracted features and labels, and for each seed performs:
    1. Train/test split (80/20, stratified) to prevent data leakage
    2. Random under-sampling on training set only
    3. Feature selection on training set only using tsfresh
    4. Classifier training with GridSearchCV (supports RF, GB, SVM, LR, NN, etc.)
    5. Evaluation on held-out test set

    This multi-seed approach provides robust feature selection and model
    evaluation by averaging results across many random data splits, reducing
    the risk of overfitting to a particular train/test split.

    Args:
        extracted_features_csv (str): Path to the extracted features CSV file.
            Can be found in:
                ``<cwd>output/{nslc}/features/all_extracted_features_{dates}.csv``, or
                ``<cwd>output/{nslc}/features/relevant_features_{dates}.csv``
        label_features_csv (str): Path to the label CSV file. Built by using LabelBuilder.
            Can be found in: ``<cwd>output/{nslc}/features/label_features_{dates}.csv``
        output_dir (str, optional): Directory for output files. Defaults to
            ``<cwd>/output/trainings``.
        n_jobs (int, optional): Number of parallel workers. Defaults to 1.
        prefix_filename (str, optional): Prefix for output filenames. Defaults to None.
        classifier (str, optional): Classifier type ("rf", "gb", "svm", "lr", "nn",
            "dt", "knn", "nb", "voting"). Defaults to "rf".
        cv_strategy (str, optional): Cross-validation strategy ("shuffle", "stratified",
            "timeseries"). Defaults to "shuffle".
        cv_splits (int, optional): Number of CV splits. Defaults to 5.
        number_of_significant_features (int, optional): Number of top features
            to save separately. If provided, creates a separate CSV with
            the most frequently occurring features across seeds. Defaults to 20.
        verbose (bool, optional): Verbose logging. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.

    Raises:
        ValueError: If features or labels are empty, or their lengths
            do not match.

    Example:
        >>> # Train with Random Forest (default)
        >>> trainer = TrainModel(
        ...     features_csv="output/features/extracted_features.csv",
        ...     label_features_csv="output/features/label_features.csv",
        ...     output_dir="output/trainings",
        ...     n_jobs=4,
        ... )
        >>> trainer.train(
        ...     random_state=0,
        ...     total_seed=100,
        ...     number_of_significant_features=20,
        ...     sampling_strategy=0.75,
        ... )

        >>> # Train with Gradient Boosting
        >>> trainer = TrainModel(
        ...     features_csv="output/features/extracted_features.csv",
        ...     label_features_csv="output/features/label_features.csv",
        ...     output_dir="output/trainings",
        ...     classifier="gb",
        ...     n_jobs=4,
        ... )

        >>> # Train with VotingClassifier ensemble and TimeSeriesSplit
        >>> trainer = TrainModel(
        ...     features_csv="output/features/extracted_features.csv",
        ...     label_features_csv="output/features/label_features.csv",
        ...     output_dir="output/trainings",
        ...     classifier="voting",
        ...     cv_strategy="shuffle",
        ...     cv_splits=5,
        ...     n_jobs=4,
        ... )

        >>> # Results saved to:
        >>> # - output/trainings/significant_features.csv (aggregated features)
        >>> # - output/trainings/models/ (trained models)
        >>> # - output/trainings/metrics/ (evaluation metrics)
        >>> # - output/trainings/all_metrics.csv (metrics from all seeds)
        >>> # - output/trainings/metrics_summary.csv (mean/std statistics)
    """

    def __init__(
        self,
        extracted_features_csv: str,
        label_features_csv: str,
        output_dir: str | None = None,
        prefix_filename: str | None = None,
        classifier: Literal[
            "svm", "knn", "dt", "rf", "gb", "nn", "nb", "lr", "voting"
        ] = "rf",
        cv_strategy: Literal["shuffle", "stratified", "timeseries"] = "shuffle",
        cv_splits: int = 5,
        number_of_significant_features: int = 20,
        feature_selection_method: Literal[
            "tsfresh", "random_forest", "combined"
        ] = "tsfresh",
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # =========================
        # Set DEFAULT parameter
        # =========================
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
        )
        classifier_name = classifier_model.name
        classifier_cv_name = classifier_model.cv_name

        # Output training dir: ``<cwd>/output/trainings``
        output_dir = output_dir or os.path.join(os.getcwd(), "output", "trainings")

        # Filtered features dir: ``<output_dir>/features``
        features_dir = os.path.join(output_dir, "features")

        # All features dir: ``<features_dir>/all_features``
        all_features_dir = os.path.join(features_dir, "all_features")

        # Significant features dir: ``<features_dir>/significant_features``
        significant_features_dir = os.path.join(features_dir, "significant_features")

        # Plot significant features dir: ``<features_dir>/figures/significant``
        figures_dir = os.path.join(features_dir, "figures")
        significant_figures_dir = os.path.join(figures_dir, "significant")

        # Classifier training dir: ``<features_dir>/<classifier_name>/<classifier_cv_name>``
        classifier_dir = os.path.join(
            output_dir, "classifier", classifier_name, classifier_cv_name
        )

        # Classifier training model dir: ``<classifier_dir>/models``
        models_dir = os.path.join(classifier_dir, "models")

        # Classifier metrics dir: ``<classifier_dir>/metrics``
        metrics_dir = os.path.join(classifier_dir, "metrics")

        # =========================
        # Set DEFAULT properties
        # =========================
        self.df_features = df_features
        self.df_labels = df_labels
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.prefix_filename = prefix_filename
        self.classifier = classifier
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.number_of_significant_features = number_of_significant_features
        self.feature_selection_method = feature_selection_method
        self.overwrite = overwrite
        self.verbose = verbose
        self.debug = debug

        # =========================
        # Set ADDITIONAL properties (derived values)
        # =========================
        self.FeatureSelector = FeatureSelector(
            method=feature_selection_method, verbose=verbose
        )
        self.ClassifierModel = classifier_model
        self.classifier_name = classifier_name
        self.features_dir = features_dir
        self.significant_features_dir = significant_features_dir
        self.all_features_dir = all_features_dir
        self.figures_dir = figures_dir
        self.significant_figures_dir = significant_figures_dir
        self.classifier_dir = classifier_dir
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir

        # =========================
        # Will be set after train() method called
        # =========================
        self.significant_features_csvs: list[str] = []
        self.df_significant_features: pd.DataFrame = pd.DataFrame()

        # Will contain pair of significant features csv and trained model filepath
        self.df: pd.DataFrame = pd.DataFrame()
        self.csv: str | None = None

        # =========================
        # Will be updated after _generate_filepath() method called
        # =========================
        self.can_skip: bool = False

        # =========================
        # Validate and create directories
        # =========================
        self.validate()
        self.create_directories()

        # =========================
        # Verbose and logging
        # =========================
        if verbose:
            logger.info(
                f"Train model using {n_jobs} jobs with {classifier_name} classifier "
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
            >>> trainer = TrainModel(
            ...     features_csv="features.csv",
            ...     label_features_csv="labels.csv"
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

    def update_grid_params(
        self, classifier: ClassifierModel, grid_params: dict[str, Any]
    ) -> ClassifierModel:
        """Updates grid parameters with new grid parameters.

        Update self.ClassifierModel.grid value.

        Args:
            classifier (ClassifierModel): Classifier model to be updated.
            grid_params (dict): Grid search parameters.

        Returns:
            self (Self): TrainModel class
        """
        current_grid = classifier.grid
        classifier.grid = grid_params
        if self.verbose:
            logger.info(
                f"Your current grid parameters {current_grid} has been updated to{grid_params}."
            )

        return classifier

    def create_directories(self) -> None:
        """Create required output directories for training results.

        Creates the main output directory and subdirectories for storing
        significant features CSVs, trained models, and metrics. Called
        automatically during initialization.

        Example:
            >>> trainer = TrainModel(...)
            >>> trainer.create_directories()  # Called in __init__
            >>> # Creates: output_dir/, significant_features/, models/, metrics/
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.significant_features_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

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
            >>> trainer = TrainModel(...)
            >>> trainer.train(total_seed=100)
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
                plot_significant(
                    df=df.reset_index(),
                    filepath=os.path.join(self.features_dir, filename),
                    overwrite=True,
                    dpi=72,
                )

        return df

    def select_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        random_state: int = 42,
        significant_filepath: str | None = None,
        all_features_filepath: str | None = None,
        all_figures_filepath: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Feature selection.

        Based on tsfresh statistical significance testing with FDR control
        or permutation importance analysis

        Args:
            features (pd.DataFrame): Resampled features to select.
            labels (pd.Series): Target labels.
            random_state (int, optional): Random seed for feature selection. Defaults to 42.
            significant_filepath (str, optional): Save path for significant features.
            all_features_filepath (str, optional): Save path for all features.
            all_figures_filepath (str, optional): Save path for all features figures.

        Returns:
            tuple[pd.DataFrame, pd.Series, pd.Series]: Selected features dataframe,
                all selected features, top (n) features
        """
        selector = self.FeatureSelector.set_random_state(random_state)

        # Reduced features/columns
        df_selected_features = selector.fit_transform(
            features, labels, top_n=self.number_of_significant_features
        )

        # Return pd.Series indexed by features name. Value column is p_value
        #
        # Example:
        #
        # ===============================
        # | feature (idx)   | p_value   |
        # ===============================
        # | feature A       | 0.02      |
        # | feature B       | 0.03      |
        # | feature C       | 0.04      |
        all_selected_features = selector.selected_features_

        top_selected_features = all_selected_features.head(
            self.number_of_significant_features
        )

        # Save TOP-N significant features
        top_selected_features.to_csv(significant_filepath, index=True)

        # Save all SELECTED features if requested
        if all_features_filepath:
            all_selected_features.to_csv(all_features_filepath, index=True)
            if all_figures_filepath:
                self._plot_significant_features(
                    all_selected_features=pd.DataFrame(
                        all_selected_features
                    ).reset_index(),
                    all_figures_filepath=all_figures_filepath,
                )

        return df_selected_features, all_selected_features, top_selected_features

    def _plot_significant_features(
        self,
        all_selected_features: pd.DataFrame,
        all_figures_filepath: str,
    ):
        plot_significant(
            df=all_selected_features,
            filepath=all_figures_filepath,
            overwrite=self.overwrite,
            dpi=150,
        )

        return None

    def _generate_filepaths(
        self,
        random_state: int,
    ) -> tuple[str, str, str, str, str, str]:
        """Generate filepaths based on random seed.

        Args:
            random_state (int): Random seed for feature selection.

        Returns:
            str: Base filename
            str: Signficant filepath
            str: All features filepath
            str: All figures filepath
            str: Model filepath
            str: Metrics filepath
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
            self.significant_figures_dir, f"{filename}.jpg"
        )
        model_filepath = os.path.join(self.models_dir, f"{filename}.pkl")
        metrics_filepath = os.path.join(self.metrics_dir, f"{filename}.json")

        self.can_skip = (
            not self.overwrite
            and os.path.isfile(significant_filepath)
            and os.path.isfile(model_filepath)
            and os.path.isfile(metrics_filepath)
            and os.path.isfile(all_features_filepath)
            and os.path.isfile(all_figures_filepath)
        )

        return (
            filename,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            model_filepath,
            metrics_filepath,
        )

    def _train(
        self,
        random_state: int,
        sampling_strategy: str | float = 0.75,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> tuple[int, str, str, dict]:
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
            tuple[int, str, str, dict]: Random state value, path to significant features CSV,
                trained model filepath, and metrics dictionary.

        Example:
            >>> trainer = TrainModel(...)
            >>> csv_path, evaluation_metrics = trainer._train(
            ...     seed=0,
            ...     random_state=42,
            ... )
            >>> print(csv_path)
            "output/trainings/significant_features/00042.csv"
            >>> print(metrics['balanced_accuracy'])
            0.8234
        """
        if self.debug:
            logger.debug(
                f"_train: seed={random_state}, random_state={random_state}, state={random_state}"
            )

        # ========== STEP 0: Preparation ==========
        (
            _,
            significant_filepath,
            all_features_filepath,
            all_figures_filepath,
            model_filepath,
            metrics_filepath,
        ) = self._generate_filepaths(random_state=random_state)

        # Skip if files already exist
        if self.can_skip or not save_features or not plot_features:
            logger.info(f"Seed {random_state:05d} already trained.")
            with open(metrics_filepath) as f:
                metrics = json.load(f)
            return random_state, significant_filepath, model_filepath, metrics

        logger.info(f"Training Seed: {random_state:05d}")

        # ========== STEP 1: Train/Test Split ==========
        # X_train, X_test, y_train, y_test
        features_train, features_test, labels_train, labels_test = train_test_split(
            self.df_features,
            self.df_labels,
            test_size=0.2,
            random_state=random_state,
            stratify=self.df_labels,
        )

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
            selected_features,
            top_selected_features,
        ) = self.select_features(
            features=features_train_resampled,
            labels=labels_train_resampled,
            random_state=random_state,
            significant_filepath=significant_filepath,
            all_features_filepath=all_features_filepath if save_features else None,
            all_figures_filepath=all_figures_filepath if plot_features else None,
        )

        # ========== STEP 4: Cross-Validation with Dynamic Classifier ==========
        # Update random state value to classifier
        clf = self.ClassifierModel.set_random_state(random_state=random_state)

        # Get model and grid from ClassifierModel
        model = clf.model
        param_grid = clf.grid
        cv = clf.get_cv_splitter()

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=1,  # prevent creating multiprocessing inside multiprocessing
            verbose=0,
        )

        # Select top-N features
        top_n_features = top_selected_features.index.tolist()
        features_train_resampled_selected = features_train_resampled_selected[
            top_n_features
        ]
        features_test_selected = features_test[top_n_features]

        grid_search.fit(features_train_resampled_selected, labels_train_resampled)
        best_model = grid_search.best_estimator_

        # ========== STEP 5: Evaluate on Test Set ==========
        labels_pred = best_model.predict(features_test_selected)

        # Get and save metrics
        metrics = get_metrics(
            classifier=clf,
            labels_test=labels_test,
            labels_pred=labels_pred,
            labels_train=labels_train_resampled,
            top_n=len(top_n_features),
            grid_search=grid_search,
            random_state=random_state,
            metrics_filepath=metrics_filepath,
        )

        # ========== STEP 6: Save Outputs ==========
        # Save model
        joblib.dump(best_model, model_filepath)

        if self.verbose:
            logger.info(f"Model {random_state:05d}: {model_filepath}")
            logger.info(f"Metrics {random_state:05d}: {metrics_filepath}")
            logger.info(
                f"Seed {random_state:05d} - Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}"
            )

        return random_state, significant_filepath, model_filepath, metrics

    def train(
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
            >>> TrainModel(...).train(
            ...     random_state=42,
            ...     total_seed=100,
            ... )
        """
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

        # Inittiate all metric values to save calculation metric
        all_metrics = []
        significant_features_and_trained_models = []

        if self.n_jobs == 1:
            # Run test split, resampler, features selection
            for job in jobs:
                (
                    _random_state,
                    significant_features_csv,
                    trained_model_filepath,
                    metrics,
                ) = self._train(*job)

                self.significant_features_csvs.append(significant_features_csv)
                significant_features_and_trained_models.append(
                    {
                        "random_state": _random_state,
                        "significant_features_csv": significant_features_csv,
                        "trained_model_filepath": trained_model_filepath,
                    }
                )
                all_metrics.append(metrics)

        if self.n_jobs > 1:
            # Run test split, resampler, features selection
            logger.info(f"Running on {self.n_jobs} job(s)")
            with Pool(self.n_jobs) as pool:
                results = pool.starmap(self._train, jobs)

                for (
                    _random_state,
                    significant_features_csv,
                    trained_model_filepath,
                    metrics,
                ) in results:
                    self.significant_features_csvs.append(significant_features_csv)
                    significant_features_and_trained_models.append(
                        {
                            "random_state": _random_state,
                            "significant_features_csv": significant_features_csv,
                            "trained_model_filepath": trained_model_filepath,
                        }
                    )
                    all_metrics.append(metrics)

        # Aggregate feature selection results
        self.df_significant_features = self.concat_significant_features(
            plot=plot_significant_features,
        )

        # Aggregate significant features and models

        # Example suffix filename: RandomForestClassifier_rs-0_ts-500_top-20
        # Where:
        #   - RandomForestClassifier the name of the classifier model
        #   - rs-0 is random state with 0 value
        #   - top-20 is number of significant features
        suffix_filename = (
            f"{self.classifier_name}_rs-{random_state}_ts-{total_seed}"
            f"_top-{self.number_of_significant_features}"
        )
        df_models = pd.DataFrame(significant_features_and_trained_models)
        df_models = df_models.set_index("random_state")

        if df_models.empty:
            raise ValueError("No significant features or trained models found.")

        models_filename = f"trained_model_{suffix_filename}.csv"
        csv = os.path.join(self.classifier_dir, models_filename)

        # Save features and model to self.classifier_dir
        df_models.to_csv(csv, index=True)

        # Aggregate and save metrics
        self._aggregate_metrics(all_metrics, suffix_filename=suffix_filename)

        # set values
        self.df = df_models
        self.csv = csv

        if self.verbose:
            logger.info(f"Models saved to: {csv}")

        return None

    def _aggregate_metrics(
        self, all_metrics: list[dict], suffix_filename: str = ""
    ) -> None:
        """Aggregate metrics across all seeds.

        Computes mean and std for each metric and saves to CSV.
        Also saves all individual metrics for detailed analysis.

        Args:
            all_metrics (list[dict]): List of metric dictionaries from each seed.

        Example:
            >>> trainer = TrainModel(...)
            >>> trainer.train(total_seed=100)
            >>> # Creates: all_metrics.csv and metrics_summary.csv
        """
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
