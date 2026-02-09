# Standard library imports
import json
import os
from multiprocessing import Pool
from typing import Literal

# Third party imports
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

# Project imports
from eruption_forecast.features.constants import (
    DATETIME_COLUMN,
    ERUPTED_COLUMN,
    ID_COLUMN,
    SIGNIFICANT_FEATURES_FILENAME,
)
from eruption_forecast.logger import logger
from eruption_forecast.model.classifier_model import ClassifierModel
from eruption_forecast.plot import plot_significant_features as plot_significant
from eruption_forecast.utils import (
    get_significant_features,
    random_under_sampler,
)


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
        features_csv (str): Path to the extracted features CSV file.
        label_csv (str): Path to the label CSV file.
        output_dir (str, optional): Directory for output files. Defaults to
            ``<cwd>/output/trainings``.
        n_jobs (int, optional): Number of parallel workers. Defaults to 1.
        prefix_filename (str, optional): Prefix for output filenames. Defaults to None.
        classifier (str, optional): Classifier type ("rf", "gb", "svm", "lr", "nn",
            "dt", "knn", "nb", "voting"). Defaults to "rf".
        cv_strategy (str, optional): Cross-validation strategy ("shuffle", "stratified",
            "timeseries"). Defaults to "shuffle".
        cv_splits (int, optional): Number of CV splits. Defaults to 5.
        verbose (bool, optional): Verbose logging. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.

    Raises:
        ValueError: If features or labels are empty, or their lengths
            do not match.

    Example:
        >>> # Train with Random Forest (default)
        >>> trainer = TrainModel(
        ...     features_csv="output/features/extracted_features.csv",
        ...     label_csv="output/features/label_features.csv",
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
        ...     label_csv="output/features/label_features.csv",
        ...     output_dir="output/trainings",
        ...     classifier="gb",
        ...     n_jobs=4,
        ... )

        >>> # Train with VotingClassifier ensemble and TimeSeriesSplit
        >>> trainer = TrainModel(
        ...     features_csv="output/features/extracted_features.csv",
        ...     label_csv="output/features/label_features.csv",
        ...     output_dir="output/trainings",
        ...     classifier="voting",
        ...     cv_strategy="timeseries",
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
        features_csv: str,
        label_csv: str,
        output_dir: str | None = None,
        n_jobs: int = 1,
        prefix_filename: str | None = None,
        classifier: Literal[
            "svm", "knn", "dt", "rf", "gb", "nn", "nb", "lr", "voting"
        ] = "rf",
        cv_strategy: Literal["shuffle", "stratified", "timeseries"] = "shuffle",
        cv_splits: int = 5,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        # Set DEFAULT parameter
        df_features = pd.read_csv(features_csv, index_col=0)
        df_labels = pd.read_csv(label_csv)
        if ID_COLUMN in df_labels.columns:
            df_labels = df_labels.set_index(ID_COLUMN)

        if DATETIME_COLUMN in df_labels.columns:
            df_labels = df_labels.drop(DATETIME_COLUMN, axis=1)

        df_labels = df_labels[ERUPTED_COLUMN]
        output_dir = output_dir or os.path.join(os.getcwd(), "output", "trainings")

        significant_features_dir = os.path.join(output_dir, "significant_features")
        all_features_dir = os.path.join(output_dir, "all_features")

        figures_dir = os.path.join(output_dir, "figures")
        significant_figures_dir = os.path.join(figures_dir, "significant")

        models_dir = os.path.join(output_dir, "models")
        metrics_dir = os.path.join(output_dir, "metrics")

        # Set DEFAULT properties
        self.df_features = df_features
        self.df_labels = df_labels
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.prefix_filename = prefix_filename
        self.classifier = classifier
        self.cv_strategy = cv_strategy
        self.cv_splits = cv_splits
        self.verbose = verbose
        self.debug = debug

        # Set ADDITIONAL properties
        self.significant_features_dir = significant_features_dir
        self.all_features_dir = all_features_dir
        self.figures_dir = figures_dir
        self.significant_figures_dir = significant_figures_dir
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        self.csvs: list[str] = []
        self.df_significant_features: pd.DataFrame = pd.DataFrame()

        # Validate and create directories
        self.validate()
        self.create_directories()

        if verbose:
            logger.info(
                f"Train model using {n_jobs} jobs with {classifier} classifier "
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
            ...     label_csv="labels.csv"
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

    def concat_significant_features(
        self, number_of_significant_features: int | None = None, plot: bool = False
    ) -> pd.DataFrame:
        """Concatenate significant features from all training seeds.

        Merges significant feature CSVs from all random seeds, aggregates
        by feature name to count occurrences across seeds, and saves the
        top-N most frequently selected features.

        Args:
            number_of_significant_features (int, optional): Number of top
                features to save separately. If provided, creates a separate
                CSV with the most frequently occurring features across seeds.
                Defaults to None.
            plot (bool, optional): If True, generates a plot of the top
                features. Defaults to False.

        Returns:
            pd.DataFrame: Concatenated significant features with counts of
                how many times each feature appeared across all seeds.

        Raises:
            ValueError: If no CSV files are found in self.csvs.
            ValueError: If concatenated DataFrame is empty.

        Example:
            >>> trainer = TrainModel(...)
            >>> trainer.train(total_seed=100)
            >>> df = trainer.concat_significant_features(
            ...     number_of_significant_features=20,
            ...     plot=True
            ... )
            >>> print(df.head())
            # Shows features and their occurrence counts across seeds
        """
        if len(self.csvs) == 0:
            raise ValueError(
                f"No significant features CSV file inside directory {self.significant_features_dir}"
            )

        df = pd.concat([pd.read_csv(file) for file in self.csvs], ignore_index=True)

        if df.empty:
            raise ValueError("No data found inside csv files.")

        filename = (
            f"{self.prefix_filename}_{SIGNIFICANT_FEATURES_FILENAME}"
            if self.prefix_filename
            else SIGNIFICANT_FEATURES_FILENAME
        )
        df.to_csv(os.path.join(self.output_dir, f"{filename}.csv"), index=False)

        # Save number_of_significant_features
        if (
            number_of_significant_features is not None
            and number_of_significant_features > 0
        ):
            filename = (
                f"{self.prefix_filename}_top_{SIGNIFICANT_FEATURES_FILENAME}"
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
                os.path.join(self.output_dir, filename),
                index=True,
            )

            if plot:
                plot_significant(
                    df=df.reset_index(),
                    filepath=os.path.join(self.output_dir, filename),
                    overwrite=True,
                    dpi=72,
                )

        return df

    def _train(
        self,
        seed: int,
        random_state: int,
        number_of_significant_features: int = 20,
        sampling_strategy: str | float = 0.75,
        overwrite: bool = False,
        save_features: bool = False,
        plot_features: bool = False,
    ) -> tuple[str, dict]:
        """Train feature selection and classifier model for a single random seed.

        Performs:
        1. Train/test split (80/20, stratified)
        2. Random under-sampling on training set only
        3. Feature selection on training set only
        4. Classifier training with GridSearchCV (using configured classifier type)
        5. Evaluation on held-out test set

        Args:
            seed (int): Seed number (0 to total_seed-1).
            random_state (int): Base random state value.
            number_of_significant_features (int, optional): Number of top
                features to select. Defaults to 20.
            sampling_strategy (str | float, optional): Under-sampling ratio.
                Defaults to 0.75.
            overwrite (bool, optional): Overwrite existing files. Defaults to False.
            save_features (bool, optional): Save all ranked features. Defaults to False.
            plot_features (bool, optional): Generate feature plots. Defaults to False.

        Returns:
            tuple[str, dict]: Path to significant features CSV and metrics dictionary.

        Example:
            >>> trainer = TrainModel(...)
            >>> csv_path, metrics = trainer._train(
            ...     seed=0,
            ...     random_state=42,
            ...     number_of_significant_features=20
            ... )
            >>> print(csv_path)
            "output/trainings/significant_features/00042.csv"
            >>> print(metrics['balanced_accuracy'])
            0.8234
        """
        state = random_state + seed
        logger.debug(f"_train: seed={seed}, random_state={random_state}, state={state}")

        # Create filename and filepath
        filename = (
            f"{self.prefix_filename}_{state:05d}"
            if self.prefix_filename
            else f"{state:05d}"
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

        # Skip if files already exist
        can_skip = (
            not overwrite
            and os.path.isfile(significant_filepath)
            and os.path.isfile(model_filepath)
            and os.path.isfile(metrics_filepath)
            and (not save_features or os.path.isfile(all_features_filepath))
            and (not plot_features or os.path.isfile(all_figures_filepath))
        )

        if can_skip:
            logger.info(f"Seed {seed:05d} already trained.")
            with open(metrics_filepath) as f:
                metrics = json.load(f)
            return significant_filepath, metrics

        logger.info(f"Training Seed: {seed:05d}")

        # ========== STEP 1: Train/Test Split ==========
        features_train, features_test, labels_train, labels_test = train_test_split(
            self.df_features,
            self.df_labels,
            test_size=0.2,
            random_state=state,
            stratify=self.df_labels,  # type: ignore
        )

        # ========== STEP 2: Resample ONLY Training Data ==========
        features_train_resampled, labels_train_resampled = random_under_sampler(
            features=features_train,
            labels=labels_train,
            sampling_strategy=sampling_strategy,
            random_state=state,
        )

        # ========== STEP 3: Feature Selection ONLY on Training Data ==========
        significant_features = get_significant_features(
            features=features_train_resampled,
            labels=labels_train_resampled,
        )

        # Save all features if requested
        if save_features:
            significant_features.to_csv(all_features_filepath, index=True)
            if plot_features:
                plot_significant(
                    df=pd.DataFrame(significant_features).reset_index(),
                    filepath=all_figures_filepath,
                    overwrite=overwrite,
                    dpi=72,
                )

        # Select top-N features
        top_features = significant_features.head(
            number_of_significant_features
        ).index.tolist()
        features_train_selected = features_train_resampled[top_features]
        features_test_selected = features_test[top_features]

        # Save top-N significant features
        significant_features.head(number_of_significant_features).to_csv(
            significant_filepath, index=True
        )

        # ========== STEP 4: Cross-Validation with Dynamic Classifier ==========
        # Create classifier model with appropriate CV strategy
        clf_model = ClassifierModel(
            classifier=self.classifier,
            random_state=state,
            cv_strategy=self.cv_strategy,
            n_splits=self.cv_splits,
        )

        # Get model and grid from ClassifierModel
        model = clf_model.model
        param_grid = clf_model.grid
        cv = clf_model.get_cv_splitter()

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=1,
            verbose=0,
        )

        grid_search.fit(features_train_selected, labels_train_resampled)  # type: ignore
        best_model = grid_search.best_estimator_

        # ========== STEP 5: Evaluate on Test Set ==========
        y_pred = best_model.predict(features_test_selected)

        metrics = {
            "seed": seed,
            "random_state": state,
            "classifier": self.classifier,
            "cv_strategy": self.cv_strategy,
            "cv_splits": self.cv_splits,
            "n_train": len(labels_train_resampled),
            "n_test": len(labels_test),
            "n_features": number_of_significant_features,
            "accuracy": accuracy_score(labels_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(labels_test, y_pred),
            "f1_score": f1_score(labels_test, y_pred),
            "precision": precision_score(labels_test, y_pred),
            "recall": recall_score(labels_test, y_pred),
            "best_params": grid_search.best_params_,
            "best_cv_score": grid_search.best_score_,
        }

        # ========== STEP 6: Save Outputs ==========
        # Save model
        joblib.dump(best_model, model_filepath)

        # Save metrics
        with open(metrics_filepath, "w") as f:
            json.dump(metrics, f, indent=2)

        if self.verbose:
            logger.info(
                f"Seed {state:05d} - Test Balanced Accuracy: {metrics['balanced_accuracy']:.4f}"
            )

        return significant_filepath, metrics

    def train(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        number_of_significant_features: int = 20,
        sampling_strategy: str | float = 0.75,
        save_all_features: bool = False,
        plot_significant_features: bool = False,
        overwrite: bool = False,
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
            number_of_significant_features (int, optional): Number of top features
                to save per seed. Defaults to 20.
            sampling_strategy (str | float, optional): Under-sampling ratio for
                balancing classes. Defaults to 0.75.
            save_all_features (bool, optional): Save all features per seed,
                not just top-N. Defaults to False.
            plot_significant_features (bool, optional): Generate feature importance
                plots. Defaults to False.
            overwrite (bool, optional): Overwrite existing output files.
                Defaults to False.

        Example:
            >>> trainer.train(
            ...     random_state=42,
            ...     total_seed=100,
            ...     number_of_significant_features=20,
            ... )
        """
        if save_all_features:
            os.makedirs(self.all_features_dir, exist_ok=True)

        if plot_significant_features:
            os.makedirs(self.significant_figures_dir, exist_ok=True)

        jobs = [
            (
                seed,
                random_state,
                number_of_significant_features,
                sampling_strategy,
                overwrite,
                save_all_features,
                plot_significant_features,
            )
            for seed in range(total_seed)
        ]

        all_metrics = []

        if self.n_jobs == 1:
            for job in jobs:
                csv, metrics = self._train(*job)
                self.csvs.append(csv)
                all_metrics.append(metrics)

        if self.n_jobs > 1:
            logger.info(f"Running on {self.n_jobs} job(s)")
            with Pool(self.n_jobs) as pool:
                results = pool.starmap(self._train, jobs)
                for csv, metrics in results:
                    self.csvs.append(csv)
                    all_metrics.append(metrics)

        # Aggregate feature selection results
        self.df_significant_features = self.concat_significant_features(
            number_of_significant_features=number_of_significant_features,
            plot=plot_significant_features,
        )

        # Aggregate and save metrics
        self._aggregate_metrics(all_metrics)

        return None

    def _aggregate_metrics(self, all_metrics: list[dict]) -> None:
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
        summary.to_csv(os.path.join(self.output_dir, "metrics_summary.csv"))

        # Save all individual metrics
        df_metrics.to_csv(os.path.join(self.output_dir, "all_metrics.csv"), index=False)

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
