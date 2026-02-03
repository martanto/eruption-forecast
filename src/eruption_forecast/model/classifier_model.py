"""RandomForest classifier trained via GridSearchCV over multiple random seeds.

Each seed runs randomised under-sampling followed by a full grid search over
a ``RandomForestClassifier`` using ``ShuffleSplit`` cross-validation.  The best
estimator from each seed is persisted as a ``joblib``-pickled ``.pkl`` file.
"""

# Standard library imports
import os
from multiprocessing import Pool

# Third party imports
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split

# Project imports
from eruption_forecast.features.constants import (
    DATETIME_COLUMN,
    ERUPTED_COLUMN,
    ID_COLUMN,
)
from eruption_forecast.logger import logger
from eruption_forecast.utils import random_under_sampler

DEFAULT_GRID_PARAMS: dict[str, list] = {
    "n_estimators": [10, 30, 100],
    "max_depth": [3, 5, 7],
    "criterion": ["gini", "entropy"],
    "max_features": ["sqrt", "log2", None],
}
"""54-combination parameter grid (3 × 3 × 2 × 3).

``"auto"`` is omitted from ``max_features`` because it was removed in
scikit-learn 1.4 (imbalanced-learn >= 0.14.1 pulls sklearn >= 1.4).
"""


class ClassifierModel:
    """Train RandomForest classifiers via GridSearchCV over multiple random seeds.

    Loads pre-extracted features and labels, runs randomised under-sampling
    followed by ``GridSearchCV(RandomForestClassifier)`` for each seed, and
    saves the best estimator per seed as a pickled model file.

    Args:
        features_csv (str): Path to the features CSV file.
        label_csv (str): Path to the label CSV file.
        output_dir (str, optional): Directory for output model files.
            Defaults to ``<cwd>/output/models/prediction``.
        overwrite (bool, optional): Overwrite existing output files.
            Defaults to False.
        n_jobs (int, optional): Number of parallel workers for the outer
            seed loop.  The inner ``GridSearchCV`` always uses ``n_jobs=1``
            to avoid fork-bomb on Windows (spawn start method).
            Defaults to 1.
        verbose (bool, optional): Verbose logging. Defaults to False.
        debug (bool, optional): Debug mode. Defaults to False.

    Raises:
        ValueError: If features or labels are empty, or their lengths
            do not match.
    """

    def __init__(
        self,
        features_csv: str,
        label_csv: str,
        output_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        df_features = pd.read_csv(features_csv, index_col=0)
        df_labels = pd.read_csv(label_csv)

        if ID_COLUMN in df_labels.columns:
            df_labels = df_labels.set_index(ID_COLUMN)

        if DATETIME_COLUMN in df_labels.columns:
            df_labels = df_labels.drop(DATETIME_COLUMN, axis=1)

        df_labels = df_labels[ERUPTED_COLUMN]

        output_dir = output_dir or os.path.join(
            os.getcwd(), "output", "models", "prediction"
        )

        self.df_features = df_features
        self.df_labels = df_labels
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.n_jobs = int(n_jobs)
        self.verbose = verbose
        self.debug = debug
        self.model_paths: list[str] = []

        self.validate()
        self.create_directories()

    def validate(self) -> None:
        """Validate that features and labels are non-empty and aligned.

        Raises:
            ValueError: If features or labels are empty, or their row
                counts do not match.
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
        """Create required output directories."""
        os.makedirs(self.output_dir, exist_ok=True)

    def _train(
        self,
        seed: int,
        random_state: int,
        sampling_strategy: str | float,
        grid_params: dict[str, list],
        n_splits: int,
        test_size: float,
        overwrite: bool,
    ) -> str:
        """Per-seed training worker.

        Args:
            seed (int): Seed offset for this iteration.
            random_state (int): Base random state.
            sampling_strategy (Union[str, float]): Under-sampling strategy
                passed to ``RandomUnderSampler``.
            grid_params (dict[str, list]): Parameter grid for GridSearchCV.
            n_splits (int): Number of ShuffleSplit folds.
            test_size (float): Fraction of data used for the validation split.
            overwrite (bool): Whether to overwrite an existing model file.

        Returns:
            str: Absolute path to the saved ``.pkl`` model file.
        """
        state = random_state + seed
        logger.debug(
            f"_train: seed={seed}, random_state={random_state}, state={state}"
        )

        model_filepath = os.path.join(self.output_dir, f"model_{state:05d}.pkl")

        if not overwrite and os.path.isfile(model_filepath):
            if self.verbose:
                logger.info(f"Model {state:05d}.pkl already exists, skipping.")
            return model_filepath

        if self.verbose:
            logger.info(f"Training seed: {state:05d}")

        features, labels = random_under_sampler(
            features=self.df_features,
            labels=self.df_labels,
            sampling_strategy=sampling_strategy,
            random_state=state,
        )

        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=state)
        clf = RandomForestClassifier(random_state=state)
        grid_search = GridSearchCV(
            estimator=clf, param_grid=grid_params, cv=cv, n_jobs=1
        )
        grid_search.fit(features, labels) # type: ignore

        joblib.dump(grid_search.best_estimator_, model_filepath)

        if self.verbose:
            logger.info(
                f"Model saved to {model_filepath} | best params: {grid_search.best_params_}"
            )

        return model_filepath

    def train(
        self,
        random_state: int = 0,
        total_seed: int = 500,
        sampling_strategy: str | float = 0.75,
        grid_params: dict[str, list] | None = None,
        n_splits: int = 5,
        test_size: float = 0.2,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> None:
        """Train RandomForest classifiers across multiple seeds.

        Each seed produces one model file: ``model_{state:05d}.pkl`` where
        ``state = random_state + seed``.

        Args:
            random_state (int, optional): Base random state. Defaults to 0.
            total_seed (int, optional): Number of seeds (models) to train.
                Defaults to 500.
            sampling_strategy (Union[str, float], optional): Under-sampling
                ratio for ``RandomUnderSampler``. Defaults to 0.75.
            grid_params (dict[str, list], optional): Parameter grid for
                ``GridSearchCV``. Defaults to ``DEFAULT_GRID_PARAMS``.
            n_splits (int, optional): Number of ``ShuffleSplit`` folds.
                Defaults to 5.
            test_size (float, optional): Fraction of data for each validation
                split. Defaults to 0.2.
            overwrite (bool, optional): Overwrite existing model files.
                Defaults to False.
            verbose (bool, optional): Verbose logging. Defaults to False.

        Returns:
            None
        """
        overwrite = overwrite or self.overwrite
        grid_params = grid_params or DEFAULT_GRID_PARAMS

        jobs = [
            (seed, random_state, sampling_strategy, grid_params, n_splits, test_size, overwrite)
            for seed in range(total_seed)
        ]

        if self.n_jobs == 1:
            for job in jobs:
                path = self._train(*job)
                self.model_paths.append(path)

        if self.n_jobs > 1:
            with Pool(self.n_jobs) as pool:
                paths = pool.starmap(self._train, jobs)
                self.model_paths.extend(paths)

        return None

    def evaluate(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        save_results: bool = True,
        output_filename: str | None = None,
    ) -> pd.DataFrame:
        """Evaluate trained models on a held-out test set.

        Loads all trained model files from the output directory and evaluates
        each on a stratified test split. Computes accuracy, precision, recall,
        F1-score, and ROC AUC for each model, then aggregates statistics.

        Args:
            test_size (float, optional): Fraction of data for test set.
                Defaults to 0.2.
            random_state (int, optional): Random state for train/test split.
                Defaults to 42.
            save_results (bool, optional): Save evaluation results to CSV.
                Defaults to True.
            output_filename (str, optional): Custom filename for results CSV.
                Defaults to ``evaluation_results.csv``.

        Returns:
            pd.DataFrame: DataFrame with evaluation metrics for each model
                and aggregate statistics (mean, std) in the last two rows.

        Raises:
            ValueError: If no trained models are found in the output directory.
        """
        # Find all model files
        model_files = sorted(
            [
                f
                for f in os.listdir(self.output_dir)
                if f.startswith("model_") and f.endswith(".pkl")
            ]
        )

        if not model_files:
            raise ValueError(
                f"No trained models found in {self.output_dir}. "
                "Run train() first to create model files."
            )

        if self.verbose:
            logger.info(f"Evaluating {len(model_files)} models...")

        # Create stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.df_features,
            self.df_labels,
            test_size=test_size,
            stratify=self.df_labels,
            random_state=random_state,
        )

        if self.verbose:
            logger.info(
                f"Test set: {len(X_test)} samples "
                f"(positive: {y_test.sum()}, negative: {len(y_test) - y_test.sum()})"
            )

        # Evaluate each model
        results: list[dict] = []

        for model_file in model_files:
            model_path = os.path.join(self.output_dir, model_file)
            model = joblib.load(model_path)

            # Predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = {
                "model": model_file,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            # Confusion matrix elements
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics["true_negatives"] = tn
            metrics["false_positives"] = fp
            metrics["false_negatives"] = fn
            metrics["true_positives"] = tp

            results.append(metrics)

            if self.debug:
                logger.debug(
                    f"{model_file}: acc={metrics['accuracy']:.3f}, "
                    f"f1={metrics['f1']:.3f}, auc={metrics['roc_auc']:.3f}"
                )

        # Create DataFrame
        df_results = pd.DataFrame(results)

        # Calculate aggregate statistics
        numeric_cols = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "true_positives",
        ]

        mean_row: dict[str, str | float] = {"model": "MEAN"}
        std_row: dict[str, str | float] = {"model": "STD"}

        for col in numeric_cols:
            mean_row[col] = float(df_results[col].mean())
            std_row[col] = float(df_results[col].std())

        df_results = pd.concat(
            [df_results, pd.DataFrame([mean_row, std_row])], ignore_index=True
        )

        if self.verbose:
            logger.info(
                f"Evaluation complete. "
                f"Mean accuracy: {mean_row['accuracy']:.3f} (+/- {std_row['accuracy']:.3f}), "
                f"Mean F1: {mean_row['f1']:.3f} (+/- {std_row['f1']:.3f}), "
                f"Mean ROC AUC: {mean_row['roc_auc']:.3f} (+/- {std_row['roc_auc']:.3f})"
            )

        # Save results
        if save_results:
            output_filename = output_filename or "evaluation_results.csv"
            output_path = os.path.join(self.output_dir, output_filename)
            df_results.to_csv(output_path, index=False)

            if self.verbose:
                logger.info(f"Evaluation results saved to {output_path}")

        return df_results

    def get_classification_report(
        self,
        model_index: int = 0,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> str:
        """Get detailed classification report for a specific model.

        Args:
            model_index (int, optional): Index of the model file to evaluate
                (sorted alphabetically). Defaults to 0 (first model).
            test_size (float, optional): Fraction of data for test set.
                Defaults to 0.2.
            random_state (int, optional): Random state for train/test split.
                Defaults to 42.

        Returns:
            str: Formatted classification report from sklearn.

        Raises:
            ValueError: If no trained models are found.
            IndexError: If model_index is out of range.
        """
        model_files = sorted(
            [
                f
                for f in os.listdir(self.output_dir)
                if f.startswith("model_") and f.endswith(".pkl")
            ]
        )

        if not model_files:
            raise ValueError(
                f"No trained models found in {self.output_dir}. "
                "Run train() first to create model files."
            )

        if model_index >= len(model_files):
            raise IndexError(
                f"model_index {model_index} is out of range. "
                f"Only {len(model_files)} models available."
            )

        model_path = os.path.join(self.output_dir, model_files[model_index])
        model = joblib.load(model_path)

        # Create stratified train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.df_features,
            self.df_labels,
            test_size=test_size,
            stratify=self.df_labels,
            random_state=random_state,
        )

        y_pred = model.predict(X_test)

        report: str = classification_report(
            y_test, y_pred, target_names=["Not Erupted", "Erupted"]
        )
        return report

    def get_feature_importances(
        self,
        model_index: int = 0,
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Get feature importances from a specific trained model.

        Args:
            model_index (int, optional): Index of the model file to use
                (sorted alphabetically). Defaults to 0 (first model).
            top_n (int, optional): Return only top N features. If None,
                returns all features. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame with feature names and importance scores,
                sorted by importance descending.

        Raises:
            ValueError: If no trained models are found.
            IndexError: If model_index is out of range.
        """
        model_files = sorted(
            [
                f
                for f in os.listdir(self.output_dir)
                if f.startswith("model_") and f.endswith(".pkl")
            ]
        )

        if not model_files:
            raise ValueError(
                f"No trained models found in {self.output_dir}. "
                "Run train() first to create model files."
            )

        if model_index >= len(model_files):
            raise IndexError(
                f"model_index {model_index} is out of range. "
                f"Only {len(model_files)} models available."
            )

        model_path = os.path.join(self.output_dir, model_files[model_index])
        model = joblib.load(model_path)

        importances = model.feature_importances_
        feature_names = self.df_features.columns.tolist()

        df_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        if top_n is not None:
            df_importance = df_importance.head(top_n)

        return df_importance.reset_index(drop=True)
