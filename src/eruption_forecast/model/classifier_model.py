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
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit

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
        grid_search.fit(features, labels)

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
