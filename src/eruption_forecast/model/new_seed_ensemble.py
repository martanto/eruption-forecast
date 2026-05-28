"""Bundle all seed models for a single classifier into one serialisable object.

Provides :class:`SeedEnsemble`, a serialisable sklearn-compatible estimator
that aggregates all per-seed estimators produced by ``ModelTrainer`` for one
classifier type.

Averaging predictions across seeds reduces variance and provides an uncertainty
estimate (standard deviation of per-seed probabilities).  This mirrors the
file-based logic in
:func:`eruption_forecast.utils.ml.compute_model_probabilities` but operates
entirely on in-memory estimators so the ``.pkl`` files do not need to be
re-loaded on every inference call.

Key capabilities:
    - ``from_registry(registry_csv)``: Construct a ``SeedEnsemble`` by loading
      all seed model ``.pkl`` files listed in a model registry CSV.
    - ``predict_proba(X)``: Average ``predict_proba`` output across seeds;
      each seed uses only its own significant feature subset.
    - ``predict_with_uncertainty(X)``: Return mean probability, standard
      deviation across seeds, confidence score, and binary prediction array.
    - ``save(path)`` / ``load(path)``: Persist and restore via joblib (inherited
      from :class:`~eruption_forecast.model.base_ensemble.BaseEnsemble`).
"""

import os
from typing import Self

import numpy as np
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import (
    save_forecast_seed,
    compute_model_probabilities,
)
from eruption_forecast.model.base_ensemble import BaseEnsemble


class SeedEnsemble(BaseEnsemble, BaseEstimator, ClassifierMixin):
    """Bundle of seed models for a single classifier type.

    Wraps all trained estimators produced by ``ModelTrainer`` for one
    classifier — along with their per-seed significant feature lists — into a
    single serialisable object.

    Attributes:
        classifier_name (str): Human-readable classifier name (e.g.
            "RandomForestClassifier").
        seeds (list[dict]): List of seed records.  Each record is a dict with
            keys ``random_state`` (int), ``model`` (fitted estimator), and
            ``feature_names`` (list[str]).

    Args:
        classifier_name (str): Human-readable classifier name.
    """

    def __init__(self, classifier_name: str) -> None:
        """Initialise an empty SeedEnsemble for the given classifier.

        Creates an empty container ready to have seeds added via
        :meth:`from_registry`.

        Args:
            classifier_name (str): Human-readable classifier name (e.g.
                "RandomForestClassifier").
        """
        self.classifier_name = classifier_name
        self.seeds: list[dict] = []

    def __len__(self) -> int:
        """Return the number of seeds in this ensemble.

        Returns:
            int: Number of seed records stored.
        """
        return len(self.seeds)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return a concise string representation of the ensemble.

        The ``N_CHAR_MAX`` parameter matches the ``BaseEstimator.__repr__``
        signature; it is accepted but not used — the representation is always
        compact.

        Args:
            N_CHAR_MAX (int): Maximum character limit inherited from
                ``BaseEstimator``.  Not applied here.  Defaults to 700.

        Returns:
            str: Representation including classifier name and seed count.
        """
        return (
            f"SeedEnsemble(classifier_name={self.classifier_name!r}, "
            f"n_seeds={len(self.seeds)})"
        )

    @classmethod
    def from_registry(
        cls,
        registry_csv: str,
        classifier_name: str | None = None,
        verbose: bool = False,
    ) -> Self:
        """Load all seed models from a registry CSV and bundle into a SeedEnsemble.

        Reads the registry CSV produced by ``ModelTrainer._save_models_registry``,
        then for each row loads the significant features CSV and the trained model
        ``.pkl`` file into memory.  The resulting ``SeedEnsemble`` contains all
        seeds as in-memory objects and can be serialised with :meth:`save`.

        Args:
            registry_csv (str): Path to the trained-model registry CSV.  Must
                have ``random_state`` as index and columns
                ``significant_features_csv`` and ``trained_model_filepath``.
            classifier_name (str, optional): Human-readable classifier name.
                Defaults to None.
            verbose (bool, optional): If ``True``, log a message for each saved file.
                Defaults to ``False``.

        Returns:
            SeedEnsemble: Populated ensemble with all seeds loaded.

        Raises:
            FileNotFoundError: If ``registry_csv`` does not exist.
            KeyError: If required columns are absent from the CSV.
        """
        if not os.path.isfile(registry_csv):
            raise FileNotFoundError(f"Registry CSV not found: {registry_csv}")

        _classifier_name = "Unknown" if classifier_name is None else classifier_name

        df = pd.read_csv(registry_csv, index_col=0)

        if df.empty:
            raise ValueError(f"Trained model CSV is empty: {registry_csv}")

        for column in ("features_csv", "model_filepath"):
            if column not in df.columns.tolist():
                raise ValueError(f"Column {column} is not present in {registry_csv}")

        seeds: list[dict] = []
        for random_state, row in df.iterrows():
            features_csv: str = row["features_csv"]
            model_filepath: str = row["model_filepath"]

            feature_names: list[str] = pd.read_csv(
                features_csv, index_col=0
            ).index.tolist()

            model = joblib.load(model_filepath)

            # Try to extract metadata from the first seed
            if _classifier_name == "Unknown":
                _classifier_name = type(model).__name__

            seeds.append(
                {
                    "random_state": int(random_state),  # ty:ignore[invalid-argument-type]
                    "model": model,
                    "feature_names": feature_names,
                }
            )

        ensemble = cls(classifier_name=_classifier_name)
        ensemble.seeds = seeds

        if verbose:
            logger.info(f"SeedEnsemble: {classifier_name} — Loaded {len(seeds)} seeds.")

        return ensemble

    def _compute_probabilities_and_predictions(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute eruption probabilities and predictions across all seeds.

        Selects each seed's feature subset from ``X``, runs ``predict_proba``
        or ``decision_function``, and returns stacked matrices.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape
                (n_samples, n_features).

        Returns:
            tuple[np.ndarray, np.ndarray]: Two arrays of shape
                ``(n_samples, n_seeds)``:

                - ``probabilities``: P(eruption) per seed.
                - ``predictions``: Binary prediction (0 or 1) per seed.

        Raises:
            RuntimeError: If an estimator supports neither ``predict_proba``
                nor ``decision_function``.
        """
        seed_probabilities: list[np.ndarray] = []
        seed_predictions: list[np.ndarray] = []

        for seed in self.seeds:
            seed_model = seed["model"]
            features_df: pd.DataFrame = X[seed["feature_names"]]

            if hasattr(seed_model, "predict_proba"):
                scores: np.ndarray = seed_model.predict_proba(features_df)
                eruption_probabilities = scores[:, 1]
                eruption_predictions = seed_model.predict(features_df)
            elif hasattr(seed_model, "decision_function"):
                scores = seed_model.decision_function(features_df)
                eruption_probabilities = 1.0 / (1.0 + np.exp(-scores))
                eruption_predictions = (eruption_probabilities >= 0.5).astype(int)
            else:
                raise RuntimeError(
                    f"{seed_model} supports neither ``predict_proba`` nor ``decision_function``."
                )

            seed_probabilities.append(eruption_probabilities)
            seed_predictions.append(eruption_predictions)

        return np.stack(seed_probabilities, axis=1), np.stack(seed_predictions, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities averaged across all seeds.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape ``(n_samples, 2)`` where column 0 is
                P(non-eruption) and column 1 is the mean P(eruption) across seeds.
        """
        probabilities, _ = self._compute_probabilities_and_predictions(X)
        mean_eruption: np.ndarray = np.mean(probabilities, axis=1)
        return np.column_stack([1.0 - mean_eruption, mean_eruption])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using a threshold on the mean probability.

        Applies the given threshold to the mean P(eruption) returned by :meth:`predict_proba`.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            threshold (float, optional): Probability threshold for eruption classification.
                Defaults to 0.5.

        Returns:
            np.ndarray: 1-D integer array of shape ``(n_samples,)`` with values
                0 (non-eruption) or 1 (eruption).
        """
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        save: bool = False,
        output_dir: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return mean probability, uncertainty, confidence, and binary predictions.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            save (bool, optional): If ``True``, save per-seed predictions to CSV files
                under ``output_dir``. Defaults to ``False``.
            output_dir (str | None, optional): Directory for per-seed CSV output.
                Required when ``save`` is ``True``. Defaults to ``None``.
            overwrite (bool, optional): If ``True``, overwrite existing CSV files.
                Defaults to ``False``.
            verbose (bool, optional): If ``True``, log a message for each saved file.
                Defaults to ``False``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four 1-D
                arrays of shape ``(n_samples,)``:

            - ``seed_probability``: Mean P(eruption) across seeds.
            - ``seed_uncertainty``: Standard deviation of P(eruption) across seeds.
            - ``seed_prediction``: Mean of per-seed binary votes (continuous, [0, 1]).
            - ``seed_confidence``: CI-like metric ``1.96 * sqrt(p * (1-p) / n_seeds)``.
        """
        probabilities, predictions = self._compute_probabilities_and_predictions(X)

        # Save probability and prediction per seed
        if save and output_dir is not None:
            for idx, seed in enumerate(self.seeds):
                save_forecast_seed(
                    output_dir=output_dir,
                    random_state=seed["random_state"],
                    probabilities=probabilities[:, idx],
                    predictions=predictions[:, idx],
                    overwrite=overwrite,
                    verbose=verbose,
                )

        return compute_model_probabilities(probabilities, predictions)
