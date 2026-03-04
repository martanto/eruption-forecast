"""SeedEnsemble: bundles all seed models for a single classifier into one object.

This module provides the ``SeedEnsemble`` class, which wraps the 500 (or any
number of) trained estimators produced by ``ModelTrainer`` — along with the
per-seed significant feature lists — into a single, serialisable object.

Instead of loading 500 ``.pkl`` files and 500 ``.csv`` files at prediction
time, callers load one file and call ``predict_proba()`` directly.
"""

import os

import numpy as np
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import (
    aggregate_seed_probabilities,
    predict_proba_from_estimator,
)
from eruption_forecast.config.constants import ERUPTION_PROBABILITY_THRESHOLD
from eruption_forecast.model.base_ensemble import BaseEnsemble


class SeedEnsemble(BaseEnsemble, BaseEstimator, ClassifierMixin):
    """Bundle of seed models for a single classifier type.

    Wraps all trained estimators produced by ``ModelTrainer`` for one
    classifier — along with their per-seed significant feature lists — into a
    single serialisable object.  Aggregating predictions across seeds reduces
    variance and mirrors the logic in
    :func:`eruption_forecast.utils.ml.compute_model_probabilities`, but
    operates on in-memory estimators instead of loading files on every call.

    Attributes:
        classifier_name (str): Human-readable classifier name (e.g.
            "RandomForestClassifier").
        cv_strategy (str): CV strategy slug used during training (e.g.
            "StratifiedShuffleSplit").
        seeds (list[dict]): List of seed records.  Each record is a dict with
            keys ``random_state`` (int), ``model`` (fitted estimator), and
            ``feature_names`` (list[str]).

    Args:
        classifier_name (str): Human-readable classifier name.
        cv_strategy (str): CV strategy slug.
    """

    def __init__(self, classifier_name: str, cv_strategy: str) -> None:
        """Initialise an empty SeedEnsemble for the given classifier.

        Creates an empty container ready to have seeds added via
        :meth:`from_registry`.  ``classifier_name`` and ``cv_strategy`` are
        stored as metadata and used in log messages.

        Args:
            classifier_name (str): Human-readable classifier name (e.g.
                "RandomForestClassifier").
            cv_strategy (str): CV strategy slug used during training (e.g.
                "StratifiedShuffleSplit").
        """
        self.classifier_name = classifier_name
        self.cv_strategy = cv_strategy
        self.seeds: list[dict] = []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_registry(cls, registry_csv: str) -> "SeedEnsemble":
        """Load all seed models from a registry CSV and bundle into a SeedEnsemble.

        Reads the registry CSV produced by ``ModelTrainer._save_models_registry``,
        then for each row loads the significant features CSV and the trained model
        ``.pkl`` file into memory.  The resulting ``SeedEnsemble`` contains all
        seeds as in-memory objects and can be serialised with :meth:`save`.

        Args:
            registry_csv (str): Path to the trained-model registry CSV.  Must
                have ``random_state`` as index and columns
                ``significant_features_csv`` and ``trained_model_filepath``.

        Returns:
            SeedEnsemble: Populated ensemble with all seeds loaded.

        Raises:
            FileNotFoundError: If ``registry_csv`` does not exist.
            KeyError: If required columns are absent from the CSV.
        """
        if not os.path.isfile(registry_csv):
            raise FileNotFoundError(f"Registry CSV not found: {registry_csv}")

        df = pd.read_csv(registry_csv, index_col=0)

        # Infer metadata from the first row's model file path via joblib peek —
        # fall back to generic labels if the model has no class name attribute.
        classifier_name = "unknown"
        cv_strategy = "unknown"

        seeds: list[dict] = []

        for random_state, row in df.iterrows():
            sig_csv: str = row["significant_features_csv"]
            model_path: str = row["trained_model_filepath"]

            feature_names: list[str] = pd.read_csv(sig_csv, index_col=0).index.tolist()

            model = joblib.load(model_path)

            # Try to extract metadata from the first seed
            if classifier_name == "unknown":
                classifier_name = type(model).__name__
                # cv_strategy stays "unknown" unless we find a better source

            seeds.append(
                {
                    "random_state": int(random_state),  # type: ignore[arg-type]
                    "model": model,
                    "feature_names": feature_names,
                }
            )

            logger.debug(
                f"[SeedEnsemble] Loaded seed {int(random_state):05d} — "  # type: ignore[arg-type]
                f"{len(feature_names)} features"
            )

        ensemble = cls(classifier_name=classifier_name, cv_strategy=cv_strategy)
        ensemble.seeds = seeds
        logger.info(f"[SeedEnsemble] Loaded {len(seeds)} seeds for {classifier_name}")
        return ensemble

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_seed_proba(seed: dict, X: pd.DataFrame) -> np.ndarray:
        """Compute eruption probability for a single seed.

        Selects the seed-specific features from ``X``, then calls
        ``predict_proba`` or ``decision_function`` on the stored estimator and
        returns a 1-D array of P(eruption) values.

        Args:
            seed (dict): Seed record with keys ``model`` and ``feature_names``.
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).

        Returns:
            np.ndarray: 1-D array of shape ``(n_samples,)`` with P(eruption).

        Raises:
            RuntimeError: If the estimator supports neither ``predict_proba``
                nor ``decision_function``.
        """
        model = seed["model"]
        feature_names: list[str] = seed["feature_names"]
        X_seed = X[feature_names]
        eruption_proba, _ = predict_proba_from_estimator(
            model, X_seed, identifier=seed["random_state"]
        )
        return eruption_proba

    # ------------------------------------------------------------------
    # Public sklearn-compatible interface
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities averaged across all seeds.

        Conforms to the sklearn ``(n_samples, n_classes)`` convention: column 0
        is P(non-eruption), column 1 is P(eruption).  The values are the mean
        across all seed estimators.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape ``(n_samples, 2)`` where column 1 is the
                mean eruption probability across seeds.
        """
        seed_probas = np.stack(
            [self._compute_seed_proba(s, X) for s in self.seeds], axis=1
        )  # (n_samples, n_seeds)
        mean_eruption: np.ndarray = seed_probas.mean(axis=1)
        result = np.column_stack([1.0 - mean_eruption, mean_eruption])
        return result

    def predict(
        self, X: pd.DataFrame, threshold: float = ERUPTION_PROBABILITY_THRESHOLD
    ) -> np.ndarray:
        """Return binary predictions using ``ERUPTION_PROBABILITY_THRESHOLD`` on mean probability.

        Applies ``ERUPTION_PROBABILITY_THRESHOLD`` to the mean P(eruption) returned by
        :meth:`predict_proba`.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            threshold (float, optional): Probability threshold for eruption classification.
                Defaults to ``ERUPTION_PROBABILITY_THRESHOLD`` which value is 0.5.

        Returns:
            np.ndarray: 1-D integer array of shape ``(n_samples,)`` with values
                0 (non-eruption) or 1 (eruption).
        """
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        threshold: float = ERUPTION_PROBABILITY_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return mean probability, uncertainty, confidence, and binary predictions.

        Aggregates per-seed eruption probabilities into four summary statistics
        that mirror the output of
        :func:`eruption_forecast.utils.ml.compute_model_probabilities`.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            threshold (float, optional): Probability threshold for eruption classification.
                Defaults to ``ERUPTION_PROBABILITY_THRESHOLD`` which value is 0.5.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four 1-D
                arrays of shape ``(n_samples,)``:

                - ``mean_probability``: Mean P(eruption) across seeds.
                - ``std``: Standard deviation of P(eruption) across seeds
                  (uncertainty).
                - ``confidence``: Fraction of seeds voting with the majority
                  decision (0.5–1.0).
                - ``prediction``: Binary predictions (0 or 1) based on
                  ``threshold``.
        """
        seed_probas = np.stack(
            [self._compute_seed_proba(seed, X) for seed in self.seeds], axis=1
        )  # (n_samples, n_seeds)

        return aggregate_seed_probabilities(seed_probas, threshold=threshold)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @classmethod
    def _load_log_msg(cls, obj: "SeedEnsemble") -> str:  # type: ignore[override]
        """Return a seed-count suffix for the load log message.

        Args:
            obj (SeedEnsemble): The just-loaded SeedEnsemble instance.

        Returns:
            str: Human-readable seed count string.
        """
        return f"{len(obj.seeds)} seeds"

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

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
            str: Representation including classifier name, CV strategy, and
                seed count.
        """
        return (
            f"SeedEnsemble(classifier_name={self.classifier_name!r}, "
            f"cv_strategy={self.cv_strategy!r}, "
            f"n_seeds={len(self.seeds)})"
        )
