"""ClassifierEnsemble: wraps multiple SeedEnsemble objects for cross-classifier consensus.

This module provides the ``ClassifierEnsemble`` class, which bundles one
``SeedEnsemble`` per classifier into a single, serialisable object and owns
the cross-classifier consensus logic previously inline in
``ModelPredictor.predict_proba()``.
"""


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from eruption_forecast.logger import logger
from eruption_forecast.config.constants import ERUPTION_PROBABILITY_THRESHOLD
from eruption_forecast.model.base_ensemble import BaseEnsemble
from eruption_forecast.model.seed_ensemble import SeedEnsemble


class ClassifierEnsemble(BaseEnsemble, BaseEstimator, ClassifierMixin):
    """Bundle of SeedEnsemble objects for multiple classifier types.

    Wraps one :class:`SeedEnsemble` per classifier into a single serialisable
    object.  Owns the cross-classifier consensus logic: for each input sample it
    collects per-classifier mean probabilities and aggregates them into a
    consensus prediction with uncertainty quantification.

    Attributes:
        ensembles (dict[str, SeedEnsemble]): Mapping from classifier name to
            its ``SeedEnsemble``.
    """

    def __init__(self) -> None:
        """Initialise an empty ClassifierEnsemble.

        Creates an empty container ready to be populated via
        :meth:`from_seed_ensembles` or :meth:`from_registry_dict`.
        """
        self.ensembles: dict[str, SeedEnsemble] = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_seed_ensembles(
        cls, ensembles: dict[str, SeedEnsemble]
    ) -> "ClassifierEnsemble":
        """Build a ClassifierEnsemble from a dict of named SeedEnsemble objects.

        Copies the provided mapping into a new ``ClassifierEnsemble`` instance
        without re-loading any models from disk.

        Args:
            ensembles (dict[str, SeedEnsemble]): Mapping from classifier name to
                its already-constructed ``SeedEnsemble``.

        Returns:
            ClassifierEnsemble: Populated ensemble containing all provided
                classifiers.

        Raises:
            ValueError: If ``ensembles`` is empty.
        """
        if not ensembles:
            raise ValueError("ensembles must not be empty.")
        obj = cls()
        obj.ensembles = dict(ensembles)
        logger.info(
            f"[ClassifierEnsemble] Built from {len(ensembles)} SeedEnsemble(s): "
            f"{list(ensembles.keys())}"
        )
        return obj

    @classmethod
    def from_registry_dict(
        cls, registry_csvs: dict[str, str]
    ) -> "ClassifierEnsemble":
        """Build a ClassifierEnsemble directly from registry CSV paths.

        Calls :meth:`SeedEnsemble.from_registry` for each entry and assembles
        the result into a :class:`ClassifierEnsemble`.

        Args:
            registry_csvs (dict[str, str]): Mapping from classifier name to the
                path of its trained-model registry CSV.

        Returns:
            ClassifierEnsemble: Populated ensemble with all seed models loaded
                from disk.

        Raises:
            ValueError: If ``registry_csvs`` is empty.
            FileNotFoundError: If any registry CSV path does not exist.
        """
        if not registry_csvs:
            raise ValueError("registry_csvs must not be empty.")
        ensembles: dict[str, SeedEnsemble] = {}
        for name, csv_path in registry_csvs.items():
            logger.info(f"[ClassifierEnsemble] Loading classifier: {name}")
            ensembles[name] = SeedEnsemble.from_registry(csv_path)
        return cls.from_seed_ensembles(ensembles)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return consensus class probabilities across all classifiers.

        Computes the mean P(eruption) across all classifiers and returns an
        sklearn-compatible ``(n_samples, 2)`` array where column 1 is the
        consensus mean eruption probability.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape ``(n_samples, 2)`` where column 1 is the
                consensus mean eruption probability.
        """
        _, _, _, _, per_clf = self.predict_with_uncertainty(X)
        consensus_mean: np.ndarray = np.stack(
            [v["mean"] for v in per_clf.values()], axis=0
        ).mean(axis=0)
        return np.column_stack([1.0 - consensus_mean, consensus_mean])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using ``ERUPTION_PROBABILITY_THRESHOLD`` on consensus probability.

        Applies ``ERUPTION_PROBABILITY_THRESHOLD`` to the consensus mean P(eruption).

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).

        Returns:
            np.ndarray: 1-D integer array of shape ``(n_samples,)`` with values
                0 (non-eruption) or 1 (eruption).
        """
        return (self.predict_proba(X)[:, 1] >= ERUPTION_PROBABILITY_THRESHOLD).astype(int)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        threshold: float = ERUPTION_PROBABILITY_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return consensus statistics and per-classifier results.

        For each classifier, calls :meth:`SeedEnsemble.predict_with_uncertainty`
        to obtain per-seed-aggregated statistics, then computes cross-classifier
        consensus by averaging mean probabilities across classifiers.

        Confidence is defined as the fraction of classifiers whose binary
        prediction agrees with the consensus prediction.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            threshold (float, optional): Probability threshold for binary
                eruption classification.  Defaults to
                ``ERUPTION_PROBABILITY_THRESHOLD``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]: A 5-tuple:

                - ``consensus_mean`` (np.ndarray): Shape ``(n_samples,)`` — mean
                  P(eruption) averaged across all classifiers.
                - ``consensus_std`` (np.ndarray): Shape ``(n_samples,)`` — std of
                  per-classifier mean probabilities (inter-classifier uncertainty).
                - ``consensus_confidence`` (np.ndarray): Shape ``(n_samples,)`` —
                  fraction of classifiers agreeing with the consensus prediction.
                - ``consensus_prediction`` (np.ndarray): Shape ``(n_samples,)`` —
                  binary predictions (0 or 1).
                - ``per_classifier_results`` (dict): Mapping from classifier name to
                  a dict with keys ``"mean"``, ``"std"``, ``"confidence"``,
                  ``"prediction"`` — each a 1-D ``np.ndarray`` of shape
                  ``(n_samples,)``.
        """
        per_clf_results: dict[str, dict[str, np.ndarray]] = {}
        for name, seed_ensemble in self.ensembles.items():
            mean, std, conf, pred = seed_ensemble.predict_with_uncertainty(
                X, threshold
            )
            per_clf_results[name] = {
                "mean": mean,
                "std": std,
                "confidence": conf,
                "prediction": pred,
            }

        # Cross-classifier consensus
        all_means = np.stack(
            [v["mean"] for v in per_clf_results.values()], axis=0
        )  # (n_classifiers, n_samples)
        consensus_mean: np.ndarray = all_means.mean(axis=0)
        consensus_std: np.ndarray = all_means.std(axis=0)
        consensus_pred: np.ndarray = (consensus_mean >= threshold).astype(int)

        n_classifiers = all_means.shape[0]
        clf_preds = np.stack(
            [v["prediction"] for v in per_clf_results.values()], axis=0
        )  # (n_classifiers, n_samples)
        votes = np.where(
            consensus_pred == 1,
            (clf_preds == 1).sum(axis=0),
            (clf_preds == 0).sum(axis=0),
        )
        consensus_conf: np.ndarray = votes / n_classifiers

        return (
            consensus_mean,
            consensus_std,
            consensus_conf,
            consensus_pred,
            per_clf_results,
        )

    def predict_with_major_voting(
        self,
        X: pd.DataFrame,
        threshold: float = ERUPTION_PROBABILITY_THRESHOLD,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return consensus major-voting statistics and per-classifier results.

        For each classifier, calls
        :meth:`SeedEnsemble.predict_with_major_voting` to obtain per-seed
        aggregated binary-vote statistics, then computes cross-classifier
        consensus by averaging vote ratios across classifiers.

        Confidence is defined as the fraction of classifiers whose binary
        prediction agrees with the consensus prediction.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape
                (n_samples, n_features).
            threshold (float, optional): Vote-ratio threshold for binary
                eruption classification. Defaults to
                ``ERUPTION_PROBABILITY_THRESHOLD``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
                A 5-tuple:

                - ``consensus_vote_ratio`` (np.ndarray): Shape ``(n_samples,)``
                  — mean vote ratio averaged across all classifiers.
                - ``consensus_std`` (np.ndarray): Shape ``(n_samples,)`` — std
                  of per-classifier vote ratios (inter-classifier uncertainty).
                - ``consensus_confidence`` (np.ndarray): Shape ``(n_samples,)``
                  — fraction of classifiers agreeing with the consensus
                  prediction.
                - ``consensus_prediction`` (np.ndarray): Shape ``(n_samples,)``
                  — binary predictions (0 or 1).
                - ``per_classifier_results`` (dict): Mapping from classifier
                  name to a dict with keys ``"vote_ratio"``, ``"std"``,
                  ``"confidence"``, ``"prediction"`` — each a 1-D
                  ``np.ndarray`` of shape ``(n_samples,)``.
        """
        per_clf_results: dict[str, dict[str, np.ndarray]] = {}
        for name, seed_ensemble in self.ensembles.items():
            vote_ratio, std, conf, pred = seed_ensemble.predict_with_major_voting(
                X, threshold
            )
            per_clf_results[name] = {
                "vote_ratio": vote_ratio,
                "std": std,
                "confidence": conf,
                "prediction": pred,
            }

        # Cross-classifier consensus
        all_ratios = np.stack(
            [v["vote_ratio"] for v in per_clf_results.values()], axis=0
        )  # (n_classifiers, n_samples)
        consensus_vote_ratio: np.ndarray = all_ratios.mean(axis=0)
        consensus_std: np.ndarray = all_ratios.std(axis=0)
        consensus_pred: np.ndarray = (consensus_vote_ratio >= threshold).astype(int)

        n_classifiers = all_ratios.shape[0]
        clf_preds = np.stack(
            [v["prediction"] for v in per_clf_results.values()], axis=0
        )  # (n_classifiers, n_samples)
        votes = np.where(
            consensus_pred == 1,
            (clf_preds == 1).sum(axis=0),
            (clf_preds == 0).sum(axis=0),
        )
        consensus_conf: np.ndarray = votes / n_classifiers

        return (
            consensus_vote_ratio,
            consensus_std,
            consensus_conf,
            consensus_pred,
            per_clf_results,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def _load_log_msg(cls, obj: "ClassifierEnsemble") -> str:  # type: ignore[override]
        """Return a classifier-count suffix for the load log message.

        Args:
            obj (ClassifierEnsemble): The just-loaded ClassifierEnsemble instance.

        Returns:
            str: Human-readable classifier count string.
        """
        return f"{len(obj)} classifier(s)"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def classifiers(self) -> list[str]:
        """Return the list of classifier names registered in this ensemble.

        Returns:
            list[str]: Names of all classifiers in insertion order.
        """
        return list(self.ensembles.keys())

    def __getitem__(self, name: str) -> SeedEnsemble:
        """Return the SeedEnsemble for the given classifier name.

        Allows dictionary-style access: ``ensemble["rf"]``.

        Args:
            name (str): Classifier name key.

        Returns:
            SeedEnsemble: The SeedEnsemble registered under ``name``.

        Raises:
            KeyError: If ``name`` is not a registered classifier.
        """
        return self.ensembles[name]

    def __len__(self) -> int:
        """Return the number of classifiers in this ensemble.

        Returns:
            int: Number of registered classifiers.
        """
        return len(self.ensembles)

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:  # noqa: N803
        """Return a concise string representation of the ensemble.

        The ``N_CHAR_MAX`` parameter matches the ``BaseEstimator.__repr__``
        signature; it is accepted but not used — the representation is always
        compact.

        Args:
            N_CHAR_MAX (int): Maximum character limit inherited from
                ``BaseEstimator``.  Not applied here.  Defaults to 700.

        Returns:
            str: Representation including classifier names and counts.
        """
        names = list(self.ensembles.keys())
        return (
            f"ClassifierEnsemble(n_classifiers={len(names)}, "
            f"classifiers={names!r})"
        )
