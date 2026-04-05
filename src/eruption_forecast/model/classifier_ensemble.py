"""Wrap multiple SeedEnsemble objects for cross-classifier consensus inference.

Provides :class:`ClassifierEnsemble`, a serialisable sklearn-compatible
estimator that bundles one :class:`~eruption_forecast.model.seed_ensemble.SeedEnsemble`
per classifier type.

Each :class:`SeedEnsemble` already averages predictions across random seeds for
a single classifier.  ``ClassifierEnsemble`` adds a second aggregation layer:
it collects the per-classifier mean probabilities and computes a consensus
prediction (mean of means) together with an uncertainty estimate (standard
deviation across classifiers).

Key capabilities:
    - ``from_seed_ensembles(ensembles)``: Construct from an existing dict of
      ``SeedEnsemble`` objects.
    - ``from_registry_dict(registry_dict)``: Load from a mapping of classifier
      names to registry CSV paths (delegates to ``SeedEnsemble.from_registry()``).
    - ``predict_proba(X)``: Return per-classifier probabilities and consensus
      probability array.
    - ``predict_with_uncertainty(X)``: Return mean probability, standard
      deviation, mean binary vote, confidence, and a per-classifier breakdown
      dict.
    - ``save(path)`` / ``load(path)``: Persist and restore via joblib (inherited
      from :class:`~eruption_forecast.model.base_ensemble.BaseEnsemble`).
"""

import os

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import compute_model_probabilities
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
    def from_registry_dict(cls, registry_csvs: dict[str, str]) -> "ClassifierEnsemble":
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
        _, _, _, _, clf_results = self.predict_with_uncertainty(X)

        classifier_probability_means = np.stack(
            [v["probability"] for v in clf_results.values()], axis=0
        )

        mean_proba = classifier_probability_means.mean(axis=0)
        return np.column_stack([1.0 - mean_proba, mean_proba])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return binary predictions using a 0.5 threshold on the consensus probability.

        Applies a 0.5 threshold to the consensus mean P(eruption).

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).

        Returns:
            np.ndarray: 1-D integer array of shape ``(n_samples,)`` with values
                0 (non-eruption) or 1 (eruption).
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        save: bool = False,
        output_dir: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Return consensus statistics and per-classifier results.

        For each classifier, calls :meth:`SeedEnsemble.predict_with_uncertainty`
        to obtain per-seed-aggregated statistics, then computes cross-classifier
        consensus by averaging mean probabilities across classifiers.

        Confidence is a CI-like metric computed as
        ``1.96 * sqrt(p * (1 - p) / n_classifiers)``, where ``p`` is the mean
        binary vote across classifiers.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            save (bool, optional): If ``True``, save per-seed predictions to CSV
                files under ``{output_dir}/{classifier_name}/``. Defaults to ``False``.
            output_dir (str | None, optional): Base directory for per-seed CSV output.
                Required when ``save`` is ``True``. Defaults to ``None``.
            overwrite (bool, optional): If ``True``, overwrite existing CSV files.
                Defaults to ``False``.
            verbose (bool, optional): If ``True``, log a message for each saved file.
                Defaults to ``False``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]: A 5-tuple:

                - ``consensus_probability`` (np.ndarray): Shape ``(n_samples,)`` —
                  mean P(eruption) averaged across all classifiers.
                - ``consensus_uncertainty`` (np.ndarray): Shape ``(n_samples,)`` —
                  std of per-classifier mean probabilities (inter-classifier
                  uncertainty).
                - ``consensus_prediction`` (np.ndarray): Shape ``(n_samples,)`` —
                  mean of per-classifier binary votes (continuous, ``[0, 1]``).
                - ``consensus_confidence`` (np.ndarray): Shape ``(n_samples,)`` —
                  CI-like metric: ``1.96 * sqrt(p * (1 - p) / n_classifiers)``.
                - ``per_classifier_results`` (dict): Mapping from classifier name to
                  a dict with keys ``"probability"``, ``"uncertainty"``,
                  ``"prediction"``, ``"confidence"`` — each a 1-D ``np.ndarray``
                  of shape ``(n_samples,)``.
        """
        clf_results: dict[str, dict[str, np.ndarray]] = {}
        for name, seed_ensemble in self.ensembles.items():
            clf_output_dir = (
                os.path.join(output_dir, name) if (save and output_dir) else None
            )

            (
                seed_probability,
                seed_uncertainty,
                seed_prediction,
                seed_confidence,
            ) = seed_ensemble.predict_with_uncertainty(
                X,
                save=save,
                output_dir=clf_output_dir,
                overwrite=overwrite,
                verbose=verbose,
            )

            clf_results[name] = {
                "probability": seed_probability,
                "uncertainty": seed_uncertainty,
                "prediction": seed_prediction,
                "confidence": seed_confidence,
            }

        # Cross-classifier consensus
        classifier_probability_means = np.stack(
            [v["probability"] for v in clf_results.values()], axis=1
        )  # (n_samples, n_classifiers)
        classifier_prediction_means = np.stack(
            [v["prediction"] for v in clf_results.values()], axis=1
        )  # (n_samples, n_classifiers)

        (
            consensus_probability,
            consensus_uncertainty,
            consensus_prediction,
            consensus_confidence,
        ) = compute_model_probabilities(
            classifier_probability_means, classifier_prediction_means
        )

        return (
            consensus_probability,
            consensus_uncertainty,
            consensus_prediction,
            consensus_confidence,
            clf_results,
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
        return f"ClassifierEnsemble(n_classifiers={len(names)}, classifiers={names!r})"
