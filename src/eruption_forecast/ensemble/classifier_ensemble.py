import os
import json
from typing import Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import compute_model_probabilities
from eruption_forecast.ensemble.base_ensemble import BaseEnsemble
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble


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

        Creates an empty container ready to be populated via one of the
        factories — :meth:`from_any` (recommended), :meth:`from_seed_ensembles`,
        :meth:`from_dict`, or :meth:`from_json`.
        """
        self.ensembles: dict[str, SeedEnsemble] = {}

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

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
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

    @property
    def classifiers(self) -> list[str]:
        """Return the list of classifier names registered in this ensemble.

        Returns:
            list[str]: Names of all classifiers in insertion order.
        """
        return list(self.ensembles.keys())

    @classmethod
    def from_any(cls, source: str | SeedEnsemble, verbose: bool = False) -> Self:
        """Load a ``ClassifierEnsemble`` from any supported model source.

        Normalises a ``SeedEnsemble`` object or a file path (``*.json``,
        ``*.pkl``, or ``*.csv``) into a ``ClassifierEnsemble``. Per-classifier
        trained-model registries (``.json`` list or ``.csv``) are routed
        through :meth:`SeedEnsemble.from_any`, which dispatches on the file
        extension to the appropriate loader.

        Args:
            source (str | SeedEnsemble): Source to load from. Accepted forms:

                - ``SeedEnsemble`` object — wrapped directly.
                - ``str`` path ending in ``".json"`` — payload type decides:

                  * Top-level ``dict`` → top-level results map produced by
                    ``TrainingModel.fit()`` at
                    ``{training_dir}/classifier/ClassifierEnsemble_{cv}.json``;
                    loaded via :meth:`from_json`.
                  * Top-level ``list`` → per-classifier trained-model registry
                    produced by
                    :func:`~eruption_forecast.utils.ml.save_model_json`;
                    loaded via :meth:`SeedEnsemble.from_any` and wrapped.
                - ``str`` path ending in ``".pkl"`` — deserialised via
                  :meth:`~BaseEnsemble.load`; if the result is a
                  ``SeedEnsemble`` it is wrapped automatically. Examples:
                  ``{training_dir}/classifiers/ClassifierEnsemble.pkl`` or
                  ``{classifier_dir}/random-forest-classifier/SeedEnsemble_RandomForestClassifier.pkl``.
                - ``str`` path ending in ``".csv"`` — legacy trained-model
                  registry CSV loaded via :meth:`SeedEnsemble.from_any` (which
                  dispatches to the legacy CSV loader) then wrapped.
            verbose (bool, optional): Whether to emit load progress logs.
                Defaults to ``False``.

        Returns:
            ClassifierEnsemble: A fully constructed ``ClassifierEnsemble``
                instance.

        Raises:
            ValueError: If ``source`` is a string with an unrecognised
                extension or a JSON payload of an unexpected shape.
        """
        if isinstance(source, SeedEnsemble):
            return cls.from_seed_ensembles({source.classifier_name: source}, verbose)

        if source.endswith(".json"):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"JSON file not found: {source}")
            with open(source) as f:
                payload = json.load(f)

            if isinstance(payload, dict):
                return cls.from_dict(payload, verbose=verbose)

            if isinstance(payload, list):
                seed_ensemble = SeedEnsemble.from_any(source, verbose=verbose)
                return cls.from_seed_ensembles(
                    {seed_ensemble.classifier_name: seed_ensemble}, verbose
                )

            raise ValueError(
                f"Unsupported JSON payload type in {source!r}: "
                f"expected dict (top-level results map) or list "
                f"(per-classifier registry), got {type(payload).__name__}."
            )

        if source.endswith(".pkl"):
            loaded = cls.load(source)
            if isinstance(loaded, SeedEnsemble):
                return cls.from_seed_ensembles(
                    {loaded.classifier_name: loaded}, verbose
                )
            return loaded

        if source.endswith(".csv"):
            seed_ensemble = SeedEnsemble.from_any(source, verbose=verbose)
            return cls.from_seed_ensembles(
                {seed_ensemble.classifier_name: seed_ensemble}
            )

        raise ValueError(f"Unsupported source type or extension: {source!r}")

    @classmethod
    def from_seed_ensembles(
        cls, ensembles: dict[str, SeedEnsemble], verbose: bool = False
    ) -> Self:
        """Build a ClassifierEnsemble from a dict of named SeedEnsemble objects.

        Copies the provided mapping into a new ``ClassifierEnsemble`` instance
        without re-loading any models from disk.

        Args:
            ensembles (dict[str, SeedEnsemble]): Mapping from classifier name to
                its already-constructed ``SeedEnsemble``.
            verbose (bool, optional): Whether to emit load progress logs.
                Defaults to ``False``.

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

        if verbose:
            logger.info(
                f"[ClassifierEnsemble] Built from {len(ensembles)} SeedEnsemble(s): "
                f"{list(ensembles.keys())}"
            )

        return obj

    @classmethod
    def from_dict(
        cls, trained_model_paths: dict[str, str], verbose: bool = False
    ) -> Self:
        """Build a ClassifierEnsemble directly from registry paths.

        Calls :meth:`SeedEnsemble.from_any` for each entry, which dispatches
        on the file extension (``.json`` for the new inline-features registry,
        ``.csv`` for the legacy registry), and assembles the result into a
        :class:`ClassifierEnsemble`.

        Args:
            trained_model_paths (dict[str, str]): Mapping from classifier name
                to the path of its trained-model registry (``.json`` or
                ``.csv``).
            verbose (bool, optional): Whether to emit load progress logs.
                Defaults to ``False``.

        Returns:
            ClassifierEnsemble: Populated ensemble with all seed models loaded
                from disk.

        Raises:
            ValueError: If ``trained_model_paths`` is empty.
            FileNotFoundError: If any registry path does not exist.
        """
        if not trained_model_paths:
            raise ValueError("trained_model_paths must not be empty.")
        ensembles: dict[str, SeedEnsemble] = {}
        for name, path in trained_model_paths.items():
            ensembles[name] = SeedEnsemble.from_any(path, classifier_name=name)
            if verbose:
                logger.info(f"[ClassifierEnsemble] Loaded {name} from: {path}")
        return cls.from_seed_ensembles(ensembles)

    @classmethod
    def from_json(cls, json_path: str, verbose: bool = False) -> Self:
        """Build a ClassifierEnsemble from a JSON results file.

        Loads the classifier-name-to-registry-path mapping produced by
        ``TrainingModel.fit()`` and delegates to :meth:`from_dict`. The
        mapping's values may point at either the new ``.json`` trained-model
        registry or the legacy ``.csv`` registry; both are dispatched through
        :meth:`SeedEnsemble.from_any`.

        Args:
            json_path (str): Path to the ``results.json`` file written by
                ``TrainingModel.fit()``.
            verbose (bool): Whether to emit load progress logs. Defaults to
                ``False``.

        Returns:
            ClassifierEnsemble: Populated ensemble with all seed models loaded
                from disk.

        Raises:
            FileNotFoundError: If ``json_path`` does not exist.
            ValueError: If the JSON file contains an empty mapping.
        """
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path) as f:
            trained_model_paths = json.load(f)
        if not isinstance(trained_model_paths, dict):
            raise ValueError(
                f"Top-level results JSON must contain a "
                f"{{classifier_name: registry_path}} mapping, got "
                f"{type(trained_model_paths).__name__}: {json_path}"
            )
        return cls.from_dict(trained_model_paths, verbose=verbose)

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
        clf_results = self.predict_per_classifier(X)

        classifier_probability_means = np.stack(
            [v["probability"] for v in clf_results.values()], axis=0
        )

        mean_proba = classifier_probability_means.mean(axis=0)
        return np.column_stack([1.0 - mean_proba, mean_proba])

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using a 0.5 threshold on the consensus probability.

        Applies a 0.5 threshold to the consensus mean P(eruption).

        Returns:
            np.ndarray: 1-D integer array of shape ``(n_samples,)`` with values
                0 (non-eruption) or 1 (eruption).
        """
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def predict_per_classifier(
        self,
        X: pd.DataFrame,
        save: bool = False,
        output_dir: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run :meth:`SeedEnsemble.predict_with_uncertainty` for every classifier.

        Loops over the registered ``SeedEnsemble`` instances and collects the
        per-seed-aggregated probability/uncertainty/prediction/confidence
        arrays. Used internally by :meth:`predict_proba` and
        :meth:`predict_with_uncertainty` to build the cross-classifier consensus.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape
                ``(n_samples, n_features)``.
            save (bool, optional): Forwarded to
                :meth:`SeedEnsemble.predict_with_uncertainty`. When ``True``,
                per-seed CSVs land under ``{output_dir}/{classifier_name}/``.
                Defaults to ``False``.
            output_dir (str | None, optional): Base directory for per-seed
                output. Required when ``save`` is ``True``. Defaults to ``None``.
            overwrite (bool, optional): Overwrite existing per-seed files.
                Defaults to ``False``.
            verbose (bool, optional): Log per-classifier progress. Defaults to
                ``False``.

        Returns:
            dict[str, dict[str, np.ndarray]]: Mapping from classifier name to a
                dict with keys ``"probability"``, ``"uncertainty"``,
                ``"prediction"``, and ``"confidence"`` — each a 1-D
                ``np.ndarray`` of shape ``(n_samples,)``.
        """
        clf_results: dict[str, dict[str, np.ndarray]] = {}
        for classifier_name, seed_ensemble in self.ensembles.items():
            clf_output_dir = (
                os.path.join(output_dir, classifier_name)
                if (save and output_dir)
                else None
            )

            if verbose:
                logger.info(
                    f"Predicting probabiities for {classifier_name} with {len(seed_ensemble)} seed(s) ..."
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

            clf_results[classifier_name] = {
                "probability": seed_probability,
                "uncertainty": seed_uncertainty,
                "prediction": seed_prediction,
                "confidence": seed_confidence,
            }

        return clf_results

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        save: bool = False,
        output_dir: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> dict[str, np.ndarray]:
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
            dict[str, np.ndarray]: Flat mapping ready to be wrapped in a
                ``pd.DataFrame`` (one ndarray of shape ``(n_samples,)`` per
                key). Contains four cross-classifier consensus columns:

                - ``"consensus_probability"`` — mean P(eruption) averaged across
                  all classifiers.
                - ``"consensus_uncertainty"`` — std of per-classifier mean
                  probabilities (inter-classifier uncertainty).
                - ``"consensus_prediction"`` — mean of per-classifier binary
                  votes (continuous, ``[0, 1]``).
                - ``"consensus_confidence"`` — CI-like metric
                  ``1.96 * sqrt(p * (1 - p) / n_classifiers)``.

                Plus four per-classifier columns for each registered
                classifier (``{name}`` is the classifier key):

                - ``"{name}_probability"``, ``"{name}_uncertainty"``,
                  ``"{name}_prediction"``, ``"{name}_confidence"``.
        """
        clf_results = self.predict_per_classifier(
            X, save=save, output_dir=output_dir, overwrite=overwrite, verbose=verbose
        )

        # Cross-classifier consensus
        classifier_probability_matrix = np.stack(
            [v["probability"] for v in clf_results.values()], axis=1
        )  # (n_samples, n_classifiers)
        classifier_prediction_matrix = np.stack(
            [v["prediction"] for v in clf_results.values()], axis=1
        )  # (n_samples, n_classifiers)

        (
            consensus_probability,
            consensus_uncertainty,
            consensus_prediction,
            consensus_confidence,
        ) = compute_model_probabilities(
            classifier_probability_matrix, classifier_prediction_matrix
        )

        results: dict[str, np.ndarray] = {
            "consensus_probability": consensus_probability,
            "consensus_uncertainty": consensus_uncertainty,
            "consensus_prediction": consensus_prediction,
            "consensus_confidence": consensus_confidence,
        }

        for classifier_name, clf_result in clf_results.items():
            results[f"{classifier_name}_probability"] = clf_result["probability"]
            results[f"{classifier_name}_uncertainty"] = clf_result["uncertainty"]
            results[f"{classifier_name}_prediction"] = clf_result["prediction"]
            results[f"{classifier_name}_confidence"] = clf_result["confidence"]

        return results

    def get_params(self, deep: bool = True) -> dict:
        """Return the ensemble's display parameters.

        Overrides :meth:`sklearn.base.BaseEstimator.get_params` so that
        Jupyter's rich HTML representation renders a single
        ``ClassifierEnsemble`` box whose Parameters section lists
        ``n_classifiers`` followed by one entry per registered classifier —
        each of which sklearn renders as a collapsible nested Parameters
        box for the underlying :class:`SeedEnsemble` (see
        :meth:`SeedEnsemble.get_params`).

        This intentionally reports synthetic params that are not accepted by
        :meth:`__init__`. It is safe here because :class:`ClassifierEnsemble`
        is a fitted container that is never passed through
        :func:`sklearn.base.clone` (which would try to re-instantiate the
        class with these kwargs).

        Args:
            deep (bool, optional): Accepted for sklearn API compatibility;
                the return value is identical regardless. Defaults to
                ``True``.

        Returns:
            dict: ``{"n_classifiers": 0}`` for an empty ensemble, otherwise
                ``{"n_classifiers": N, <classifier_name>: <SeedEnsemble>, ...}``
                with one entry per registered classifier in insertion order.
        """
        params: dict = {"n_classifiers": len(self.ensembles)}
        params.update(self.ensembles)
        return params
