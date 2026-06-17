import os
import json
from typing import Self

import numpy as np
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils._repr_html.estimator import _VisualBlock

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import (
    save_forecast_seed,
    compute_model_probabilities,
)
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.ensemble.base_ensemble import BaseEnsemble


class SeedEnsemble(BaseEnsemble, BaseEstimator, ClassifierMixin):
    """Bundle of seed models for a single classifier type.

    Wraps all trained estimators produced by ``TrainingModel.fit()`` for one
    classifier — along with their per-seed top-N feature lists — into a
    single serialisable object.

    Attributes:
        classifier_name (str): Human-readable classifier name (e.g.
            "RandomForestClassifier").
        seeds (list[dict]): List of seed records.  Each record is a dict with
            keys ``random_state`` (int), ``model`` (fitted estimator), and
            ``feature_names`` (list[str]).
        probabilities (pd.DataFrame | None): Per-seed eruption probability
            matrix cached by :meth:`save_matrices`. Shape
            ``(n_samples, n_seeds)`` with columns ``seed_{random_state:05d}``.
            ``None`` until the first prediction call.
        predictions (pd.DataFrame | None): Per-seed binary prediction matrix
            cached by :meth:`save_matrices`. Same shape and column scheme as
            ``probabilities``; values cast to ``int8``. ``None`` until the
            first prediction call.

    Args:
        classifier_name (str): Human-readable classifier name.
    """

    def __init__(self, classifier_name: str) -> None:
        """Initialise an empty SeedEnsemble for the given classifier.

        Creates an empty container ready to have seeds added via one of the
        registry factories — :meth:`from_any` (recommended; dispatches on
        extension), :meth:`from_json` (new JSON registry), or
        :meth:`from_registry` (legacy CSV registry).

        Args:
            classifier_name (str): Human-readable classifier name (e.g.
                "RandomForestClassifier").
        """
        self.classifier_name = classifier_name
        self.seeds: list[dict] = []
        self.probabilities: pd.DataFrame | None = None
        self.predictions: pd.DataFrame | None = None

    def __getitem__(self, seed_index: int):
        return self.seeds[seed_index]

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
        """Load all seed models from a legacy CSV registry and bundle into a SeedEnsemble.

        Reads the legacy registry CSV produced by older training runs (prior
        to the JSON registry switch), then for each row loads the per-seed
        top-N feature CSV (column ``features_csv``) and the trained model
        ``.pkl`` (column ``model_filepath``) into memory. The resulting
        ``SeedEnsemble`` contains all seeds as in-memory objects and can be
        serialised with :meth:`save`. New runs should use :meth:`from_json`
        (or the format-agnostic :meth:`from_any`) instead.

        Args:
            registry_csv (str): Path to the legacy trained-model registry CSV.
                Must have ``random_state`` as index and columns
                ``features_csv`` and ``model_filepath``.
            classifier_name (str, optional): Human-readable classifier name.
                If ``None``, derived from ``type(model).__name__`` of the
                first loaded estimator. Defaults to None.
            verbose (bool, optional): If ``True``, log a load summary.
                Defaults to ``False``.

        Returns:
            SeedEnsemble: Populated ensemble with all seeds loaded.

        Raises:
            FileNotFoundError: If ``registry_csv`` does not exist.
            ValueError: If the CSV is empty or a required column is missing.
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
            logger.info(
                f"SeedEnsemble: {_classifier_name} — Loaded {len(seeds)} "
                f"seeds from CSV registry."
            )

        return ensemble

    @classmethod
    def from_json(
        cls,
        trained_model_json: str,
        classifier_name: str | None = None,
        verbose: bool = False,
    ) -> Self:
        """Build a SeedEnsemble from a JSON trained-model registry.

        Reads the JSON registry produced by
        :func:`~eruption_forecast.utils.ml.save_model_json`, then for each
        record loads the trained model ``.pkl`` and takes the inline
        ``features`` list as the per-seed feature subset. Avoids the extra
        per-seed CSV read that :meth:`from_registry` performs against the
        legacy CSV registry.

        Args:
            trained_model_json (str): Path to the trained-model registry JSON
                file. Must contain a list of per-seed records, each with
                ``random_state`` (int), ``features`` (list[str]), and
                ``model_filepath`` (str).
            classifier_name (str, optional): Human-readable classifier name.
                If ``None``, derived from ``type(model).__name__`` of the
                first loaded estimator. Defaults to ``None``.
            verbose (bool, optional): If ``True``, log a load summary.
                Defaults to ``False``.

        Returns:
            SeedEnsemble: Populated ensemble with all seeds loaded.

        Raises:
            FileNotFoundError: If ``trained_model_json`` does not exist.
            ValueError: If the JSON payload is empty or any record is missing
                a required key.
        """
        if not os.path.isfile(trained_model_json):
            raise FileNotFoundError(
                f"Trained-model JSON not found: {trained_model_json}"
            )

        with open(trained_model_json) as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(
                f"Trained-model JSON must contain a list of records, got "
                f"{type(records).__name__}: {trained_model_json}"
            )

        if not records:
            raise ValueError(f"Trained-model JSON is empty: {trained_model_json}")

        required_keys = {"random_state", "features", "model_filepath"}
        _classifier_name = "Unknown" if classifier_name is None else classifier_name

        seeds: list[dict] = []
        for record in records:
            missing = required_keys - set(record)
            if missing:
                raise ValueError(
                    f"Record missing required key(s) {sorted(missing)} "
                    f"in {trained_model_json}"
                )

            model = joblib.load(record["model_filepath"])

            if _classifier_name == "Unknown":
                _classifier_name = type(model).__name__

            seeds.append(
                {
                    "random_state": int(record["random_state"]),
                    "model": model,
                    "feature_names": list(record["features"]),
                }
            )

        ensemble = cls(classifier_name=_classifier_name)
        ensemble.seeds = seeds

        if verbose:
            logger.info(
                f"SeedEnsemble: {_classifier_name} — Loaded {len(seeds)} "
                f"seeds from JSON registry."
            )

        return ensemble

    @classmethod
    def from_any(
        cls,
        trained_model_path: str,
        classifier_name: str | None = None,
        verbose: bool = False,
    ) -> Self:
        """Build a SeedEnsemble from any supported trained-model registry.

        Dispatches on the file extension — ``.json`` routes to
        :meth:`from_json` (new inline-features registry), ``.csv`` routes to
        :meth:`from_registry` (legacy registry). Callers can stop branching
        on the on-disk format.

        Args:
            trained_model_path (str): Path to a trained-model registry file.
                Must end in ``.json`` or ``.csv``.
            classifier_name (str, optional): Forwarded to the underlying
                factory. Defaults to ``None``.
            verbose (bool, optional): Forwarded to the underlying factory.
                Defaults to ``False``.

        Returns:
            SeedEnsemble: Populated ensemble with all seeds loaded.

        Raises:
            ValueError: If the file extension is neither ``.json`` nor
                ``.csv``.
        """
        ext = os.path.splitext(trained_model_path)[1].lower()
        if ext == ".json":
            return cls.from_json(
                trained_model_path,
                classifier_name=classifier_name,
                verbose=verbose,
            )
        if ext == ".csv":
            return cls.from_registry(
                trained_model_path,
                classifier_name=classifier_name,
                verbose=verbose,
            )
        raise ValueError(
            f"Unsupported trained-model registry extension {ext!r}. "
            f"Expected '.json' or '.csv'."
        )

    def compute_probabilities_and_predictions(
        self, X: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute eruption probabilities and predictions across all seeds.

        Selects each seed's feature subset from ``X``, runs ``predict_proba``
        or ``decision_function``, and returns stacked matrices. Pure with
        respect to ``self`` — no caching is performed.

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
                eruption_predictions = np.asarray(eruption_probabilities >= 0.5).astype(
                    int
                )
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
        probabilities, _ = self.compute_probabilities_and_predictions(X)
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

        Side effects:
            When ``save`` is ``True`` and ``output_dir`` is set, writes per-seed
            CSV files under ``{output_dir}/seeds/`` and additionally persists
            the full ``(n_samples, n_seeds)`` matrices as
            ``{output_dir}/seed_probabilities.parquet`` and
            ``{output_dir}/seed_predictions.parquet`` via
            :meth:`save_seed_matrices`.
        """
        probabilities, predictions = self.compute_probabilities_and_predictions(X)

        output_dir = (
            output_dir
            if output_dir is not None
            else os.path.join(
                os.getcwd(), "output", "prediction", "results", self.classifier_name
            )
        )

        # Save probability and prediction per seed
        if save:
            seed_output_dir = os.path.join(output_dir, "seeds")
            ensure_dir(seed_output_dir)

            for idx, seed in enumerate(self.seeds):
                save_forecast_seed(
                    output_dir=seed_output_dir,
                    random_state=seed["random_state"],
                    probabilities=probabilities[:, idx],
                    predictions=predictions[:, idx],
                    overwrite=overwrite,
                    verbose=False,
                )

        self.save_matrices(
            output_dir=output_dir,
            probabilities=probabilities,
            predictions=predictions,
            index=X.index,
            verbose=verbose,
        )

        return compute_model_probabilities(probabilities, predictions)

    def save_matrices(
        self,
        output_dir: str,
        probabilities: np.ndarray,
        predictions: np.ndarray,
        index: pd.Index | None = None,
        verbose: bool = False,
    ) -> tuple[str, str]:
        """Persist the per-seed probability and prediction matrices as Parquet.

        Writes the two ``(n_samples, n_seeds)`` matrices produced by
        :meth:`compute_probabilities_and_predictions` as separate Parquet
        files under ``output_dir``:

            - ``seed_probabilities.parquet`` — per-seed P(eruption).
            - ``seed_predictions.parquet``  — per-seed binary predictions (0/1).

        Columns are named ``seed_{random_state:05d}`` in the order of
        ``self.seeds``. The prediction matrix is cast to ``int8`` to keep the
        file compact. Existing files are skipped unless ``overwrite`` is
        ``True``.

        Also caches the assembled DataFrames on ``self.probabilities`` /
        ``self.predictions`` so downstream callers that already hold the
        ensemble in memory (e.g. SHAP waterfall builders) can read the data
        directly instead of re-loading the Parquet file.

        Args:
            output_dir (str): Directory where the Parquet files are written.
                Created if it does not exist.
            probabilities (np.ndarray): Array of shape ``(n_samples, n_seeds)``
                with per-seed P(eruption) values.
            predictions (np.ndarray): Array of shape ``(n_samples, n_seeds)``
                with per-seed binary predictions (0 or 1).
            index (pd.Index | None, optional): Row index for the resulting
                DataFrames (typically the feature-matrix index used to compute
                the matrices). When ``None``, a default ``RangeIndex`` is used.
                Defaults to ``None``.
            verbose (bool, optional): If ``True``, log the written paths.
                Defaults to ``False``.

        Returns:
            tuple[str, str]: ``(probabilities_path, predictions_path)``.
        """
        ensure_dir(output_dir)

        seed_columns = [f"seed_{seed['random_state']:05d}" for seed in self.seeds]

        probabilities_path = os.path.join(
            output_dir, f"{self.classifier_name}_seed_probabilities.parquet"
        )
        predictions_path = os.path.join(
            output_dir, f"{self.classifier_name}_seed_predictions.parquet"
        )

        probabilities_df = pd.DataFrame(
            probabilities, index=index, columns=seed_columns
        )
        predictions_df = pd.DataFrame(
            predictions.astype(np.int8), index=index, columns=seed_columns
        )

        self.probabilities = probabilities_df
        self.predictions = predictions_df

        # Save seed probabilities and predictions
        probabilities_df.to_parquet(probabilities_path)
        predictions_df.to_parquet(predictions_path)

        if verbose:
            logger.info(f"Saved seed probabilities matrix: {probabilities_path}")
            logger.info(f"Saved seed predictions matrix: {predictions_path}")

        return probabilities_path, predictions_path

    def _sk_visual_block_(self) -> _VisualBlock:
        """Return a sklearn ``_VisualBlock`` describing the bundled seed models.

        Picked up by :func:`sklearn.utils.estimator_html_repr` (and therefore by
        ``BaseEstimator._repr_html_``) so that Jupyter renders the ensemble as
        a single nested estimator. All seeds share the same classifier type
        (only ``random_state`` differs), so a single representative model is
        sufficient and avoids the unreadable explosion of 100–500 boxes that a
        one-per-seed layout would produce.

        Returns:
            _VisualBlock: A ``"single"`` block whose nested estimator is the
                first seed's fitted model. When the ensemble is empty, the
                block wraps ``self`` instead so the rich display still renders.
        """
        if not self.seeds:
            return _VisualBlock(
                "single",
                self,
                names=self.classifier_name,
                name_details="n_seeds=0",
            )

        representative = self.seeds[0]["model"]
        first_random_state = self.seeds[0]["random_state"]
        last_random_state = self.seeds[-1]["random_state"]
        return _VisualBlock(
            "single",
            representative,
            names=f"{self.classifier_name}",
            name_details=(
                f"n_seeds={len(self.seeds)}, "
                f"random_states=[{first_random_state}...{last_random_state}]"
            ),
        )
