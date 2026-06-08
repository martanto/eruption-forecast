"""Per-seed metrics computation for a :class:`ClassifierEnsemble`.

Provides :class:`MetricsEnsemble`, the ensemble-level metric engine that lifts
the per-classifier / per-seed metric loop out of
:class:`~eruption_forecast.model.evaluation_model.EvaluationModel`.

For every seed of every classifier in a fitted ``ClassifierEnsemble`` it:
    - Computes ``y_proba`` and ``y_pred`` against the supplied ``features_df``
      by delegating to
      :meth:`~eruption_forecast.ensemble.seed_ensemble.SeedEnsemble.compute_probabilities_and_predictions`.
    - Runs :class:`~eruption_forecast.model.metrics_computer.MetricsComputer`
      to obtain scalar metrics (accuracy, balanced_accuracy, precision, recall,
      F1, MCC, ROC-AUC, PR-AUC, average precision, plus G-mean and threshold-
      optimised variants).
    - Persists per-seed metrics as ``{classifiers_dir}/{clf}/metrics/json/{seed:05d}.json``
      so the existing
      :class:`~eruption_forecast.model.multi_model_evaluator.MultiModelEvaluator`
      aggregator and downstream plots keep working unchanged.
    - Persists the raw ``(n_samples, n_seeds)`` ``y_proba`` and ``y_pred``
      matrices plus the aligned ``y_true`` vector under
      ``{classifiers_dir}/{clf}/predictions/`` so downstream plotting can be
      rebuilt from disk without re-predicting.

The class intentionally only depends on ``ClassifierEnsemble`` so it stays in
the ``ensemble/`` package. The caller (typically ``EvaluationModel``) is
responsible for resolving ``y_true`` from labels and ``eruption_dates``.
"""

import os
import json
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.model.metrics_computer import MetricsComputer
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


if TYPE_CHECKING:
    from eruption_forecast.model.training_model import TrainingModel
    from eruption_forecast.model.prediction_model import PredictionModel


class MetricsEnsemble:
    """Compute and persist per-seed metrics for every classifier in an ensemble.

    Wraps a fitted ``ClassifierEnsemble`` together with the feature matrix and
    ground-truth labels needed to evaluate it. :meth:`compute` produces, for
    every classifier:

        - a per-seed scalar metrics dictionary saved to JSON,
        - a ``(n_samples, n_seeds)`` ``y_proba`` and ``y_pred`` CSV matrix,
        - a ``mean ± std`` aggregate summary CSV and a tidy per-seed CSV.

    Attributes:
        ClassifierEnsemble (ClassifierEnsemble): Fitted ensemble whose seeds
            are evaluated.
        features_df (pd.DataFrame): Feature matrix indexed by window id.
        y_true (pd.Series): Ground-truth ``is_erupted`` labels aligned to the
            index of ``features_df``.
        n_jobs (int): Number of parallel workers used for the per-seed metric
            computation (``joblib.Parallel``).
        output_dir (str): Root directory for evaluation artefacts. The class
            writes under ``{output_dir}/classifiers/{classifier}/``.
        classifiers_dir (str): ``{output_dir}/classifiers``.
        basename (str | None): Optional suffix appended to aggregate filenames
            (``metrics_summary_{basename}.csv`` etc.). When ``None`` the
            filenames carry no suffix.
        overwrite (bool): When ``True``, recompute and overwrite cached JSONs
            and prediction CSVs. When ``False``, cached per-seed JSONs are
            reused and prediction matrices are re-read from disk.
        verbose (bool): Emit per-classifier progress messages.
        metrics (dict[str, pd.DataFrame]): Populated by :meth:`compute`.
            Classifier name → DataFrame with one row per seed.
        aggregates (dict[str, pd.DataFrame]): Populated by :meth:`compute`.
            Classifier name → ``DataFrame.describe().T`` summary.
        predictions (dict[str, dict[str, pd.DataFrame | pd.Series]]): Populated
            by :meth:`compute`. Classifier name → ``{"y_proba": df, "y_pred":
            df, "y_true": series}``.

    Args:
        classifier_ensemble (ClassifierEnsemble): Fitted ensemble to evaluate.
        features_df (pd.DataFrame): Feature matrix indexed by window id.
        y_true (pd.Series | np.ndarray): Ground-truth labels. When an
            ``np.ndarray`` is passed it is assumed to be aligned positionally
            with ``features_df.index``.
        n_jobs (int, optional): Parallel workers for the per-seed metric loop.
            Defaults to ``1``.
        output_dir (str | None, optional): Root output directory. Required for
            :meth:`compute` to persist artefacts. Defaults to ``None``.
        basename (str | None, optional): Suffix appended to aggregate CSV
            filenames. Defaults to ``None``.
        overwrite (bool, optional): Recompute and overwrite cached artefacts.
            Defaults to ``False``.
        verbose (bool, optional): Verbose logging. Defaults to ``False``.

    Raises:
        ValueError: If ``output_dir`` is ``None``.

    Examples:
        >>> me = MetricsEnsemble(
        ...     classifier_ensemble=fm.TrainingModel.ClassifierEnsemble,
        ...     features_df=fm.TrainingModel.features_df,
        ...     y_true=fm.TrainingModel.labels,
        ...     output_dir="output/VG.OJN.00.EHZ/evaluation/training",
        ...     basename="2025-03-16_2025-03-22",
        ...     n_jobs=4,
        ... ).compute()
        >>> me.metrics["rf"].head()
    """

    def __init__(
        self,
        classifier_ensemble: ClassifierEnsemble,
        features_df: pd.DataFrame,
        y_true: pd.Series | np.ndarray,
        n_jobs: int = 1,
        output_dir: str | None = None,
        basename: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        """Validate inputs, align ``y_true`` to a ``pd.Series``, and resolve output paths.

        Args:
            classifier_ensemble (ClassifierEnsemble): Fitted ensemble to evaluate.
            features_df (pd.DataFrame): Feature matrix indexed by window id.
            y_true (pd.Series | np.ndarray): Ground-truth ``is_erupted`` labels.
            n_jobs (int, optional): Parallel workers. Defaults to ``1``.
            output_dir (str | None, optional): Root output directory. Required
                for persistence. Defaults to ``None``.
            basename (str | None, optional): Suffix for aggregate filenames.
                Defaults to ``None``.
            overwrite (bool, optional): Recompute and overwrite cached files.
                Defaults to ``False``.
            verbose (bool, optional): Verbose logging. Defaults to ``False``.

        Raises:
            ValueError: If ``output_dir`` is ``None``.
        """
        if output_dir is None:
            raise ValueError(
                "output_dir is required — MetricsEnsemble persists per-seed "
                "JSONs and (n_samples, n_seeds) prediction matrices to disk."
            )

        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(
                np.asarray(y_true), index=features_df.index, name="is_erupted"
            )

        self.ClassifierEnsemble: ClassifierEnsemble = classifier_ensemble
        self.features_df: pd.DataFrame = features_df
        self.y_true: pd.Series = y_true
        self.n_jobs: int = n_jobs
        self.output_dir: str = output_dir
        self.basename: str | None = basename
        self.overwrite: bool = overwrite
        self.verbose: bool = verbose

        self.classifiers_dir: str = os.path.join(output_dir, "classifiers")

        self.metrics: dict[str, pd.DataFrame] = {}
        self.aggregates: dict[str, pd.DataFrame] = {}
        self.predictions: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}

    @classmethod
    def from_training_model(
        cls,
        training_model: "TrainingModel",
        n_jobs: int = 1,
        output_dir: str | None = None,
        basename: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Self:
        """Build a :class:`MetricsEnsemble` from a completed ``TrainingModel``.

        Pulls ``training_model.ClassifierEnsemble``, ``features_df``, and
        ``labels`` (in-sample ground truth) into a new instance.

        Args:
            training_model (TrainingModel): Completed training stage.
            n_jobs (int, optional): Parallel workers. Defaults to ``1``.
            output_dir (str | None, optional): Output directory. Defaults to
                ``None``.
            basename (str | None, optional): Suffix for aggregate filenames.
                Defaults to ``None``.
            overwrite (bool, optional): Recompute cached files. Defaults to
                ``False``.
            verbose (bool, optional): Verbose logging. Defaults to ``False``.

        Returns:
            MetricsEnsemble: Ready for :meth:`compute`.

        Raises:
            ValueError: If ``training_model.ClassifierEnsemble`` is ``None``
                (training has not been run yet).
        """
        ensemble = training_model.ClassifierEnsemble
        if ensemble is None:
            raise ValueError(
                "training_model.ClassifierEnsemble is None — call fit() first."
            )
        return cls(
            classifier_ensemble=ensemble,
            features_df=training_model.features_df,
            y_true=training_model.labels,
            n_jobs=n_jobs,
            output_dir=output_dir,
            basename=basename,
            overwrite=overwrite,
            verbose=verbose,
        )

    @classmethod
    def from_prediction_model(
        cls,
        prediction_model: "PredictionModel",
        y_true: pd.Series | np.ndarray,
        n_jobs: int = 1,
        output_dir: str | None = None,
        basename: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Self:
        """Build a :class:`MetricsEnsemble` from a completed ``PredictionModel``.

        ``y_true`` must be supplied by the caller — ``PredictionModel.labels``
        carries only the window-id mapping with all-zero ``is_erupted``, so
        ground-truth labels have to be rebuilt upstream (typically by
        ``EvaluationModel.build_label()`` from ``eruption_dates``).

        Args:
            prediction_model (PredictionModel): Completed prediction stage.
            y_true (pd.Series | np.ndarray): Ground-truth labels aligned to
                ``prediction_model.features_df``.
            n_jobs (int, optional): Parallel workers. Defaults to ``1``.
            output_dir (str | None, optional): Output directory. Defaults to
                ``None``.
            basename (str | None, optional): Suffix for aggregate filenames.
                Defaults to ``None``.
            overwrite (bool, optional): Recompute cached files. Defaults to
                ``False``.
            verbose (bool, optional): Verbose logging. Defaults to ``False``.

        Returns:
            MetricsEnsemble: Ready for :meth:`compute`.

        Raises:
            ValueError: If ``prediction_model.ClassifierEnsemble`` is ``None``.
        """
        ensemble = prediction_model.ClassifierEnsemble
        if ensemble is None:
            raise ValueError(
                "prediction_model.ClassifierEnsemble is None — "
                "load a model before constructing PredictionModel."
            )
        return cls(
            classifier_ensemble=ensemble,
            features_df=prediction_model.features_df,
            y_true=y_true,
            n_jobs=n_jobs,
            output_dir=output_dir,
            basename=basename,
            overwrite=overwrite,
            verbose=verbose,
        )

    @classmethod
    def from_file(
        cls,
        model_filepath: str,
        features_filepath: str,
        y_true_filepath: str,
        n_jobs: int = 1,
        output_dir: str | None = None,
        basename: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> Self:
        """Build a :class:`MetricsEnsemble` by loading inputs from disk.

        ``model_filepath`` is normalised via
        :meth:`ClassifierEnsemble.from_any` so it accepts ``.pkl``, ``.json``,
        and registry ``.csv`` paths uniformly.

        Args:
            model_filepath (str): Path to a serialised ``ClassifierEnsemble``
                (``.pkl`` / ``.json``) or a trained-model registry ``.csv``.
            features_filepath (str): Path to a CSV of extracted features with
                the window id as the index.
            y_true_filepath (str): Path to a CSV holding the ground-truth
                ``is_erupted`` column. When the column ``is_erupted`` is not
                present the first column is used.
            n_jobs (int, optional): Parallel workers. Defaults to ``1``.
            output_dir (str | None, optional): Output directory. Defaults to
                ``None``.
            basename (str | None, optional): Suffix for aggregate filenames.
                Defaults to ``None``.
            overwrite (bool, optional): Recompute cached files. Defaults to
                ``False``.
            verbose (bool, optional): Verbose logging. Defaults to ``False``.

        Returns:
            MetricsEnsemble: Ready for :meth:`compute`.

        Raises:
            FileNotFoundError: If any of the three input files does not exist.
        """
        for label, path in (
            ("model", model_filepath),
            ("features", features_filepath),
            ("y_true", y_true_filepath),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{label} file not found: {path}")

        ce = ClassifierEnsemble.from_any(model_filepath, verbose=verbose)

        features_df = pd.read_csv(features_filepath, index_col=0)
        y_true_df = pd.read_csv(y_true_filepath, index_col=0)
        y_true = (
            y_true_df["is_erupted"]
            if "is_erupted" in y_true_df.columns
            else y_true_df.iloc[:, 0]
        )

        return cls(
            classifier_ensemble=ce,
            features_df=features_df,
            y_true=y_true,
            n_jobs=n_jobs,
            output_dir=output_dir,
            basename=basename,
            overwrite=overwrite,
            verbose=verbose,
        )

    def compute(self) -> Self:
        """Compute, persist, and aggregate per-seed metrics for every classifier.

        Aligns ``features_df`` and ``y_true`` on their common index, then for
        each classifier in ``self.ClassifierEnsemble.ensembles``:

            - Runs ``compute_probabilities_and_predictions`` once to obtain
              ``(n_samples, n_seeds)`` matrices.
            - Writes ``y_proba.csv`` / ``y_pred.csv`` / ``y_true.csv`` under
              ``{classifiers_dir}/{classifier}/predictions/``.
            - Computes scalar metrics per seed via :class:`MetricsComputer`
              and persists ``{classifiers_dir}/{classifier}/metrics/json/{seed:05d}.json``.
            - Writes ``metrics_summary[_{basename}].csv`` and
              ``all_metrics[_{basename}].csv`` under
              ``{classifiers_dir}/{classifier}/``.

        Populates ``self.metrics``, ``self.aggregates``, and
        ``self.predictions``.

        Returns:
            Self: The current instance, enabling method chaining.

        Raises:
            ValueError: If ``y_true`` or ``features_df`` is empty.
            ValueError: If the index intersection of ``features_df`` and
                ``y_true`` is empty.
        """
        if self.y_true.empty:
            raise ValueError("y_true is empty — nothing to evaluate.")
        if self.features_df.empty:
            raise ValueError("features_df is empty — nothing to evaluate.")

        common_index = self.features_df.index.intersection(self.y_true.index)
        if len(common_index) == 0:
            raise ValueError(
                "features_df and y_true have no overlapping index entries."
            )

        features_df = self.features_df.loc[common_index]
        y_true_series = self.y_true.loc[common_index].astype(int)
        y_true_array = y_true_series.to_numpy()

        for classifier_name, seed_ensemble in self.ClassifierEnsemble.ensembles.items():
            self._compute_for_classifier(
                classifier_name=classifier_name,
                seed_ensemble=seed_ensemble,
                features_df=features_df,
                y_true_series=y_true_series,
                y_true_array=y_true_array,
            )

        return self

    def _compute_for_classifier(
        self,
        classifier_name: str,
        seed_ensemble: SeedEnsemble,
        features_df: pd.DataFrame,
        y_true_series: pd.Series,
        y_true_array: np.ndarray,
    ) -> None:
        """Evaluate every seed of one classifier and persist artefacts.

        Args:
            classifier_name (str): Classifier identifier used in the output
                directory tree.
            seed_ensemble (SeedEnsemble): Per-classifier bundle of seed models.
            features_df (pd.DataFrame): Index-aligned feature matrix.
            y_true_series (pd.Series): Index-aligned ground-truth labels.
            y_true_array (np.ndarray): Same as ``y_true_series`` as an ndarray
                (cached to avoid repeated conversion inside the per-seed loop).
        """
        clf_dir = os.path.join(self.classifiers_dir, classifier_name)
        metrics_dir = os.path.join(clf_dir, "metrics")
        json_dir = os.path.join(metrics_dir, "json")
        predictions_dir = os.path.join(clf_dir, "predictions")
        ensure_dir(json_dir)
        ensure_dir(predictions_dir)

        seed_states: list[int] = [seed["random_state"] for seed in seed_ensemble.seeds]
        json_paths: list[str] = [
            os.path.join(json_dir, f"{rs:05d}.json") for rs in seed_states
        ]
        y_proba_path = os.path.join(predictions_dir, "y_proba.csv")
        y_pred_path = os.path.join(predictions_dir, "y_pred.csv")
        y_true_path = os.path.join(predictions_dir, "y_true.csv")

        # Fast path: every per-seed JSON and the proba matrix already exist.
        # Reuse them without re-predicting.
        cache_hit = (
            not self.overwrite
            and len(json_paths) > 0
            and all(os.path.isfile(p) for p in json_paths)
            and os.path.isfile(y_proba_path)
            and os.path.isfile(y_pred_path)
        )

        if cache_hit:
            all_metrics: list[dict[str, Any]] = []
            for json_path in json_paths:
                with open(json_path) as f:
                    all_metrics.append(json.load(f))
            y_proba_df = pd.read_csv(y_proba_path, index_col=0)
            y_pred_df = pd.read_csv(y_pred_path, index_col=0)
            if self.verbose:
                logger.info(
                    f"{classifier_name}: loaded cached metrics for {len(all_metrics)} seeds."
                )
        else:
            try:
                y_proba_matrix, y_pred_matrix = (
                    seed_ensemble.compute_probabilities_and_predictions(features_df)
                )
            except KeyError as exc:
                logger.warning(
                    f"{classifier_name}: feature mismatch ({exc!s}); "
                    "skipping classifier."
                )
                return

            seed_columns = [f"seed_{rs:05d}" for rs in seed_states]
            y_proba_df = pd.DataFrame(
                y_proba_matrix, index=features_df.index, columns=seed_columns
            )
            y_pred_df = pd.DataFrame(
                y_pred_matrix.astype(int),
                index=features_df.index,
                columns=seed_columns,
            )

            y_proba_df.to_csv(y_proba_path)
            y_pred_df.to_csv(y_pred_path)
            y_true_series.to_csv(y_true_path, header=True)

            all_metrics = self._compute_seed_metrics(
                classifier_name=classifier_name,
                seed_states=seed_states,
                json_paths=json_paths,
                y_proba_matrix=y_proba_matrix,
                y_pred_matrix=y_pred_matrix,
                y_true_array=y_true_array,
            )

        self._finalize_classifier(
            classifier_name=classifier_name,
            clf_dir=clf_dir,
            all_metrics=all_metrics,
        )
        self.predictions[classifier_name] = {
            "y_proba": y_proba_df,
            "y_pred": y_pred_df,
            "y_true": y_true_series,
        }

    def _compute_seed_metrics(
        self,
        classifier_name: str,
        seed_states: list[int],
        json_paths: list[str],
        y_proba_matrix: np.ndarray,
        y_pred_matrix: np.ndarray,
        y_true_array: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Run :class:`MetricsComputer` per seed and persist the JSON files.

        Honours ``self.overwrite``: a per-seed JSON that already exists is
        loaded from disk instead of recomputed. The metric computation itself
        is parallelised with ``joblib.Parallel`` when ``self.n_jobs > 1``.

        Args:
            classifier_name (str): Classifier name (added to each metric dict
                under ``model_name``).
            seed_states (list[int]): Random-state ints, one per seed.
            json_paths (list[str]): Per-seed JSON output paths, aligned with
                ``seed_states``.
            y_proba_matrix (np.ndarray): Shape ``(n_samples, n_seeds)``.
            y_pred_matrix (np.ndarray): Shape ``(n_samples, n_seeds)``.
            y_true_array (np.ndarray): Shape ``(n_samples,)``.

        Returns:
            list[dict[str, Any]]: Per-seed metrics in seed order.
        """
        n_jobs = max(1, self.n_jobs)
        overwrite = self.overwrite
        verbose = self.verbose

        if n_jobs > 1:
            return joblib.Parallel(n_jobs=n_jobs, backend="loky")(
                joblib.delayed(_compute_one_seed_metrics)(
                    classifier_name=classifier_name,
                    random_state=seed_states[i],
                    json_path=json_paths[i],
                    y_proba=y_proba_matrix[:, i],
                    y_pred=y_pred_matrix[:, i],
                    y_true=y_true_array,
                    overwrite=overwrite,
                    verbose=verbose,
                )
                for i in range(len(seed_states))
            )

        return [
            _compute_one_seed_metrics(
                classifier_name=classifier_name,
                random_state=seed_states[i],
                json_path=json_paths[i],
                y_proba=y_proba_matrix[:, i],
                y_pred=y_pred_matrix[:, i],
                y_true=y_true_array,
                overwrite=overwrite,
                verbose=verbose,
            )
            for i in range(len(seed_states))
        ]

    def _finalize_classifier(
        self,
        classifier_name: str,
        clf_dir: str,
        all_metrics: list[dict[str, Any]],
    ) -> None:
        """Aggregate per-seed metrics and write the summary CSVs.

        Computes ``DataFrame.describe().T`` across seeds, writes
        ``metrics_summary[_{basename}].csv`` and ``all_metrics[_{basename}].csv``,
        and logs the ``mean ± std`` of the headline metrics.

        Args:
            classifier_name (str): Classifier name for log output and the
                ``self.metrics`` / ``self.aggregates`` keys.
            clf_dir (str): Per-classifier output directory.
            all_metrics (list[dict[str, Any]]): Per-seed metric dicts.
        """
        if not all_metrics:
            logger.warning(f"{classifier_name}: no seed results — skipping.")
            return

        ensure_dir(clf_dir)
        df_metrics = pd.DataFrame(all_metrics)

        suffix = f"_{self.basename}" if self.basename else ""
        summary = df_metrics.describe().T
        summary_filepath = os.path.join(clf_dir, f"metrics_summary{suffix}.csv")
        summary.to_csv(summary_filepath)

        all_metrics_filepath = os.path.join(clf_dir, f"all_metrics{suffix}.csv")
        if "random_state" in df_metrics.columns:
            df_metrics_indexed = df_metrics.set_index("random_state")
        else:
            df_metrics_indexed = df_metrics
        df_metrics_indexed.to_csv(all_metrics_filepath, index=True)

        logger.info("=" * 60)
        logger.info(f"Metrics Summary — {classifier_name} (mean ± std across seeds)")
        logger.info("=" * 60)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "f1_score",
            "precision",
            "recall",
        ):
            if metric not in df_metrics_indexed.columns:
                continue
            mean = df_metrics_indexed[metric].mean()
            std = df_metrics_indexed[metric].std()
            logger.info(f"{metric:20s}: {mean:.4f} ± {std:.4f}")
        logger.info("=" * 60)

        if self.verbose:
            logger.info(f"Summary metrics saved to: {summary_filepath}")
            logger.info(f"All metrics saved to: {all_metrics_filepath}")

        self.metrics[classifier_name] = df_metrics
        self.aggregates[classifier_name] = summary


def _compute_one_seed_metrics(
    classifier_name: str,
    random_state: int,
    json_path: str,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
    overwrite: bool,
    verbose: bool,
) -> dict[str, Any]:
    """Compute and persist the metric dict for a single seed.

    Top-level function so it can be pickled by ``joblib.Parallel`` (a method
    bound to ``MetricsEnsemble`` would re-serialise ``self`` for every worker).

    Args:
        classifier_name (str): Stored in the output dict under ``model_name``.
        random_state (int): Seed identifier. Stored under ``random_state``.
        json_path (str): Output path. Loaded instead of recomputed when the
            file already exists and ``overwrite`` is ``False``.
        y_proba (np.ndarray): Positive-class probabilities for this seed,
            shape ``(n_samples,)``.
        y_pred (np.ndarray): Binary predictions for this seed, shape
            ``(n_samples,)``.
        y_true (np.ndarray): Ground-truth labels, shape ``(n_samples,)``.
        overwrite (bool): Recompute even when ``json_path`` exists.
        verbose (bool): When ``True``, log cache hits.

    Returns:
        dict[str, Any]: The metrics dictionary written to ``json_path``.
    """
    if not overwrite and os.path.isfile(json_path):
        with open(json_path) as f:
            cached: dict[str, Any] = json.load(f)
        if verbose:
            logger.info(f"{classifier_name}/{random_state:05d}: cached metrics loaded.")
        return cached

    metrics: dict[str, Any] = {
        **MetricsComputer(
            y_true=y_true,
            y_proba=y_proba,
            y_pred=y_pred,
        ).compute_all_metrics(),
        "random_state": int(random_state),
        "model_name": classifier_name,
    }

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
