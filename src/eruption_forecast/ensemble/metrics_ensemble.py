import os
from typing import Self, Literal
from os.path import dirname
from datetime import datetime

import numpy as np
import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import build_y_true, compute_seed
from eruption_forecast.utils.pathutils import (
    ensure_dir,
    load_pickle,
    resolve_output_dir,
)
from eruption_forecast.plots.evaluation_plots import (
    PER_SEED_PLOT_DISPATCHER,
    AGGREGATE_PLOT_DISPATCHER,
    render_one_plot,
    render_one_aggregate_plot,
)
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class MetricsEnsemble:
    """Per-seed metric engine over a fitted ``ClassifierEnsemble``.

    Wraps a trained ``ClassifierEnsemble`` with the feature matrix and
    ground-truth labels needed to compute, persist, and plot per-seed
    metrics. :meth:`compute` materialises the ``(n_samples, n_seeds)``
    probability and prediction matrices for every classifier and writes
    the per-classifier CSV artefacts under :attr:`classifiers_dir`;
    downstream plotting methods reuse the cached arrays in memory.

    Attributes:
        ClassifierEnsemble (ClassifierEnsemble): Fitted ensemble whose
            per-classifier ``SeedEnsemble`` instances drive the metric
            loop.
        features_df (pd.DataFrame): Feature matrix aligned with
            :attr:`y_true`. Shape ``(n_samples, n_features)``.
        y_true (np.ndarray): Ground-truth binary labels of length
            ``n_samples``.
        kind (Literal["prediction", "training"]): Reuse mode. Drives
            the default ``output_dir`` subpath.
        output_dir (str): Resolved evaluation root for this instance.
        classifiers_dir (str): ``{output_dir}/classifiers`` —
            per-classifier artefact root.
        n_jobs (int): Outer joblib worker count.
        overwrite (bool): When ``True``, regenerate per-seed and aggregate
            plot artefacts even if the destination files already exist.
            When ``False``, existing ``.png`` (and ``.csv`` for aggregate
            plots) outputs are kept and the render call is short-circuited.
        verbose (bool): When ``True``, emits progress logs.
        metrics (dict[str, pd.DataFrame]): Per-classifier per-seed
            metric tables, keyed by classifier name. Populated by
            :meth:`compute`.
        y_probas (dict[str, np.ndarray]): Per-classifier
            ``(n_samples, n_seeds)`` probability matrices. Populated by
            :meth:`compute`.
        y_preds (dict[str, np.ndarray]): Per-classifier
            ``(n_samples, n_seeds)`` thresholded prediction matrices.
            Populated by :meth:`compute`.

    Example:
        >>> from eruption_forecast.ensemble.metrics_ensemble import (
        ...     MetricsEnsemble,
        ... )
        >>> me = MetricsEnsemble.from_file(
        ...     model_filepath="output/VG.OJN.00.EHZ/ClassifierEnsemble.json",
        ...     features_path="output/VG.OJN.00.EHZ/features-matrix.parquet",
        ...     features_label_csv="output/VG.OJN.00.EHZ/labels.csv",
        ...     eruption_dates=["2025-03-20"],
        ...     n_jobs=4,
        ... )
        >>> me.compute().plot_aggregate()
    """

    def __init__(
        self,
        classifier_ensemble: ClassifierEnsemble,
        features_df: pd.DataFrame,
        y_true: pd.Series | np.ndarray,
        kind: Literal["prediction", "training"] = "prediction",
        output_dir: str | None = None,
        root_dir: str | None = None,
        overwrite: bool = False,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        """Bind a fitted ensemble to the data needed to score it.

        Args:
            classifier_ensemble (ClassifierEnsemble): Fitted ensemble
                whose per-classifier ``SeedEnsemble`` instances drive
                the metric loop.
            features_df (pd.DataFrame): Feature matrix aligned with
                ``y_true``. Shape ``(n_samples, n_features)``.
            y_true (pd.Series | np.ndarray): Ground-truth binary labels.
                ``pd.Series`` inputs are converted to ``np.ndarray``.
            kind (Literal["prediction", "training"], optional): Reuse
                mode used to derive the default ``output_dir`` subpath.
                Defaults to ``"prediction"``.
            output_dir (str | None, optional): Explicit evaluation root.
                When ``None``, resolves to
                ``{root_dir}/output/evaluation/{kind}``. Defaults to
                ``None``.
            root_dir (str | None, optional): Project root used to
                resolve ``output_dir`` when no explicit path is given.
                Defaults to ``None``.
            overwrite (bool, optional): When ``True``, force-regenerate
                per-seed and aggregate plot artefacts even when the
                target files already exist; when ``False``, existing
                ``.png`` (and ``.csv`` for aggregate plots) outputs are
                kept and rendering is skipped. Defaults to ``False``.
            n_jobs (int, optional): Outer joblib worker count for the
                metric and plot loops. Defaults to ``1``.
            verbose (bool, optional): When ``True``, emits progress
                logs. Defaults to ``False``.
        """
        output_dir = resolve_output_dir(
            output_dir=output_dir,
            root_dir=root_dir,
            default_subpath=os.path.join("output", "evaluation", kind),
        )

        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()

        self.ClassifierEnsemble = classifier_ensemble
        self.features_df = features_df
        self.y_true: np.ndarray = y_true
        self.kind: Literal["prediction", "training"] = kind
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.classifiers_dir = os.path.join(output_dir, "classifiers")
        self.overwrite = overwrite
        self.verbose = verbose

        self.metrics: dict[str, pd.DataFrame] = {}
        self.y_probas: dict[str, np.ndarray] = {}
        self.y_preds: dict[str, np.ndarray] = {}

    @classmethod
    def from_file(
        cls,
        model_filepath: str,
        features_path: str,
        features_label_csv: str,
        eruption_dates: list[str] | list[datetime],
        kind: Literal["prediction", "training"] = "prediction",
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "MetricsEnsemble":
        """Construct a ``MetricsEnsemble`` from on-disk artefacts.

        Loads a serialised ``ClassifierEnsemble`` together with the
        feature matrix and the label CSV needed to derive ``y_true``,
        then forwards everything to the standard constructor.

        Args:
            model_filepath (str): Path to a ``ClassifierEnsemble``
                artefact — ``.json``, ``.pkl``, or a trained-model
                registry CSV — accepted by
                :meth:`ClassifierEnsemble.from_any`.
            features_path (str): Path to the merged feature matrix
                Parquet used for scoring (the ``features-matrix_*.parquet``
                produced by :class:`FeaturesBuilder`).
            features_label_csv (str): Path to the label CSV consumed by
                :func:`build_y_true` to derive the binary ground truth.
            eruption_dates (list[str] | list[datetime]): Eruption dates
                forwarded to :func:`build_y_true` to mark positive
                samples.
            kind (Literal["prediction", "training"], optional): Reuse
                mode. Defaults to ``"prediction"``.
            output_dir (str | None, optional): Explicit evaluation
                root. Defaults to ``None``.
            root_dir (str | None, optional): Project root used to
                resolve ``output_dir`` when no explicit path is given.
                Defaults to ``None``.
            n_jobs (int, optional): Outer joblib worker count. Defaults
                to ``1``.
            verbose (bool, optional): When ``True``, emits progress
                logs. Defaults to ``False``.

        Returns:
            MetricsEnsemble: A fresh instance ready for :meth:`compute`.

        Raises:
            FileNotFoundError: If ``model_filepath``, ``features_path``,
                or ``features_label_csv`` does not exist.

        Example:
            >>> me = MetricsEnsemble.from_file(
            ...     model_filepath="output/VG.OJN.00.EHZ/ClassifierEnsemble.json",
            ...     features_path="output/VG.OJN.00.EHZ/features-matrix.parquet",
            ...     features_label_csv="output/VG.OJN.00.EHZ/labels.csv",
            ...     eruption_dates=["2025-03-20"],
            ... )
            >>> me.compute()
        """
        for label, path in (
            ("Model Filepath", model_filepath),
            ("Features Matrix", features_path),
            ("Features Label CSV", features_label_csv),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} file not found: {path}")

        classifier_ensemble = ClassifierEnsemble.from_any(
            model_filepath, verbose=verbose
        )
        features_df = pd.read_parquet(features_path)
        y_true = build_y_true(features_label_csv, eruption_dates)

        return cls(
            classifier_ensemble=classifier_ensemble,
            features_df=features_df,
            y_true=y_true,
            kind=kind,
            output_dir=output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        )

    @classmethod
    def load(cls, path: str) -> "MetricsEnsemble":
        """Reconstitute a saved MetricsEnsemble from a joblib ``.pkl``.

        Args:
            path (str): Path to a ``.pkl`` previously written by :meth:`save`.

        Returns:
            MetricsEnsemble: The deserialised instance, with ``metrics`` /
                ``y_probas`` / ``y_preds`` populated as of the save.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            TypeError: If the loaded object is not a ``MetricsEnsemble``.

        Example:
            >>> me = MetricsEnsemble.load(
            ...     "output/evaluation/prediction/MetricsEnsemble.pkl"
            ... )
            >>> me.plot_aggregate()
        """
        obj = load_pickle(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is not a MetricsEnsemble (got {type(obj).__name__})."
            )
        return obj

    def compute(self) -> Self:
        """Materialise per-seed metrics and probability matrices.

        Aligns :attr:`features_df` and :attr:`y_true` on their shared
        index, dispatches the per-classifier metric loop to
        :meth:`_compute_job`, and writes per-classifier ``y_proba`` /
        ``y_pred`` CSVs via :meth:`_persist_predictions`. Idempotent —
        once :attr:`y_probas` is populated, subsequent calls return
        without recomputing.

        Returns:
            Self: This instance, with :attr:`metrics`, :attr:`y_probas`,
                and :attr:`y_preds` populated.

        Raises:
            ValueError: If :attr:`features_df` and the index-aligned
                ``y_true`` share no common index entries.

        Example:
            >>> me = MetricsEnsemble.from_file(...)
            >>> me.compute()
            >>> me.metrics["rf"].head()
        """
        if self.y_probas:
            if self.verbose:
                logger.info(
                    "MetricsEnsemble.compute(): y_probas already populated; "
                    "skipping recomputation."
                )
            return self

        y_true = pd.Series(self.y_true, index=self.features_df.index, name="is_erupted")

        common_index = self.features_df.index.intersection(y_true.index)
        if len(common_index) == 0:
            raise ValueError(
                "features_df and y_true have no overlapping index entries. "
                "Check your features_df.index and y_true.index."
            )

        features_df = self.features_df.loc[common_index]
        y_true_series = y_true.loc[common_index].astype(int)
        self.y_true = y_true_series.to_numpy()

        (
            self.metrics,
            self.y_probas,
            self.y_preds,
        ) = self._compute_job(X=features_df, y_true=self.y_true)

        self._persist_predictions(features_df=features_df)

        return self

    def plot_seed(
        self,
        include_plots: list[str] | None = None,
        exclude_plots: list[str] | None = None,
    ):
        """Render per-seed plots registered in the dispatcher.

        Auto-runs :meth:`compute` first if probabilities have not been
        materialised yet. ``include_plots`` narrows to a positive subset (when
        provided); ``exclude_plots`` then drops names from whatever remains.

        Args:
            include_plots (list[str] | None): Positive opt-in list of plot
                names. When ``None``, starts from the full dispatcher
                registry. Defaults to ``None``.
            exclude_plots (list[str] | None): Plot names to omit. Applied
                after ``include_plots``. Defaults to ``None``.

        Returns:
            list[str]: Saved figure filepath stems, one per executed job.

        Example:
            >>> me = MetricsEnsemble.from_file(...)
            >>> me.plot_seed(include_plots=["roc_curve", "confusion_matrix"])
        """
        if len(self.y_probas) == 0:
            self.compute()

        plots = list(PER_SEED_PLOT_DISPATCHER.keys())
        if include_plots:
            plots = [name for name in plots if name in include_plots]
        if exclude_plots:
            plots = [name for name in plots if name not in exclude_plots]

        return self._plot_seed_jobs(plots)

    def plot_aggregate(
        self,
        include_plots: list[str] | None = None,
        exclude_plots: list[str] | None = None,
    ) -> list[str]:
        """Render aggregate plots registered in the dispatcher.

        For each ``(classifier_name, plot_name)`` pair, hands the full
        ``(n_samples, n_seeds)`` probability matrix to
        :func:`render_one_aggregate_plot`, which slices it into per-seed
        columns and writes both the figure and the returned mean/std
        DataFrame to disk. Auto-runs :meth:`compute` first if probabilities
        have not been materialised yet. ``include_plots`` narrows to a
        positive subset (when provided); ``exclude_plots`` then drops names
        from whatever remains.

        Args:
            include_plots (list[str] | None): Positive opt-in list of plot
                names. When ``None``, starts from the full dispatcher
                registry. Defaults to ``None``.
            exclude_plots (list[str] | None): Plot names to omit. Applied
                after ``include_plots``. Defaults to ``None``.

        Returns:
            list[str]: Saved figure filepath stems, one per executed job.
                The ``.png`` figure and the ``.csv`` data table share each
                stem.

        Example:
            >>> me = MetricsEnsemble.from_file(...)
            >>> me.plot_aggregate(exclude_plots=["calibration_curve"])
        """
        if len(self.y_probas) == 0:
            self.compute()

        plots = list(AGGREGATE_PLOT_DISPATCHER.keys())
        if include_plots:
            plots = [name for name in plots if name in include_plots]
        if exclude_plots:
            plots = [name for name in plots if name not in exclude_plots]

        return self._plot_aggregate_jobs(plots)

    def save(self, path: str | None = None) -> str:
        """Persist the computed MetricsEnsemble to disk via joblib.

        Writes a single ``.pkl`` containing the full instance — including the
        embedded ``ClassifierEnsemble``, ``features_df``, ``y_true``, and the
        populated ``metrics`` / ``y_probas`` / ``y_preds`` dicts — so
        :meth:`load` can reconstitute everything without recomputation.

        Args:
            path (str | None, optional): Destination ``.pkl`` path. ``None``
                resolves to ``{self.output_dir}/MetricsEnsemble.pkl``.
                Defaults to ``None``.

        Returns:
            str: The absolute path the instance was written to.

        Example:
            >>> me = MetricsEnsemble.from_file(...).compute()
            >>> me.save()
            '.../output/evaluation/prediction/MetricsEnsemble.pkl'
        """
        if path is None:
            os.makedirs(self.output_dir, exist_ok=True)
            path = os.path.join(self.output_dir, "MetricsEnsemble.pkl")
        else:
            parent = dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        joblib.dump(self, path)
        if self.verbose:
            logger.info(f"Saved MetricsEnsemble to {path}")

        return path

    def _plot_seed_jobs(self, plots: list[str]) -> list[str]:
        """Render the requested per-seed plots in parallel.

        Builds a flat list of ``(classifier_name, seed_idx, plot_name)`` jobs
        and dispatches them to ``joblib.Parallel`` with the ``loky`` backend.
        Caller is responsible for restricting ``plots`` to names registered
        in ``PER_SEED_PLOT_DISPATCHER``.

        Args:
            plots (list[str]): Per-seed plot names to render.

        Returns:
            list[str]: Saved figure filepath stems, one per executed job.
        """
        jobs: list[tuple[str, int, str]] = []
        for classifier_name, y_probas in self.y_probas.items():
            n_seeds = y_probas.shape[1]
            for seed_idx in range(n_seeds):
                for plot_name in plots:
                    jobs.append((classifier_name, seed_idx, plot_name))

        return joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(
            joblib.delayed(render_one_plot)(
                classifier_name=classifier_name,
                random_state=int(self.metrics[classifier_name].index[seed_idx]),
                plot_name=plot_name,
                y_true=self.y_true,
                y_proba=self.y_probas[classifier_name][:, seed_idx],
                output_dir=self.classifiers_dir,
                overwrite=self.overwrite,
                verbose=self.verbose,
            )
            for classifier_name, seed_idx, plot_name in jobs
        )

    def _plot_aggregate_jobs(self, plots: list[str]) -> list[str]:
        """Render the requested aggregate plots in parallel.

        Builds a flat list of ``(classifier_name, plot_name)`` jobs and
        dispatches them to ``joblib.Parallel`` with the ``loky`` backend.
        Caller is responsible for restricting ``plots`` to names registered
        in ``AGGREGATE_PLOT_DISPATCHER``.

        Args:
            plots (list[str]): Aggregate plot names to render.

        Returns:
            list[str]: Saved figure filepath stems, one per executed job.
        """
        jobs: list[tuple[str, str]] = [
            (cls, name) for cls in self.y_probas for name in plots
        ]

        return joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(
            joblib.delayed(render_one_aggregate_plot)(
                classifier_name=cls,
                plot_name=name,
                y_true=self.y_true,
                y_probas=self.y_probas[cls],
                output_dir=self.classifiers_dir,
                overwrite=self.overwrite,
                verbose=self.verbose,
            )
            for cls, name in jobs
        )

    def _compute_job(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
    ) -> tuple[
        dict[str, pd.DataFrame],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        """Handling parallel job to calculate metrics.

        Args:
            X (pd.DataFrame): Extracted features DataFrame of shape
                ``(n_samples, n_features)``.
            y_true (np.ndarray | pd.Series): Ground-truth binary labels
                aligned positionally with ``X``. Length must equal
                ``n_samples``.

        Returns:
            tuple: A 3-tuple of per-classifier dicts —
                ``(metrics, y_probas, y_preds)`` — keyed by classifier name.
        """
        classifier_names = list(self.ClassifierEnsemble.ensembles.keys())
        seed_ensembles = list(self.ClassifierEnsemble.ensembles.values())

        results = joblib.Parallel(n_jobs=self.n_jobs, backend="loky")(
            joblib.delayed(compute_seed)(
                seed_ensemble=seed_ensemble,
                X=X,
                y_true=y_true,
                output_dir=self.classifiers_dir,
                verbose=self.verbose,
            )
            for seed_ensemble in seed_ensembles
        )

        metrics_dfs, y_probas_arrs, y_preds_arrs = zip(*results, strict=True)

        metrics = dict(zip(classifier_names, metrics_dfs, strict=True))
        y_probas = dict(zip(classifier_names, y_probas_arrs, strict=True))
        y_preds = dict(zip(classifier_names, y_preds_arrs, strict=True))

        return metrics, y_probas, y_preds

    def _persist_predictions(
        self,
        features_df: pd.DataFrame,
    ) -> None:
        """Write per-classifier ``y_proba`` / ``y_pred`` / ``y_true`` CSVs.

        For every classifier in :attr:`y_probas`, materialises the
        ``(n_samples, n_seeds)`` probability and prediction matrices as
        DataFrames indexed by ``features_df.index`` with one column per seed
        (``seed_{random_state:05d}``), and writes them alongside the aligned
        ground-truth Series under
        ``{classifiers_dir}/{classifier_name}/predictions/``.

        Args:
            features_df (pd.DataFrame): Index-aligned feature matrix; supplies
                the row index for the saved matrices.
        """
        for classifier_name, y_proba_matrix in self.y_probas.items():
            predictions_dir = os.path.join(
                self.classifiers_dir, classifier_name, "predictions"
            )
            ensure_dir(predictions_dir)

            seed_columns = [
                f"seed_{int(rs):05d}" for rs in self.metrics[classifier_name].index
            ]
            y_proba_df = pd.DataFrame(
                y_proba_matrix,
                index=features_df.index,
                columns=seed_columns,
            )
            y_pred_df = pd.DataFrame(
                self.y_preds[classifier_name].astype(int),
                index=features_df.index,
                columns=seed_columns,
            )

            y_proba_df.to_csv(os.path.join(predictions_dir, "y_proba.csv"))
            y_pred_df.to_csv(os.path.join(predictions_dir, "y_pred.csv"))

            if self.verbose:
                logger.info(
                    f"{classifier_name}: y_proba and y_pred saved to {predictions_dir}"
                )
