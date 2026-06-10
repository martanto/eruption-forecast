import os
from typing import Self, Literal
from os.path import dirname
from datetime import datetime

import numpy as np
import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import build_y_true, compute_seed
from eruption_forecast.utils.pathutils import ensure_dir, resolve_output_dir
from eruption_forecast.plots.evaluation_plots import (
    PER_SEED_PLOT_DISPATCHER,
    AGGREGATE_PLOT_DISPATCHER,
    render_one_plot,
    render_one_aggregate_plot,
)
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


class MetricsEnsemble:
    def __init__(
        self,
        classifier_ensemble: ClassifierEnsemble,
        features_df: pd.DataFrame,
        y_true: pd.Series | np.ndarray,
        kind: Literal["prediction", "training"] = "prediction",
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
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
        self.verbose = verbose

        self.metrics: dict[str, pd.DataFrame] = {}
        self.y_probas: dict[str, np.ndarray] = {}
        self.y_preds: dict[str, np.ndarray] = {}

    @classmethod
    def from_file(
        cls,
        model_filepath: str,
        features_csv: str,
        features_label_csv: str,
        eruption_dates: list[str] | list[datetime],
        kind: Literal["prediction", "training"] = "prediction",
        output_dir: str | None = None,
        root_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "MetricsEnsemble":
        for label, path in (
            ("Model Filepath", model_filepath),
            ("Features CSV", features_csv),
            ("Features Label CSV", features_label_csv),
        ):
            if not os.path.exists(path):
                raise FileNotFoundError(f"{label} file not found: {path}")

        classifier_ensemble = ClassifierEnsemble.from_any(
            model_filepath, verbose=verbose
        )
        features_df = pd.read_csv(features_csv, index_col=0)
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
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"MetricsEnsemble file not found: {path}")

        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is not a MetricsEnsemble (got {type(obj).__name__})."
            )
        return obj

    def compute(self) -> Self:
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

    def plot(
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
        """
        if len(self.y_probas) == 0:
            self.compute()

        plots = list(PER_SEED_PLOT_DISPATCHER.keys())
        if include_plots:
            plots = [name for name in plots if name in include_plots]
        if exclude_plots:
            plots = [name for name in plots if name not in exclude_plots]

        return self._plot_jobs(plots)

    def _plot_jobs(self, plots: list[str]) -> list[str]:
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
                verbose=self.verbose,
            )
            for classifier_name, seed_idx, plot_name in jobs
        )

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
        """
        if len(self.y_probas) == 0:
            self.compute()

        plots = list(AGGREGATE_PLOT_DISPATCHER.keys())
        if include_plots:
            plots = [name for name in plots if name in include_plots]
        if exclude_plots:
            plots = [name for name in plots if name not in exclude_plots]

        return self._plot_aggregate_jobs(plots)

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
