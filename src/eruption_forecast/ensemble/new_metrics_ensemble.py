import os
from typing import Self, Literal
from datetime import datetime

import numpy as np
import joblib
import pandas as pd

from eruption_forecast.utils.ml import build_y_true, compute_seed
from eruption_forecast.utils.pathutils import resolve_output_dir
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
            default_subpath=os.path.join("output", "evaluation"),
        )

        output_dir = os.path.join(output_dir, kind)

        if isinstance(y_true, pd.Series):
            y_true = y_true.to_numpy()

        self.ClassifierEnsemble = classifier_ensemble
        self.features_df = features_df
        self.y_true: np.ndarray = y_true
        self.n_jobs = n_jobs
        self.output_dir = output_dir
        self.classifiers_dir = os.path.join(output_dir, "classifiers")
        self.verbose = verbose

        self.metrics: dict[str, pd.DataFrame] = {}
        self.y_probas: dict[str, np.ndarray] = {}

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

    def compute(self) -> Self:
        y_true = pd.Series(self.y_true, index=self.features_df.index, name="is_erupted")

        common_index = self.features_df.index.intersection(y_true.index)
        if len(common_index) == 0:
            raise ValueError(
                "features_df and y_true have no overlapping index entries. "
                "Check your features_df.index and y_true.index."
            )

        features_df = self.features_df.loc[common_index]
        self.y_true = y_true.loc[common_index].astype(int).to_numpy()

        (
            self.metrics,
            self.y_probas,
        ) = self._compute_job(X=features_df, y_true=self.y_true)

        return self

    def plot(
        self,
        exclude_plots: list[str] | None = None,
    ):
        """Render every per-seed plot registered in the dispatcher.

        Auto-runs :meth:`compute` first if probabilities have not been
        materialised yet. Plot names in ``exclude_plots`` are skipped.

        Args:
            exclude_plots (list[str] | None): Plot names to omit. Defaults to
                ``None`` (render all).

        Returns:
            list[str]: Saved figure filepath stems, one per executed job.
        """
        if len(self.y_probas) == 0:
            self.compute()

        plots = list(PER_SEED_PLOT_DISPATCHER.keys())
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
                classifier_name=cls,
                random_state=int(self.metrics[cls].index[idx]),
                plot_name=name,
                y_true=self.y_true,
                y_proba=self.y_probas[cls][:, idx],
                output_dir=self.classifiers_dir,
                verbose=self.verbose,
            )
            for cls, idx, name in jobs
        )

    def plot_aggregate(
        self,
        exclude_plots: list[str] | None = None,
    ) -> list[str]:
        """Render every aggregate plot registered in the dispatcher.

        For each ``(classifier_name, plot_name)`` pair, hands the full
        ``(n_samples, n_seeds)`` probability matrix to
        :func:`render_one_aggregate_plot`, which slices it into per-seed
        columns and writes both the figure and the returned mean/std
        DataFrame to disk. Auto-runs :meth:`compute` first if probabilities
        have not been materialised yet.

        Args:
            exclude_plots (list[str] | None): Plot names to omit. Defaults to
                ``None`` (render all).

        Returns:
            list[str]: Saved figure filepath stems, one per executed job.
                The ``.png`` figure and the ``.csv`` data table share each
                stem.
        """
        if len(self.y_probas) == 0:
            self.compute()

        plots = list(AGGREGATE_PLOT_DISPATCHER.keys())
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
    ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
        """Handling parallel job to calculate metrics.

        Args:
            X (pd.DataFrame): Extracted features DataFrame of shape
                ``(n_samples, n_features)``.
            y_true (np.ndarray | pd.Series): Ground-truth binary labels
                aligned positionally with ``X``. Length must equal
                ``n_samples``.
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

        metrics_dfs, y_probas = zip(*results, strict=True)

        metrics = dict(zip(classifier_names, metrics_dfs, strict=True))
        y_probas = dict(zip(classifier_names, y_probas, strict=True))

        return metrics, y_probas
