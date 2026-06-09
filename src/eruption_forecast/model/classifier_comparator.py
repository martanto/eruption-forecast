"""Cross-classifier comparison for volcanic eruption forecasting models.

This module provides ``ClassifierComparator``, which consumes a fitted
:class:`MetricsEnsemble` and produces side-by-side comparison plots and a
ranking table across the classifiers it covers.
"""

import os
import math
from typing import Self, Literal
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    configure_spine,
    apply_nature_style,
)
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.ensemble.new_metrics_ensemble import MetricsEnsemble


#: Metrics shown by default when ``metrics=None`` is passed to the constructor
#: or any plot method.  Only scalar, normalised scores are included; raw counts
#: (true_positives, etc.) and threshold-specific values are excluded.
DEFAULT_METRICS: list[str] = [
    "f1_score",
    "roc_auc",
    "pr_auc",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "g_mean",
    "mcc",
]


class ClassifierComparator:
    """Compare multiple classifiers side-by-side using a fitted ``MetricsEnsemble``.

    Consumes a computed :class:`MetricsEnsemble` (per-classifier per-seed
    metric DataFrames plus the ``(n_samples, n_seeds)`` ``y_probas`` matrices)
    and exposes methods for ranking, aggregate tables, and comparison plots.

    Attributes:
        output_dir (str): Root directory for saved outputs.
        figures_dir (str): Subdirectory for saved plot files.
        metrics_dir (str): Subdirectory for saved CSV files.
        metrics (list[str]): Ordered list of metrics to compare.

    Args:
        metrics_ensemble (MetricsEnsemble): A ``MetricsEnsemble`` whose
            ``compute()`` has already populated ``metrics`` and ``y_probas``.
        metrics (str | list[str] | None, optional): Metric or ordered list of
            metrics used for ranking and plots. The first entry is the default
            ranking metric. When None, uses ``DEFAULT_METRICS``. Defaults to
            None.
        output_dir (str | None, optional): Root output directory. When None,
            falls back to ``metrics_ensemble.output_dir``; the comparator
            always appends ``comparison/`` to whichever root is resolved.
            Defaults to None.

    Raises:
        ValueError: If ``metrics_ensemble.metrics`` is empty (``compute()``
            was not called).

    Examples:
        >>> me = MetricsEnsemble(
        ...     classifier_ensemble=ce, features_df=X, y_true=y
        ... ).compute()
        >>> comparator = ClassifierComparator(me, metrics=["f1_score", "roc_auc"])
        >>> ranking = comparator.get_ranking()
        >>> figures = comparator.plot_all()
    """

    def __init__(
        self,
        metrics_ensemble: MetricsEnsemble,
        metrics: str | list[str] | None = None,
        output_dir: str | None = None,
    ) -> None:
        if not metrics_ensemble.metrics:
            raise ValueError(
                "metrics_ensemble.metrics is empty. "
                "Call metrics_ensemble.compute() before constructing ClassifierComparator."
            )

        if metrics is None:
            self.metrics: list[str] = list(DEFAULT_METRICS)
        elif isinstance(metrics, str):
            self.metrics = [metrics]
        else:
            self.metrics = list(metrics)

        self._metrics_ensemble: MetricsEnsemble = metrics_ensemble
        self._classifier_names: list[str] = list(metrics_ensemble.metrics.keys())

        output_dir = output_dir or metrics_ensemble.output_dir
        self.output_dir = os.path.join(output_dir, "comparison")
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        self._metrics_table_cache: pd.DataFrame | None = None

    @classmethod
    def from_classifier_ensemble(
        cls,
        classifier_ensemble: ClassifierEnsemble,
        features_df: pd.DataFrame,
        y_true: pd.Series | np.ndarray,
        kind: Literal["training", "prediction"] = "training",
        output_dir: str | None = None,
        root_dir: str | None = None,
        metrics: str | list[str] | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> Self:
        """Build a comparator directly from a fitted ``ClassifierEnsemble``.

        Constructs a fresh :class:`MetricsEnsemble`, runs its ``compute()``,
        and returns a ``ClassifierComparator`` wired to its in-memory results.

        Args:
            classifier_ensemble (ClassifierEnsemble): Fitted ensemble whose
                seeds are evaluated.
            features_df (pd.DataFrame): Feature matrix aligned to ``y_true``.
            y_true (pd.Series | np.ndarray): Ground-truth binary labels.
            kind (Literal["training", "prediction"]): Evaluation context;
                stored on the constructed ``MetricsEnsemble`` and, when
                ``output_dir`` is provided, appended to it so artefacts land
                under ``{output_dir}/{kind}/``. Defaults to ``"training"``.
            output_dir (str | None, optional): Root output directory passed to
                ``MetricsEnsemble``. ``kind`` is appended automatically when
                this is set. Defaults to None.
            root_dir (str | None, optional): Project root passed to
                ``MetricsEnsemble`` for path resolution. Defaults to None.
            metrics (str | list[str] | None, optional): Metric or list of
                metrics for ranking and plots. Defaults to None.
            n_jobs (int, optional): Parallel workers passed to
                ``MetricsEnsemble``. Defaults to 1.
            verbose (bool, optional): Emit progress logs. Defaults to False.

        Returns:
            ClassifierComparator: Initialised comparator backed by a freshly
                computed ``MetricsEnsemble``.

        Examples:
            >>> comparator = ClassifierComparator.from_classifier_ensemble(
            ...     classifier_ensemble=training_model.ClassifierEnsemble,
            ...     features_df=training_model.features_df,
            ...     y_true=training_model.labels,
            ...     output_dir=training_model.output_dir,
            ... )
        """
        resolved_output_dir = (
            os.path.join(output_dir, kind) if output_dir is not None else None
        )

        metrics_ensemble = MetricsEnsemble(
            classifier_ensemble=classifier_ensemble,
            features_df=features_df,
            y_true=y_true,
            kind=kind,
            output_dir=resolved_output_dir,
            root_dir=root_dir,
            n_jobs=n_jobs,
            verbose=verbose,
        ).compute()

        return cls(
            metrics_ensemble=metrics_ensemble,
            output_dir=resolved_output_dir,
            metrics=metrics,
        )

    def _save_figure(self, fig: plt.Figure, filename: str, dpi: int) -> None:
        """Save a figure to the figures subdirectory.

        Args:
            fig (plt.Figure): Figure to save.
            filename (str): PNG filename (without path).
            dpi (int): Resolution in dots per inch.
        """
        ensure_dir(self.figures_dir)
        path = os.path.join(self.figures_dir, filename)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved comparison figure: {path}")

    def _color_cycle(self) -> list[str]:
        """Return one ``OKABE_ITO`` colour per classifier, cycling if needed."""
        n = len(self._classifier_names)
        colors = list(OKABE_ITO)
        return [colors[i % len(colors)] for i in range(n)]

    @staticmethod
    def _validate_metrics(metrics_list: list[str], table: pd.DataFrame) -> None:
        """Raise ``ValueError`` for any metric not present in ``table``.

        Args:
            metrics_list (list[str]): Metric names to validate.
            table (pd.DataFrame): Aggregate metrics table produced by
                :meth:`get_metrics_table`.

        Raises:
            ValueError: If any metric name is absent from ``table``.
        """
        available = {
            col.rsplit("_", 1)[0] for col in table.columns if col.endswith("_mean")
        }
        invalid = [m for m in metrics_list if m not in available]
        if invalid:
            raise ValueError(
                f"Unknown metric(s): {invalid}. Available metrics: {sorted(available)}"
            )

    def _resolve_metrics(self, metrics: str | list[str] | None) -> list[str]:
        """Normalise a metrics argument to a list."""
        if metrics is None:
            return list(self.metrics)
        if isinstance(metrics, str):
            return [metrics]
        return list(metrics)

    @staticmethod
    def _build_legend_patches(
        clf_names: list[str],
        clf_colors: list[str],
    ) -> list[mpatches.Patch]:
        """Build one legend patch per classifier."""
        return [
            mpatches.Patch(facecolor=clf_colors[i], label=name)
            for i, name in enumerate(clf_names)
        ]

    @staticmethod
    def _attach_legend(
        fig: plt.Figure,
        handles: list[mpatches.Patch],
        ncol: int,
        bbox_to_anchor: tuple[float, float],
    ) -> None:
        """Attach a classifier legend to the bottom of a figure."""
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=ncol,
            fontsize=7,
            frameon=False,
            bbox_to_anchor=bbox_to_anchor,
        )

    def _per_seed_records(self) -> dict[str, list[dict]]:
        """Return per-seed metric records keyed by classifier name.

        Converts each ``metrics_ensemble.metrics[clf]`` DataFrame into a list
        of row-dicts. The ``random_state`` index is surfaced as a column so it
        appears in each record alongside the metric values.
        """
        return {
            name: df.reset_index().to_dict(orient="records")
            for name, df in self._metrics_ensemble.metrics.items()
        }

    def get_metrics_table(self) -> pd.DataFrame:
        """Return a DataFrame of aggregate metrics for every classifier.

        Aggregates ``mean``, ``std``, ``min``, and ``max`` across seeds from
        each numeric column of ``metrics_ensemble.metrics[clf]`` and flattens
        them into ``{metric}_{stat}`` columns.

        Returns:
            pd.DataFrame: DataFrame indexed by classifier name with one
                ``{metric}_{stat}`` column per (metric, statistic) pair.

        Examples:
            >>> df = comparator.get_metrics_table()
            >>> print(df[["f1_score_mean", "f1_score_std"]])
        """
        if self._metrics_table_cache is not None:
            return self._metrics_table_cache

        rows: dict[str, dict[str, float]] = {}
        for name, df in self._metrics_ensemble.metrics.items():
            numeric = df.select_dtypes(include="number")
            flat: dict[str, float] = {}
            for metric_name in numeric.columns:
                col = numeric[metric_name]
                flat[f"{metric_name}_mean"] = float(col.mean())
                # ``ddof=0`` so a single-seed run yields 0.0 rather than NaN.
                flat[f"{metric_name}_std"] = float(col.std(ddof=0))
                flat[f"{metric_name}_min"] = float(col.min())
                flat[f"{metric_name}_max"] = float(col.max())
            rows[name] = flat

        self._metrics_table_cache = pd.DataFrame.from_dict(rows, orient="index")
        return self._metrics_table_cache

    def get_ranking(
        self,
        metric: str = "recall",
        by: str = "mean",
    ) -> pd.DataFrame:
        """Return classifiers ranked by a chosen metric, descending.

        Args:
            metric (str, optional): Metric to rank by. Defaults to ``"recall"``.
            by (str, optional): Aggregation statistic to sort on — one of
                ``"mean"``, ``"std"``, ``"min"``, or ``"max"``. Defaults to
                ``"mean"``.

        Returns:
            pd.DataFrame: Ranked DataFrame with a ``rank`` column prepended.

        Raises:
            KeyError: If the requested ``{metric}_{by}`` column does not exist
                in the metrics table.
        """
        col = f"{metric}_{by}"
        table = self.get_metrics_table()

        if col not in table.columns:
            raise KeyError(
                f"Column '{col}' not found in metrics table. "
                f"Available columns: {list(table.columns)}"
            )

        ranked = table.sort_values(col, ascending=False).copy()
        ranked.insert(0, "rank", range(1, len(ranked) + 1))

        ensure_dir(self.metrics_dir)
        csv_path = os.path.join(self.metrics_dir, f"ranking_{metric}.csv")
        ranked.to_csv(csv_path)
        logger.info(f"Saved ranking table: {csv_path}")

        return ranked

    def _iter_seed_roc_data(self, name: str):
        """Yield ``(y_true, y_proba)`` pairs for every seed of one classifier.

        Reads directly from ``metrics_ensemble.y_probas[name]`` columns and
        ``metrics_ensemble.y_true`` — no per-seed model reload. The
        ``y_true`` conversion to ``np.ndarray`` happens once outside the
        loop and the same array is yielded each iteration.

        Args:
            name (str): Classifier name; must be a key of
                ``metrics_ensemble.y_probas``.

        Yields:
            tuple[np.ndarray, np.ndarray]: ``(y_true_array, y_proba_array)``
                for each seed column.
        """
        y_probas = self._metrics_ensemble.y_probas[name]
        y_true = np.asarray(self._metrics_ensemble.y_true)
        for seed_idx in range(y_probas.shape[1]):
            yield y_true, y_probas[:, seed_idx]

    def _compute_classifier_roc(
        self,
        name: str,
        mean_fpr: np.ndarray,
        color: str,
        ax: plt.Axes,
        show_individual: bool,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Compute and draw the mean ROC curve for one classifier.

        Args:
            name (str): Classifier name used for the legend label and warnings.
            mean_fpr (np.ndarray): Common FPR grid onto which TPR values are
                interpolated before averaging.
            color (str): Hex colour string for this classifier's curves.
            ax (plt.Axes): Axes to draw on.
            show_individual (bool): Whether to draw per-seed curves as thin
                dashed lines.

        Returns:
            tuple[np.ndarray, np.ndarray, float] | None:
                ``(mean_tpr, std_tpr, mean_auc)`` if at least one seed was
                processed, else ``None``.
        """
        tprs: list[np.ndarray] = []
        aucs: list[float] = []

        for y_true, y_proba in self._iter_seed_roc_data(name):
            try:
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                aucs.append(roc_auc_score(y_true, y_proba))
                tprs.append(np.interp(mean_fpr, fpr, tpr))
            except (ValueError, KeyError):
                logger.warning(f"Skipped a seed for '{name}' in plot_roc.")
                continue

            if show_individual:
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    alpha=0.15,
                    linewidth=0.6,
                    linestyle="--",
                )

        if not tprs:
            return None

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = float(np.mean(aucs))

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            linewidth=1.5,
            label=f"{name} (AUC={mean_auc:.3f})",
        )
        ax.fill_between(
            mean_fpr,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            alpha=0.12,
            color=color,
        )

        return mean_tpr, std_tpr, mean_auc

    def _compute_classifier_pr(
        self,
        name: str,
        mean_recall: np.ndarray,
        color: str,
        ax: plt.Axes,
        show_individual: bool,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Compute and draw the mean PR curve for one classifier.

        Iterates over per-seed ``(y_true, y_proba)`` pairs from
        :meth:`_iter_seed_roc_data`, calls
        :func:`sklearn.metrics.precision_recall_curve`, reverses the
        descending-recall arrays so they are monotonically increasing for
        ``np.interp``, and aggregates precision onto the common ``mean_recall``
        grid. Per-seed AP is computed via
        :func:`sklearn.metrics.average_precision_score`.

        Args:
            name (str): Classifier name used for the legend label.
            mean_recall (np.ndarray): Common recall grid onto which precision
                values are interpolated before averaging.
            color (str): Hex colour string for this classifier's curves.
            ax (plt.Axes): Axes to draw on.
            show_individual (bool): Whether to draw per-seed PR curves as
                thin dashed lines.

        Returns:
            tuple[np.ndarray, np.ndarray, float] | None:
                ``(mean_precision, std_precision, mean_ap)`` if at least one
                seed was processed, else ``None``.
        """
        precisions_interp: list[np.ndarray] = []
        aps: list[float] = []

        for y_true, y_proba in self._iter_seed_roc_data(name):
            try:
                precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
                aps.append(average_precision_score(y_true, y_proba))
            except (ValueError, KeyError):
                logger.warning(f"Skipped a seed for '{name}' in plot_pr.")
                continue

            # precision_recall_curve returns recalls in decreasing order;
            # reverse both arrays so np.interp sees a monotonically
            # increasing x-grid.
            recalls_asc = recalls[::-1]
            precisions_paired = precisions[::-1]
            precisions_interp.append(
                np.interp(mean_recall, recalls_asc, precisions_paired)
            )

            if show_individual:
                ax.plot(
                    recalls,
                    precisions,
                    color=color,
                    alpha=0.15,
                    linewidth=0.6,
                    linestyle="--",
                )

        if not precisions_interp:
            return None

        mean_precision = np.mean(precisions_interp, axis=0)
        std_precision = np.std(precisions_interp, axis=0)
        mean_ap = float(np.mean(aps))

        ax.plot(
            mean_recall,
            mean_precision,
            color=color,
            linewidth=1.5,
            label=f"{name} (AP={mean_ap:.3f})",
        )
        ax.fill_between(
            mean_recall,
            np.clip(mean_precision - std_precision, 0, 1),
            np.clip(mean_precision + std_precision, 0, 1),
            alpha=0.12,
            color=color,
        )

        return mean_precision, std_precision, mean_ap

    def plot_pr(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        show_individual: bool = False,
    ) -> plt.Figure:
        """Plot one PR subplot per classifier in a grid (≤4 columns).

        Mirrors :meth:`plot_roc` but uses
        :func:`sklearn.metrics.precision_recall_curve` and
        :func:`sklearn.metrics.average_precision_score`. The horizontal
        baseline is the positive-class prevalence (``y_true.mean()``).

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override output filename. Defaults
                to ``"comparison_pr.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            show_individual (bool, optional): Draw per-seed curves as thin
                dashed lines. Defaults to False.

        Returns:
            plt.Figure: The matplotlib figure.
        """
        colors = self._color_cycle()
        mean_recall = np.linspace(0, 1, 200)

        y_true = np.asarray(self._metrics_ensemble.y_true)
        baseline = float(y_true.mean()) if len(y_true) else 0.0

        n = len(self._classifier_names)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        with apply_nature_style():
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 3.2, nrows * 3.2),
                squeeze=False,
            )

            for i, name in enumerate(self._classifier_names):
                ax = axes.flat[i]

                ax.hlines(
                    baseline,
                    0,
                    1,
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                )

                self._compute_classifier_pr(
                    name, mean_recall, colors[i], ax, show_individual
                )

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1.02)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(name)
                ax.legend(fontsize=6, loc="lower left")
                configure_spine(ax)

            for ax in axes.flat[n:]:
                ax.set_visible(False)

            fig.set_layout_engine("tight")

        if save:
            fname = filename or "comparison_pr.png"
            self._save_figure(fig, fname, dpi)

        plt.close(fig)
        return fig

    @staticmethod
    def _draw_metric_bars(
        ax: plt.Axes,
        clf_names: list[str],
        clf_colors: list[str],
        means: np.ndarray,
        stds: np.ndarray,
        metric: str,
    ) -> None:
        """Draw a bar chart of mean ± std onto an existing Axes."""
        bar_width = 0.6

        for ci, (_clf, clr) in enumerate(zip(clf_names, clf_colors, strict=True)):
            ax.bar(
                ci,
                means[ci],
                width=bar_width,
                yerr=stds[ci],
                capsize=3,
                color=clr,
                edgecolor="black",
                linewidth=0.5,
                error_kw={"elinewidth": 0.8, "ecolor": "black"},
            )
            ax.text(
                ci,
                means[ci] + stds[ci] * 0.05 + 0.005,
                f"{means[ci]:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )

        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_ylabel("Score")
        ax.set_title(metric.replace("_", " ").title(), fontsize=9)
        configure_spine(ax)

    def _build_combined_bar_figure(
        self,
        metrics_list: list[str],
        table: pd.DataFrame,
        clf_names: list[str],
        clf_colors: list[str],
        legend_handles: list[mpatches.Patch],
        save: bool,
        dpi: int,
    ) -> plt.Figure:
        """Build the combined overview bar figure for all metrics."""
        n_clf = len(clf_names)
        n_cols = min(4, len(metrics_list))
        n_rows = int(np.ceil(len(metrics_list) / n_cols))

        with apply_nature_style():
            fig_all, axes_all = plt.subplots(
                n_rows,
                n_cols,
                figsize=(n_cols * max(2.5, n_clf * 0.65), n_rows * 3.2),
                squeeze=False,
            )

            for i, m in enumerate(metrics_list):
                row, col = divmod(i, n_cols)
                ax = axes_all[row][col]
                mean_col = f"{m}_mean"
                std_col = f"{m}_std"
                means = table[mean_col].to_numpy()
                stds = (
                    table[std_col].to_numpy()
                    if std_col in table.columns
                    else np.zeros(len(means))
                )
                self._draw_metric_bars(ax, clf_names, clf_colors, means, stds, m)

            for j in range(len(metrics_list), n_rows * n_cols):
                row, col = divmod(j, n_cols)
                axes_all[row][col].set_visible(False)

        self._attach_legend(fig_all, legend_handles, len(clf_names), (0.5, -0.02))
        fig_all.set_layout_engine("tight")

        if save:
            self._save_figure(fig_all, "metric_bar_all.png", dpi)

        plt.close(fig_all)
        return fig_all

    def plot_metric_bar(
        self,
        metrics: str | list[str] | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> dict[str, plt.Figure]:
        """Plot one bar chart per metric, each showing mean ± std per classifier.

        Args:
            metrics (str | list[str] | None, optional): Metric or list of
                metrics to plot. Defaults to ``self.metrics``.
            save (bool, optional): Whether to save each figure to disk.
                Defaults to True.
            filename (str | None, optional): Override filename prefix. When
                set, used only for the first metric; remaining figures use the
                default ``metric_bar_{metric}.png`` naming.
            dpi (int, optional): Figure resolution in dots per inch. Defaults
                to 150.

        Returns:
            dict[str, plt.Figure]: Mapping of metric name to its figure.
                When more than one metric is requested, an additional combined
                overview figure is stored under the key ``"all"``.
        """
        metrics_list = self._resolve_metrics(metrics)

        table = self.get_metrics_table()
        self._validate_metrics(metrics_list, table)

        clf_names = list(table.index)
        clf_colors = self._color_cycle()
        n_clf = len(clf_names)

        legend_handles = self._build_legend_patches(clf_names, clf_colors)

        figures: dict[str, plt.Figure] = {}

        for idx, m in enumerate(metrics_list):
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"

            means = table[mean_col].to_numpy()
            stds = (
                table[std_col].to_numpy()
                if std_col in table.columns
                else np.zeros(len(means))
            )

            with apply_nature_style():
                fig, ax = plt.subplots(figsize=(max(3.5, n_clf * 0.7), 3.5))
                self._draw_metric_bars(ax, clf_names, clf_colors, means, stds, m)

            self._attach_legend(fig, legend_handles, len(clf_names), (0.5, -0.08))
            fig.set_layout_engine("tight")

            if save:
                fname = filename if idx == 0 and filename else f"metric_bar_{m}.png"
                self._save_figure(fig, fname, dpi)

            plt.close(fig)
            figures[m] = fig

        if len(metrics_list) > 1:
            figures["all"] = self._build_combined_bar_figure(
                metrics_list, table, clf_names, clf_colors, legend_handles, save, dpi
            )

        return figures

    @staticmethod
    def _draw_stability_violin(
        ax: plt.Axes,
        clf_names: list[str],
        clf_colors: list[str],
        all_records: dict[str, list[dict]],
        metric: str,
        rng: np.random.Generator,
    ) -> None:
        """Draw a violin + strip plot for one metric onto an existing Axes."""
        data_list = [
            [r[metric] for r in all_records[name] if metric in r] for name in clf_names
        ]
        positions = np.arange(1, len(clf_names) + 1)

        non_empty = [(d, p) for d, p in zip(data_list, positions, strict=True) if d]
        if not non_empty:
            ax.set_xticks([])
            ax.set_ylabel("Score")
            ax.set_title(metric.replace("_", " ").title(), fontsize=9)
            configure_spine(ax)
            return

        data_list_filt, positions_filt = zip(*non_empty, strict=True)
        data_list_filt = list(data_list_filt)
        positions_filt = np.array(positions_filt)

        parts: dict = ax.violinplot(
            data_list_filt,
            positions=positions_filt,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )

        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(clf_colors[i])
            pc.set_alpha(0.6)

        for i, vals in enumerate(data_list_filt):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                positions_filt[i] + jitter,
                vals,
                color="grey",
                s=12,
                alpha=0.7,
                zorder=3,
                linewidths=0,
            )

        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_ylabel("Score")
        ax.set_title(metric.replace("_", " ").title(), fontsize=9)
        configure_spine(ax)

    def _build_combined_stability_figure(
        self,
        metrics_list: list[str],
        clf_names: list[str],
        clf_colors: list[str],
        all_records: dict[str, list[dict]],
        legend_patches: list[mpatches.Patch],
        save: bool,
        dpi: int,
    ) -> plt.Figure:
        """Build the combined overview stability figure for all metrics."""
        n_cols = min(4, len(metrics_list))
        n_rows = int(np.ceil(len(metrics_list) / n_cols))
        rng_all = np.random.default_rng(42)

        with apply_nature_style():
            fig_all, axes_all = plt.subplots(
                n_rows,
                n_cols,
                figsize=(n_cols * max(2.5, len(clf_names) * 0.85), n_rows * 3.2),
                squeeze=False,
            )

            for i, m in enumerate(metrics_list):
                row, col = divmod(i, n_cols)
                self._draw_stability_violin(
                    axes_all[row][col], clf_names, clf_colors, all_records, m, rng_all
                )

            for j in range(len(metrics_list), n_rows * n_cols):
                row, col = divmod(j, n_cols)
                axes_all[row][col].set_visible(False)

        self._attach_legend(fig_all, legend_patches, len(clf_names), (0.5, -0.02))
        fig_all.set_layout_engine("tight")

        if save:
            self._save_figure(fig_all, "seed_stability_all.png", dpi)

        plt.close(fig_all)
        return fig_all

    def plot_seed_stability(
        self,
        metrics: str | list[str] | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> dict[str, plt.Figure]:
        """Plot per-seed metric distributions as violin + strip plots.

        Args:
            metrics (str | list[str] | None, optional): Metric or list of
                metrics to visualise. Defaults to ``self.metrics``.
            save (bool, optional): Whether to save the figures. Defaults to
                True.
            filename (str | None, optional): Override output filename applied
                to the first metric only. Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            dict[str, plt.Figure]: Mapping of metric name to its figure.
                When more than one metric is requested, an additional combined
                overview figure is stored under the key ``"all"``.
        """
        metrics_list = self._resolve_metrics(metrics)

        table = self.get_metrics_table()
        self._validate_metrics(metrics_list, table)

        clf_names = list(self._classifier_names)
        clf_colors = self._color_cycle()

        all_records = self._per_seed_records()

        legend_patches = self._build_legend_patches(clf_names, clf_colors)

        figures: dict[str, plt.Figure] = {}
        rng = np.random.default_rng(42)

        for idx, m in enumerate(metrics_list):
            with apply_nature_style():
                fig, ax = plt.subplots(figsize=(max(3.5, len(clf_names) * 0.9), 3.5))
                self._draw_stability_violin(
                    ax, clf_names, clf_colors, all_records, m, rng
                )

            self._attach_legend(fig, legend_patches, len(clf_names), (0.5, -0.08))
            fig.set_layout_engine("tight")

            if save:
                fname = filename if idx == 0 and filename else f"seed_stability_{m}.png"
                self._save_figure(fig, fname, dpi)

            plt.close(fig)
            figures[m] = fig

        if len(metrics_list) > 1:
            figures["all"] = self._build_combined_stability_figure(
                metrics_list,
                clf_names,
                clf_colors,
                all_records,
                legend_patches,
                save,
                dpi,
            )

        return figures

    def _draw_grid_cell(
        self,
        ax: plt.Axes,
        vals: list[float],
        color: str,
        row: int,
        col: int,
        metric: str,
        rng: np.random.Generator,
    ) -> None:
        """Draw a violin + strip plot for one (classifier, metric) grid cell."""
        clf_name = self._classifier_names[row]

        if vals:
            parts: dict = ax.violinplot(
                [vals],
                positions=[1],
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )

            parts["bodies"][0].set_facecolor(color)
            parts["bodies"][0].set_alpha(0.55)

            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                np.ones(len(vals)) + jitter,
                vals,
                color=color,
                s=6,
                alpha=0.6,
                zorder=3,
                linewidths=0,
            )

        if row == 0:
            ax.set_title(metric.replace("_", " ").title(), fontsize=8, pad=4)

        if col == 0:
            ax.set_ylabel(clf_name, fontsize=8)
        else:
            ax.set_ylabel("")

        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

        configure_spine(ax)

    def plot_comparison_grid(
        self,
        metrics: str | list[str] | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot a classifier × metric grid of per-seed distributions.

        Args:
            metrics (str | list[str] | None, optional): Metric or list of
                metrics to include as columns. Defaults to ``self.metrics``.
            save (bool, optional): Whether to save the figure. Defaults to
                True.
            filename (str | None, optional): Override output filename. Defaults
                to ``"comparison_grid.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure: The matplotlib figure.
        """
        metrics_list = self._resolve_metrics(metrics)

        self._validate_metrics(metrics_list, self.get_metrics_table())

        clf_names = list(self._classifier_names)
        clf_colors = self._color_cycle()
        n_clf = len(clf_names)
        n_met = len(metrics_list)

        all_records = self._per_seed_records()

        rng = np.random.default_rng(42)

        with apply_nature_style():
            fig, axes = plt.subplots(
                n_clf,
                n_met,
                figsize=(max(2.0, n_met * 2.2), max(4.0, n_clf * 1.8)),
                squeeze=False,
            )

            for row, (clf_name, color) in enumerate(
                zip(clf_names, clf_colors, strict=True)
            ):
                records = all_records[clf_name]

                for col, m in enumerate(metrics_list):
                    ax = axes[row][col]
                    vals = [r[m] for r in records if m in r]
                    self._draw_grid_cell(ax, vals, color, row, col, m, rng)

            fig.suptitle("Classifier × Metric Comparison", fontsize=9, y=1.01)
            fig.set_layout_engine("tight")

        if save:
            fname = filename or "comparison_grid.png"
            self._save_figure(fig, fname, dpi)

        plt.close(fig)
        return fig

    def plot_roc(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        show_individual: bool = False,
    ) -> plt.Figure:
        """Plot one ROC subplot per classifier in a grid (≤4 columns).

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override output filename. Defaults
                to ``"comparison_roc.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            show_individual (bool, optional): Draw per-seed curves as thin
                dashed lines. Defaults to False.

        Returns:
            plt.Figure: The matplotlib figure.
        """
        colors = self._color_cycle()
        mean_fpr = np.linspace(0, 1, 200)

        n = len(self._classifier_names)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        with apply_nature_style():
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 3.2, nrows * 3.2),
                squeeze=False,
            )

            for i, name in enumerate(self._classifier_names):
                ax = axes.flat[i]

                ax.plot(
                    [0, 1],
                    [0, 1],
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                )

                self._compute_classifier_roc(
                    name, mean_fpr, colors[i], ax, show_individual
                )

                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(name)
                ax.legend(fontsize=6, loc="lower right")
                configure_spine(ax)

            for ax in axes.flat[n:]:
                ax.set_visible(False)

            fig.set_layout_engine("tight")

        if save:
            fname = filename or "comparison_roc.png"
            self._save_figure(fig, fname, dpi)

        plt.close(fig)
        return fig

    def plot_all(
        self,
        dpi: int = 150,
        save: bool = True,
        parallel: bool = True,
    ) -> dict:
        """Run all plot methods and return the resulting figures.

        Args:
            dpi (int, optional): Figure resolution passed to each plot method.
                Defaults to 150.
            save (bool, optional): Whether sub-plot calls write their figures
                and CSVs to disk. Piped through to each plot method.
                Defaults to ``True``.
            parallel (bool, optional): Run the plot tasks concurrently via a
                :class:`ThreadPoolExecutor`. Set to ``False`` if matplotlib
                rcParams contention causes visual artefacts. Defaults to
                ``True``.

        Returns:
            dict: Dictionary with keys ``"metric_bar"``, ``"seed_stability"``
                (each a ``dict[str, plt.Figure]``), ``"comparison_grid"``,
                ``"roc"``, ``"pr"`` (each a ``plt.Figure``), and
                ``"ranking"`` (``pd.DataFrame``). Failed tasks store
                ``None`` for the corresponding key.
        """
        tasks = {
            "metric_bar": lambda: self.plot_metric_bar(dpi=dpi, save=save),
            "seed_stability": lambda: self.plot_seed_stability(dpi=dpi, save=save),
            "comparison_grid": lambda: self.plot_comparison_grid(dpi=dpi, save=save),
            "roc": lambda: self.plot_roc(dpi=dpi, save=save),
            "pr": lambda: self.plot_pr(dpi=dpi, save=save),
            "ranking": self.get_ranking,
        }

        if not parallel:
            results: dict = {}
            for name, fn in tasks.items():
                try:
                    results[name] = fn()
                except Exception as exc:
                    logger.warning(f"Plot {name!r} failed: {exc}")
                    results[name] = None
            return results

        results = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(fn): name for name, fn in tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.warning(f"Plot {name!r} failed: {exc}")
                    results[name] = None
        return results
