"""Cross-classifier comparison for volcanic eruption forecasting models.

This module provides ``ClassifierComparator``, which accepts a mapping of
classifier names to trained-model CSV paths, builds a ``MultiModelEvaluator``
per classifier, and produces side-by-side comparison plots and a ranking table.
"""

import os
import json
import math
from typing import Any, Self, cast

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, roc_auc_score

from eruption_forecast.logger import logger
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    configure_spine,
    apply_nature_style,
)
from eruption_forecast.utils.pathutils import ensure_dir
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator


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
    "sensitivity",
]


class ClassifierComparator:
    """Compare multiple classifiers side-by-side using aggregate seed metrics.

    Accepts a mapping of classifier names to trained-model registry CSV paths
    (produced by ``ModelTrainer.evaluate()``), constructs one
    ``MultiModelEvaluator`` per classifier, and exposes methods for generating
    comparison tables and plots.

    Attributes:
        output_dir (str): Root directory for saved outputs.
        figures_dir (str): Subdirectory for saved plot files.
        metrics_dir (str): Subdirectory for saved CSV files.
        metrics (list[str]): Ordered list of metrics to compare.

    Args:
        classifiers (dict[str, str]): Mapping of classifier name to the path
            of its trained-model registry CSV. Example:
            ``{"rf": "output/.../trained_model_rf.csv"}``.
        output_dir (str | None, optional): Root output directory. When None,
            defaults to ``{cwd}/output/comparison/``. An explicit path is used
            as-is without appending a subdirectory. Defaults to None.
        metrics (str | list[str] | None, optional): Metric or ordered list of
            metrics used for ranking and plots. The first entry is the default
            ranking metric. When None, uses ``DEFAULT_METRICS`` (f1_score,
            roc_auc, pr_auc, balanced_accuracy, precision, recall,
            specificity, sensitivity). Defaults to None.

    Raises:
        ValueError: If ``classifiers`` is empty.
        ValueError: If any metric name in ``metrics`` is not present in the
            aggregate metrics table.
        FileNotFoundError: If any CSV path in ``classifiers`` does not exist.

    Examples:
        >>> comparator = ClassifierComparator(
        ...     classifiers={"rf": "output/.../trained_model_rf.csv",
        ...                  "xgb": "output/.../trained_model_xgb.csv"},
        ...     metrics=["f1_score", "roc_auc"],
        ... )
        >>> df = comparator.get_ranking()
        >>> figs = comparator.plot_all()
    """

    def __init__(
        self,
        classifiers: dict[str, str],
        output_dir: str | None = None,
        metrics: str | list[str] | None = None,
    ) -> None:
        """Initialize the ClassifierComparator.

        Validates that the classifiers dict is non-empty and that every CSV
        path exists, then builds a MultiModelEvaluator for each classifier.
        When metrics is None, the full DEFAULT_METRICS list is used. Any metric
        name not present in the aggregate metrics table raises a ValueError.

        Args:
            classifiers (dict[str, str]): Mapping of classifier name to its
                trained-model registry CSV path.
            output_dir (str | None, optional): Root directory for outputs.
                When None, defaults to ``{cwd}/output/comparison/``. An
                explicit path is used as-is. Defaults to None.
            metrics (str | list[str] | None, optional): Metric or list of
                metrics for comparison. Single strings are wrapped in a list.
                The first entry is used as the default ranking metric. When
                None, defaults to ``DEFAULT_METRICS``. Defaults to None.

        Raises:
            ValueError: If ``classifiers`` is empty.
            ValueError: If a requested metric does not exist in the data.
            FileNotFoundError: If any CSV path does not exist on disk.
        """
        if not classifiers:
            raise ValueError("classifiers dict must not be empty.")

        for name, csv_path in classifiers.items():
            if not os.path.isfile(csv_path):
                raise FileNotFoundError(
                    f"Trained model CSV for '{name}' not found: {csv_path}"
                )

        if metrics is None:
            self.metrics: list[str] = list(DEFAULT_METRICS)
        elif isinstance(metrics, str):
            self.metrics = [metrics]
        else:
            self.metrics = list(metrics)

        output_dir = output_dir or os.path.join(os.getcwd(), "output")
        self.output_dir = os.path.join(output_dir, "comparison")

        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        self._metrics_table_cache: pd.DataFrame | None = None

        self._evaluators: dict[str, MultiModelEvaluator] = {}
        for name, csv_path in classifiers.items():
            # Metrics JSON files live in a ``metrics/`` sibling to the registry CSV.
            csv_dir = os.path.dirname(os.path.abspath(csv_path))
            clf_metrics_dir = os.path.join(csv_dir, "metrics")
            self._evaluators[name] = MultiModelEvaluator(
                metrics_dir=clf_metrics_dir if os.path.isdir(clf_metrics_dir) else None,
                trained_model_csv=csv_path,
            )

    @classmethod
    def from_json(
        cls,
        json_path: str,
        output_dir: str | None = None,
        metrics: str | list[str] | None = None,
    ) -> Self:
        """Create a ClassifierComparator from a JSON file of classifier paths.

        Loads a JSON file whose keys are classifier names and whose values are
        absolute paths to their trained-model registry CSVs, then constructs
        and returns a ``ClassifierComparator`` with those classifiers.

        The expected JSON format is:

        .. code-block:: json

            {
                "RandomForestClassifier": "/path/to/trained_model_rf.csv",
                "XGBClassifier":          "/path/to/trained_model_xgb.csv"
            }

        Args:
            json_path (str): Path to the JSON file.
            output_dir (str | None, optional): Root output directory passed to
                the constructor. Defaults to None.
            metrics (str | list[str] | None, optional): Metrics passed to the
                constructor. Defaults to None.

        Returns:
            ClassifierComparator: Initialised comparator loaded from the JSON.

        Raises:
            FileNotFoundError: If ``json_path`` does not exist.
            ValueError: If the JSON does not contain a non-empty object.

        Examples:
            >>> comparator = ClassifierComparator.from_json(
            ...     "output/VG.OJN.00.EHZ/evaluations_trained_models.json",
            ...     metrics=["f1_score", "roc_auc"],
            ... )
        """
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path) as f:
            classifiers = json.load(f)

        if not isinstance(classifiers, dict) or not classifiers:
            raise ValueError(
                f"JSON file must contain a non-empty object mapping classifier "
                f"names to CSV paths. Got: {type(classifiers).__name__}"
            )

        return cls(classifiers=classifiers, output_dir=output_dir, metrics=metrics)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_figure(self, fig: plt.Figure, filename: str, dpi: int) -> None:
        """Save a figure to the figures subdirectory.

        Creates the figures directory if it does not already exist and writes
        the figure as a PNG at the requested resolution.

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
        """Return a list of OKABE_ITO colors, cycling if needed.

        Returns:
            list[str]: List of hex color strings with length equal to the
                number of classifiers.
        """
        n = len(self._evaluators)
        colors = list(OKABE_ITO)
        return [colors[i % len(colors)] for i in range(n)]

    def _validate_metrics(self, metrics_list: list[str], table: pd.DataFrame) -> None:
        """Raise ValueError for any metric not present in the metrics table.

        Checks that ``{metric}_mean`` exists as a column in ``table`` for
        every entry in ``metrics_list``.  This catches typos or unsupported
        metric names early, before any plotting begins.

        Args:
            metrics_list (list[str]): Metric names to validate.
            table (pd.DataFrame): Aggregate metrics table produced by
                ``get_metrics_table()``.

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
        """Normalise a metrics argument to a list.

        Converts ``None`` to ``self.metrics``, a bare string to a one-element
        list, and passes any existing list through unchanged.

        Args:
            metrics (str | list[str] | None): Raw metrics argument.

        Returns:
            list[str]: Resolved list of metric names.
        """
        if metrics is None:
            return list(self.metrics)
        if isinstance(metrics, str):
            return [metrics]
        return list(metrics)

    def _build_legend_patches(
        self,
        clf_names: list[str],
        clf_colors: list[str],
    ) -> list[mpatches.Patch]:
        """Build a list of legend patch handles for the classifier colour palette.

        Produces one ``mpatches.Patch`` per classifier, used as handles in
        ``fig.legend()`` calls so that classifier identity can be read from
        the legend rather than x-axis tick labels.

        Args:
            clf_names (list[str]): Ordered classifier names.
            clf_colors (list[str]): Ordered hex colour strings aligned with
                ``clf_names``.

        Returns:
            list[mpatches.Patch]: One patch per classifier in the same order
                as ``clf_names``.
        """
        return [
            mpatches.Patch(facecolor=clf_colors[i], label=name)
            for i, name in enumerate(clf_names)
        ]

    def _attach_legend(
        self,
        fig: plt.Figure,
        handles: list[mpatches.Patch],
        ncol: int,
        bbox_to_anchor: tuple[float, float],
    ) -> None:
        """Attach a classifier legend to the bottom of a figure.

        Places ``fig.legend()`` at the lower centre of the figure using a
        consistent style: no frame, 7 pt font, horizontally laid out patches.

        Args:
            fig (plt.Figure): Figure to attach the legend to.
            handles (list[mpatches.Patch]): Legend patch handles.
            ncol (int): Number of columns in the legend layout.
            bbox_to_anchor (tuple[float, float]): ``(x, y)`` anchor for the
                legend in figure coordinates.
        """
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=ncol,
            fontsize=7,
            frameon=False,
            bbox_to_anchor=bbox_to_anchor,
        )

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
        """Build the combined overview bar figure for all metrics.

        Lays out all metrics in a grid of subplots (max 4 columns), calls
        ``_draw_metric_bars`` for each cell, hides any unused axes in the
        last row, attaches the classifier legend at the bottom, and optionally
        saves the figure as ``metric_bar_all.png``.

        Args:
            metrics_list (list[str]): Ordered list of metric names to include.
            table (pd.DataFrame): Aggregate metrics table from
                ``get_metrics_table()``.
            clf_names (list[str]): Ordered classifier names.
            clf_colors (list[str]): Ordered hex colour strings aligned with
                ``clf_names``.
            legend_handles (list[mpatches.Patch]): Pre-built legend patches.
            save (bool): Whether to save the figure to disk.
            dpi (int): Resolution in dots per inch.

        Returns:
            plt.Figure: The combined overview figure.
        """
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

            # Hide any unused axes in the last row.
            for j in range(len(metrics_list), n_rows * n_cols):
                row, col = divmod(j, n_cols)
                axes_all[row][col].set_visible(False)

        self._attach_legend(fig_all, legend_handles, len(clf_names), (0.5, -0.02))
        fig_all.set_layout_engine("tight")

        if save:
            self._save_figure(fig_all, "metric_bar_all.png", dpi)

        plt.close(fig_all)
        return fig_all

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
        """Build the combined overview stability figure for all metrics.

        Lays out all metrics in a grid of subplots (max 4 columns), calls
        ``_draw_stability_violin`` for each cell, hides any unused axes, attaches
        the classifier legend at the bottom, and optionally saves the figure as
        ``seed_stability_all.png``.

        Args:
            metrics_list (list[str]): Ordered list of metric names to include.
            clf_names (list[str]): Ordered classifier names.
            clf_colors (list[str]): Ordered hex colour strings aligned with
                ``clf_names``.
            all_records (dict[str, list[dict]]): Per-seed metric dicts keyed by
                classifier name.
            legend_patches (list[mpatches.Patch]): Pre-built legend patches.
            save (bool): Whether to save the figure to disk.
            dpi (int): Resolution in dots per inch.

        Returns:
            plt.Figure: The combined overview figure.
        """
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

    def _compute_classifier_roc(
        self,
        evaluator: MultiModelEvaluator,
        name: str,
        mean_fpr: np.ndarray,
        color: str,
        ax: plt.Axes,
        show_individual: bool,
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Compute and draw the mean ROC curve for one classifier.

        Iterates over every seed in the classifier's registry, computes per-seed
        ROC data, optionally draws individual seed curves as thin dashed lines,
        then plots the interpolated mean curve with a shaded std-deviation band.
        Returns None if no valid seed data was found.

        Args:
            evaluator (MultiModelEvaluator): Evaluator for the classifier.
            name (str): Classifier name used for the legend label and warnings.
            mean_fpr (np.ndarray): Common FPR grid onto which TPR values are
                interpolated before averaging.
            color (str): Hex colour string for this classifier's curves.
            ax (plt.Axes): Axes to draw on.
            show_individual (bool): Whether to draw per-seed curves as thin
                dashed lines.

        Returns:
            tuple[np.ndarray, np.ndarray, float] | None: A
                ``(mean_tpr, std_tpr, mean_auc)`` triple if at least one seed
                was successfully processed, or ``None`` if the registry had no
                usable data.
        """
        registry = evaluator._require_registry()
        tprs: list[np.ndarray] = []
        aucs: list[float] = []

        for _, row in registry.iterrows():
            try:
                _, _X_test, y_true, y_proba = evaluator._load_seed_data(row)
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc_val = roc_auc_score(y_true, y_proba)
                aucs.append(auc_val)
                tprs.append(np.interp(mean_fpr, fpr, tpr))

                if show_individual:
                    ax.plot(
                        fpr,
                        tpr,
                        color=color,
                        alpha=0.15,
                        linewidth=0.6,
                        linestyle="--",
                    )
            except (OSError, ValueError, KeyError):
                logger.warning(f"Skipped a seed for '{name}' in plot_roc.")

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
        """Draw a violin + strip plot for one (classifier, metric) grid cell.

        Renders a violin body with a jittered scatter overlay when ``vals`` is
        non-empty.  Sets the column header title on the first row and the
        classifier name as y-axis label on the first column.  Hides x-axis ticks
        and configures spines via ``configure_spine``.

        Args:
            ax (plt.Axes): Axes for this grid cell.
            vals (list[float]): Per-seed metric values for the cell.
            color (str): Hex colour string for violin and scatter points.
            row (int): Zero-based row index in the grid (used for title
                placement on the first row).
            col (int): Zero-based column index in the grid (used for y-label
                placement on the first column).
            metric (str): Metric name; used as column header on row 0.
            rng (np.random.Generator): Random number generator for jitter.
        """
        clf_name = list(self._evaluators.keys())[row]

        if vals:
            parts = cast(
                dict[str, Any],
                ax.violinplot(
                    [vals],
                    positions=[1],
                    showmeans=False,
                    showmedians=True,
                    showextrema=True,
                ),
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

        # Column header: metric name on first row only.
        if row == 0:
            ax.set_title(metric.replace("_", " ").title(), fontsize=8, pad=4)

        # Row header: classifier name as y-label on first column only.
        if col == 0:
            ax.set_ylabel(clf_name, fontsize=8)
        else:
            ax.set_ylabel("")

        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

        configure_spine(ax)

    # ------------------------------------------------------------------
    # Metrics table and ranking
    # ------------------------------------------------------------------

    def get_metrics_table(self) -> pd.DataFrame:
        """Return a DataFrame of aggregate metrics for every classifier.

        Calls ``MultiModelEvaluator.get_aggregate_metrics()`` for each
        classifier and stacks the results into a single wide DataFrame where
        each row represents one classifier and columns are flattened
        ``{metric}_{stat}`` pairs (e.g. ``f1_score_mean``, ``f1_score_std``).

        Returns:
            pd.DataFrame: DataFrame indexed by classifier name with columns
                for each ``{metric}_{stat}`` combination.

        Raises:
            ValueError: If any evaluator cannot load its metrics files.

        Examples:
            >>> df = comparator.get_metrics_table()
            >>> print(df[["f1_score_mean", "f1_score_std"]])
        """
        if self._metrics_table_cache is not None:
            return self._metrics_table_cache
        rows: dict[str, dict[str, float]] = {}
        for name, evaluator in self._evaluators.items():
            agg = evaluator.get_aggregate_metrics()
            flat: dict[str, float] = {}
            for metric_name in agg.index:
                for stat in agg.columns:
                    flat[f"{metric_name}_{stat}"] = agg.loc[metric_name, stat]
            rows[name] = flat

        self._metrics_table_cache = pd.DataFrame.from_dict(rows, orient="index")
        return self._metrics_table_cache

    def get_ranking(
        self,
        metric: str = "recall",
        by: str = "mean",
    ) -> pd.DataFrame:
        """Return classifiers ranked by a chosen metric, descending.

        Builds the full metrics table, sorts by ``{metric}_{by}`` descending,
        inserts a ``rank`` column, and saves the result to
        ``{metrics_dir}/ranking_{metric}.csv``.

        Args:
            metric (str, optional): Metric to rank by. Defaults to
                ``"recall"``.
            by (str, optional): Aggregation statistic to sort on — one of
                ``"mean"``, ``"std"``, ``"min"``, or ``"max"``. Defaults to
                ``"mean"``.

        Returns:
            pd.DataFrame: Ranked DataFrame with a ``rank`` column prepended.

        Raises:
            KeyError: If the requested ``{metric}_{by}`` column does not exist
                in the metrics table.

        Examples:
            >>> df = comparator.get_ranking()
            >>> df = comparator.get_ranking(metric="roc_auc", by="mean")
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

    # ------------------------------------------------------------------
    # Comparison plots
    # ------------------------------------------------------------------

    def plot_metric_bar(
        self,
        metrics: str | list[str] | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> dict[str, plt.Figure]:
        """Plot one bar chart per metric, each showing mean ± std per classifier.

        Produces a separate figure for every metric in ``metrics``.  Within
        each figure, one bar per classifier is drawn, coloured by classifier
        using the OKABE_ITO palette.  The classifier legend is placed outside
        the figure at the bottom.  Each figure is saved as
        ``metric_bar_{metric}.png``.  When more than one metric is requested, an
        additional combined overview figure (max 4 columns) is produced and
        stored under the key ``"all"``; saved as ``metric_bar_all.png``.

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

        Examples:
            >>> figs = comparator.plot_metric_bar(metrics="f1_score")
            >>> figs = comparator.plot_metric_bar(metrics=["f1_score", "roc_auc"])
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

        # Combined overview figure — all metrics in a single subplot grid.
        if len(metrics_list) > 1:
            figures["all"] = self._build_combined_bar_figure(
                metrics_list, table, clf_names, clf_colors, legend_handles, save, dpi
            )

        return figures

    def _draw_metric_bars(
        self,
        ax: plt.Axes,
        clf_names: list[str],
        clf_colors: list[str],
        means: np.ndarray,
        stds: np.ndarray,
        metric: str,
    ) -> None:
        """Draw a bar chart of mean ± std values onto an existing Axes.

        Places one bar per classifier with error bars and a value annotation
        above each bar. x-axis ticks are hidden; classifier identity is
        conveyed via the caller's legend.

        Args:
            ax (plt.Axes): Axes to draw on.
            clf_names (list[str]): Ordered classifier names.
            clf_colors (list[str]): Ordered hex colour strings aligned with
                ``clf_names``.
            means (np.ndarray): Mean metric value per classifier.
            stds (np.ndarray): Standard deviation per classifier.
            metric (str): Metric name used as the subplot title.
        """
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
                fontsize=7,
            )

        ax.set_xticks([])
        ax.set_ylabel("Score")
        ax.set_title(metric.replace("_", " ").title(), fontsize=9)
        configure_spine(ax)

    def plot_seed_stability(
        self,
        metrics: str | list[str] | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> dict[str, plt.Figure]:
        """Plot per-seed metric distributions as violin + strip plots.

        Collects raw per-seed metric values from each classifier's
        ``MultiModelEvaluator.get_metrics_list()`` and produces one figure per
        metric, each showing a violin + strip plot per classifier.  Classifier
        identity is shown via a legend of coloured patches rather than x-axis
        tick labels.  When more than one metric is requested, an additional
        combined overview figure (max 4 columns) is produced and stored under
        the key ``"all"``; saved as ``seed_stability_all.png``.

        Args:
            metrics (str | list[str] | None, optional): Metric or list of
                metrics to visualise. Defaults to ``self.metrics``.
            save (bool, optional): Whether to save the figures. Defaults to
                True.
            filename (str | None, optional): Override output filename applied
                to the first metric only. Subsequent metrics always use the
                default ``"seed_stability_{metric}.png"`` pattern. Defaults to
                None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            dict[str, plt.Figure]: Mapping of metric name to its figure.

        Examples:
            >>> figs = comparator.plot_seed_stability(metrics="roc_auc")
            >>> figs = comparator.plot_seed_stability(metrics=["f1_score", "roc_auc"])
        """
        metrics_list = self._resolve_metrics(metrics)

        table = self.get_metrics_table()
        self._validate_metrics(metrics_list, table)

        clf_names = list(self._evaluators.keys())
        clf_colors = self._color_cycle()

        # Pre-load per-seed records once for all classifiers.
        all_records: dict[str, list[dict]] = {
            name: evaluator.get_metrics_list()
            for name, evaluator in self._evaluators.items()
        }

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

        # Combined overview figure — all metrics in a single subplot grid.
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

    def _draw_stability_violin(
        self,
        ax: plt.Axes,
        clf_names: list[str],
        clf_colors: list[str],
        all_records: dict[str, list[dict]],
        metric: str,
        rng: np.random.Generator,
    ) -> None:
        """Draw a violin + strip plot for one metric onto an existing Axes.

        Renders one violin per classifier with a jittered strip overlay.
        x-axis ticks are hidden; classifier identity is conveyed via the
        caller's legend.

        Args:
            ax (plt.Axes): Axes to draw on.
            clf_names (list[str]): Ordered classifier names.
            clf_colors (list[str]): Ordered hex colour strings aligned with
                ``clf_names``.
            all_records (dict[str, list[dict]]): Per-seed metric dicts keyed
                by classifier name.
            metric (str): Metric key to extract from each record dict.
            rng (np.random.Generator): Random number generator for jitter.
        """
        data_list = [
            [r[metric] for r in all_records[name] if metric in r] for name in clf_names
        ]
        positions = np.arange(1, len(clf_names) + 1)

        # Filter out classifiers with no seed data for this metric.
        non_empty = [(d, p) for d, p in zip(data_list, positions, strict=True) if d]
        if not non_empty:
            ax.set_xticks([])
            ax.set_ylabel("Score")
            ax.set_title(metric.replace("_", " ").title(), fontsize=9)
            configure_spine(ax)
            return
        data_list, positions = zip(*non_empty, strict=True)
        data_list = list(data_list)
        positions = np.array(positions)

        parts = ax.violinplot(
            data_list,
            positions=positions,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )

        for i, pc in enumerate(parts["bodies"]):  # ty:ignore[invalid-argument-type]
            pc.set_facecolor(clf_colors[i])
            pc.set_alpha(0.6)

        for i, vals in enumerate(data_list):
            jitter = rng.uniform(-0.08, 0.08, size=len(vals))
            ax.scatter(
                positions[i] + jitter,
                vals,
                color=clf_colors[i],
                s=12,
                alpha=0.7,
                zorder=3,
                linewidths=0,
            )

        ax.set_xticks([])
        ax.set_ylabel("Score")
        ax.set_title(metric.replace("_", " ").title(), fontsize=9)
        configure_spine(ax)

    def plot_roc(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        show_individual: bool = False,
    ) -> plt.Figure:
        """Plot one ROC subplot per classifier in a grid (≤4 columns).

        Loads per-seed ROC data via each ``MultiModelEvaluator._load_seed_data()``,
        interpolates to a common FPR grid, and plots mean AUC curves. A gray
        diagonal chance line is drawn on every subplot. The grid has at most
        four columns; unused cells are hidden.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override output filename. Defaults
                to ``"comparison_roc.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            show_individual (bool, optional): Draw per-seed curves as thin
                dashed lines. Defaults to False.

        Returns:
            plt.Figure: The matplotlib figure.

        Examples:
            >>> fig = comparator.plot_roc(show_individual=True)
        """
        colors = self._color_cycle()
        mean_fpr = np.linspace(0, 1, 200)

        n = len(self._evaluators)
        ncols = min(4, n)
        nrows = math.ceil(n / ncols)

        with apply_nature_style():
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(ncols * 3.2, nrows * 3.2),
                squeeze=False,
            )

            for i, (name, evaluator) in enumerate(self._evaluators.items()):
                ax = axes.flat[i]

                # Diagonal chance line
                ax.plot(
                    [0, 1],
                    [0, 1],
                    color="gray",
                    linestyle="--",
                    linewidth=0.8,
                    label="Chance",
                )

                self._compute_classifier_roc(
                    evaluator, name, mean_fpr, colors[i], ax, show_individual
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

    def plot_comparison_grid(
        self,
        metrics: str | list[str] | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot a classifier × metric grid of per-seed distributions.

        Creates a figure where each row corresponds to one classifier and each
        column corresponds to one metric.  Every cell shows the per-seed
        distribution for that (classifier, metric) combination as a violin plot
        overlaid with jittered strip points, so both the shape and individual
        seed values are visible.

        Column headers (metric names) are shown as titles on the top row only.
        Row headers (classifier names) are shown as y-axis labels on the
        leftmost column only.  All other axes labels are hidden to reduce
        clutter.

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

        Examples:
            >>> fig = comparator.plot_comparison_grid()
            >>> fig = comparator.plot_comparison_grid(metrics=["f1_score", "roc_auc"])
        """
        metrics_list = self._resolve_metrics(metrics)

        self._validate_metrics(metrics_list, self.get_metrics_table())

        clf_names = list(self._evaluators.keys())
        clf_colors = self._color_cycle()
        n_clf = len(clf_names)
        n_met = len(metrics_list)

        # Pre-load all per-seed records once.
        all_records: dict[str, list[dict]] = {
            name: evaluator.get_metrics_list()
            for name, evaluator in self._evaluators.items()
        }

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

    def plot_all(self, dpi: int = 150) -> dict[str, Any]:
        """Run all plot methods and return the resulting figures.

        Calls ``plot_metric_bar()``, ``plot_seed_stability()``,
        ``plot_comparison_grid()``, ``plot_roc()``, and ``get_ranking()`` in
        sequence and bundles their outputs into a dict.  All configured metrics
        are passed to the bar, stability, and grid plots.

        Args:
            dpi (int, optional): Figure resolution passed to each plot method.
                Defaults to 150.

        Returns:
            dict[str, Any]: Dictionary with keys ``"metric_bar"``
                (dict[str, plt.Figure] — one figure per metric),
                ``"comparison_grid"``, ``"roc"`` (plt.Figure),
                ``"seed_stability"`` (dict[str, plt.Figure] — one figure per
                metric), and ``"ranking"`` (pd.DataFrame ranked by recall).

        Examples:
            >>> results = comparator.plot_all()
            >>> results["ranking"].head()
        """
        return {
            "metric_bar": self.plot_metric_bar(dpi=dpi),
            "seed_stability": self.plot_seed_stability(dpi=dpi),
            "comparison_grid": self.plot_comparison_grid(dpi=dpi),
            "roc": self.plot_roc(dpi=dpi),
            "ranking": self.get_ranking(),
        }
