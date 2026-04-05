"""Multi-seed aggregate model evaluation for volcanic eruption forecasting.

This module provides ``MultiModelEvaluator``, which aggregates per-seed
evaluation results — either from JSON metrics files or from a model registry
CSV — and generates ensemble-level statistics and plots.
"""

import os
import glob
import json
import functools
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import shap
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

from eruption_forecast.logger import logger
from eruption_forecast.utils.array import predict_proba_from_estimator
from eruption_forecast.utils.pathutils import (
    save_data,
    ensure_dir,
    save_figure,
    resolve_output_dir,
)
from eruption_forecast.plots.shap_plots import (
    compute_shap_explanation,
    plot_aggregate_shap_summary,
    plot_aggregate_shap_waterfall,
)
from eruption_forecast.plots.feature_plots import plot_frequency_band_contribution
from eruption_forecast.plots.evaluation_plots import (
    plot_seed_stability,
    plot_aggregate_roc_curve as _plot_roc_styled,
    plot_aggregate_calibration as _plot_cal_styled,
    plot_aggregate_g_mean_curve as _plot_gmean_styled,
    plot_aggregate_learning_curve as _plot_agg_lc_styled,
    plot_aggregate_confusion_matrix as _plot_cm_styled,
    plot_aggregate_feature_importance as _plot_fi_styled,
    plot_aggregate_threshold_analysis as _plot_threshold_styled,
    plot_aggregate_precision_recall_curve as _plot_pr_styled,
    plot_aggregate_prediction_distribution as _plot_pred_dist_styled,
)


class MultiModelEvaluator:
    """Aggregate evaluation across multiple seeds of a trained classifier.

    Accepts per-seed JSON metrics files (written by
    ``ModelEvaluator.save_metrics()``) and/or a model registry CSV (written
    by ``ModelTrainer.evaluate()``) to compute ensemble-level
    statistics and generate aggregate plots.

    Attributes:
        output_dir (str): Directory where figures and CSVs are saved.

    Args:
        metrics_dir (str | None, optional): Directory containing per-seed
            ``*.json`` metrics files. Globs for all JSON files in the
            directory. Defaults to None.
        metrics_files (list[str] | None, optional): Explicit list of paths to
            per-seed JSON metrics files. Takes precedence over
            ``metrics_dir``. Defaults to None.
        trained_model_csv (str | None, optional): Path to the model registry CSV
            produced by ``ModelTrainer.evaluate()``. Required for
            plot methods. Defaults to None.
        root_dir (str | None, optional): Root directory used to anchor
            output_dir resolution. Defaults to None.
        output_dir (str | None, optional): Directory for saved figures and
            CSVs. If None, resolved in priority order: (1) when
            ``trained_model_csv`` is given →
            ``<trained_model_csv_dir>/figures/`` (e.g.
            ``output/{station_dir}/trainings/evaluations/{classifier}/{cv}/figures/``);
            (2) when ``metrics_dir`` is given →
            ``<parent_of_metrics_dir>/figures/``; (3) fallback →
            ``<cwd>/output/evaluation/figures/``. Defaults to None.

    Raises:
        ValueError: If none of ``metrics_dir``, ``metrics_files``, or
            ``trained_model_csv`` is provided.

    Examples:
        >>> # Aggregate from JSON metrics files
        >>> evaluator = MultiModelEvaluator(metrics_dir="output/eval/")
        >>> df = evaluator.get_aggregate_metrics()
        >>> evaluator.save_aggregate_metrics()

        >>> # Aggregate plots from registry CSV
        >>> evaluator = MultiModelEvaluator(trained_model_csv="output/.../trained_model_registry.csv")
        >>> figs = evaluator.plot_all()
    """

    def __init__(
        self,
        metrics_dir: str | None = None,
        metrics_files: list[str] | None = None,
        trained_model_csv: str | None = None,
        classifier_name: str = "model",
        plot_shap: bool = False,
        root_dir: str | None = None,
        output_dir: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the MultiModelEvaluator with one or more metrics sources.

        Exactly one of metrics_dir, metrics_files, or trained_model_csv must be
        provided. The output directory is inferred from the supplied source when
        output_dir is not given explicitly.

        Args:
            metrics_dir (str | None, optional): Directory containing per-seed JSON
                metrics files (*.json). Defaults to None.
            metrics_files (list[str] | None, optional): Explicit list of paths to
                per-seed JSON metrics files. Defaults to None.
            trained_model_csv (str | None, optional): Path to a trained model registry
                CSV produced by ModelTrainer. Used for aggregate plot generation.
                Defaults to None.
            classifier_name (str): Classifier name used as prefix filename.
                Defaults to "model".
            plot_shap (bool, optional): If ``True``, generate aggregate SHAP
                summary plots. Defaults to ``False``.
            root_dir (str | None, optional): Root directory used to anchor output_dir
                resolution. See class-level docs for resolution priority. Defaults to None.
            output_dir (str | None, optional): Directory for saving evaluation outputs.
                When None, resolved from the provided source path. Defaults to None.
            verbose (bool, optional): Emit progress log messages. Defaults to False.

        Raises:
            ValueError: If none of metrics_dir, metrics_files, or trained_model_csv
                is provided.
        """
        if metrics_dir is None and metrics_files is None and trained_model_csv is None:
            raise ValueError(
                "At least one of metrics_dir, metrics_files, or trained_model_csv must be provided."
            )

        self.classifier_name = classifier_name
        self.plot_shap = plot_shap
        self._metrics_dir = metrics_dir
        self._metrics_files = metrics_files
        self._trained_model_csv = trained_model_csv

        output_dir = resolve_output_dir(
            output_dir, root_dir, os.path.join("output", "evaluations")
        )
        self.output_dir = os.path.join(output_dir, "aggregate")
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_metrics_files(self) -> list[str]:
        """Return the resolved list of per-seed JSON metrics file paths.

        Prefers explicit ``metrics_files`` when provided; otherwise globs
        the ``metrics_dir`` for ``*.json`` files.

        Returns:
            list[str]: Sorted list of JSON file paths.

        Raises:
            ValueError: If neither ``metrics_files`` nor ``metrics_dir`` was
                provided at construction time.
        """
        if self._metrics_files is not None:
            return list(self._metrics_files)
        if self._metrics_dir is not None:
            paths = sorted(glob.glob(os.path.join(self._metrics_dir, "*.json")))
            return paths
        raise ValueError("No metrics source: provide metrics_files or metrics_dir.")

    @staticmethod
    def _load_seed_data(
        row: pd.Series,
    ) -> tuple[BaseEstimator, pd.DataFrame, np.ndarray, np.ndarray]:
        """Load model, filtered test features, true labels, and probabilities for one seed.

        Reads the trained model, X_test, y_test, and significant-features
        list from paths stored in a registry row, then runs ``predict_proba``
        (or ``decision_function`` as fallback) to produce probability estimates.

        Args:
            row (pd.Series): A single row from the model registry DataFrame
                with at least the keys ``trained_model_filepath``,
                ``X_test_filepath``, ``y_test_filepath``, and
                ``significant_features_csv``.

        Returns:
            tuple[BaseEstimator, pd.DataFrame, np.ndarray, np.ndarray]: A 4-tuple of
                ``(model, X_test_filtered, y_true, y_proba)`` where
                ``X_test_filtered`` is a DataFrame containing only the
                significant features selected for this seed.
        """
        model = joblib.load(row["trained_model_filepath"])
        X_test = pd.read_csv(row["X_test_filepath"], index_col=0)
        y_true = pd.read_csv(row["y_test_filepath"], index_col=0).iloc[:, 0].to_numpy()

        sig_features = pd.read_csv(
            row["significant_features_csv"], index_col=0
        ).index.tolist()

        # Checking if X_test_filtered columns has all significant features
        try:
            X_test_filtered = X_test[sig_features]
        except KeyError as e:
            raise KeyError(
                f"{e}. significant_features_csv: {row['significant_features_csv']}"
            )

        proba, _, _ = predict_proba_from_estimator(model, X_test_filtered)

        return model, X_test_filtered, y_true, proba

    def _require_registry(self) -> pd.DataFrame:
        """Load and return the model registry CSV, raising if unavailable.

        Returns:
            pd.DataFrame: Registry DataFrame indexed by seed.

        Raises:
            ValueError: If ``trained_model_csv`` was not provided at construction.
        """
        if self._trained_model_csv is None:
            raise ValueError(
                "trained_model_csv is required for plot methods. "
                "Provide it when constructing MultiModelEvaluator."
            )

        df = pd.read_csv(self._trained_model_csv, index_col=0)

        if df.empty:
            raise ValueError("Registry CSV is empty; no seeds to process.")

        return df

    def _save_outputs(
        self,
        fig: plt.Figure,
        data: pd.DataFrame | shap.Explanation | None,
        fig_filename: str,
        data_filename: str | None,
        dpi: int,
        save: bool,
    ) -> None:
        """Save a figure and optional data artifact to the output directory.

        Delegates figure saving to
        :func:`~eruption_forecast.utils.pathutils.save_figure` and data
        saving to :func:`~eruption_forecast.utils.pathutils.save_data`.
        The figure is always closed after a save attempt. Pass filenames
        WITHOUT a file extension; extensions are appended automatically
        (``".png"`` for figures, ``".csv"`` for DataFrames, ``".pkl"`` for
        ``shap.Explanation`` objects).

        Args:
            fig (plt.Figure): The figure to save and close.
            data (pd.DataFrame | shap.Explanation | None): Aggregate data
                to persist. Skipped when None or when ``data_filename``
                is None.
            fig_filename (str): Figure basename WITHOUT extension, joined
                with ``self.output_dir``.
            data_filename (str | None): Data basename WITHOUT extension.
                If None, no data file is written.
            dpi (int): Figure resolution in dots per inch.
            save (bool): When False, nothing is written to disk.

        Returns:
            None
        """
        # Strip any caller-provided extension so save_figure/save_data can
        # append the correct one without doubling it.
        fig_path = os.path.join(self.output_dir, os.path.splitext(fig_filename)[0])
        if save:
            save_figure(fig, fig_path, dpi)
        else:
            plt.close(fig)

        if save and data is not None and data_filename is not None:
            data_path = os.path.join(
                self.output_dir, os.path.splitext(data_filename)[0]
            )
            filetype = "pkl" if isinstance(data, shap.Explanation) else "csv"
            save_data(data, data_path, filetype=filetype)

    @functools.cached_property
    def _predictions(
        self,
    ) -> tuple[
        list[BaseEstimator], list[pd.DataFrame], list[np.ndarray], list[np.ndarray]
    ]:
        """Load and cache models, X_tests, y_trues, and y_probas from all registry seeds.

        Iterates over every row in the model registry, loads the fitted model,
        filtered test features, true labels, and predicted probabilities via
        ``_load_seed_data``, and returns them as four parallel lists.

        The result is cached on the instance after the first call so that
        subsequent plot methods do not repeat the expensive disk I/O.

        Returns:
            tuple[list[BaseEstimator], list[pd.DataFrame], list[np.ndarray], list[np.ndarray]]:
                A 4-tuple of ``(models, x_tests, y_trues, y_probas)`` where each
                list has one entry per registry seed.
        """
        registry = self._require_registry()
        models: list[BaseEstimator] = []
        x_tests: list[pd.DataFrame] = []
        y_trues: list[np.ndarray] = []
        y_probas: list[np.ndarray] = []
        for _, row in registry.iterrows():
            model, X_test, y_true, y_proba = self._load_seed_data(row)
            models.append(model)
            x_tests.append(X_test)
            y_trues.append(y_true)
            y_probas.append(y_proba)
        return models, x_tests, y_trues, y_probas

    # ------------------------------------------------------------------
    # Aggregate metrics (from JSON files)
    # ------------------------------------------------------------------

    def get_aggregate_metrics(self) -> pd.DataFrame:
        """Compute summary statistics across all per-seed metrics JSON files.

        Loads every JSON file resolved from ``metrics_files`` or
        ``metrics_dir``, stacks the records into a DataFrame (one row per
        seed), and computes mean, std, min, and max for each numeric metric.

        Returns:
            pd.DataFrame: Summary DataFrame with the metric name as the index
            and columns ``mean``, ``std``, ``min``, ``max``. Non-numeric
            fields (e.g. ``model_name``) are excluded.

        Raises:
            ValueError: If no metrics files are found or resolvable.

        Examples:
            >>> df = evaluator.get_aggregate_metrics()
            >>> print(df.loc["f1_score"])
        """
        if self.verbose:
            logger.info(f"Get classifier aggregate metrics for {self.classifier_name}")

        paths = self._resolve_metrics_files()
        if not paths:
            raise ValueError("No JSON metrics files found.")

        records: list[dict[str, Any]] = []
        for p in paths:
            with open(p) as f:
                records.append(json.load(f))

        df = pd.DataFrame(records)
        numeric = df.select_dtypes(include="number")
        summary = pd.DataFrame(
            {
                "mean": numeric.mean(),
                "std": numeric.std(),
                "min": numeric.min(),
                "max": numeric.max(),
            }
        )
        return summary

    def save_aggregate_metrics(
        self,
        filename: str = "aggregate_metrics.csv",
    ) -> str:
        """Save aggregate metric summary to a CSV file.

        Calls ``get_aggregate_metrics()`` and writes the result to
        ``{output_dir}/{filename}``.

        Args:
            filename (str, optional): Output CSV filename. Defaults to
                ``"aggregate_metrics.csv"``.

        Returns:
            str: Absolute path to the saved CSV file.

        Examples:
            >>> path = evaluator.save_aggregate_metrics()
            >>> path = evaluator.save_aggregate_metrics("my_summary.csv")
        """
        ensure_dir(self.output_dir)
        df = self.get_aggregate_metrics()
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path)
        logger.info(f"Saved aggregate metrics: {path}")
        return path

    # ------------------------------------------------------------------
    # Aggregate plots (require trained_model_csv)
    # ------------------------------------------------------------------

    def plot_roc(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        title: str | None = None,
        show_individual: bool = True,
    ) -> plt.Figure:
        """Generate an aggregate ROC curve across all seeds.

        Loads every seed's model and test data from the registry, collects
        per-seed ROC curves, and plots the mean ± std band.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename. Defaults to
                ``"aggregate_roc_curve.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            title (str | None, optional): Plot title. Defaults to None.
            show_individual (bool, optional): Draw per-seed curves as thin
                background lines. Defaults to True.

        Returns:
            plt.Figure: Matplotlib figure with the aggregate ROC curve.
        """
        _, _, y_trues, y_probas = self._predictions

        fig, data = _plot_roc_styled(
            y_trues=y_trues,
            y_probas=y_probas,
            show_individual=show_individual,
            title=title,
            label_classifier=self.classifier_name,
            dpi=dpi,
        )
        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_roc_curve",
            f"{self.classifier_name}_aggregate_roc_curve",
            dpi,
            save,
        )
        return fig

    def plot_precision_recall(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        title: str | None = None,
        show_individual: bool = True,
    ) -> plt.Figure:
        """Generate an aggregate Precision-Recall curve across all seeds.

        Loads every seed's model and test data from the registry and plots
        the mean PR curve with a ±1 std confidence band.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename. Defaults to
                ``"aggregate_pr_curve.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            title (str | None, optional): Plot title. Defaults to None.
            show_individual (bool, optional): Draw per-seed curves as thin
                background lines. Defaults to True.

        Returns:
            plt.Figure: Matplotlib figure with the aggregate PR curve.
        """
        _, _, y_trues, y_probas = self._predictions

        fig, data = _plot_pr_styled(
            y_trues=y_trues,
            y_probas=y_probas,
            show_individual=show_individual,
            title=title,
            dpi=dpi,
        )
        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_pr_curve",
            f"{self.classifier_name}_aggregate_pr_curve",
            dpi,
            save,
        )
        return fig

    def plot_calibration(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        title: str | None = None,
        n_bins: int = 10,
    ) -> plt.Figure:
        """Generate an aggregate calibration curve across all seeds.

        Averages per-seed calibration curves on a shared probability grid and
        plots the mean with a ±1 std band against the perfect-calibration
        diagonal.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename. Defaults to
                ``"aggregate_calibration.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            title (str | None, optional): Plot title. Defaults to None.
            n_bins (int, optional): Number of calibration bins. Defaults to 10.

        Returns:
            plt.Figure: Matplotlib figure with the aggregate calibration curve.
        """
        _, _, y_trues, y_probas = self._predictions

        fig, data = _plot_cal_styled(
            y_trues=y_trues,
            y_probas=y_probas,
            n_bins=n_bins,
            title=title,
            dpi=dpi,
        )
        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_calibration",
            f"{self.classifier_name}_aggregate_calibration",
            dpi,
            save,
        )
        return fig

    def plot_prediction_distribution(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        title: str | None = None,
    ) -> plt.Figure:
        """Generate an aggregate prediction distribution plot across all seeds.

        Pools predicted probabilities from all seeds by true class and plots
        KDE distributions, giving an ensemble-level view of model
        discrimination.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename. Defaults to
                ``"aggregate_prediction_distribution.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            title (str | None, optional): Plot title. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure with aggregate KDE distributions.
        """
        _, _, y_trues, y_probas = self._predictions

        fig, data = _plot_pred_dist_styled(
            y_trues=y_trues,
            y_probas=y_probas,
            title=title,
            dpi=dpi,
        )
        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_prediction_distribution",
            f"{self.classifier_name}_aggregate_prediction_distribution",
            dpi,
            save,
        )
        return fig

    def plot_confusion_matrix(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        title: str | None = None,
        normalize: str | None = None,
    ) -> plt.Figure:
        """Generate an aggregate confusion matrix summed across all seeds.

        Accumulates raw confusion matrices from every seed and optionally
        normalises the result before displaying as a heatmap.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename. Defaults to
                ``"aggregate_confusion_matrix.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            title (str | None, optional): Plot title. Defaults to None.
            normalize (str | None, optional): Normalisation mode: ``"true"``,
                ``"pred"``, ``"all"``, or None. Defaults to None.

        Returns:
            plt.Figure: Matplotlib figure with the aggregate confusion matrix.
        """
        models, x_tests, y_trues, _ = self._predictions
        y_preds: list[np.ndarray] = [
            model.predict(X_test) for model, X_test in zip(models, x_tests, strict=True)
        ]

        fig, data = _plot_cm_styled(
            y_trues=y_trues,
            y_preds=y_preds,
            normalize=normalize,
            title=title,
            dpi=dpi,
        )

        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_confusion_matrix",
            f"{self.classifier_name}_aggregate_confusion_matrix",
            dpi,
            save,
        )
        return fig

    def plot_threshold_analysis(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        title: str | None = None,
        show_individual: bool = True,
    ) -> plt.Figure:
        """Generate an aggregate threshold analysis plot across all seeds.

        Sweeps decision thresholds from 0 to 1 per seed and plots the mean
        F1, precision, recall, and balanced accuracy with ±1 std bands.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename. Defaults to
                ``"aggregate_threshold_analysis.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            title (str | None, optional): Plot title. Defaults to None.
            show_individual (bool, optional): Draw per-seed curves as thin
                background lines. Defaults to True.

        Returns:
            plt.Figure: Matplotlib figure with the aggregate threshold analysis.
        """
        _, _, y_trues, y_probas = self._predictions

        fig, data = _plot_threshold_styled(
            y_trues=y_trues,
            y_probas=y_probas,
            show_individual=show_individual,
            title=title,
            dpi=dpi,
        )
        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_threshold_analysis",
            f"{self.classifier_name}_aggregate_threshold_analysis",
            dpi,
            save,
        )
        return fig

    def plot_g_mean_curve(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        title: str | None = None,
        show_individual: bool = True,
    ) -> plt.Figure:
        """Generate an aggregate G-mean curve across all seeds.

        Computes per-seed G-mean, sensitivity, and specificity curves, then
        aggregates using the median and IQR (25th–75th percentile shading)
        for robustness against outlier seeds. Marks the default threshold and
        the threshold at peak median G-mean.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename. Defaults to
                ``"{classifier_name}_aggregate_g_mean_curve"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            title (str | None, optional): Plot title. Defaults to None.
            show_individual (bool, optional): Draw per-seed G-mean curves as thin
                background lines. Defaults to True.

        Returns:
            plt.Figure: Matplotlib figure with the aggregate G-mean curve.
        """
        _, _, y_trues, y_probas = self._predictions

        fig, data = _plot_gmean_styled(
            y_trues=y_trues,
            y_probas=y_probas,
            show_individual=show_individual,
            title=title,
            dpi=dpi,
        )
        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_g_mean_curve",
            f"{self.classifier_name}_aggregate_g_mean_curve",
            dpi,
            save,
        )
        return fig

    def plot_feature_importance(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
        title: str | None = None,
        top_n: int = 20,
    ) -> plt.Figure | None:
        """Generate an aggregate feature importance plot across all seeds.

        Collects feature importances from every seed's model and plots the
        mean importance with ±1 std error bars for the top-N features.
        Returns None when no model exposes ``feature_importances_``.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename. Defaults to
                ``"aggregate_feature_importance.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.
            title (str | None, optional): Plot title. Defaults to None.
            top_n (int, optional): Number of top features to display.
                Defaults to 20.

        Returns:
            plt.Figure | None: Matplotlib figure, or None if no model exposes
                ``feature_importances_``.
        """
        registry = self._require_registry()
        models: list[BaseEstimator] = []
        feature_names: list[str] = []
        for _, row in registry.iterrows():
            model, _, _, _ = self._load_seed_data(row)
            models.append(model)
            if not feature_names:
                feature_names = pd.read_csv(
                    row["significant_features_csv"], index_col=0
                ).index.tolist()

        if not feature_names:
            logger.warning("No feature names found; skipping feature importance plot.")
            return None

        result = _plot_fi_styled(
            models=models,
            feature_names=feature_names,
            top_n=top_n,
            title=title,
            dpi=dpi,
        )
        if result is None:
            logger.warning(
                "No model exposes feature_importances_; "
                "skipping aggregate feature importance plot"
            )
            return None

        fig, data = result
        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_feature_importance",
            f"{self.classifier_name}_aggregate_feature_importance",
            dpi,
            save,
        )
        return fig

    def plot_learning_curve(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> tuple[plt.Figure, pd.DataFrame] | None:
        """Plot an aggregate learning curve across all seeds.

        Loads per-seed learning-curve JSON files from
        ``{metrics_dir}/learning_curve/*.json``, interpolates them onto a
        shared training-size grid, and plots mean ± std bands for train and
        validation balanced accuracy.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename.
                Defaults to ``"aggregate_learning_curve.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            tuple[plt.Figure, pd.DataFrame] | None: Figure and summary DataFrame,
                or None if no learning-curve JSON files are found.

        Examples:
            >>> result = evaluator.plot_learning_curve()
        """
        if self._metrics_dir is None:
            logger.warning("plot_learning_curve requires metrics_dir; skipping.")
            return None

        lc_dir = os.path.join(self._metrics_dir, "learning_curve")
        json_paths = sorted(glob.glob(os.path.join(lc_dir, "*.json")))
        if not json_paths:
            logger.warning(f"No learning-curve JSON files found in {lc_dir}; skipping.")
            return None

        all_data: list[dict] = []
        for p in json_paths:
            with open(p) as f:
                all_data.append(json.load(f))

        fig_path = os.path.join(
            self.output_dir, filename or "aggregate_learning_curve.png"
        )
        fig, df = _plot_agg_lc_styled(
            all_data=all_data,
            filepath=fig_path,
            overwrite=save,
            dpi=dpi,
        )
        if save:
            ensure_dir(self.output_dir)
            logger.info(f"Saved aggregate learning curve: {fig_path}")
            df.to_csv(fig_path.replace(".png", ".csv"))
            plt.close(fig)
        return fig, df

    def plot_all(
        self,
        dpi: int = 150,
        show_individual: bool = True,
    ) -> dict[str, plt.Figure | tuple[plt.Figure, pd.DataFrame] | None]:
        """Generate and save all aggregate evaluation plots.

        Runs every individual ``plot_*`` method and collects the resulting
        figures. Each figure is saved to ``output_dir/figures/`` automatically.

        Requires ``trained_model_csv`` to have been provided at construction time.

        Args:
            dpi (int, optional): Figure resolution applied to all plots.
                Defaults to 150.
            show_individual (bool, optional): Draw per-seed curves as thin
                background lines on curve-based plots. Defaults to True.

        Returns:
            dict[str, plt.Figure | None]: Mapping of plot name to figure.
                Keys: ``"roc_curve"``, ``"pr_curve"``, ``"calibration"``,
                ``"prediction_distribution"``, ``"confusion_matrix"``,
                ``"threshold_analysis"``, ``"g_mean_curve"``, ``"feature_importance"``,
                ``"shap_summary"``, ``"seed_stability"``,
                ``"frequency_band_contribution"``, ``"learning_curve"``.
                Values are None when a plot cannot be generated (e.g.,
                feature importance for models without ``feature_importances_``).

        Examples:
            >>> figs = evaluator.plot_all(dpi=200)
            >>> figs["roc_curve"].savefig("custom_roc.png")
        """
        # Pre-warm the prediction cache once on the main thread before spawning
        # workers, so all plot methods share a single round of disk I/O.
        if self._trained_model_csv is not None:
            _ = self._predictions

        plot_tasks: dict[str, Any] = {
            "roc_curve": lambda: self.plot_roc(
                dpi=dpi, show_individual=show_individual
            ),
            "pr_curve": lambda: self.plot_precision_recall(
                dpi=dpi, show_individual=show_individual
            ),
            "calibration": lambda: self.plot_calibration(dpi=dpi),
            "prediction_distribution": lambda: self.plot_prediction_distribution(
                dpi=dpi
            ),
            "confusion_matrix": lambda: self.plot_confusion_matrix(dpi=dpi),
            "threshold_analysis": lambda: self.plot_threshold_analysis(
                dpi=dpi, show_individual=show_individual
            ),
            "g_mean_curve": lambda: self.plot_g_mean_curve(
                dpi=dpi, show_individual=show_individual
            ),
            "feature_importance": lambda: self.plot_feature_importance(dpi=dpi),
            "seed_stability": lambda: self.plot_seed_stability(dpi=dpi),
            "frequency_band_contribution": lambda: (
                self.plot_frequency_band_contribution(dpi=dpi)
            ),
            "learning_curve": lambda: self.plot_learning_curve(dpi=dpi),
        }

        results: dict[str, plt.Figure | tuple[plt.Figure, pd.DataFrame] | None] = {}
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(fn): name for name, fn in plot_tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.warning(f"Plot {name!r} failed: {exc}")
                    results[name] = None

        # SHAP plots use matplotlib global state and are not thread-safe.
        # Run them sequentially after the parallel block to prevent figures
        # from being drawn onto each other's canvas.
        for shap_name, shap_fn in (
            ("shap_summary", lambda: self.plot_shap_summary(dpi=dpi)),
            ("shap_waterfall", lambda: self.plot_shap_waterfall(dpi=dpi)),
        ):
            if not self.plot_shap:
                results[shap_name] = None
                continue
            try:
                results[shap_name] = shap_fn()
            except Exception as exc:
                logger.warning(f"Plot {shap_name!r} failed: {exc}")
                results[shap_name] = None

        return results

    # ------------------------------------------------------------------
    # New visualizations: SHAP, seed stability, frequency band
    # ------------------------------------------------------------------

    def get_metrics_list(self) -> list[dict[str, Any]]:
        """Load and return all per-seed metrics as a list of dicts.

        Reads each JSON file resolved from ``metrics_files`` or
        ``metrics_dir`` and returns the raw records without aggregation.
        Useful for feeding ``plot_classifier_comparison`` and
        ``plot_seed_stability`` when combining multiple evaluators.

        Returns:
            list[dict[str, Any]]: List of per-seed metrics dicts, one per
                JSON file. Each dict matches the output of
                ``ModelEvaluator.save_metrics()``.

        Raises:
            ValueError: If no metrics source was provided at construction.

        Examples:
            >>> records = evaluator.get_metrics_list()
            >>> f1_scores = [r["f1_score"] for r in records]
        """
        paths = self._resolve_metrics_files()
        records: list[dict[str, Any]] = []
        for p in paths:
            with open(p) as f:
                records.append(json.load(f))
        return records

    def _collect_shap_explanations(
        self,
        models: list[BaseEstimator],
        x_tests: list[pd.DataFrame],
    ) -> list[shap.Explanation | None]:
        """Load per-seed SHAP explanations from disk, recomputing when absent.

        Iterates the registry rows in order, attempting to load each seed's
        cached SHAP Explanation object from the path recorded in the
        ``shap_explanation_filepath`` column. Falls back to computing the
        explanation on the fly when the file is missing or unreadable. Seeds
        whose computation also fails are represented as ``None`` and will be
        skipped by ``plot_aggregate_shap_summary``.

        Args:
            models (list[BaseEstimator]): Fitted models in registry order.
            x_tests (list[pd.DataFrame]): Corresponding test feature DataFrames.

        Returns:
            list[shap.Explanation | None]: One entry per seed. Each entry is
            either a ``shap.Explanation`` or ``None`` when unavailable.
        """
        registry = self._require_registry()
        col_exists = "shap_explanation_filepath" in registry.columns
        explanations: list[shap.Explanation | None] = []

        for (_, row), model, X in zip(
            registry.iterrows(), models, x_tests, strict=False
        ):
            filepath = row.get("shap_explanation_filepath") if col_exists else None
            if filepath and os.path.isfile(str(filepath)):
                try:
                    explanations.append(joblib.load(filepath))
                    continue
                except Exception as e:
                    logger.warning(
                        f"Failed to load SHAP explanation from {filepath}: {e}"
                    )
            try:
                explanations.append(compute_shap_explanation(model, X))
            except Exception as e:
                logger.warning(f"SHAP computation failed for seed: {e}")
                explanations.append(None)

        return explanations

    def plot_shap_summary(
        self,
        max_display: int = 20,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot a beeswarm of SHAP values aggregated across all seeds.

        Loads every seed's model and test set from the registry, loads cached
        SHAP Explanation objects from disk (recomputing only when missing),
        then renders a beeswarm showing direction and magnitude of feature
        contributions across all seeds.

        Args:
            max_display (int, optional): Number of top features to show,
                sorted by mean |SHAP| descending. Defaults to 20.
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename.
                Defaults to ``"aggregate_shap_summary.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure: Matplotlib figure with the aggregate SHAP beeswarm plot.

        Examples:
            >>> fig = evaluator.plot_shap_summary(max_display=15)
        """
        models, x_tests, _, _ = self._predictions
        # Each seed may select different features; collect names per seed.
        per_seed_feature_names: list[list[str]] = [
            X_test.columns.tolist() for X_test in x_tests
        ]

        explanations = self._collect_shap_explanations(models, x_tests)

        fig, data = plot_aggregate_shap_summary(
            models=models,
            X_tests=x_tests,
            feature_names=per_seed_feature_names,
            max_display=max_display,
            dpi=dpi,
            explanations=explanations,
        )
        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_shap_summary",
            f"{self.classifier_name}_aggregate_shap_summary",
            dpi,
            save,
        )
        return fig

    def plot_shap_waterfall(
        self,
        max_display: int = 20,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot an aggregate SHAP waterfall across all seeds.

        For each seed, selects the sample with the highest predicted eruption
        probability, zero-pads its SHAP values to the union feature space, then
        averages across seeds to produce a single representative waterfall.
        Reuses cached SHAP Explanation objects from disk when available.

        Args:
            max_display (int, optional): Maximum number of features to display.
                Defaults to 20.
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename.
                Defaults to ``"{classifier_name}_aggregate_shap_waterfall"``.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure: Matplotlib figure with the aggregate SHAP waterfall.

        Examples:
            >>> fig = evaluator.plot_shap_waterfall()
        """
        models, x_tests, _, y_probas = self._predictions
        per_seed_feature_names: list[list[str]] = [
            X_test.columns.tolist() for X_test in x_tests
        ]

        explanations = self._collect_shap_explanations(models, x_tests)

        fig, data = plot_aggregate_shap_waterfall(
            models=models,
            X_tests=x_tests,
            feature_names=per_seed_feature_names,
            y_probas=y_probas,
            max_display=max_display,
            title=f"Aggregate SHAP Waterfall — {self.classifier_name}",
            dpi=dpi,
            explanations=explanations,
        )

        self._save_outputs(
            fig,
            data,
            filename or f"{self.classifier_name}_aggregate_shap_waterfall",
            f"{self.classifier_name}_aggregate_shap_waterfall",
            dpi,
            save,
        )
        return fig

    def plot_seed_stability(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> None:
        """Plot seed stability violin for this single-classifier evaluator.

        Shows the distribution of ``metric`` across random seeds as a violin
        with individual seed dots and a mean line. Requires JSON metrics files
        to have been provided at construction.

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename.
                Defaults to ``"seed_stability_{metric}.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Examples:
            >>> fig = evaluator.plot_seed_stability(metric="balanced_accuracy")
        """
        if self._metrics_files is None and self._metrics_dir is None:
            raise ValueError(
                "plot_seed_stability requires JSON metrics files. "
                "Provide metrics_dir or metrics_files when constructing MultiModelEvaluator."
            )
        records = self.get_metrics_list()

        # Use model_name from first record as classifier label, fall back to "model"
        clf_name = records[0].get("model_name", "model") if records else "model"
        metrics_by_clf = {clf_name: records}

        metrics: list[str] = [
            "f1_score",
            "roc_auc",
            "pr_auc",
            "balanced_accuracy",
            "precision",
            "recall",
            "specificity",
            "sensitivity",
        ]

        for metric in metrics:
            fig, data = plot_seed_stability(
                metrics_by_classifier=metrics_by_clf,
                figsize=(2, 4),
                metric=metric,
                dpi=dpi,
            )
            fig_filename = f"{self.classifier_name}_seed_stability_{metric}"
            data_filename = f"{self.classifier_name}_seed_stability_{metric}"
            self._save_outputs(
                fig, data, filename or fig_filename, data_filename, dpi, save
            )

        return None

    def plot_frequency_band_contribution(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot the frequency band contribution across all seeds.

        Loads the significant-features CSV for each seed from the registry,
        collects per-seed feature lists, and passes them to
        ``plot_frequency_band_contribution`` for multi-seed aggregation
        (mean ± std count per band).

        Args:
            save (bool, optional): Whether to save the figure. Defaults to True.
            filename (str | None, optional): Override figure filename.
                Defaults to ``"frequency_band_contribution.png"``.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure: Matplotlib figure with the frequency band bar chart.

        Examples:
            >>> fig = evaluator.plot_frequency_band_contribution()
        """
        registry = self._require_registry()
        per_seed_features: list[list[str]] = []

        for _, row in registry.iterrows():
            features = pd.read_csv(
                row["significant_features_csv"], index_col=0
            ).index.tolist()
            per_seed_features.append(features)

        fig, data = plot_frequency_band_contribution(
            feature_names=per_seed_features,
            dpi=dpi,
        )
        self._save_outputs(
            fig,
            data,
            filename or "frequency_band_contribution",
            "frequency_band_contribution",
            dpi,
            save,
        )
        return fig
