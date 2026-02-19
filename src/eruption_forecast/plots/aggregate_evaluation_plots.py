"""Aggregate evaluation plots for multi-seed model ensembles.

This module provides functions to generate ensemble-level evaluation plots
from a model registry CSV produced by ``ModelTrainer.train_and_evaluate()``.
Each function loads every seed's held-out test data and model, computes
aggregate statistics (mean ± std), and saves both a PNG figure and a CSV
data file to the ``plots/`` directory.

All plots follow Nature/Science journal standards with consistent styling,
colorblind-safe palettes, and high-DPI output.
"""

import os
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eruption_forecast.logger import logger
from eruption_forecast.utils.aggregate import load_seed_data
from eruption_forecast.plots.evaluation_plots import (
    plot_aggregate_roc_curve as _plot_roc_styled,
    plot_aggregate_calibration as _plot_cal_styled,
    plot_aggregate_confusion_matrix as _plot_cm_styled,
    plot_aggregate_feature_importance as _plot_fi_styled,
    plot_aggregate_threshold_analysis as _plot_threshold_styled,
    plot_aggregate_precision_recall_curve as _plot_pr_styled,
    plot_aggregate_prediction_distribution as _plot_pred_dist_styled,
)


def save_aggregate_outputs(
    fig: plt.Figure,
    data: pd.DataFrame | None,
    registry_csv: str,
    output_dir: str | None,
    fig_filename: str,
    data_filename: str | None,
    dpi: int,
    save: bool,
) -> None:
    """Save an aggregate plot figure and its underlying data to disk.

    Resolves the output directory to the ``plots/`` subdirectory of the
    registry CSV directory when ``output_dir`` is None, then writes the
    figure as a PNG and (optionally) the data as a CSV.

    Args:
        fig (plt.Figure): The figure to save.
        data (pd.DataFrame | None): Aggregate data to save alongside the
            figure. If None, no CSV is written.
        registry_csv (str): Path to the model registry CSV (used to derive
            the default ``plots/`` output directory).
        output_dir (str | None): Explicit output directory. If None, defaults
            to ``<registry_dir>/plots/``.
        fig_filename (str): Output filename for the figure (PNG).
        data_filename (str | None): Output filename for the data CSV. If None,
            no CSV is written.
        dpi (int): Figure resolution in dots per inch.
        save (bool): Whether to write any files.
    """
    if not save:
        return
    out = output_dir or os.path.join(os.path.dirname(registry_csv), "plots")
    os.makedirs(out, exist_ok=True)
    fig_path = os.path.join(out, fig_filename)
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    logger.info(f"Saved aggregate plot: {fig_path}")
    if data is not None and data_filename is not None:
        data_path = os.path.join(out, data_filename)
        data.to_csv(data_path)
        logger.info(f"Saved aggregate data: {data_path}")


def plot_aggregate_roc(
    registry_csv: str,
    output_dir: str | None = None,
    save: bool = True,
    filename: str | None = None,
    dpi: int = 150,
    title: str | None = None,
    show_individual: bool = True,
) -> plt.Figure:
    """Generate an aggregate ROC curve across all seeds in the registry.

    Loads every seed's model and held-out test data from the registry,
    collects per-seed ROC curves, and plots the mean ± std band.
    Saves both a PNG figure and a CSV data file to the ``plots/`` directory.

    Args:
        registry_csv (str): Path to the model registry CSV produced by
            ``ModelTrainer.train_and_evaluate()``.
        output_dir (str | None, optional): Directory for saved outputs.
            Defaults to ``<registry_dir>/plots/``.
        save (bool, optional): Whether to save outputs. Defaults to True.
        filename (str | None, optional): Override figure filename. Defaults to
            ``"aggregate_roc_curve.png"``.
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. Defaults to None.
        show_individual (bool, optional): Draw per-seed curves as thin
            background lines. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure with the aggregate ROC curve.
    """
    registry = pd.read_csv(registry_csv, index_col=0)
    y_trues: list[np.ndarray] = []
    y_probas: list[np.ndarray] = []
    for _, row in registry.iterrows():
        _, _, y_true, y_proba = load_seed_data(row)
        y_trues.append(y_true)
        y_probas.append(y_proba)

    fig, data = _plot_roc_styled(
        y_trues=y_trues,
        y_probas=y_probas,
        show_individual=show_individual,
        title=title,
        dpi=dpi,
    )
    save_aggregate_outputs(
        fig, data, registry_csv, output_dir,
        filename or "aggregate_roc_curve.png", "aggregate_roc_curve.csv", dpi, save,
    )
    return fig


def plot_aggregate_precision_recall(
    registry_csv: str,
    output_dir: str | None = None,
    save: bool = True,
    filename: str | None = None,
    dpi: int = 150,
    title: str | None = None,
    show_individual: bool = True,
) -> plt.Figure:
    """Generate an aggregate Precision-Recall curve across all seeds.

    Loads every seed's model and test data from the registry and plots
    mean PR curve with a ±1 std confidence band.
    Saves both a PNG figure and a CSV data file to the ``plots/`` directory.

    Args:
        registry_csv (str): Path to the model registry CSV.
        output_dir (str | None, optional): Directory for saved outputs.
            Defaults to ``<registry_dir>/plots/``.
        save (bool, optional): Whether to save outputs. Defaults to True.
        filename (str | None, optional): Override figure filename. Defaults to
            ``"aggregate_pr_curve.png"``.
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. Defaults to None.
        show_individual (bool, optional): Draw per-seed curves as thin
            background lines. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure with the aggregate PR curve.
    """
    registry = pd.read_csv(registry_csv, index_col=0)
    y_trues: list[np.ndarray] = []
    y_probas: list[np.ndarray] = []
    for _, row in registry.iterrows():
        _, _, y_true, y_proba = load_seed_data(row)
        y_trues.append(y_true)
        y_probas.append(y_proba)

    fig, data = _plot_pr_styled(
        y_trues=y_trues,
        y_probas=y_probas,
        show_individual=show_individual,
        title=title,
        dpi=dpi,
    )
    save_aggregate_outputs(
        fig, data, registry_csv, output_dir,
        filename or "aggregate_pr_curve.png", "aggregate_pr_curve.csv", dpi, save,
    )
    return fig


def plot_aggregate_calibration(
    registry_csv: str,
    output_dir: str | None = None,
    save: bool = True,
    filename: str | None = None,
    dpi: int = 150,
    title: str | None = None,
    n_bins: int = 10,
) -> plt.Figure:
    """Generate an aggregate calibration curve across all seeds.

    Averages per-seed calibration curves on a shared probability grid and
    plots the mean with a ±1 std band against the perfect calibration
    diagonal. Saves both a PNG figure and a CSV data file.

    Args:
        registry_csv (str): Path to the model registry CSV.
        output_dir (str | None, optional): Directory for saved outputs.
            Defaults to ``<registry_dir>/plots/``.
        save (bool, optional): Whether to save outputs. Defaults to True.
        filename (str | None, optional): Override figure filename. Defaults to
            ``"aggregate_calibration.png"``.
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. Defaults to None.
        n_bins (int, optional): Number of calibration bins. Defaults to 10.

    Returns:
        plt.Figure: Matplotlib figure with the aggregate calibration curve.
    """
    registry = pd.read_csv(registry_csv, index_col=0)
    y_trues: list[np.ndarray] = []
    y_probas: list[np.ndarray] = []
    for _, row in registry.iterrows():
        _, _, y_true, y_proba = load_seed_data(row)
        y_trues.append(y_true)
        y_probas.append(y_proba)

    fig, data = _plot_cal_styled(
        y_trues=y_trues,
        y_probas=y_probas,
        n_bins=n_bins,
        title=title,
        dpi=dpi,
    )
    save_aggregate_outputs(
        fig, data, registry_csv, output_dir,
        filename or "aggregate_calibration.png", "aggregate_calibration.csv", dpi, save,
    )
    return fig


def plot_aggregate_prediction_distribution(
    registry_csv: str,
    output_dir: str | None = None,
    save: bool = True,
    filename: str | None = None,
    dpi: int = 150,
    title: str | None = None,
) -> plt.Figure:
    """Generate an aggregate prediction distribution plot across all seeds.

    Pools predicted probabilities from all seeds by true class and plots
    KDE distributions for each class, giving an ensemble-level view of
    model discrimination. Saves both a PNG figure and a CSV data file.

    Args:
        registry_csv (str): Path to the model registry CSV.
        output_dir (str | None, optional): Directory for saved outputs.
            Defaults to ``<registry_dir>/plots/``.
        save (bool, optional): Whether to save outputs. Defaults to True.
        filename (str | None, optional): Override figure filename. Defaults to
            ``"aggregate_prediction_distribution.png"``.
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. Defaults to None.

    Returns:
        plt.Figure: Matplotlib figure with aggregate KDE distributions.
    """
    registry = pd.read_csv(registry_csv, index_col=0)
    y_trues: list[np.ndarray] = []
    y_probas: list[np.ndarray] = []
    for _, row in registry.iterrows():
        _, _, y_true, y_proba = load_seed_data(row)
        y_trues.append(y_true)
        y_probas.append(y_proba)

    fig, data = _plot_pred_dist_styled(
        y_trues=y_trues,
        y_probas=y_probas,
        title=title,
        dpi=dpi,
    )
    save_aggregate_outputs(
        fig, data, registry_csv, output_dir,
        filename or "aggregate_prediction_distribution.png",
        "aggregate_prediction_distribution.csv", dpi, save,
    )
    return fig


def plot_aggregate_confusion_matrix(
    registry_csv: str,
    output_dir: str | None = None,
    save: bool = True,
    filename: str | None = None,
    dpi: int = 150,
    title: str | None = None,
    normalize: str | None = None,
) -> plt.Figure:
    """Generate an aggregate confusion matrix summed across all seeds.

    Accumulates raw confusion matrices from every seed and optionally
    normalises the result before displaying as a heatmap. Saves both a PNG
    figure and a CSV data file (always raw counts, regardless of normalization).

    Args:
        registry_csv (str): Path to the model registry CSV.
        output_dir (str | None, optional): Directory for saved outputs.
            Defaults to ``<registry_dir>/plots/``.
        save (bool, optional): Whether to save outputs. Defaults to True.
        filename (str | None, optional): Override figure filename. Defaults to
            ``"aggregate_confusion_matrix.png"``.
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. Defaults to None.
        normalize (str | None, optional): Normalisation mode: ``"true"``,
            ``"pred"``, ``"all"``, or None. Defaults to None.

    Returns:
        plt.Figure: Matplotlib figure with the aggregate confusion matrix.
    """
    registry = pd.read_csv(registry_csv, index_col=0)
    y_trues: list[np.ndarray] = []
    y_preds: list[np.ndarray] = []
    for _, row in registry.iterrows():
        model, X_test, y_true, _ = load_seed_data(row)
        y_preds.append(model.predict(X_test))
        y_trues.append(y_true)

    fig, data = _plot_cm_styled(
        y_trues=y_trues,
        y_preds=y_preds,
        normalize=normalize,
        title=title,
        dpi=dpi,
    )
    save_aggregate_outputs(
        fig, data, registry_csv, output_dir,
        filename or "aggregate_confusion_matrix.png",
        "aggregate_confusion_matrix.csv", dpi, save,
    )
    return fig


def plot_aggregate_threshold_analysis(
    registry_csv: str,
    output_dir: str | None = None,
    save: bool = True,
    filename: str | None = None,
    dpi: int = 150,
    title: str | None = None,
    show_individual: bool = True,
) -> plt.Figure:
    """Generate an aggregate threshold analysis plot across all seeds.

    Sweeps decision thresholds from 0 to 1 per seed and plots the mean
    F1, precision, recall, and balanced accuracy with ±1 std bands.
    Saves both a PNG figure and a CSV data file.

    Args:
        registry_csv (str): Path to the model registry CSV.
        output_dir (str | None, optional): Directory for saved outputs.
            Defaults to ``<registry_dir>/plots/``.
        save (bool, optional): Whether to save outputs. Defaults to True.
        filename (str | None, optional): Override figure filename. Defaults to
            ``"aggregate_threshold_analysis.png"``.
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. Defaults to None.
        show_individual (bool, optional): Draw per-seed curves as thin
            background lines. Defaults to True.

    Returns:
        plt.Figure: Matplotlib figure with the aggregate threshold analysis.
    """
    registry = pd.read_csv(registry_csv, index_col=0)
    y_trues: list[np.ndarray] = []
    y_probas: list[np.ndarray] = []
    for _, row in registry.iterrows():
        _, _, y_true, y_proba = load_seed_data(row)
        y_trues.append(y_true)
        y_probas.append(y_proba)

    fig, data = _plot_threshold_styled(
        y_trues=y_trues,
        y_probas=y_probas,
        show_individual=show_individual,
        title=title,
        dpi=dpi,
    )
    save_aggregate_outputs(
        fig, data, registry_csv, output_dir,
        filename or "aggregate_threshold_analysis.png",
        "aggregate_threshold_analysis.csv", dpi, save,
    )
    return fig


def plot_aggregate_feature_importance(
    registry_csv: str,
    output_dir: str | None = None,
    save: bool = True,
    filename: str | None = None,
    dpi: int = 150,
    title: str | None = None,
    top_n: int = 20,
) -> plt.Figure | None:
    """Generate an aggregate feature importance plot across all seeds.

    Collects feature importances from every seed's model and plots the
    mean importance with ±1 std error bars for the top-N features. Saves
    both a PNG figure and a CSV data file (all features, not just top-N).
    Returns None when no model exposes ``feature_importances_``.

    Args:
        registry_csv (str): Path to the model registry CSV.
        output_dir (str | None, optional): Directory for saved outputs.
            Defaults to ``<registry_dir>/plots/``.
        save (bool, optional): Whether to save outputs. Defaults to True.
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
    registry = pd.read_csv(registry_csv, index_col=0)
    models: list[Any] = []
    feature_names: list[str] = []
    for _, row in registry.iterrows():
        model, _, _, _ = load_seed_data(row)
        models.append(model)
        if not feature_names:
            feature_names = pd.read_csv(
                row["significant_features_csv"], index_col=0
            ).index.tolist()

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
    save_aggregate_outputs(
        fig, data, registry_csv, output_dir,
        filename or "aggregate_feature_importance.png",
        "aggregate_feature_importance.csv", dpi, save,
    )
    return fig


def plot_all_aggregate(
    registry_csv: str,
    output_dir: str | None = None,
    dpi: int = 150,
    show_individual: bool = True,
) -> dict[str, plt.Figure | None]:
    """Generate and save all aggregate evaluation plots for the full seed ensemble.

    Convenience function equivalent to running all seven individual
    ``plot_aggregate_*`` functions. Each plot is saved to the ``plots/``
    subdirectory alongside its data CSV.

    Args:
        registry_csv (str): Path to the model registry CSV produced by
            ``ModelTrainer.train_and_evaluate()``.
        output_dir (str | None, optional): Directory for saved outputs.
            Defaults to ``<registry_dir>/plots/``.
        dpi (int, optional): Figure resolution applied to all plots.
            Defaults to 150.
        show_individual (bool, optional): Draw per-seed curves as thin
            background lines on curve-based plots. Defaults to True.

    Returns:
        dict[str, plt.Figure | None]: Mapping of plot name to figure.
            Keys: ``"roc_curve"``, ``"pr_curve"``, ``"calibration"``,
            ``"prediction_distribution"``, ``"confusion_matrix"``,
            ``"threshold_analysis"``, ``"feature_importance"``.
            Values are None when a plot cannot be generated (e.g.,
            feature importance for models without ``feature_importances_``).

    Examples:
        >>> from eruption_forecast.plots.aggregate_evaluation_plots import plot_all_aggregate
        >>> figs = plot_all_aggregate(
        ...     registry_csv="output/.../trained_model_*.csv",
        ...     output_dir="output/eval/aggregate",
        ...     dpi=150,
        ...     show_individual=True,
        ... )
        >>> figs["roc_curve"].savefig("custom_roc.png")
    """
    return {
        "roc_curve": plot_aggregate_roc(
            registry_csv, output_dir=output_dir, dpi=dpi,
            show_individual=show_individual,
        ),
        "pr_curve": plot_aggregate_precision_recall(
            registry_csv, output_dir=output_dir, dpi=dpi,
            show_individual=show_individual,
        ),
        "calibration": plot_aggregate_calibration(
            registry_csv, output_dir=output_dir, dpi=dpi,
        ),
        "prediction_distribution": plot_aggregate_prediction_distribution(
            registry_csv, output_dir=output_dir, dpi=dpi,
        ),
        "confusion_matrix": plot_aggregate_confusion_matrix(
            registry_csv, output_dir=output_dir, dpi=dpi,
        ),
        "threshold_analysis": plot_aggregate_threshold_analysis(
            registry_csv, output_dir=output_dir, dpi=dpi,
            show_individual=show_individual,
        ),
        "feature_importance": plot_aggregate_feature_importance(
            registry_csv, output_dir=output_dir, dpi=dpi,
        ),
    }
