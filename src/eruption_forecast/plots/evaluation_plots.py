"""Model evaluation plots with Nature/Science journal styling.

Provides a comprehensive suite of per-seed and aggregate classifier evaluation
charts. Single-seed functions accept pre-computed arrays or dicts; aggregate
functions accept lists of per-seed results and plot mean ± std envelopes.

Key functions (single-seed):

- ``plot_confusion_matrix`` — annotated normalised confusion matrix heatmap.
- ``plot_roc_curve`` — ROC curve with AUC annotation.
- ``plot_precision_recall_curve`` — precision-recall curve with average precision.
- ``plot_calibration`` — calibration (reliability) diagram.
- ``plot_threshold_analysis`` — precision/recall/F1 vs. decision threshold.
- ``plot_feature_importance`` — horizontal bar chart of top-N features.
- ``plot_prediction_distribution`` — histogram of predicted probabilities by class.
- ``plot_learning_curve`` — training vs. validation score against training set size.
- ``plot_seed_stability`` — scatter/box plot of metric values across seeds.
- ``plot_classifier_comparison`` — side-by-side metric bars across classifiers.

Key functions (aggregate / multi-seed):

- ``plot_aggregate_roc_curve``, ``plot_aggregate_precision_recall_curve``,
  ``plot_aggregate_calibration``, ``plot_aggregate_confusion_matrix``,
  ``plot_aggregate_threshold_analysis``, ``plot_aggregate_feature_importance``,
  ``plot_aggregate_prediction_distribution``, ``plot_aggregate_learning_curve``,
  ``plot_learning_curve_grid`` — ensemble-level variants with mean ± std shading.
"""

import os
import json
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    f1_score,
    roc_curve,
    recall_score,
    roc_auc_score,
    precision_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import calibration_curve

from eruption_forecast.config import CLASS_LABELS, ERUPTION_PROBABILITY_THRESHOLD
from eruption_forecast.utils.ml import compute_threshold_metrics
from eruption_forecast.plots.styles import (
    OKABE_ITO,
    NATURE_COLORS,
    nature_figure,
    configure_spine,
    apply_nature_style,
)
from eruption_forecast.utils.pathutils import ensure_dir


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot a confusion matrix heatmap with Nature/Science styling.

    Creates a 2×2 heatmap showing classification results with optional normalization.
    Labels are "Not Erupted" (0) and "Erupted" (1). Uses blue colormap.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1). Shape: (n_samples,).
        y_pred (np.ndarray): Predicted binary labels (0 or 1). Shape: (n_samples,).
        normalize (str | None, optional): Normalization mode. Options:
            - None: Raw counts (default)
            - "true": Normalize over true labels (rows)
            - "pred": Normalize over predicted labels (columns)
            - "all": Normalize over entire matrix
            Defaults to None.
        title (str | None, optional): Plot title. If None, uses "Confusion Matrix".
            Defaults to None.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. Defaults to (6, 5).
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object with the confusion matrix heatmap.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 0, 1, 1, 0])
        >>> y_pred = np.array([0, 1, 1, 1, 0, 0])
        >>> fig = plot_confusion_matrix(y_true, y_pred)
        >>> fig.savefig("confusion_matrix.png")
        >>>
        >>> # Normalized by rows (shows recall per class)
        >>> fig = plot_confusion_matrix(y_true, y_pred, normalize="true")
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    labels = CLASS_LABELS
    fmt = ".2f" if normalize else "d"

    with nature_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Normalized" if normalize else "Count"},
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title or "Confusion Matrix")

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    label_classifier: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot ROC curve with AUC annotation and Nature/Science styling.

    Creates a Receiver Operating Characteristic (ROC) curve showing true positive
    rate vs. false positive rate. Includes diagonal reference line for random
    classifier and AUC score in legend.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1). Shape: (n_samples,).
        y_proba (np.ndarray): Predicted probabilities for the positive class (1).
            Values should be in [0, 1]. Shape: (n_samples,).
        title (str | None, optional): Plot title. If None, uses "ROC Curve".
            Defaults to None.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. Defaults to (6, 5).
        label_classifier (str | None, optional): Label for legend. Defaults to None.
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object with the ROC curve.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 0, 1, 1])
        >>> y_proba = np.array([0.1, 0.8, 0.3, 0.9, 0.6])
        >>> fig = plot_roc_curve(y_true, y_proba)
        >>> fig.savefig("roc_curve.png")
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    with nature_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Plot ROC curve
        ax.plot(
            fpr,
            tpr,
            color=OKABE_ITO[4],  # Blue
            linewidth=2.0,
            label=f"ROC (AUC = {auc:.3f})",
        )

        # Add diagonal reference line
        ax.plot(
            [0, 1],
            [0, 1],
            color=NATURE_COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label=label_classifier or "Random Classifier",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "ROC Curve")
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot Precision-Recall curve with AP annotation and Nature/Science styling.

    Creates a Precision-Recall (PR) curve showing precision vs. recall across
    decision thresholds. Includes horizontal baseline for random classifier and
    Average Precision (AP) score in legend.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1). Shape: (n_samples,).
        y_proba (np.ndarray): Predicted probabilities for the positive class (1).
            Values should be in [0, 1]. Shape: (n_samples,).
        title (str | None, optional): Plot title. If None, uses
            "Precision-Recall Curve". Defaults to None.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. Defaults to (6, 5).
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object with the PR curve.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 0, 1, 1])
        >>> y_proba = np.array([0.1, 0.8, 0.3, 0.9, 0.6])
        >>> fig = plot_precision_recall_curve(y_true, y_proba)
        >>> fig.savefig("pr_curve.png")
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    with nature_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Plot PR curve
        ax.plot(
            recall,
            precision,
            color=OKABE_ITO[0],  # Orange
            linewidth=2.0,
            label=f"PR (AP = {ap:.3f})",
        )

        # Add baseline (random classifier)
        baseline = y_true.sum() / len(y_true)
        ax.axhline(
            baseline,
            color=NATURE_COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label=f"Baseline (AP = {baseline:.3f})",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or "Precision-Recall Curve")
        ax.legend(loc="best", frameon=False)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))

    return fig


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> plt.Figure:
    """Plot precision, recall, F1, and balanced accuracy vs. decision threshold.

    Creates a multi-metric threshold analysis plot showing how classification metrics
    vary across decision thresholds from 0 to 1. Marks default threshold ``ERUPTION_PROBABILITY_THRESHOLD`` and
    optimal F1 threshold with vertical lines.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1). Shape: (n_samples,).
        y_proba (np.ndarray): Predicted probabilities for the positive class (1).
            Values should be in [0, 1]. Shape: (n_samples,).
        title (str | None, optional): Plot title. If None, uses
            "Threshold Analysis". Defaults to None.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. Defaults to (10, 6).
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object with threshold analysis plot showing
            precision (blue), recall (orange), F1 score (green), and balanced
            accuracy (pink) curves.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 0, 1, 1, 0, 1])
        >>> y_proba = np.array([0.1, 0.8, 0.3, 0.9, 0.6, 0.2, 0.7])
        >>> fig = plot_threshold_analysis(y_true, y_proba)
        >>> fig.savefig("threshold_analysis.png")
    """
    thresholds, metrics = compute_threshold_metrics(y_true, y_proba)

    # Find optimal threshold (max F1)
    optimal_idx = np.argmax(metrics["f1"])
    optimal_threshold = thresholds[optimal_idx]

    with nature_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Plot metrics with distinct colors from Okabe-Ito palette
        ax.plot(
            thresholds,
            metrics["precision"],
            label="Precision",
            color=OKABE_ITO[4],  # Blue
            linewidth=2.0,
        )
        ax.plot(
            thresholds,
            metrics["recall"],
            label="Recall",
            color=OKABE_ITO[0],  # Orange
            linewidth=2.0,
        )
        ax.plot(
            thresholds,
            metrics["f1"],
            label="F1 Score",
            color=OKABE_ITO[2],  # Green
            linewidth=2.0,
        )
        ax.plot(
            thresholds,
            metrics["balanced_accuracy"],
            label="Balanced Accuracy",
            color=OKABE_ITO[6],  # Pink
            linewidth=2.0,
        )

        # Mark default threshold ERUPTION_PROBABILITY_THRESHOLD (0.5)
        ax.axvline(
            ERUPTION_PROBABILITY_THRESHOLD,
            color=NATURE_COLORS["gray"],
            linestyle=":",
            linewidth=1.5,
            label=f"Default ({ERUPTION_PROBABILITY_THRESHOLD})",
            alpha=0.7,
        )

        # Mark optimal threshold
        ax.axvline(
            optimal_threshold,
            color=NATURE_COLORS["red"],
            linestyle="--",
            linewidth=1.5,
            label=f"Optimal F1 ({optimal_threshold:.2f})",
            alpha=0.8,
        )

        configure_spine(ax)
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title(title or "Threshold Analysis")
        ax.legend(loc="best", frameon=False, ncol=2)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))

    return fig


def plot_feature_importance(
    model: BaseEstimator,
    feature_names: list[str],
    top_n: int = 20,
    title: str | None = None,
    dpi: int = 150,
) -> plt.Figure | None:
    """Plot horizontal bar chart of feature importances with Nature/Science styling.

    Creates a horizontal bar chart showing the top-N most important features from
    a trained model. Features are sorted by importance (highest to lowest from top
    to bottom). Supports tree-based models and VotingClassifier (averages across
    estimators).

    Args:
        model (BaseEstimator): Fitted scikit-learn model with feature_importances_
            attribute. Supports RandomForest, GradientBoosting, XGBoost, LightGBM,
            DecisionTree, and VotingClassifier.
        feature_names (list[str]): List of feature names corresponding to model
            features. Length must match model's n_features_in_.
        top_n (int, optional): Number of top features to display. Features are
            sorted by importance. Defaults to 20.
        title (str | None, optional): Plot title. If None, uses
            "Top {top_n} Feature Importances". Defaults to None.
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure | None: Matplotlib figure object, or None if model lacks
            feature_importances_ attribute and is not a VotingClassifier with
            tree-based estimators.

    Examples:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> # Train model
        >>> rf = RandomForestClassifier(random_state=0)
        >>> rf.fit(X_train, y_train)
        >>> # Plot importances
        >>> fig = plot_feature_importance(rf, feature_names=X_train.columns)
        >>> fig.savefig("feature_importance.png") if fig else None
        >>>
        >>> # Plot top 10 features only
        >>> fig = plot_feature_importance(rf, feature_names, top_n=10)
    """
    # Extract importances
    importances: np.ndarray
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_  # type: ignore[assignment]
    elif isinstance(model, VotingClassifier):
        # Average importances from voting estimators
        all_importances = []
        for est in model.estimators_:
            if hasattr(est, "feature_importances_"):
                all_importances.append(est.feature_importances_)
        if not all_importances:
            return None
        importances = np.mean(all_importances, axis=0)
    else:
        return None

    # Sort and select top-N
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Reverse for bottom-to-top display
    top_features = top_features[::-1]
    top_importances = top_importances[::-1]

    figheight = max(4, top_n * 0.35)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=(8, figheight), dpi=dpi)

        bars = ax.barh(
            range(top_n),
            top_importances,
            color=OKABE_ITO[4],  # Blue
            alpha=0.8,
        )
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features)

        configure_spine(ax)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Feature")
        ax.set_title(title or f"Top {top_n} Feature Importances")

        # Add value labels
        for i, (_bar, val) in enumerate(zip(bars, top_importances, strict=False)):
            ax.text(
                val,
                i,
                f"  {val:.4f}",
                va="center",
                ha="left",
                fontsize=8,
                color=NATURE_COLORS["blue"],
            )

    return fig


def plot_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot calibration curve with Nature/Science styling.

    Creates a calibration plot showing predicted probability vs. observed frequency.
    Well-calibrated models have points along the diagonal. Bins probabilities into
    n_bins and plots fraction of positives vs. mean predicted probability per bin.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1). Shape: (n_samples,).
        y_proba (np.ndarray): Predicted probabilities for the positive class (1).
            Values should be in [0, 1]. Shape: (n_samples,).
        n_bins (int, optional): Number of bins for calibration curve. More bins
            give finer resolution but require more data. Defaults to 10.
        title (str | None, optional): Plot title. If None, uses "Calibration Curve".
            Defaults to None.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. Defaults to (6, 5).
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object with calibration curve showing model
            performance (blue line with square markers) vs. perfect calibration
            (diagonal gray dashed line).

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        >>> y_proba = np.array([0.1, 0.9, 0.2, 0.85, 0.7, 0.3, 0.8, 0.15])
        >>> fig = plot_calibration(y_true, y_proba, n_bins=5)
        >>> fig.savefig("calibration.png")
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    with nature_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Plot calibration curve
        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            marker="s",
            markersize=6,
            linewidth=2.0,
            color=OKABE_ITO[4],  # Blue
            label="Model",
        )

        # Add perfectly calibrated reference line
        ax.plot(
            [0, 1],
            [0, 1],
            color=NATURE_COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label="Perfectly Calibrated",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title or "Calibration Curve")
        ax.legend(loc="best", frameon=False)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))

    return fig


def plot_prediction_distribution(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot histogram of predicted probabilities by true class with Nature/Science styling.

    Creates overlapping histograms showing the distribution of predicted probabilities
    for each true class. Well-separated distributions indicate good model discrimination.
    Includes vertical line at ``ERUPTION_PROBABILITY_THRESHOLD``.

    Args:
        y_true (np.ndarray): True binary labels (0 or 1). Shape: (n_samples,).
        y_proba (np.ndarray): Predicted probabilities for the positive class (1).
            Values should be in [0, 1]. Shape: (n_samples,).
        title (str | None, optional): Plot title. If None, uses
            "Prediction Distribution by True Class". Defaults to None.
        figsize (tuple[float, float], optional): Figure size as (width, height)
            in inches. Defaults to (8, 5).
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object with overlapping histograms for
            "Not Erupted" (blue) and "Erupted" (orange) classes.

    Examples:
        >>> import numpy as np
        >>> y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        >>> y_proba = np.array([0.1, 0.9, 0.2, 0.85, 0.7, 0.3, 0.8, 0.15])
        >>> fig = plot_prediction_distribution(y_true, y_proba)
        >>> fig.savefig("prediction_distribution.png")
    """
    proba_0 = y_proba[y_true == 0]
    proba_1 = y_proba[y_true == 1]

    # Cap bins to unique values in each class to avoid "too many bins" errors
    # when predicted probabilities cluster in a narrow range.
    bins = max(1, min(20, len(np.unique(proba_0)), len(np.unique(proba_1))))

    with nature_figure(figsize=figsize, dpi=dpi) as (fig, ax):
        # Plot histograms for each class
        ax.hist(
            proba_0,
            bins=bins,
            alpha=0.6,
            color=OKABE_ITO[4],  # Blue
            label=f"Not Erupted (n={len(proba_0)})",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.hist(
            proba_1,
            bins=bins,
            alpha=0.6,
            color=OKABE_ITO[0],  # Orange
            label=f"Erupted (n={len(proba_1)})",
            edgecolor="black",
            linewidth=0.5,
        )

        # Mark ERUPTION_PROBABILITY_THRESHOLD
        ax.axvline(
            ERUPTION_PROBABILITY_THRESHOLD,
            color=NATURE_COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({ERUPTION_PROBABILITY_THRESHOLD})",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_title(title or "Prediction Distribution by True Class")
        ax.legend(loc="best", frameon=False)
        ax.set_xlim((0.0, 1.0))

    return fig


# ---------------------------------------------------------------------------
# Aggregate (multi-seed) plot functions
# ---------------------------------------------------------------------------


def plot_aggregate_roc_curve(
    y_trues: np.ndarray | list[np.ndarray],
    y_probas: list[np.ndarray],
    show_individual: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    label_classifier: str | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot aggregate ROC curves across multiple seeds with a mean ± std band.

    Interpolates each seed's ROC curve onto a shared FPR grid, then plots the
    mean curve as a bold line with a ±1 std confidence band. Individual seed
    curves can optionally be drawn as thin background lines.

    Args:
        y_trues (np.ndarray | list[np.ndarray]): True binary labels. Either a
            single array broadcast to all seeds, or a list of per-seed arrays.
        y_probas (list[np.ndarray]): List of per-seed predicted probabilities
            for the positive class.
        show_individual (bool, optional): Draw individual seed curves as thin
            background lines. Defaults to True.
        title (str | None, optional): Plot title. Defaults to
            "Aggregate ROC Curve".
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to (6, 5).
        label_classifier (str | None, optional): Label for legend. Defaults to None.
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame
            with columns ``fpr``, ``mean_tpr``, ``std_tpr``.

    Examples:
        >>> fig, data = plot_aggregate_roc_curve(y_true, y_probas_list, dpi=150)
        >>> fig.savefig("aggregate_roc.png")
        >>> data.to_csv("roc_data.csv", index=False)
    """
    fpr_grid = np.linspace(0, 1, 200)
    tprs: list[np.ndarray] = []
    aucs: list[float] = []

    # Broadcast single y_true to all seeds
    if isinstance(y_trues, np.ndarray):
        y_trues = [y_trues] * len(y_probas)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for y_true, y_proba in zip(y_trues, y_probas, strict=False):
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            aucs.append(roc_auc_score(y_true, y_proba))

            if show_individual:
                ax.plot(
                    fpr_grid,
                    tpr_interp,
                    color=OKABE_ITO[4],
                    alpha=0.15,
                    linewidth=0.5,
                )

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        ax.plot(
            fpr_grid,
            mean_tpr,
            color=OKABE_ITO[4],
            linewidth=2.0,
            label=f"Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})",
        )
        ax.fill_between(
            fpr_grid,
            np.clip(mean_tpr - std_tpr, 0, 1),
            np.clip(mean_tpr + std_tpr, 0, 1),
            color=OKABE_ITO[4],
            alpha=0.2,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color=NATURE_COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label=label_classifier or "Random Classifier",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "Aggregate ROC Curve")
        ax.legend(loc="lower right", frameon=False)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))

    data = pd.DataFrame({"fpr": fpr_grid, "mean_tpr": mean_tpr, "std_tpr": std_tpr})
    return fig, data


def plot_aggregate_precision_recall_curve(
    y_trues: np.ndarray | list[np.ndarray],
    y_probas: list[np.ndarray],
    show_individual: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot aggregate Precision-Recall curves across multiple seeds.

    Interpolates each seed's PR curve onto a shared recall grid, then plots
    the mean curve with a ±1 std confidence band.

    Args:
        y_trues (np.ndarray | list[np.ndarray]): True binary labels. Either a
            single array broadcast to all seeds, or a list of per-seed arrays.
        y_probas (list[np.ndarray]): List of per-seed predicted probabilities
            for the positive class.
        show_individual (bool, optional): Draw individual seed curves as thin
            background lines. Defaults to True.
        title (str | None, optional): Plot title. Defaults to
            "Aggregate Precision-Recall Curve".
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to (6, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame
            with columns ``recall``, ``mean_precision``, ``std_precision``.

    Examples:
        >>> fig, data = plot_aggregate_precision_recall_curve(y_true, y_probas_list)
        >>> fig.savefig("aggregate_pr.png")
        >>> data.to_csv("pr_data.csv", index=False)
    """
    recall_grid = np.linspace(0, 1, 200)
    precisions: list[np.ndarray] = []
    aps: list[float] = []

    if isinstance(y_trues, np.ndarray):
        y_trues = [y_trues] * len(y_probas)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for y_true, y_proba in zip(y_trues, y_probas, strict=False):
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            # precision_recall_curve returns decreasing recall; flip for interp
            prec_interp = np.interp(recall_grid, recall[::-1], precision[::-1])
            precisions.append(prec_interp)
            aps.append(average_precision_score(y_true, y_proba))

            if show_individual:
                ax.plot(
                    recall_grid,
                    prec_interp,
                    color=OKABE_ITO[0],
                    alpha=0.15,
                    linewidth=0.5,
                )

        mean_prec = np.mean(precisions, axis=0)
        std_prec = np.std(precisions, axis=0)
        mean_ap = float(np.mean(aps))
        std_ap = float(np.std(aps))

        ax.plot(
            recall_grid,
            mean_prec,
            color=OKABE_ITO[0],
            linewidth=2.0,
            label=f"Mean PR (AP = {mean_ap:.3f} ± {std_ap:.3f})",
        )
        ax.fill_between(
            recall_grid,
            np.clip(mean_prec - std_prec, 0, 1),
            np.clip(mean_prec + std_prec, 0, 1),
            color=OKABE_ITO[0],
            alpha=0.2,
        )

        configure_spine(ax)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or "Aggregate Precision-Recall Curve")
        ax.legend(loc="best", frameon=False)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))

    data = pd.DataFrame(
        {"recall": recall_grid, "mean_precision": mean_prec, "std_precision": std_prec}
    )
    return fig, data


def plot_aggregate_calibration(
    y_trues: np.ndarray | list[np.ndarray],
    y_probas: list[np.ndarray],
    n_bins: int = 10,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot an aggregate calibration curve averaged across multiple seeds.

    Computes each seed's calibration curve on a uniform bin grid, then
    plots the mean fraction of positives with a ±1 std band against the
    perfect calibration diagonal.

    Args:
        y_trues (np.ndarray | list[np.ndarray]): True binary labels. Either a
            single array broadcast to all seeds, or a list of per-seed arrays.
        y_probas (list[np.ndarray]): List of per-seed predicted probabilities
            for the positive class.
        n_bins (int, optional): Number of calibration bins. Defaults to 10.
        title (str | None, optional): Plot title. Defaults to
            "Aggregate Calibration Curve".
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to (6, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame
            with columns ``prob_bin``, ``mean_frac_positives``,
            ``std_frac_positives``.

    Examples:
        >>> fig, data = plot_aggregate_calibration(y_true, y_probas_list, n_bins=10)
        >>> fig.savefig("aggregate_calibration.png")
        >>> data.to_csv("calibration_data.csv", index=False)
    """
    if isinstance(y_trues, np.ndarray):
        y_trues = [y_trues] * len(y_probas)

    # Use uniform mean_predicted_value grid for consistent alignment
    prob_grid = np.linspace(0, 1, n_bins)
    fracs: list[np.ndarray] = []

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for y_true, y_proba in zip(y_trues, y_probas, strict=False):
            frac, mean_pred = calibration_curve(
                y_true, y_proba, n_bins=n_bins, strategy="uniform"
            )
            frac_interp = np.interp(prob_grid, mean_pred, frac)
            fracs.append(frac_interp)

        mean_frac = np.mean(fracs, axis=0)
        std_frac = np.std(fracs, axis=0)

        ax.plot(
            prob_grid,
            mean_frac,
            marker="s",
            markersize=5,
            linewidth=2.0,
            color=OKABE_ITO[4],
            label="Mean Model",
        )
        ax.fill_between(
            prob_grid,
            np.clip(mean_frac - std_frac, 0, 1),
            np.clip(mean_frac + std_frac, 0, 1),
            color=OKABE_ITO[4],
            alpha=0.2,
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color=NATURE_COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label="Perfectly Calibrated",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title(title or "Aggregate Calibration Curve")
        ax.legend(loc="best", frameon=False)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.0))

    data = pd.DataFrame(
        {
            "prob_bin": prob_grid,
            "mean_frac_positives": mean_frac,
            "std_frac_positives": std_frac,
        }
    )
    return fig, data


def plot_aggregate_prediction_distribution(
    y_trues: np.ndarray | list[np.ndarray],
    y_probas: list[np.ndarray],
    title: str | None = None,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot overlaid KDE distributions of predicted probabilities across all seeds.

    Pools predicted probabilities from all seeds by true class and plots
    kernel density estimates. This gives a smooth ensemble-level view of
    model discrimination.

    Args:
        y_trues (np.ndarray | list[np.ndarray]): True binary labels. Either a
            single array broadcast to all seeds, or a list of per-seed arrays.
        y_probas (list[np.ndarray]): List of per-seed predicted probabilities
            for the positive class.
        title (str | None, optional): Plot title. Defaults to
            "Aggregate Prediction Distribution".
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to (8, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame
            with columns ``y_proba`` and ``y_true`` containing the pooled
            probabilities and labels from all seeds.

    Examples:
        >>> fig, data = plot_aggregate_prediction_distribution(y_true, y_probas_list)
        >>> fig.savefig("aggregate_pred_dist.png")
        >>> data.to_csv("prediction_distribution_data.csv", index=False)
    """
    if isinstance(y_trues, np.ndarray):
        y_trues = [y_trues] * len(y_probas)

    all_proba_0: list[float] = []
    all_proba_1: list[float] = []

    for y_true, y_proba in zip(y_trues, y_probas, strict=False):
        all_proba_0.extend(y_proba[y_true == 0].tolist())
        all_proba_1.extend(y_proba[y_true == 1].tolist())

    proba_0 = np.array(all_proba_0)
    proba_1 = np.array(all_proba_1)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        sns.kdeplot(
            proba_0,
            ax=ax,
            color=OKABE_ITO[4],
            fill=True,
            alpha=0.4,
            linewidth=2.0,
            label=f"Not Erupted (n={len(proba_0)})",
        )
        sns.kdeplot(
            proba_1,
            ax=ax,
            color=OKABE_ITO[0],
            fill=True,
            alpha=0.4,
            linewidth=2.0,
            label=f"Erupted (n={len(proba_1)})",
        )
        ax.axvline(
            ERUPTION_PROBABILITY_THRESHOLD,
            color=NATURE_COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label=f"Threshold ({ERUPTION_PROBABILITY_THRESHOLD})",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Density")
        ax.set_title(title or "Aggregate Prediction Distribution")
        ax.legend(loc="best", frameon=False)
        ax.set_xlim((0.0, 1.0))

    pooled_probas = np.concatenate([proba_0, proba_1])
    pooled_trues = np.concatenate([np.zeros(len(proba_0)), np.ones(len(proba_1))])
    data = pd.DataFrame({"y_proba": pooled_probas, "y_true": pooled_trues.astype(int)})
    return fig, data


def plot_aggregate_confusion_matrix(
    y_trues: np.ndarray | list[np.ndarray],
    y_preds: list[np.ndarray],
    normalize: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot a summed confusion matrix across all seeds.

    Accumulates raw confusion matrices from every seed, then optionally
    normalises the result and displays it as a heatmap. The returned DataFrame
    always contains raw (un-normalised) counts so the data can be re-analysed
    independently of the display normalization.

    Args:
        y_trues (np.ndarray | list[np.ndarray]): True binary labels. Either a
            single array broadcast to all seeds, or a list of per-seed arrays.
        y_preds (list[np.ndarray]): List of per-seed binary predictions.
        normalize (str | None, optional): Normalisation mode: ``"true"``,
            ``"pred"``, ``"all"``, or None (raw counts). Defaults to None.
        title (str | None, optional): Plot title. Defaults to
            "Aggregate Confusion Matrix".
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to (6, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame
            containing the raw summed confusion matrix with index and columns
            ``["not_erupted", "erupted"]``.

    Examples:
        >>> fig, data = plot_aggregate_confusion_matrix(y_true, y_preds_list)
        >>> fig.savefig("aggregate_cm.png")
        >>> data.to_csv("confusion_matrix_data.csv")
    """
    if isinstance(y_trues, np.ndarray):
        y_trues = [y_trues] * len(y_preds)

    # Sum raw confusion matrices across all seeds
    cm_sum = np.zeros((2, 2), dtype=np.int64)
    for y_true, y_pred in zip(y_trues, y_preds, strict=False):
        cm_sum += confusion_matrix(y_true, y_pred)

    labels = CLASS_LABELS

    if normalize == "true":
        cm_display = cm_sum.astype(float) / cm_sum.sum(axis=1, keepdims=True)
        fmt = ".2f"
    elif normalize == "pred":
        cm_display = cm_sum.astype(float) / cm_sum.sum(axis=0, keepdims=True)
        fmt = ".2f"
    elif normalize == "all":
        cm_display = cm_sum.astype(float) / cm_sum.sum()
        fmt = ".2f"
    else:
        cm_display = cm_sum
        fmt = "d"

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.heatmap(
            cm_display,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Normalized" if normalize else "Count (summed)"},
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title or "Aggregate Confusion Matrix")

    # Always save raw counts regardless of display normalization
    data = pd.DataFrame(
        cm_sum,
        index=["not_erupted", "erupted"],
        columns=["not_erupted", "erupted"],
    )
    return fig, data


def plot_aggregate_threshold_analysis(
    y_trues: np.ndarray | list[np.ndarray],
    y_probas: list[np.ndarray],
    show_individual: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot mean metric curves vs. decision threshold across multiple seeds.

    Sweeps thresholds from 0 to 1 and computes F1, precision, recall, and
    balanced accuracy per seed. Plots the mean of each metric with a ±1 std
    confidence band.

    Args:
        y_trues (np.ndarray | list[np.ndarray]): True binary labels. Either a
            single array broadcast to all seeds, or a list of per-seed arrays.
        y_probas (list[np.ndarray]): List of per-seed predicted probabilities
            for the positive class.
        show_individual (bool, optional): Draw individual seed curves as thin
            background lines. Defaults to True.
        title (str | None, optional): Plot title. Defaults to
            "Aggregate Threshold Analysis".
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to (10, 6).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame
            with columns ``threshold``, ``mean_precision``, ``std_precision``,
            ``mean_recall``, ``std_recall``, ``mean_f1``, ``std_f1``,
            ``mean_balanced_accuracy``, ``std_balanced_accuracy``.

    Examples:
        >>> fig, data = plot_aggregate_threshold_analysis(y_true, y_probas_list)
        >>> fig.savefig("aggregate_threshold.png")
        >>> data.to_csv("threshold_data.csv", index=False)
    """
    thresholds = np.linspace(0, 1, 101)
    metric_keys = ["precision", "recall", "f1", "balanced_accuracy"]
    colors = {
        "precision": OKABE_ITO[4],
        "recall": OKABE_ITO[0],
        "f1": OKABE_ITO[2],
        "balanced_accuracy": OKABE_ITO[6],
    }
    labels_map = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
        "balanced_accuracy": "Balanced Accuracy",
    }

    if isinstance(y_trues, np.ndarray):
        y_trues = [y_trues] * len(y_probas)

    # all_curves[key] is a list of 1-D arrays, one per seed
    all_curves: dict[str, list[np.ndarray]] = {k: [] for k in metric_keys}

    for y_true, y_proba in zip(y_trues, y_probas, strict=False):
        curves: dict[str, list[float]] = {k: [] for k in metric_keys}
        for t in thresholds:
            y_pred_t = (y_proba >= t).astype(int)
            curves["precision"].append(
                precision_score(y_true, y_pred_t, zero_division=0)
            )
            curves["recall"].append(recall_score(y_true, y_pred_t, zero_division=0))
            curves["f1"].append(f1_score(y_true, y_pred_t, zero_division=0))
            curves["balanced_accuracy"].append(
                balanced_accuracy_score(y_true, y_pred_t)
            )
        for k in metric_keys:
            all_curves[k].append(np.array(curves[k]))

    # Compute mean and std per metric for the data DataFrame
    mean_curves: dict[str, np.ndarray] = {}
    std_curves: dict[str, np.ndarray] = {}
    for key in metric_keys:
        matrix = np.stack(all_curves[key], axis=0)
        mean_curves[key] = matrix.mean(axis=0)
        std_curves[key] = matrix.std(axis=0)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        for key in metric_keys:
            mean_curve = mean_curves[key]
            std_curve = std_curves[key]

            if show_individual:
                for seed_curve in all_curves[key]:
                    ax.plot(
                        thresholds,
                        seed_curve,
                        color=colors[key],
                        alpha=0.10,
                        linewidth=0.5,
                    )

            ax.plot(
                thresholds,
                mean_curve,
                color=colors[key],
                linewidth=2.0,
                label=labels_map[key],
            )
            ax.fill_between(
                thresholds,
                np.clip(mean_curve - std_curve, 0, 1),
                np.clip(mean_curve + std_curve, 0, 1),
                color=colors[key],
                alpha=0.15,
            )

        ax.axvline(
            ERUPTION_PROBABILITY_THRESHOLD,
            color=NATURE_COLORS["gray"],
            linestyle=":",
            linewidth=1.5,
            label=r"Default ({ERUPTION_PROBABILITY_THRESHOLD})",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Score")
        ax.set_title(title or "Aggregate Threshold Analysis")
        ax.legend(loc="best", frameon=False, ncol=2)
        ax.set_xlim((0.0, 1.0))
        ax.set_ylim((0.0, 1.05))

    data = pd.DataFrame(
        {
            "threshold": thresholds,
            "mean_precision": mean_curves["precision"],
            "std_precision": std_curves["precision"],
            "mean_recall": mean_curves["recall"],
            "std_recall": std_curves["recall"],
            "mean_f1": mean_curves["f1"],
            "std_f1": std_curves["f1"],
            "mean_balanced_accuracy": mean_curves["balanced_accuracy"],
            "std_balanced_accuracy": std_curves["balanced_accuracy"],
        }
    )
    return fig, data


def plot_aggregate_feature_importance(
    models: list[BaseEstimator],
    feature_names: list[str],
    top_n: int = 20,
    title: str | None = None,
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame] | None:
    """Plot mean feature importances across multiple seeds with ±1 std error bars.

    Extracts feature importances from each seed's model, computes the mean and
    standard deviation across seeds, and displays a horizontal bar chart for the
    top-N features. Returns None if no model exposes feature importances. The
    returned DataFrame contains all features (not just top-N) so users can
    re-filter or re-rank without recomputing.

    Args:
        models (list[BaseEstimator]): List of fitted scikit-learn estimators.
            Supports tree-based models and VotingClassifier.
        feature_names (list[str]): Feature names matching the model's input
            columns.
        top_n (int, optional): Number of top features to display. Defaults to 20.
        title (str | None, optional): Plot title. Defaults to
            "Aggregate Top-{top_n} Feature Importances".
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame] | None: Matplotlib figure and a
            DataFrame with columns ``feature``, ``mean_importance``,
            ``std_importance`` sorted by descending mean importance
            (all features, not just top-N). Returns None if no model exposes
            ``feature_importances_``.

    Examples:
        >>> result = plot_aggregate_feature_importance(models_list, feature_names)
        >>> if result:
        ...     fig, data = result
        ...     fig.savefig("aggregate_fi.png")
        ...     data.to_csv("feature_importance_data.csv", index=False)
    """
    all_importances: list[np.ndarray] = []

    for model in models:
        importances: np.ndarray | None = None
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_  # type: ignore[assignment]
        elif isinstance(model, VotingClassifier):
            sub_imps = [
                est.feature_importances_
                for est in model.estimators_
                if hasattr(est, "feature_importances_")
            ]
            if sub_imps:
                importances = np.mean(sub_imps, axis=0)

        if importances is not None:
            all_importances.append(importances)

    if not all_importances:
        return None

    stacked = np.stack(all_importances, axis=0)  # shape (n_seeds, n_features)
    mean_imp = stacked.mean(axis=0)
    std_imp = stacked.std(axis=0)

    # Select top-N by mean importance for display
    indices = np.argsort(mean_imp)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_mean = mean_imp[indices]
    top_std = std_imp[indices]

    # Reverse for bottom-to-top display
    top_features = top_features[::-1]
    top_mean = top_mean[::-1]
    top_std = top_std[::-1]

    figheight = max(4, top_n * 0.35)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=(8, figheight), dpi=dpi)

        ax.barh(
            range(len(top_features)),
            top_mean,
            xerr=top_std,
            color=OKABE_ITO[4],
            alpha=0.8,
            error_kw={"ecolor": NATURE_COLORS["gray"], "capsize": 3, "linewidth": 1.0},
        )
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)

        configure_spine(ax)
        ax.set_xlabel("Mean Importance Score ± Std")
        ax.set_ylabel("Feature")
        ax.set_title(title or f"Aggregate Top-{top_n} Feature Importances")

    # Save all features (not just top-N) for downstream filtering
    all_sorted_idx = np.argsort(mean_imp)[::-1]
    data = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in all_sorted_idx],
            "mean_importance": mean_imp[all_sorted_idx],
            "std_importance": std_imp[all_sorted_idx],
        }
    )
    return fig, data


# ---------------------------------------------------------------------------
# Cross-classifier comparison plots
# ---------------------------------------------------------------------------

_DEFAULT_METRICS = [
    "balanced_accuracy",
    "f1_score",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
]


def plot_classifier_comparison(
    metrics_by_classifier: dict[str, list[dict[str, Any]]],
    metrics_to_show: list[str] | None = None,
    figsize: tuple[float, float] = (12, 5),
    dpi: int = 150,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot a heatmap comparing metrics across multiple classifiers.

    Computes mean and standard deviation of each metric across seeds for
    every classifier, then renders a heatmap where each cell shows
    ``mean ± std``. Classifiers are sorted by mean F1 descending.

    Args:
        metrics_by_classifier (dict[str, list[dict[str, Any]]]): Mapping
            from classifier name to a list of per-seed metrics dicts.
            Each dict must contain numeric keys matching ``metrics_to_show``.
        metrics_to_show (list[str] | None, optional): Metric keys to include
            as heatmap columns. Defaults to
            ``["balanced_accuracy", "f1_score", "precision", "recall",
            "roc_auc", "pr_auc"]``.
        figsize (tuple[float, float], optional): Figure size in inches.
            Defaults to (12, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. If None, uses
            "Classifier Comparison". Defaults to None.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a summary
            DataFrame with MultiIndex ``(classifier, metric)`` and columns
            ``mean`` and ``std``.

    Examples:
        >>> metrics_by_clf = {
        ...     "rf":  [{"f1_score": 0.8, "precision": 0.75, ...}, ...],
        ...     "xgb": [{"f1_score": 0.85, "precision": 0.80, ...}, ...],
        ... }
        >>> fig, df = plot_classifier_comparison(metrics_by_clf)
        >>> fig.savefig("classifier_comparison.png")
        >>> df.to_csv("comparison_summary.csv")
    """
    if metrics_to_show is None:
        metrics_to_show = _DEFAULT_METRICS

    # Build summary: mean and std per (classifier, metric)
    records: list[dict[str, Any]] = []
    for clf_name, seeds_list in metrics_by_classifier.items():
        for metric in metrics_to_show:
            values = [s[metric] for s in seeds_list if metric in s]
            if values:
                records.append(
                    {
                        "classifier": clf_name,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                )

    summary_df = pd.DataFrame(records).set_index(["classifier", "metric"])

    # Build mean matrix (rows = classifiers, cols = metrics)
    classifiers = list(metrics_by_classifier.keys())

    # Sort classifiers by mean F1
    def _mean_f1(classifier: str) -> float:
        """Return the mean F1 score for a classifier from the summary DataFrame.

        Used as a sort key to order classifiers by descending mean F1 score.
        Returns 0.0 when the classifier has no F1 entry in the summary.

        Args:
            classifier (str): Classifier name as it appears in the summary_df index.

        Returns:
            float: Mean F1 score, or 0.0 if the entry is missing.
        """
        try:
            return float(summary_df.loc[(classifier, "f1_score"), "mean"])
        except KeyError:
            return 0.0

    classifiers = sorted(classifiers, key=_mean_f1, reverse=True)

    mean_matrix = pd.DataFrame(index=classifiers, columns=metrics_to_show, dtype=float)
    std_matrix = pd.DataFrame(index=classifiers, columns=metrics_to_show, dtype=float)
    annot_matrix = pd.DataFrame(index=classifiers, columns=metrics_to_show, dtype=str)

    for clf in classifiers:
        for metric in metrics_to_show:
            try:
                m = float(summary_df.loc[(clf, metric), "mean"])
                s = float(summary_df.loc[(clf, metric), "std"])
            except KeyError:
                m, s = float("nan"), 0.0
            mean_matrix.loc[clf, metric] = m
            std_matrix.loc[clf, metric] = s
            annot_matrix.loc[clf, metric] = f"{m:.3f}\n±{s:.3f}"

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        sns.heatmap(
            mean_matrix.astype(float),
            annot=annot_matrix,
            fmt="",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            ax=ax,
            linewidths=0.5,
            cbar_kws={"label": "Mean Score"},
            annot_kws={"size": 8},
        )
        ax.set_xlabel("Metric")
        ax.set_ylabel("Classifier")
        ax.set_title(title or "Classifier Comparison")

    return fig, summary_df


def plot_learning_curve(
    json_filepath: str,
    plot_filepath: str,
    scoring: str = "balanced_accuracy",
    overwrite: bool = False,
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Load a learning-curve JSON and render it as a train vs. validation plot.

    Reads the per-seed JSON file produced by
    ``ModelTrainer._compute_all_learning_curves``, then renders train and
    validation scores for the requested ``scoring`` metric as a function of
    training-set size with ±1 std shaded bands around each line.

    Backward-compatible: if the JSON uses the old flat format (no ``"metrics"``
    key), it falls back to reading the top-level score keys directly.

    Args:
        json_filepath (str): Path to the learning-curve JSON file.
        plot_filepath (str): Destination path for the saved PNG file.
        scoring (str, optional): Scoring metric to plot. Must be one of the
            keys in the ``"metrics"`` dict. Defaults to
            ``"balanced_accuracy"``.
        overwrite (bool, optional): When False, skip saving if the PNG already
            exists. Defaults to False.
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame with
            columns ``train_sizes``, ``train_scores_mean``, ``train_scores_std``,
            ``test_scores_mean``, ``test_scores_std``.
    """
    with open(json_filepath) as f:
        data = json.load(f)

    train_sizes = np.array(data["train_sizes"])

    # Support both new multi-metric format and legacy flat format.
    if "metrics" in data:
        scores = data["metrics"][scoring]
    else:
        scores = data

    train_mean = np.array(scores["train_scores_mean"])
    train_std = np.array(scores["train_scores_std"])
    test_mean = np.array(scores["test_scores_mean"])
    test_std = np.array(scores["test_scores_std"])

    ylabel = scoring.replace("_", " ").title()

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

        ax.plot(
            train_sizes, train_mean, color=OKABE_ITO[0], linewidth=2.0, label="Train"
        )
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color=OKABE_ITO[0],
        )

        ax.plot(
            train_sizes,
            test_mean,
            color=OKABE_ITO[1],
            linewidth=2.0,
            linestyle="--",
            label="Validation",
        )
        ax.fill_between(
            train_sizes,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.2,
            color=OKABE_ITO[1],
        )

        ax.set_xlabel("Training set size")
        ax.set_ylabel(ylabel)
        ax.set_title("Learning Curve")
        ax.legend(frameon=False, fontsize=8)
        configure_spine(ax)

    df = pd.DataFrame(
        {
            "train_sizes": train_sizes,
            "train_scores_mean": train_mean,
            "train_scores_std": train_std,
            "test_scores_mean": test_mean,
            "test_scores_std": test_std,
        }
    )

    if overwrite or not os.path.isfile(plot_filepath):
        ensure_dir(os.path.dirname(plot_filepath))
        fig.savefig(plot_filepath, dpi=dpi, bbox_inches="tight")

    return fig, df


def plot_aggregate_learning_curve(
    all_data: list[dict],
    filepath: str,
    scoring: str = "balanced_accuracy",
    overwrite: bool = False,
    dpi: int = 150,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot aggregate learning curves across multiple seeds with mean ± std bands.

    Interpolates each seed's learning curve onto a shared training-size grid,
    then plots mean train and validation score lines with ±1 std shaded bands.

    Backward-compatible: if dicts use the old flat format (no ``"metrics"``
    key), it falls back to reading the top-level score keys directly.

    Args:
        all_data (list[dict]): List of per-seed dicts in the new multi-metric
            format (keys: ``train_sizes``, ``metrics``) or the legacy flat
            format (keys: ``train_sizes``, ``train_scores_mean``, etc.).
        filepath (str): Destination path for the saved PNG file.
        scoring (str, optional): Scoring metric to plot. Must be one of the
            keys in each seed's ``"metrics"`` dict. Defaults to
            ``"balanced_accuracy"``.
        overwrite (bool, optional): When False, skip saving if the file already
            exists. Defaults to False.
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a DataFrame with
            columns ``train_sizes``, ``mean_train``, ``std_train``, ``mean_val``,
            ``std_val``.
    """
    # Build shared grid from the largest train_sizes array
    max_sizes = max(len(d["train_sizes"]) for d in all_data)
    size_grid = np.linspace(
        max(d["train_sizes"][0] for d in all_data),
        min(d["train_sizes"][-1] for d in all_data),
        max_sizes,
    )

    train_interps: list[np.ndarray] = []
    test_interps: list[np.ndarray] = []

    for d in all_data:
        sizes = np.array(d["train_sizes"])
        # Support both new multi-metric format and legacy flat format.
        scores = d["metrics"][scoring] if "metrics" in d else d
        train_interps.append(np.interp(size_grid, sizes, scores["train_scores_mean"]))
        test_interps.append(np.interp(size_grid, sizes, scores["test_scores_mean"]))

    mean_train = np.mean(train_interps, axis=0)
    std_train = np.std(train_interps, axis=0)
    mean_test = np.mean(test_interps, axis=0)
    std_test = np.std(test_interps, axis=0)

    ylabel = scoring.replace("_", " ").title()

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)

        ax.plot(size_grid, mean_train, color=OKABE_ITO[0], linewidth=2.0, label="Train")
        ax.fill_between(
            size_grid,
            mean_train - std_train,
            mean_train + std_train,
            alpha=0.2,
            color=OKABE_ITO[0],
        )

        ax.plot(
            size_grid,
            mean_test,
            color=OKABE_ITO[1],
            linewidth=2.0,
            linestyle="--",
            label="Validation",
        )
        ax.fill_between(
            size_grid,
            mean_test - std_test,
            mean_test + std_test,
            alpha=0.2,
            color=OKABE_ITO[1],
        )

        ax.set_xlabel("Training set size")
        ax.set_ylabel(ylabel)
        ax.set_title("Aggregate Learning Curve")
        ax.legend(frameon=False, fontsize=8)
        configure_spine(ax)

    df = pd.DataFrame(
        {
            "train_sizes": size_grid,
            "mean_train": mean_train,
            "std_train": std_train,
            "mean_val": mean_test,
            "std_val": std_test,
        }
    )

    if overwrite or not os.path.isfile(filepath):
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")

    return fig, df


def plot_learning_curve_grid(
    json_filepath: str,
    plot_filepath: str,
    scorings: list[str],
    overwrite: bool = False,
    dpi: int = 150,
) -> plt.Figure:
    """Plot all scoring metrics as side-by-side subplots in a single figure.

    Reads the multi-metric learning-curve JSON produced by
    ``ModelTrainer._compute_all_learning_curves`` and renders one subplot per
    scoring metric, each with train and validation lines and ±1 std shaded
    bands.

    Backward-compatible: if the JSON uses the old flat format (no ``"metrics"``
    key), only one subplot is rendered using the top-level score keys.

    Args:
        json_filepath (str): Path to the learning-curve JSON file.
        plot_filepath (str): Destination path for the saved PNG file.
        scorings (list[str]): Ordered list of scoring keys to plot
            (e.g. ``["balanced_accuracy", "recall", "f1_weighted"]``).
        overwrite (bool, optional): When False, skip saving if the file already
            exists. Defaults to False.
        dpi (int, optional): Figure resolution in dots per inch. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure containing one subplot per scoring metric.
    """
    with open(json_filepath) as f:
        data = json.load(f)

    train_sizes = np.array(data["train_sizes"])
    n = len(scorings)

    with apply_nature_style():
        fig, axes = plt.subplots(
            1, n, figsize=(5 * n, 5), dpi=dpi, layout="constrained"
        )
        if n == 1:
            axes = [axes]

        for ax, scoring in zip(axes, scorings, strict=True):
            scores = data["metrics"][scoring] if "metrics" in data else data

            train_mean = np.array(scores["train_scores_mean"])
            train_std = np.array(scores["train_scores_std"])
            test_mean = np.array(scores["test_scores_mean"])
            test_std = np.array(scores["test_scores_std"])
            ylabel = scoring.replace("_", " ").title()

            ax.plot(
                train_sizes,
                train_mean,
                color=OKABE_ITO[0],
                linewidth=2.0,
                label="Train",
            )
            ax.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.2,
                color=OKABE_ITO[0],
            )
            ax.plot(
                train_sizes,
                test_mean,
                color=OKABE_ITO[1],
                linewidth=2.0,
                linestyle="--",
                label="Validation",
            )
            ax.fill_between(
                train_sizes,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.2,
                color=OKABE_ITO[1],
            )

            ax.set_xlabel("Training set size")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Learning Curve — {ylabel}")
            ax.legend(frameon=False, fontsize=8)
            configure_spine(ax)

    if overwrite or not os.path.isfile(plot_filepath):
        ensure_dir(os.path.dirname(plot_filepath))
        fig.savefig(plot_filepath, dpi=dpi, bbox_inches="tight")

    return fig


def plot_seed_stability(
    metrics_by_classifier: dict[str, list[dict[str, Any]]],
    metric: str = "f1_score",
    figsize: tuple[float, float] | None = None,
    dpi: int = 150,
    title: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot a violin plot showing metric stability across random seeds per classifier.

    Renders one violin per classifier with individual seed dots overlaid as a
    jittered strip plot. Classifiers are sorted by median descending. The mean
    across seeds is marked with a horizontal white line.

    Args:
        metrics_by_classifier (dict[str, list[dict[str, Any]]]): Mapping
            from classifier name to a list of per-seed metrics dicts.
        metric (str, optional): Metric key to plot. Must be present in each
            seed dict. Defaults to ``"f1_score"``.
        figsize (tuple[float, float] | None, optional): Figure size in inches.
            If None, auto-sizes based on number of classifiers using
            ``(10, max(5, n * 0.6))``. Defaults to None.
        dpi (int, optional): Figure resolution. Defaults to 150.
        title (str | None, optional): Plot title. If None, uses
            ``"Seed Stability — {metric}"``. Defaults to None.

    Returns:
        tuple[plt.Figure, pd.DataFrame]: Matplotlib figure and a long-form
            DataFrame with columns ``classifier``, ``seed_idx``, and
            ``value``.

    Examples:
        >>> metrics_by_clf = {
        ...     "rf":  [{"f1_score": 0.8}, {"f1_score": 0.82}, ...],
        ...     "xgb": [{"f1_score": 0.85}, {"f1_score": 0.84}, ...],
        ... }
        >>> fig, df = plot_seed_stability(metrics_by_clf, metric="balanced_accuracy")
        >>> fig.savefig("seed_stability.png")
    """
    # Build long-form DataFrame
    rows: list[dict[str, Any]] = []
    for clf_name, seeds_list in metrics_by_classifier.items():
        for seed_idx, seed_metrics in enumerate(seeds_list):
            if metric in seed_metrics:
                rows.append(
                    {
                        "classifier": clf_name,
                        "seed_idx": seed_idx,
                        "value": seed_metrics[metric],
                    }
                )
    long_df = pd.DataFrame(rows)

    # Sort classifiers by median descending
    order = (
        long_df.groupby("classifier")["value"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    n_classifiers = len(order)
    if figsize is None:
        figsize = (10, max(5, n_classifiers * 0.6))

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        sns.violinplot(
            data=long_df,
            x="classifier",
            y="value",
            order=order,
            ax=ax,
            color=OKABE_ITO[4],
            alpha=0.7,
            inner=None,
        )
        sns.stripplot(
            data=long_df,
            x="classifier",
            y="value",
            order=order,
            ax=ax,
            color="black",
            size=3,
            alpha=0.5,
            jitter=True,
        )

        # Mark mean per classifier
        for i, clf in enumerate(order):
            mean_val = long_df.loc[long_df["classifier"] == clf, "value"].mean()
            ax.hlines(mean_val, i - 0.3, i + 0.3, colors="white", linewidths=1.0)

        configure_spine(ax)
        ax.set_xlabel("Classifier")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title or f"Seed Stability — {metric}")

    return fig, long_df
