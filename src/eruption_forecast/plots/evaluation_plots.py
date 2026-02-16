"""Model evaluation plots with Nature/Science journal styling.

This module provides publication-quality visualizations for ML model evaluation:
- Confusion matrices
- ROC and Precision-Recall curves
- Threshold analysis
- Feature importance
- Calibration curves
- Prediction distributions

All plots follow Nature/Science journal standards with consistent styling.
"""

import numpy as np
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

from eruption_forecast.plots.styles import (
    OKABE_ITO,
    NATURE_COLORS,
    configure_spine,
    apply_nature_style,
)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: str | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot a confusion matrix heatmap with Nature/Science styling.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        normalize (str | None, optional): Normalization mode ("true", "pred",
            "all", or None). Defaults to None (raw counts).
        title (str | None, optional): Plot title. Defaults to "Confusion Matrix".
        figsize (tuple[float, float], optional): Figure size. Defaults to (6, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    labels = ["Not Erupted", "Erupted"]
    fmt = ".2f" if normalize else "d"

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize)
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
        plt.tight_layout()

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str | None = None,
    figsize: tuple[float, float] = (6, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plot ROC curve with AUC annotation and Nature/Science styling.

    Args:
        y_true (np.ndarray): True labels.
        y_proba (np.ndarray): Predicted probabilities for positive class.
        title (str | None, optional): Plot title. Defaults to "ROC Curve".
        figsize (tuple[float, float], optional): Figure size. Defaults to (6, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize)

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
            label="Random Classifier",
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

    Args:
        y_true (np.ndarray): True labels.
        y_proba (np.ndarray): Predicted probabilities for positive class.
        title (str | None, optional): Plot title. Defaults to
            "Precision-Recall Curve".
        figsize (tuple[float, float], optional): Figure size. Defaults to (6, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize)

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
    """Plot precision, recall, F1, and balanced accuracy vs. threshold.

    Args:
        y_true (np.ndarray): True labels.
        y_proba (np.ndarray): Predicted probabilities for positive class.
        title (str | None, optional): Plot title. Defaults to
            "Threshold Analysis".
        figsize (tuple[float, float], optional): Figure size. Defaults to (10, 6).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    thresholds = np.linspace(0, 1, 101)
    metrics = {"precision": [], "recall": [], "f1": [], "balanced_accuracy": []}

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        metrics["precision"].append(precision_score(y_true, y_pred_t, zero_division=0))
        metrics["recall"].append(recall_score(y_true, y_pred_t, zero_division=0))
        metrics["f1"].append(f1_score(y_true, y_pred_t, zero_division=0))
        metrics["balanced_accuracy"].append(balanced_accuracy_score(y_true, y_pred_t))

    # Find optimal threshold (max F1)
    optimal_idx = np.argmax(metrics["f1"])
    optimal_threshold = thresholds[optimal_idx]

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize)

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

        # Mark default threshold (0.5)
        ax.axvline(
            0.5,
            color=NATURE_COLORS["gray"],
            linestyle=":",
            linewidth=1.5,
            label="Default (0.5)",
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

    Args:
        model (BaseEstimator): Fitted model with feature_importances_ attribute.
        feature_names (list[str]): List of feature names.
        top_n (int, optional): Number of top features to display. Defaults to 20.
        title (str | None, optional): Plot title. Defaults to "Feature Importance".
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure | None: Matplotlib figure object, or None if model lacks
            feature_importances_.
    """
    # Extract importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
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
        fig, ax = plt.subplots(figsize=(8, figheight))

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

    Args:
        y_true (np.ndarray): True labels.
        y_proba (np.ndarray): Predicted probabilities for positive class.
        n_bins (int, optional): Number of bins for calibration. Defaults to 10.
        title (str | None, optional): Plot title. Defaults to "Calibration Curve".
        figsize (tuple[float, float], optional): Figure size. Defaults to (6, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="uniform"
    )

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize)

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

    Args:
        y_true (np.ndarray): True labels.
        y_proba (np.ndarray): Predicted probabilities for positive class.
        title (str | None, optional): Plot title. Defaults to
            "Prediction Distribution".
        figsize (tuple[float, float], optional): Figure size. Defaults to (8, 5).
        dpi (int, optional): Figure resolution. Defaults to 150.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    proba_0 = y_proba[y_true == 0]
    proba_1 = y_proba[y_true == 1]

    with apply_nature_style():
        fig, ax = plt.subplots(figsize=figsize)

        # Plot histograms for each class
        ax.hist(
            proba_0,
            bins=20,
            alpha=0.6,
            color=OKABE_ITO[4],  # Blue
            label=f"Not Erupted (n={len(proba_0)})",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.hist(
            proba_1,
            bins=20,
            alpha=0.6,
            color=OKABE_ITO[0],  # Orange
            label=f"Erupted (n={len(proba_1)})",
            edgecolor="black",
            linewidth=0.5,
        )

        # Mark 0.5 threshold
        ax.axvline(
            0.5,
            color=NATURE_COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label="Threshold (0.5)",
            alpha=0.7,
        )

        configure_spine(ax)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Count")
        ax.set_title(title or "Prediction Distribution by True Class")
        ax.legend(loc="best", frameon=False)
        ax.set_xlim((0.0, 1.0))

    return fig
