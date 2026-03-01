"""
Metrics computation for model evaluation.

This module extracts the metrics calculation logic from ModelEvaluator
to follow the Single Responsibility Principle.
"""

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    precision_recall_curve,
    balanced_accuracy_score,
)

from eruption_forecast.utils.ml import compute_threshold_metrics


class MetricsComputer:
    """
    Computes evaluation metrics for binary classification models.

    This class encapsulates all metric calculation logic, providing a clean
    interface for computing standard classification metrics, ROC/PR curves,
    and optimal threshold analysis.

    Attributes:
        y_true: Ground truth binary labels (0 or 1).
        y_proba: Predicted probabilities for the positive class.
        y_pred: Binary predictions (0 or 1) at default 0.5 threshold.

    Examples:
        >>> computer = MetricsComputer(y_true, y_proba, y_pred)
        >>> metrics = computer.compute_all_metrics()
        >>> print(metrics["accuracy"])
        0.85
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: np.ndarray,
    ):
        """
        Initialize MetricsComputer.

        Args:
            y_true: Ground truth binary labels.
            y_proba: Predicted probabilities for positive class.
            y_pred: Binary predictions at threshold 0.5.
        """
        self.y_true = np.asarray(y_true)
        self.y_proba = np.asarray(y_proba)
        self.y_pred = np.asarray(y_pred)

    def compute_all_metrics(self) -> dict[str, float]:
        """
        Compute all standard classification metrics.

        Returns:
            Dictionary containing:
                - accuracy, balanced_accuracy
                - precision, recall, f1_score
                - sensitivity, specificity
                - roc_auc, pr_auc
                - true_positives, true_negatives, false_positives, false_negatives
                - optimal_threshold, f1_at_optimal, recall_at_optimal, precision_at_optimal
        """
        metrics = {
            "accuracy": accuracy_score(self.y_true, self.y_pred),
            "balanced_accuracy": balanced_accuracy_score(self.y_true, self.y_pred),
            "precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "f1_score": f1_score(self.y_true, self.y_pred, zero_division=0),
        }

        # Basic metrics

        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        # Sensitivity and specificity
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # AUC metrics
        metrics["roc_auc"] = roc_auc_score(self.y_true, self.y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(
            self.y_true, self.y_proba
        )
        metrics["pr_auc"] = auc(recall_curve, precision_curve)

        # Optimal threshold metrics
        optimal_metrics = self.optimize_threshold()
        metrics.update(optimal_metrics)

        return metrics

    def optimize_threshold(self) -> dict[str, float]:
        """
        Find optimal classification threshold by maximizing F1 score.

        Returns:
            Dictionary with:
                - optimal_threshold: Threshold that maximizes F1
                - f1_at_optimal: F1 score at optimal threshold
                - recall_at_optimal: Recall at optimal threshold
                - precision_at_optimal: Precision at optimal threshold
        """
        thresholds, metrics = compute_threshold_metrics(self.y_true, self.y_proba)

        # Find optimal threshold
        optimal_idx = np.argmax(metrics["f1"])

        return {
            "optimal_threshold": float(thresholds[optimal_idx]),
            "f1_at_optimal": float(metrics["f1"][optimal_idx]),
            "recall_at_optimal": float(metrics["recall"][optimal_idx]),
            "precision_at_optimal": float(metrics["precision"][optimal_idx]),
            "balanced_accuracy_at_optimal": float(
                metrics["balanced_accuracy"][optimal_idx]
            ),
        }
