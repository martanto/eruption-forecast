"""Metrics computation for model evaluation.

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
    matthews_corrcoef,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
)

from eruption_forecast.utils.ml import compute_g_mean, compute_threshold_metrics


class MetricsComputer:
    """Computes evaluation metrics for binary classification models.

    This class encapsulates all metric calculation logic, providing a clean
    interface for computing standard classification metrics, ROC/PR curves,
    and optimal threshold analysis.

    Attributes:
        y_true (np.ndarray): Ground truth binary labels (0 or 1).
        y_proba (np.ndarray): Predicted probabilities for the positive class.
        y_pred (np.ndarray): Binary predictions (0 or 1) at default 0.5 threshold.

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
        """Initialize MetricsComputer with ground truth and prediction arrays.

        Args:
            y_true (np.ndarray): Ground truth binary labels (0 or 1).
            y_proba (np.ndarray): Predicted probabilities for the positive class.
            y_pred (np.ndarray): Binary predictions (0 or 1) at threshold 0.5.
        """
        self.y_true = np.asarray(y_true)
        self.y_proba = np.asarray(y_proba)
        self.y_pred = np.asarray(y_pred)

    def compute_all_metrics(self) -> dict[str, float]:
        """Compute all standard classification metrics.

        Returns:
            dict[str, float]: Dictionary containing:
                - ``accuracy``, ``balanced_accuracy``
                - ``precision``, ``recall``, ``f1_score``
                - ``sensitivity``, ``specificity``
                - ``g_mean`` (float): G-mean at the default 0.5 threshold.
                - ``roc_auc``, ``pr_auc``, ``average_precision``
                - ``mcc``
                - ``true_positives``, ``true_negatives``, ``false_positives``, ``false_negatives``
                - ``optimal_threshold``, ``f1_at_optimal``, ``recall_at_optimal``, ``precision_at_optimal``
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

        # G-mean at default (0.5) threshold
        metrics["g_mean"] = float(
            compute_g_mean(
                {
                    "recall": [metrics["sensitivity"]],
                    "specificity": [metrics["specificity"]],
                }
            )[0]
        )

        # AUC metrics
        metrics["roc_auc"] = roc_auc_score(self.y_true, self.y_proba)
        precision_curve, recall_curve, _ = precision_recall_curve(
            self.y_true, self.y_proba
        )
        metrics["pr_auc"] = auc(recall_curve, precision_curve)
        metrics["average_precision"] = average_precision_score(
            self.y_true, self.y_proba
        )

        # MCC — best single metric for imbalanced binary classification
        metrics["mcc"] = matthews_corrcoef(self.y_true, self.y_pred)

        # Optimal threshold metrics
        optimal_metrics = self.optimize_threshold()
        metrics.update(optimal_metrics)

        return metrics

    def optimize_threshold(self) -> dict[str, float]:
        """Find optimal classification threshold by maximizing G-mean.

        G-mean (geometric mean of sensitivity and specificity) is the preferred
        objective for eruption forecasting — it equally penalizes missing eruptions
        and false alarms without being inflated by class imbalance, unlike F1.

        Returns:
            dict[str, float]: Dictionary with keys:
                - ``optimal_threshold`` (float): Threshold that maximizes G-mean.
                - ``g_mean_at_optimal`` (float): G-mean at optimal threshold.
                - ``f1_at_optimal`` (float): F1 score at optimal threshold.
                - ``recall_at_optimal`` (float): Recall at optimal threshold.
                - ``precision_at_optimal`` (float): Precision at optimal threshold.
                - ``balanced_accuracy_at_optimal`` (float): Balanced accuracy at optimal threshold.
        """
        thresholds, metrics = compute_threshold_metrics(self.y_true, self.y_proba)

        g_mean = compute_g_mean(metrics)
        optimal_idx = np.argmax(g_mean)
        optimal_threshold = float(thresholds[optimal_idx])

        return {
            "optimal_threshold": optimal_threshold,
            "g_mean_at_optimal": float(g_mean[optimal_idx]),
            "f1_at_optimal": float(metrics["f1"][optimal_idx]),
            "recall_at_optimal": float(metrics["recall"][optimal_idx]),
            "precision_at_optimal": float(metrics["precision"][optimal_idx]),
            "balanced_accuracy_at_optimal": float(
                metrics["balanced_accuracy"][optimal_idx]
            ),
        }
