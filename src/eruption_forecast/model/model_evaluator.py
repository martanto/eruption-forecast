import os
from typing import Any

import numpy as np
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    f1_score,
    roc_curve,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import GridSearchCV

from eruption_forecast.logger import logger


class ModelEvaluator:
    """Evaluate a fitted classifier on a test set.

    Wraps a trained model and test data to provide metrics, a text summary,
    and plots. Predictions are computed once on construction and cached.

    Args:
        model: Fitted sklearn estimator or GridSearchCV (best_estimator_ is used).
        X_test: Test features.
        y_test: True test labels.
        model_name: Name used in plot titles and filenames. Defaults to "model".
        output_dir: Directory for saved plots. Defaults to ``<cwd>/output/evaluation``.
        selected_features: If provided, filter X_test to these columns before predicting.

    Example:
        >>> evaluator = ModelEvaluator(model, X_test, y_test, model_name="rf_seed_42")
        >>> print(evaluator.summary())
        >>> evaluator.plot_all()
    """

    def __init__(
        self,
        model: BaseEstimator | GridSearchCV,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model",
        output_dir: str | None = None,
        selected_features: list[str] | None = None,
    ) -> None:
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_

        if selected_features is not None:
            X_test = X_test[selected_features]

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.output_dir = output_dir or os.path.join(os.getcwd(), "output", "evaluation")

        os.makedirs(self.output_dir, exist_ok=True)

        self._y_pred: np.ndarray = model.predict(X_test)  # type: ignore[union-attr]
        self._y_proba: np.ndarray | None = self._get_proba()
        self._metrics: dict[str, Any] | None = None

    @classmethod
    def from_files(
        cls,
        model_path: str,
        X_test: pd.DataFrame | str,
        y_test: pd.Series | str,
        selected_features: list[str] | None = None,
        model_name: str = "model",
        output_dir: str | None = None,
    ) -> "ModelEvaluator":
        """Load model and data from files and construct a ModelEvaluator.

        Args:
            model_path: Path to a joblib-saved model file.
            X_test: Test features DataFrame or path to CSV.
            y_test: True test labels Series or path to CSV.
            selected_features: If provided, filter X_test to these columns.
            model_name: Name identifier for the model. Defaults to "model".
            output_dir: Directory for saved plots. Defaults to None.

        Returns:
            ModelEvaluator ready for metrics and plots.
        """
        model = joblib.load(model_path)

        if isinstance(X_test, str):
            X_test = pd.read_csv(X_test, index_col=0)
        if isinstance(y_test, str):
            y_test = pd.read_csv(y_test, index_col=0).iloc[:, 0]

        return cls(model, X_test, y_test, model_name, output_dir, selected_features)

    def _get_proba(self) -> np.ndarray | None:
        if hasattr(self.model, "predict_proba"):
            proba: np.ndarray = self.model.predict_proba(self.X_test)  # type: ignore[union-attr]
            return proba[:, 1] if proba.ndim > 1 else proba
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(self.X_test)  # type: ignore[union-attr]
        return None

    def get_metrics(self) -> dict[str, Any]:
        """Compute and return evaluation metrics (cached after first call).

        Returns:
            dict with accuracy, balanced_accuracy, precision, recall, f1_score,
            roc_auc, pr_auc, and confusion matrix components.
        """
        if self._metrics is not None:
            return self._metrics

        y_true, y_pred, y_proba = self.y_test, self._y_pred, self._y_proba

        metrics: dict[str, Any] = {
            "model_name": self.model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
                metrics["pr_auc"] = average_precision_score(y_true, y_proba)
            except ValueError:
                metrics["roc_auc"] = np.nan
                metrics["pr_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            })

        self._metrics = metrics
        return metrics

    def summary(self) -> str:
        """Return a formatted text summary of evaluation metrics."""
        m = self.get_metrics()
        lines = [
            "=" * 50,
            f"Model: {self.model_name}",
            "=" * 50,
            f"  Accuracy:          {m['accuracy']:.4f}",
            f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}",
            f"  Precision:         {m['precision']:.4f}",
            f"  Recall:            {m['recall']:.4f}",
            f"  F1 Score:          {m['f1_score']:.4f}",
        ]
        if not np.isnan(m["roc_auc"]):
            lines += [
                f"  ROC-AUC:           {m['roc_auc']:.4f}",
                f"  PR-AUC:            {m['pr_auc']:.4f}",
            ]
        if "sensitivity" in m:
            lines += [
                f"  Sensitivity:       {m['sensitivity']:.4f}",
                f"  Specificity:       {m['specificity']:.4f}",
                f"  TP/TN/FP/FN:       "
                f"{m['true_positives']}/{m['true_negatives']}/"
                f"{m['false_positives']}/{m['false_negatives']}",
            ]
        lines.append("=" * 50)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        normalize: str | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(self.y_test, self._y_pred, normalize=normalize)
        labels = ["Not Erupted", "Erupted"]
        fmt = ".2f" if normalize else "d"

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {self.model_name}")
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, filename or f"{self.model_name}_confusion_matrix.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

    def plot_roc_curve(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot ROC curve. Returns None if probabilities are unavailable."""
        if self._y_proba is None:
            logger.warning("ROC curve requires probability predictions")
            return None

        fpr, tpr, _ = roc_curve(self.y_test, self._y_proba)
        auc = roc_auc_score(self.y_test, self._y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve — {self.model_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, filename or f"{self.model_name}_roc_curve.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

    def plot_precision_recall_curve(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot Precision-Recall curve. Returns None if probabilities are unavailable."""
        if self._y_proba is None:
            logger.warning("PR curve requires probability predictions")
            return None

        precision, recall, _ = precision_recall_curve(self.y_test, self._y_proba)
        ap = average_precision_score(self.y_test, self._y_proba)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve — {self.model_name}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save:
            path = os.path.join(self.output_dir, filename or f"{self.model_name}_pr_curve.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

    def plot_all(self, dpi: int = 150) -> dict[str, plt.Figure | None]:
        """Generate and save all plots. Returns a dict of figure objects."""
        return {
            "confusion_matrix": self.plot_confusion_matrix(dpi=dpi),
            "roc_curve": self.plot_roc_curve(dpi=dpi),
            "pr_curve": self.plot_precision_recall_curve(dpi=dpi),
        }
