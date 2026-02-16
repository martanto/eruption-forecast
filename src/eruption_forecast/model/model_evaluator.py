import os
from typing import Any, Literal

import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    confusion_matrix,
    average_precision_score,
    balanced_accuracy_score,
)
from sklearn.model_selection import GridSearchCV

from eruption_forecast.utils import resolve_output_dir
from eruption_forecast.logger import logger

# Import new styled plotting functions
from eruption_forecast.plots.evaluation_plots import (
    plot_roc_curve as _plot_roc_styled,
    plot_calibration as _plot_cal_styled,
    plot_confusion_matrix as _plot_cm_styled,
    plot_feature_importance as _plot_fi_styled,
    plot_threshold_analysis as _plot_threshold_styled,
    plot_precision_recall_curve as _plot_pr_styled,
    plot_prediction_distribution as _plot_pred_dist_styled,
)


class ModelEvaluator:
    """Evaluate a fitted classifier on a test set.

    Wraps a trained model and test data to provide metrics, a text summary,
    and plots. Predictions are computed once on construction and cached.

    Args:
        model: Fitted sklearn estimator or GridSearchCV (best_estimator_ is used).
        X_test: Test features.
        y_test: True test labels.
        model_name: Name used in plot titles and filenames. Defaults to "model".
        output_dir: Directory for saved plots. If None, defaults to
            ``root_dir/output/evaluation``. Relative paths are resolved against
            ``root_dir`` (or ``os.getcwd()`` when ``root_dir`` is None). Absolute
            paths are used as-is. Defaults to None.
        selected_features: If provided, filter X_test to these columns before predicting.
        root_dir: Anchor directory for resolving relative ``output_dir`` values.
            Defaults to None (uses ``os.getcwd()``).

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
        root_dir: str | None = None,
    ) -> None:
        if isinstance(model, GridSearchCV):
            model = model.best_estimator_

        if selected_features is not None:
            X_test = X_test[selected_features]

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.output_dir = resolve_output_dir(output_dir, root_dir, os.path.join("output", "evaluation"))

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
        """Retrieve predicted probabilities or decision scores for the test set.

        Tries ``predict_proba`` first (returns the positive-class column),
        then falls back to ``decision_function``. Returns None when neither
        method is available on the model.

        Returns:
            np.ndarray | None: 1-D array of probability estimates or
            decision scores, or None if the model does not support either.
        """
        if hasattr(self.model, "predict_proba"):
            proba: np.ndarray = self.model.predict_proba(self.X_test)  # type: ignore[union-attr]
            return proba[:, 1] if proba.ndim > 1 else proba
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(self.X_test)  # type: ignore[union-attr]
        return None

    def get_metrics(self) -> dict[str, Any]:
        """Compute and return evaluation metrics (cached after first call).

        Calculates classification performance metrics from the cached
        predictions. Results are memoised — subsequent calls return the
        same dict without recomputing.

        Returns:
            dict[str, Any]: Metrics dictionary with keys:

            - ``model_name`` (str): Name of the model.
            - ``accuracy`` (float): Overall accuracy.
            - ``balanced_accuracy`` (float): Accuracy adjusted for class
              imbalance.
            - ``precision`` (float): Positive predictive value.
            - ``recall`` (float): True positive rate (sensitivity).
            - ``f1_score`` (float): Harmonic mean of precision and recall.
            - ``roc_auc`` (float): Area under the ROC curve (nan if not
              available).
            - ``pr_auc`` (float): Area under the precision-recall curve
              (nan if not available).
            - ``true_positives`` (int): TP count (binary labels only).
            - ``true_negatives`` (int): TN count (binary labels only).
            - ``false_positives`` (int): FP count (binary labels only).
            - ``false_negatives`` (int): FN count (binary labels only).
            - ``sensitivity`` (float): Same as recall (binary labels only).
            - ``specificity`` (float): True negative rate (binary labels
              only).
            - ``optimal_threshold`` (float): Decision threshold that
              maximises F1 (nan if probabilities unavailable).
            - ``f1_at_optimal`` (float): F1 at the optimal threshold.
            - ``recall_at_optimal`` (float): Recall at the optimal threshold.
            - ``precision_at_optimal`` (float): Precision at the optimal
              threshold.

        Examples:
            >>> m = evaluator.get_metrics()
            >>> print(f"F1: {m['f1_score']:.3f}, AUC: {m['roc_auc']:.3f}")
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

        # Optimal threshold analysis
        if y_proba is not None:
            opt_thresh, opt_metrics = self.optimize_threshold(criterion="f1")
            metrics["optimal_threshold"] = opt_thresh
            metrics["f1_at_optimal"] = opt_metrics["f1"]
            metrics["recall_at_optimal"] = opt_metrics["recall"]
            metrics["precision_at_optimal"] = opt_metrics["precision"]
        else:
            metrics["optimal_threshold"] = np.nan
            metrics["f1_at_optimal"] = np.nan
            metrics["recall_at_optimal"] = np.nan
            metrics["precision_at_optimal"] = np.nan

        self._metrics = metrics
        return metrics

    def summary(self) -> str:
        """Return a formatted text summary of evaluation metrics.

        Returns:
            str: Multi-line string with accuracy, balanced accuracy, precision,
            recall, F1, ROC-AUC, PR-AUC, sensitivity/specificity, and
            confusion matrix counts (where available).

        Examples:
            >>> print(evaluator.summary())
            ==================================================
            Model: rf_seed_42
            ==================================================
              Accuracy:          0.9200
              ...
        """
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
        """Plot a confusion matrix heatmap.

        Args:
            normalize (str | None, optional): Normalisation mode passed to
                ``sklearn.metrics.confusion_matrix``. Use ``"true"`` to
                normalise by row. Defaults to None (raw counts).
            save (bool, optional): If True, save the figure to
                ``output_dir``. Defaults to True.
            filename (str | None, optional): Output filename. If None,
                defaults to ``"<model_name>_confusion_matrix.png"``.
                Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure: Matplotlib figure object.
        """
        # Delegate to styled plotting function
        fig = _plot_cm_styled(
            y_true=self.y_test,
            y_pred=self._y_pred,
            normalize=normalize,
            title=f"Confusion Matrix — {self.model_name}",
            figsize=(6, 5),
            dpi=dpi,
        )

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
        """Plot the ROC curve with AUC annotation.

        Args:
            save (bool, optional): If True, save the figure to
                ``output_dir``. Defaults to True.
            filename (str | None, optional): Output filename. If None,
                defaults to ``"<model_name>_roc_curve.png"``.
                Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure | None: Matplotlib figure, or None if probability
            predictions are unavailable.
        """
        if self._y_proba is None:
            logger.warning("ROC curve requires probability predictions")
            return None

        # Delegate to styled plotting function
        fig = _plot_roc_styled(
            y_true=self.y_test,
            y_proba=self._y_proba,
            title=f"ROC Curve — {self.model_name}",
            figsize=(6, 5),
            dpi=dpi,
        )

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
        """Plot the Precision-Recall curve with average precision annotation.

        Args:
            save (bool, optional): If True, save the figure to
                ``output_dir``. Defaults to True.
            filename (str | None, optional): Output filename. If None,
                defaults to ``"<model_name>_pr_curve.png"``.
                Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure | None: Matplotlib figure, or None if probability
            predictions are unavailable.
        """
        if self._y_proba is None:
            logger.warning("PR curve requires probability predictions")
            return None

        # Delegate to styled plotting function
        fig = _plot_pr_styled(
            y_true=self.y_test,
            y_proba=self._y_proba,
            title=f"Precision-Recall Curve — {self.model_name}",
            figsize=(6, 5),
            dpi=dpi,
        )

        if save:
            path = os.path.join(self.output_dir, filename or f"{self.model_name}_pr_curve.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

    def optimize_threshold(
        self,
        criterion: Literal["f1", "balanced_accuracy", "recall", "precision"] = "f1",
    ) -> tuple[float, dict[str, float]]:
        """Sweep thresholds and return the one maximising the given criterion.

        Args:
            criterion: Metric to optimise. One of ``"f1"``, ``"balanced_accuracy"``,
                ``"recall"``, or ``"precision"``.

        Returns:
            Tuple of ``(threshold, metrics_at_threshold)`` where
            ``metrics_at_threshold`` contains f1, balanced_accuracy, recall, and
            precision at the optimal threshold.

        Raises:
            ValueError: If probabilities are not available.
        """
        if self._y_proba is None:
            raise ValueError("optimize_threshold requires probability predictions")

        thresholds = np.linspace(0.0, 1.0, 101)
        best_thresh = 0.5
        best_score = -1.0

        for t in thresholds:
            y_pred_t = (self._y_proba >= t).astype(int)
            if criterion == "f1":
                score = f1_score(self.y_test, y_pred_t, zero_division=0)
            elif criterion == "balanced_accuracy":
                score = balanced_accuracy_score(self.y_test, y_pred_t)
            elif criterion == "recall":
                score = recall_score(self.y_test, y_pred_t, zero_division=0)
            else:  # precision
                score = precision_score(self.y_test, y_pred_t, zero_division=0)

            if score > best_score:
                best_score = score
                best_thresh = float(t)

        y_pred_best = (self._y_proba >= best_thresh).astype(int)
        metrics_at = {
            "f1": f1_score(self.y_test, y_pred_best, zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(self.y_test, y_pred_best),
            "recall": recall_score(self.y_test, y_pred_best, zero_division=0),
            "precision": precision_score(self.y_test, y_pred_best, zero_division=0),
        }
        return best_thresh, metrics_at

    def plot_threshold_analysis(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot precision, recall, F1, and balanced accuracy vs decision threshold.

        Marks the default 0.5 threshold and the optimal F1 threshold.

        Args:
            save (bool, optional): If True, save the figure to
                ``output_dir``. Defaults to True.
            filename (str | None, optional): Output filename. If None,
                defaults to ``"<model_name>_threshold_analysis.png"``.
                Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure | None: Matplotlib figure, or None if probability
            predictions are unavailable.
        """
        if self._y_proba is None:
            logger.warning("plot_threshold_analysis requires probability predictions")
            return None

        # Delegate to styled plotting function
        fig = _plot_threshold_styled(
            y_true=self.y_test,
            y_proba=self._y_proba,
            title=f"Threshold Analysis — {self.model_name}",
            figsize=(10, 6),
            dpi=dpi,
        )

        if save:
            path = os.path.join(
                self.output_dir, filename or f"{self.model_name}_threshold_analysis.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

    def plot_feature_importance(
        self,
        top_n: int = 20,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot a horizontal bar chart of the top-N feature importances.

        Works for tree-based models (RF, GB, XGB, DT) and VotingClassifier
        when sub-estimators expose ``feature_importances_``. Returns None with
        a warning for models that do not support this attribute.

        Args:
            top_n (int, optional): Number of top features to display.
                Defaults to 20.
            save (bool, optional): If True, save the figure to
                ``output_dir``. Defaults to True.
            filename (str | None, optional): Output filename. If None,
                defaults to ``"<model_name>_feature_importance.png"``.
                Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure | None: Matplotlib figure, or None if the model does
            not expose ``feature_importances_``.
        """
        # Delegate to styled plotting function
        fig = _plot_fi_styled(
            model=self.model,
            feature_names=list(self.X_test.columns),
            top_n=top_n,
            title=f"Top-{top_n} Feature Importances — {self.model_name}",
            dpi=dpi,
        )

        if fig is None:
            logger.warning(
                f"{type(self.model).__name__} does not expose feature_importances_; "
                "skipping plot_feature_importance"
            )
            return None

        if save:
            path = os.path.join(
                self.output_dir, filename or f"{self.model_name}_feature_importance.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

    def plot_calibration(
        self,
        n_bins: int = 10,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot a reliability diagram (calibration curve).

        Compares predicted probabilities against the empirical fraction of
        positives to assess how well-calibrated the model is.

        Args:
            n_bins (int, optional): Number of bins for the calibration
                curve. Defaults to 10.
            save (bool, optional): If True, save the figure to
                ``output_dir``. Defaults to True.
            filename (str | None, optional): Output filename. If None,
                defaults to ``"<model_name>_calibration.png"``.
                Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure | None: Matplotlib figure, or None if probability
            predictions are unavailable.
        """
        if self._y_proba is None:
            logger.warning("plot_calibration requires probability predictions")
            return None

        # Delegate to styled plotting function
        fig = _plot_cal_styled(
            y_true=self.y_test,
            y_proba=self._y_proba,
            n_bins=n_bins,
            title=f"Calibration Curve — {self.model_name}",
            figsize=(6, 5),
            dpi=dpi,
        )

        if save:
            path = os.path.join(
                self.output_dir, filename or f"{self.model_name}_calibration.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

    def plot_prediction_distribution(
        self,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot histograms of predicted probabilities split by true class.

        Class 0 (not erupted) is shown in blue and class 1 (erupted) in
        orange. A dashed line marks the default 0.5 decision threshold.

        Args:
            save (bool, optional): If True, save the figure to
                ``output_dir``. Defaults to True.
            filename (str | None, optional): Output filename. If None,
                defaults to ``"<model_name>_prediction_distribution.png"``.
                Defaults to None.
            dpi (int, optional): Figure resolution. Defaults to 150.

        Returns:
            plt.Figure | None: Matplotlib figure, or None if probability
            predictions are unavailable.
        """
        if self._y_proba is None:
            logger.warning("plot_prediction_distribution requires probability predictions")
            return None

        # Delegate to styled plotting function
        fig = _plot_pred_dist_styled(
            y_true=self.y_test,
            y_proba=self._y_proba,
            title=f"Prediction Distribution — {self.model_name}",
            figsize=(8, 5),
            dpi=dpi,
        )

        if save:
            path = os.path.join(
                self.output_dir, filename or f"{self.model_name}_prediction_distribution.png"
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

        if save:
            path = os.path.join(
                self.output_dir,
                filename or f"{self.model_name}_prediction_distribution.png",
            )
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved: {path}")

        return fig

    def plot_all(self, dpi: int = 150) -> dict[str, plt.Figure | None]:
        """Generate and save all evaluation plots.

        Runs every individual plot method and collects the resulting figures.
        Each figure is also saved to ``output_dir`` automatically.

        Args:
            dpi (int, optional): Figure resolution applied to all plots.
                Defaults to 150.

        Returns:
            dict[str, plt.Figure | None]: Mapping of plot name to figure
            object. Keys: ``"confusion_matrix"``, ``"roc_curve"``,
            ``"pr_curve"``, ``"threshold_analysis"``,
            ``"feature_importance"``, ``"calibration"``,
            ``"prediction_distribution"``. Values are None when a plot
            could not be generated (e.g. probabilities unavailable).

        Examples:
            >>> figs = evaluator.plot_all(dpi=200)
            >>> figs["roc_curve"].savefig("custom_roc.png")
        """
        return {
            "confusion_matrix": self.plot_confusion_matrix(dpi=dpi),
            "roc_curve": self.plot_roc_curve(dpi=dpi),
            "pr_curve": self.plot_precision_recall_curve(dpi=dpi),
            "threshold_analysis": self.plot_threshold_analysis(dpi=dpi),
            "feature_importance": self.plot_feature_importance(dpi=dpi),
            "calibration": self.plot_calibration(dpi=dpi),
            "prediction_distribution": self.plot_prediction_distribution(dpi=dpi),
        }
