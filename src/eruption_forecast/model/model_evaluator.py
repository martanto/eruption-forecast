# Standard library imports
import os
from typing import Any, Literal, Protocol, Union

# Third party imports
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    VotingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve

# Project imports
from eruption_forecast.logger import logger


class SupportsPredict(Protocol):
    """Protocol for models that support predict method."""

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...


class SupportsPredictProba(Protocol):
    """Protocol for models that support predict_proba method."""

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...


class SupportsDecisionFunction(Protocol):
    """Protocol for models that support decision_function method."""

    def decision_function(self, X: np.ndarray | pd.DataFrame) -> np.ndarray: ...


class ModelEvaluator:
    """Comprehensive model evaluation with metrics, export, and plotting.

    Provides detailed evaluation metrics, model persistence, and visualization
    capabilities for eruption prediction classifiers.

    Args:
        model: A fitted sklearn classifier or GridSearchCV object.
        X_test: Test features (array-like of shape (n_samples, n_features)).
        y_test: True labels for test data (array-like of shape (n_samples,)).
        X_train: Training features (optional, for learning curves).
        y_train: Training labels (optional, for learning curves).
        model_name: Name identifier for the model. Defaults to "model".
        output_dir: Directory for saving outputs. Defaults to "output/evaluation".

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.model_selection import train_test_split
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        >>> model = RandomForestClassifier(random_state=42)
        >>> model.fit(X_train, y_train)
        >>> evaluator = ModelEvaluator(model, X_test, y_test, X_train, y_train)
        >>> metrics = evaluator.get_metrics()
        >>> evaluator.plot_all()
        >>> evaluator.export_all()
    """

    def __init__(
        self,
        model: BaseEstimator | GridSearchCV,
        X_test: np.ndarray | pd.DataFrame,
        y_test: np.ndarray | pd.Series,
        X_train: np.ndarray | pd.DataFrame | None = None,
        y_train: np.ndarray | pd.Series | None = None,
        model_name: str = "model",
        output_dir: str | None = None,
    ) -> None:
        # Handle GridSearchCV objects
        if isinstance(model, GridSearchCV):
            self._fitted_model: BaseEstimator = model.best_estimator_
            self._grid_search: GridSearchCV | None = model
        else:
            self._fitted_model = model
            self._grid_search = None

        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.model_name = model_name
        self.output_dir = output_dir or os.path.join(
            os.getcwd(), "output", "evaluation"
        )

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)

        # Generate predictions
        if not hasattr(self._fitted_model, "predict"):
            raise TypeError(
                f"Model {type(self._fitted_model).__name__} does not support predict method"
            )
        self._y_pred: np.ndarray = self._fitted_model.predict(X_test)  # type: ignore[attr-defined]
        self._y_proba = self._get_probabilities()

        # Cache for metrics
        self._metrics: dict[str, Any] | None = None

    @property
    def fitted_model(self) -> BaseEstimator:
        """Get the fitted model (unwrapped from GridSearchCV if applicable)."""
        return self._fitted_model

    @property
    def y_pred(self) -> np.ndarray:
        """Get predicted labels."""
        return self._y_pred

    @property
    def y_proba(self) -> np.ndarray | None:
        """Get predicted probabilities for positive class."""
        return self._y_proba

    def _get_probabilities(self) -> np.ndarray | None:
        """Extract prediction probabilities if available."""
        if hasattr(self._fitted_model, "predict_proba"):
            proba: np.ndarray = self._fitted_model.predict_proba(self.X_test)  # type: ignore[attr-defined]
            # Return probability for positive class (class 1)
            return proba[:, 1] if proba.ndim > 1 else proba
        elif hasattr(self._fitted_model, "decision_function"):
            decision: np.ndarray = self._fitted_model.decision_function(self.X_test)  # type: ignore[attr-defined]
            return decision
        return None

    # =========================================================================
    # Evaluation Metrics
    # =========================================================================

    def get_metrics(self, as_dataframe: bool = False) -> dict[str, Any] | pd.DataFrame:
        """Calculate comprehensive evaluation metrics.

        Computes accuracy, precision, recall, F1-score, balanced accuracy,
        ROC-AUC, PR-AUC, and confusion matrix statistics.

        Args:
            as_dataframe: If True, returns metrics as a DataFrame.

        Returns:
            Dictionary or DataFrame containing all metrics.

        Example:
            >>> metrics = evaluator.get_metrics()
            >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
            >>> print(f"ROC-AUC: {metrics['roc_auc']:.3f}")

            >>> # As DataFrame for easy export
            >>> df = evaluator.get_metrics(as_dataframe=True)
            >>> df.to_csv("metrics.csv")
        """
        if self._metrics is not None:
            return pd.DataFrame([self._metrics]) if as_dataframe else self._metrics

        y_true = self.y_test
        y_pred = self._y_pred
        y_proba = self._y_proba

        # Basic classification metrics
        metrics: dict[str, Any] = {
            "model_name": self.model_name,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(ly_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            ),
        }

        # Probability-based metrics (if available)
        if y_proba is not None:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics["roc_auc"] = np.nan

            try:
                metrics["pr_auc"] = average_precision_score(y_true, y_proba)
            except ValueError:
                metrics["pr_auc"] = np.nan
        else:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan

        # Confusion matrix components
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics["npv"] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            metrics["ppv"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Sample counts
        metrics["n_samples"] = len(y_true)
        metrics["n_positive"] = int(np.sum(y_true == 1))
        metrics["n_negative"] = int(np.sum(y_true == 0))

        self._metrics = metrics

        return pd.DataFrame([metrics]) if as_dataframe else metrics

    def get_classification_report(
        self, output_dict: bool = False
    ) -> str | dict[str, Any]:
        """Get sklearn classification report.

        Args:
            output_dict: If True, returns report as dictionary.

        Returns:
            Classification report as string or dictionary.

        Example:
            >>> print(evaluator.get_classification_report())
            >>> report_dict = evaluator.get_classification_report(output_dict=True)
        """
        return classification_report(
            self.y_test,
            self._y_pred,
            output_dict=output_dict,
            zero_division=0,
        )

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix.

        Returns:
            Confusion matrix as numpy array.

        Example:
            >>> cm = evaluator.get_confusion_matrix()
            >>> print(f"True Positives: {cm[1, 1]}")
        """
        return confusion_matrix(self.y_test, self._y_pred)

    def get_feature_importances(
        self, feature_names: list[str] | None = None, top_n: int | None = None
    ) -> pd.DataFrame | None:
        """Get feature importances from tree-based models.

        Args:
            feature_names: Names of features. If None, uses column names from
                X_test if DataFrame, else generic names.
            top_n: Return only top N features by importance.

        Returns:
            DataFrame with feature names and importances, sorted descending.
            Returns None if model doesn't support feature importances.

        Example:
            >>> importances = evaluator.get_feature_importances(top_n=20)
            >>> print(importances.head(10))
        """
        model: BaseEstimator = self._fitted_model

        # Handle VotingClassifier - get importances from first tree-based estimator
        if isinstance(model, VotingClassifier):
            for name, estimator in model.named_estimators_.items():
                if hasattr(estimator, "feature_importances_"):
                    model = estimator
                    break
            else:
                return None

        if not hasattr(model, "feature_importances_"):
            logger.warning(
                f"Model {type(model).__name__} does not support feature importances"
            )
            return None

        importances: np.ndarray = model.feature_importances_  # type: ignore[attr-defined]

        # Determine feature names
        if feature_names is None:
            if isinstance(self.X_test, pd.DataFrame):
                feature_names = self.X_test.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        if top_n is not None:
            df = df.head(top_n)

        return df

    def cross_validate(
        self,
        cv: int = 5,
        scoring: str | list[str] = "balanced_accuracy",
    ) -> dict[str, Any]:
        """Perform cross-validation on training data.

        Requires X_train and y_train to be provided during initialization.

        Args:
            cv: Number of cross-validation folds.
            scoring: Scoring metric(s) to use.

        Returns:
            Dictionary with cross-validation scores.

        Example:
            >>> cv_scores = evaluator.cross_validate(cv=5)
            >>> print(f"Mean CV Score: {cv_scores['test_score'].mean():.3f}")
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("X_train and y_train required for cross-validation")

        scores: np.ndarray = cross_val_score(
            self._fitted_model,  # type: ignore[arg-type]
            self.X_train,
            self.y_train,
            cv=cv,
            scoring=scoring if isinstance(scoring, str) else scoring[0],
        )

        return {
            "test_score": scores,
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
        }

    # =========================================================================
    # Export Methods
    # =========================================================================

    def export_model(self, filename: str | None = None, compress: int = 3) -> str:
        """Export trained model using joblib.

        Args:
            filename: Output filename. Defaults to "{model_name}.joblib".
            compress: Compression level (0-9). Defaults to 3.

        Returns:
            Path to saved model file.

        Example:
            >>> path = evaluator.export_model()
            >>> # Later: model = joblib.load(path)
        """
        filename = filename or f"{self.model_name}.joblib"
        filepath = os.path.join(self.output_dir, filename)

        joblib.dump(self._fitted_model, filepath, compress=compress)  # type: ignore[arg-type]
        logger.info(f"Model exported to: {filepath}")

        return filepath

    def export_metrics(
        self, filename: str | None = None, format: Literal["csv", "json"] = "csv"
    ) -> str:
        """Export evaluation metrics to file.

        Args:
            filename: Output filename. Defaults to "{model_name}_metrics.{format}".
            format: Output format ("csv" or "json").

        Returns:
            Path to saved metrics file.

        Example:
            >>> path = evaluator.export_metrics(format="csv")
        """
        metrics = self.get_metrics()
        filename = filename or f"{self.model_name}_metrics.{format}"
        filepath = os.path.join(self.output_dir, filename)

        df = pd.DataFrame([metrics])

        if format == "csv":
            df.to_csv(filepath, index=False)
        elif format == "json":
            df.to_json(filepath, orient="records", indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Metrics exported to: {filepath}")
        return filepath

    def export_classification_report(self, filename: str | None = None) -> str:
        """Export classification report to text file.

        Args:
            filename: Output filename. Defaults to "{model_name}_report.txt".

        Returns:
            Path to saved report file.
        """
        filename = filename or f"{self.model_name}_report.txt"
        filepath = os.path.join(self.output_dir, filename)

        report = self.get_classification_report()
        with open(filepath, "w") as f:
            f.write(f"Classification Report - {self.model_name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(str(report))

        logger.info(f"Classification report exported to: {filepath}")
        return filepath

    def export_confusion_matrix(self, filename: str | None = None) -> str:
        """Export confusion matrix to CSV.

        Args:
            filename: Output filename. Defaults to "{model_name}_confusion_matrix.csv".

        Returns:
            Path to saved CSV file.
        """
        filename = filename or f"{self.model_name}_confusion_matrix.csv"
        filepath = os.path.join(self.output_dir, filename)

        cm = self.get_confusion_matrix()
        df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"],
        )
        df.to_csv(filepath)

        logger.info(f"Confusion matrix exported to: {filepath}")
        return filepath

    def export_feature_importances(
        self,
        filename: str | None = None,
        feature_names: list[str] | None = None,
        top_n: int | None = None,
    ) -> str | None:
        """Export feature importances to CSV.

        Args:
            filename: Output filename.
            feature_names: Names of features.
            top_n: Export only top N features.

        Returns:
            Path to saved CSV file, or None if not supported.
        """
        importances = self.get_feature_importances(feature_names, top_n)
        if importances is None:
            return None

        filename = filename or f"{self.model_name}_feature_importances.csv"
        filepath = os.path.join(self.output_dir, filename)

        importances.to_csv(filepath, index=False)
        logger.info(f"Feature importances exported to: {filepath}")
        return filepath

    def export_all(
        self,
        feature_names: list[str] | None = None,
        top_n_features: int | None = 50,
    ) -> dict[str, str | None]:
        """Export all results (model, metrics, reports, plots).

        Args:
            feature_names: Names of features for importance export.
            top_n_features: Number of top features to export.

        Returns:
            Dictionary mapping export type to file path.

        Example:
            >>> paths = evaluator.export_all()
            >>> print(paths['model'])  # Path to saved model
        """
        paths: dict[str, str | None] = {
            "model": self.export_model(),
            "metrics_csv": self.export_metrics(format="csv"),
            "metrics_json": self.export_metrics(format="json"),
            "classification_report": self.export_classification_report(),
            "confusion_matrix": self.export_confusion_matrix(),
            "feature_importances": self.export_feature_importances(
                feature_names=feature_names, top_n=top_n_features
            ),
        }

        logger.info(f"All exports saved to: {self.output_dir}")
        return paths

    # =========================================================================
    # Plotting Methods
    # =========================================================================

    def plot_confusion_matrix(
        self,
        normalize: Literal["true", "pred", "all"] | None = None,
        cmap: str = "Blues",
        figsize: tuple[int, int] = (8, 6),
        title: str | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot confusion matrix heatmap.

        Args:
            normalize: Normalize over 'true' (rows), 'pred' (columns), 'all', or None.
            cmap: Colormap for heatmap.
            figsize: Figure size (width, height).
            title: Plot title.
            save: Whether to save the figure.
            filename: Output filename.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure object.

        Example:
            >>> fig = evaluator.plot_confusion_matrix(normalize="true")
        """
        fig, ax = plt.subplots(figsize=figsize)

        cm = confusion_matrix(self.y_test, self._y_pred, normalize=normalize)

        # Use seaborn heatmap for better visualization
        labels = ["Not Erupted (0)", "Erupted (1)"]
        fmt = ".2f" if normalize else "d"

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"label": "Count" if not normalize else "Proportion"},
        )

        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(
            title or f"Confusion Matrix - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        if save:
            filename = filename or f"{self.model_name}_confusion_matrix.png"
            filepath = os.path.join(self.figures_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.info(f"Confusion matrix plot saved to: {filepath}")

        return fig

    def plot_roc_curve(
        self,
        figsize: tuple[int, int] = (8, 6),
        title: str | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot ROC (Receiver Operating Characteristic) curve.

        Args:
            figsize: Figure size (width, height).
            title: Plot title.
            save: Whether to save the figure.
            filename: Output filename.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure object, or None if probabilities unavailable.

        Example:
            >>> fig = evaluator.plot_roc_curve()
        """
        if self._y_proba is None:
            logger.warning("ROC curve requires probability predictions")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.y_test, self._y_proba)
        roc_auc = roc_auc_score(self.y_test, self._y_proba)

        # Plot ROC curve
        ax.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (AUC = {roc_auc:.3f})",
        )

        # Plot diagonal (random classifier)
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(
            title or f"ROC Curve - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = filename or f"{self.model_name}_roc_curve.png"
            filepath = os.path.join(self.figures_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.info(f"ROC curve saved to: {filepath}")

        return fig

    def plot_precision_recall_curve(
        self,
        figsize: tuple[int, int] = (8, 6),
        title: str | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot Precision-Recall curve.

        Args:
            figsize: Figure size (width, height).
            title: Plot title.
            save: Whether to save the figure.
            filename: Output filename.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure object, or None if probabilities unavailable.

        Example:
            >>> fig = evaluator.plot_precision_recall_curve()
        """
        if self._y_proba is None:
            logger.warning("PR curve requires probability predictions")
            return None

        fig, ax = plt.subplots(figsize=figsize)

        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self._y_proba
        )
        pr_auc = average_precision_score(self.y_test, self._y_proba)

        # Plot PR curve
        ax.plot(
            recall,
            precision,
            color="darkorange",
            lw=2,
            label=f"PR curve (AP = {pr_auc:.3f})",
        )

        # Plot baseline (proportion of positive class)
        baseline = np.sum(self.y_test == 1) / len(self.y_test)
        ax.axhline(
            y=baseline,
            color="navy",
            lw=2,
            linestyle="--",
            label=f"Baseline ({baseline:.3f})",
        )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(
            title or f"Precision-Recall Curve - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = filename or f"{self.model_name}_pr_curve.png"
            filepath = os.path.join(self.figures_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.info(f"Precision-Recall curve saved to: {filepath}")

        return fig

    def plot_feature_importances(
        self,
        feature_names: list[str] | None = None,
        top_n: int = 20,
        figsize: tuple[int, int] | None = None,
        title: str | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot feature importances as horizontal bar chart.

        Args:
            feature_names: Names of features.
            top_n: Number of top features to plot.
            figsize: Figure size. Defaults to dynamic based on top_n.
            title: Plot title.
            save: Whether to save the figure.
            filename: Output filename.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure object, or None if not supported.

        Example:
            >>> fig = evaluator.plot_feature_importances(top_n=15)
        """
        importances = self.get_feature_importances(feature_names, top_n)
        if importances is None:
            return None

        # Dynamic figure size based on number of features
        figsize = figsize or (10, int(max(6.0, top_n * 0.4)))
        fig, ax = plt.subplots(figsize=figsize)

        # Reverse for horizontal bar (top feature at top)
        importances_reversed = importances.iloc[::-1]

        # Use seaborn for consistent styling
        sns.barplot(
            data=importances_reversed,
            x="importance",
            y="feature",
            hue="feature",
            ax=ax,
            palette="viridis",
            legend=False,
        )

        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(
            title or f"Top {top_n} Feature Importances - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, axis="x", alpha=0.3)

        plt.tight_layout()

        if save:
            filename = filename or f"{self.model_name}_feature_importances.png"
            filepath = os.path.join(self.figures_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.info(f"Feature importances plot saved to: {filepath}")

        return fig

    def plot_learning_curve(
        self,
        cv: int = 5,
        train_sizes: np.ndarray | None = None,
        scoring: str = "balanced_accuracy",
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure | None:
        """Plot learning curve to diagnose bias/variance.

        Requires X_train and y_train to be provided during initialization.

        Args:
            cv: Number of cross-validation folds.
            train_sizes: Array of training set sizes to evaluate.
            scoring: Scoring metric.
            figsize: Figure size.
            title: Plot title.
            save: Whether to save the figure.
            filename: Output filename.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure object, or None if training data unavailable.

        Example:
            >>> fig = evaluator.plot_learning_curve(cv=5)
        """
        if self.X_train is None or self.y_train is None:
            logger.warning("Learning curve requires X_train and y_train")
            return None

        train_sizes = (
            train_sizes if train_sizes is not None else np.linspace(0.1, 1.0, 10)
        )

        result = learning_curve(
            self._fitted_model,  # type: ignore[arg-type]
            self.X_train,
            self.y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        train_sizes_abs, train_scores, test_scores = result[0], result[1], result[2]

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=figsize)

        # Plot training score
        ax.fill_between(
            train_sizes_abs,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.2,
            color="blue",
        )
        ax.plot(
            train_sizes_abs,
            train_mean,
            "o-",
            color="blue",
            label="Training score",
        )

        # Plot validation score
        ax.fill_between(
            train_sizes_abs,
            test_mean - test_std,
            test_mean + test_std,
            alpha=0.2,
            color="orange",
        )
        ax.plot(
            train_sizes_abs,
            test_mean,
            "o-",
            color="orange",
            label="Cross-validation score",
        )

        ax.set_xlabel("Training Set Size", fontsize=12)
        ax.set_ylabel(f"Score ({scoring})", fontsize=12)
        ax.set_title(
            title or f"Learning Curve - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = filename or f"{self.model_name}_learning_curve.png"
            filepath = os.path.join(self.figures_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.info(f"Learning curve saved to: {filepath}")

        return fig

    def plot_metrics_summary(
        self,
        figsize: tuple[int, int] = (10, 6),
        title: str | None = None,
        save: bool = True,
        filename: str | None = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Plot summary bar chart of key metrics.

        Args:
            figsize: Figure size.
            title: Plot title.
            save: Whether to save the figure.
            filename: Output filename.
            dpi: Figure resolution.

        Returns:
            Matplotlib Figure object.

        Example:
            >>> fig = evaluator.plot_metrics_summary()
        """
        metrics = self.get_metrics()

        # Select key metrics to display
        display_metrics = {
            "Accuracy": metrics["accuracy"],
            "Balanced Acc": metrics["balanced_accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
        }

        if not np.isnan(metrics["roc_auc"]):
            display_metrics["ROC-AUC"] = metrics["roc_auc"]
        if not np.isnan(metrics["pr_auc"]):
            display_metrics["PR-AUC"] = metrics["pr_auc"]

        fig, ax = plt.subplots(figsize=figsize)

        names = list(display_metrics.keys())
        values = list(display_metrics.values())
        colors = sns.color_palette("viridis", len(names))

        bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            title or f"Performance Metrics - {self.model_name}",
            fontsize=14,
            fontweight="bold",
        )
        ax.axhline(
            y=0.5, color="red", linestyle="--", alpha=0.5, label="Random baseline"
        )
        ax.legend(loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        if save:
            filename = filename or f"{self.model_name}_metrics_summary.png"
            filepath = os.path.join(self.figures_dir, filename)
            fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
            logger.info(f"Metrics summary plot saved to: {filepath}")

        return fig

    def plot_all(
        self,
        feature_names: list[str] | None = None,
        top_n_features: int = 20,
        dpi: int = 150,
    ) -> dict[str, plt.Figure | None]:
        """Generate all available plots.

        Args:
            feature_names: Names of features for importance plot.
            top_n_features: Number of top features to plot.
            dpi: Figure resolution for all plots.

        Returns:
            Dictionary mapping plot name to Figure object.

        Example:
            >>> figures = evaluator.plot_all()
            >>> plt.show()  # Display all figures
        """
        figures: dict[str, plt.Figure | None] = {
            "confusion_matrix": self.plot_confusion_matrix(dpi=dpi),
            "confusion_matrix_normalized": self.plot_confusion_matrix(
                normalize="true",
                filename=f"{self.model_name}_confusion_matrix_normalized.png",
                title=f"Normalized Confusion Matrix - {self.model_name}",
                dpi=dpi,
            ),
            "roc_curve": self.plot_roc_curve(dpi=dpi),
            "pr_curve": self.plot_precision_recall_curve(dpi=dpi),
            "feature_importances": self.plot_feature_importances(
                feature_names=feature_names, top_n=top_n_features, dpi=dpi
            ),
            "learning_curve": self.plot_learning_curve(dpi=dpi),
            "metrics_summary": self.plot_metrics_summary(dpi=dpi),
        }

        logger.info(f"All plots saved to: {self.figures_dir}")
        return figures

    def summary(self) -> str:
        """Generate text summary of evaluation results.

        Returns:
            Formatted string with key metrics and model information.

        Example:
            >>> print(evaluator.summary())
        """
        metrics = self.get_metrics()

        lines = [
            "=" * 60,
            f"Model Evaluation Summary: {self.model_name}",
            "=" * 60,
            "",
            "Classification Metrics:",
            f"  Accuracy:          {metrics['accuracy']:.4f}",
            f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}",
            f"  Precision:         {metrics['precision']:.4f}",
            f"  Recall:            {metrics['recall']:.4f}",
            f"  F1 Score:          {metrics['f1_score']:.4f}",
            "",
        ]

        if not np.isnan(metrics["roc_auc"]):
            lines.extend(
                [
                    "Probability Metrics:",
                    f"  ROC-AUC:           {metrics['roc_auc']:.4f}",
                    f"  PR-AUC:            {metrics['pr_auc']:.4f}",
                    "",
                ]
            )

        if "true_positives" in metrics:
            lines.extend(
                [
                    "Confusion Matrix:",
                    f"  True Positives:    {metrics['true_positives']}",
                    f"  True Negatives:    {metrics['true_negatives']}",
                    f"  False Positives:   {metrics['false_positives']}",
                    f"  False Negatives:   {metrics['false_negatives']}",
                    f"  Sensitivity:       {metrics['sensitivity']:.4f}",
                    f"  Specificity:       {metrics['specificity']:.4f}",
                    "",
                ]
            )

        lines.extend(
            [
                "Sample Information:",
                f"  Total Samples:     {metrics['n_samples']}",
                f"  Positive Class:    {metrics['n_positive']}",
                f"  Negative Class:    {metrics['n_negative']}",
                "",
                "=" * 60,
            ]
        )

        return "\n".join(lines)
