# Standard library imports
from typing import Any, Literal, Self

# Third party imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ClassifierModel:
    """Manages machine learning classifiers and their hyperparameter grids.

    Provides a unified interface for selecting and configuring classifiers
    for eruption prediction. Each classifier comes with predefined
    hyperparameter grids optimized for GridSearchCV.

    Supported classifiers:
        - svm: Support Vector Machine (SVC with balanced class weights)
        - knn: K-Nearest Neighbors
        - dt: Decision Tree (with balanced class weights)
        - rf: Random Forest (with balanced class weights)
        - nn: Multi-Layer Perceptron Neural Network
        - nb: Gaussian Naive Bayes
        - lr: Logistic Regression (with balanced class weights)

    Args:
        classifier: Classifier type identifier.

    Example:
        >>> # Create a Random Forest classifier
        >>> clf = ClassifierModel("rf")
        >>> model, grid = clf.model_and_grid
        >>> print(grid)
        {'n_estimators': [10, 30, 100], 'max_depth': [3, 5, 7], ...}

        >>> # Custom grid parameters
        >>> clf = ClassifierModel("rf")
        >>> clf.grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10]}

        >>> # Chain model and grid updates
        >>> clf = ClassifierModel("svm").update_model_and_grid(
        ...     SVC(kernel="rbf"),
        ...     {"C": [0.1, 1, 10]}
        ... )
    """

    def __init__(self, classifier: Literal["svm", "knn", "dt", "rf", "nn", "nb", "lr"]):
        self.classifier = classifier
        self._model: (
            SVC
            | KNeighborsClassifier
            | DecisionTreeClassifier
            | RandomForestClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
            | None
        ) = None
        self._grid: dict[str, Any] | None = None

    @property
    def grid(self) -> dict[str, Any]:
        """Hyperparameter grid for cross-validation.

        Returns default grid if none set, otherwise returns custom grid.

        Returns:
            dict[str, Any]: Dictionary mapping parameter names to lists of values.

        Example:
            >>> clf = ClassifierModel("rf")
            >>> print(clf.grid["n_estimators"])
            [10, 30, 100]
        """
        if self._grid is not None:
            return self._grid

        if self.classifier == "svm" or isinstance(self.model, SVC):
            return {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "kernel": ["poly", "rbf", "sigmoid"],
                "degree": [2, 3, 4, 5],
                "decision_function_shape": ["ovo", "ovr"],
            }

        if self.classifier == "knn" or isinstance(self.model, KNeighborsClassifier):
            return {
                "n_neighbors": [3, 6, 12, 24],
                "weights": ["uniform", "distance"],
                "p": [1, 2, 3],
            }

        if self.classifier == "dt" or isinstance(self.model, DecisionTreeClassifier):
            return {
                "max_depth": [3, 5, 7],
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", "log2", None],
            }

        if self.classifier == "rf" or isinstance(self.model, RandomForestClassifier):
            return {
                "n_estimators": [10, 30, 100],
                "max_depth": [3, 5, 7],
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", "log2", None],
            }

        if self.classifier == "nn" or isinstance(self.model, MLPClassifier):
            return {
                "activation": ["identity", "logistic", "tanh", "relu"],
                "hidden_layer_sizes": [10, 100],
            }

        if self.classifier == "nb" or isinstance(self.model, GaussianNB):
            return {"var_smoothing": [1.0]}

        if self.classifier == "lr" or isinstance(self.model, LogisticRegression):
            return {
                "penalty": ["l2", "l1", "elasticnet"],
                "C": [0.001, 0.01, 0.1, 1, 10],
            }

        raise ValueError(f"Unknown classifier: {self.classifier}")

    @grid.setter
    def grid(self, grid: dict[str, Any]):
        """Set grid parameters"""
        self._grid = grid

    @property
    def model(
        self,
    ) -> (
        SVC
        | KNeighborsClassifier
        | DecisionTreeClassifier
        | RandomForestClassifier
        | MLPClassifier
        | GaussianNB
        | LogisticRegression
    ):
        """Get the classifier model instance.

        Returns a new instance of the classifier with default settings
        if none has been set. Most classifiers use balanced class weights
        to handle the imbalanced eruption/non-eruption data.

        Returns:
            Classifier instance (SVC, KNeighborsClassifier, DecisionTreeClassifier,
            RandomForestClassifier, MLPClassifier, GaussianNB, or LogisticRegression).

        Example:
            >>> clf = ClassifierModel("rf")
            >>> model = clf.model
            >>> print(type(model).__name__)
            'RandomForestClassifier'
        """
        if self._model is not None:
            return self._model

        if self.classifier == "svm":
            return SVC(class_weight="balanced")

        if self.classifier == "knn":
            return KNeighborsClassifier()

        if self.classifier == "dt":
            return DecisionTreeClassifier(class_weight="balanced")

        if self.classifier == "rf":
            return RandomForestClassifier(class_weight="balanced")

        if self.classifier == "nn":
            return MLPClassifier(alpha=1, max_iter=1000)

        if self.classifier == "nb":
            return GaussianNB()

        if self.classifier == "lr":
            return LogisticRegression(class_weight="balanced")

        raise ValueError(f"Unknown classifier: {self.classifier}")

    @model.setter
    def model(
        self,
        model: (
            SVC
            | KNeighborsClassifier
            | DecisionTreeClassifier
            | RandomForestClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
        ),
    ):
        """Set model classifier"""
        self._model = model

    @property
    def model_and_grid(self):
        return self.model, self.grid

    def update_model_and_grid(
        self,
        model: (
            SVC
            | KNeighborsClassifier
            | DecisionTreeClassifier
            | RandomForestClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
        ),
        grid: dict[str, Any],
    ) -> Self:
        """Update model classifier and grid parameters.

        Args:
            model (Union[
                SVC,
                KNeighborsClassifier,
                DecisionTreeClassifier,
                RandomForestClassifier,
                MLPClassifier,
                GaussianNB, LogisticRegression
            ]): Model classifier
            grid (dict[str, Any]): Grid parameters

        Returns:
            Self: ClassifierModel
        """
        self._model = model
        self._grid = grid
        return self
