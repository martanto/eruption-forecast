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
        """Grid parameter for cross-validation

        Returns:
            dict[str, Any]: Grid parameters
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
                "max_features": ["auto", "sqrt", "log2", None],
            }

        if self.classifier == "rf" or isinstance(self.model, RandomForestClassifier):
            return {
                "n_estimators": [10, 30, 100],
                "max_depth": [3, 5, 7],
                "criterion": ["gini", "entropy"],
                "max_features": ["auto", "sqrt", "log2", None],
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
        """Model classifier

        Returns:
            Model:
                SVC: Support Vector Machine
                KNeighborsClassifier: K-Nearest Neighbors
                DecisionTreeClassifier: Decision Tree
                RandomForestClassifier: Random Forest Classifier
                MLPClassifier: MLP (Neural Network)
                GaussianNB: Gaussian Naive Bayes
                LogisticRegression: Logistic Regression
            Grid (dict[str, any]): Grid Parameters
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
            grid (dict[str, any]): Grid parameters

        Returns:
            Self: ClassifierModel
        """
        self._model = model
        self._grid = grid
        return self
