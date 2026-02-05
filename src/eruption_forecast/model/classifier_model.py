# Standard library imports
from typing import Any, Literal, Self

# Third party imports
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)
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
        - gb: Gradient Boosting (handles imbalanced data well)
        - nn: Multi-Layer Perceptron Neural Network
        - nb: Gaussian Naive Bayes
        - lr: Logistic Regression (with balanced class weights)
        - voting: Ensemble VotingClassifier combining rf, gb, lr, and svm

    Args:
        classifier: Classifier type identifier.
        random_state: Random seed for reproducibility. Applies to classifiers
            that support it (rf, gb, dt, nn, lr, svm). Defaults to None.
        cv_strategy: Cross-validation strategy. Options:
            - "stratified": StratifiedKFold (default, preserves class distribution)
            - "timeseries": TimeSeriesSplit (for temporal data, prevents data leakage)
        n_splits: Number of cross-validation splits. Defaults to 5.

    Example:
        >>> # Create a Random Forest classifier
        >>> clf = ClassifierModel("rf")
        >>> model, grid = clf.model_and_grid
        >>> print(grid)
        {'n_estimators': [10, 30, 100], 'max_depth': [3, 5, 7], ...}

        >>> # Create a Gradient Boosting classifier with random state
        >>> clf = ClassifierModel("gb", random_state=42)
        >>> model, grid = clf.model_and_grid

        >>> # Use TimeSeriesSplit for temporal data (prevents data leakage)
        >>> clf = ClassifierModel("rf", cv_strategy="timeseries", n_splits=5)
        >>> cv = clf.get_cv_splitter()

        >>> # Create VotingClassifier ensemble
        >>> clf = ClassifierModel("voting", random_state=42)
        >>> model, grid = clf.model_and_grid

        >>> # Custom grid parameters
        >>> clf = ClassifierModel("rf", random_state=123)
        >>> clf.grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10]}

        >>> # Chain model and grid updates
        >>> clf = ClassifierModel("svm").update_model_and_grid(
        ...     SVC(kernel="rbf"),
        ...     {"C": [0.1, 1, 10]}
        ... )
    """

    def __init__(
        self,
        classifier: Literal["svm", "knn", "dt", "rf", "gb", "nn", "nb", "lr", "voting"],
        random_state: int | None = None,
        cv_strategy: Literal["shuffle", "stratified", "timeseries"] = "shuffle",
        n_splits: int = 5,
        test_size: float = 0.2,
    ):
        self.classifier = classifier
        self.random_state = random_state
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.test_size = test_size
        self._model: (
            SVC
            | KNeighborsClassifier
            | DecisionTreeClassifier
            | RandomForestClassifier
            | GradientBoostingClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
            | VotingClassifier
            | None
        ) = None
        self._grid: dict[str, Any] | None = None

    def get_cv_splitter(self) -> BaseCrossValidator:
        """Get the cross-validation splitter based on cv_strategy.

        Returns the appropriate cross-validator for the configured strategy:
        - "stratified": StratifiedKFold (preserves class distribution)
        - "timeseries": TimeSeriesSplit (for temporal data)

        Returns:
            BaseCrossValidator: sklearn cross-validation splitter instance.

        Example:
            >>> # Use StratifiedKFold (default)
            >>> clf = ClassifierModel("rf", cv_strategy="stratified")
            >>> cv = clf.get_cv_splitter()
            >>> print(type(cv).__name__)
            'StratifiedKFold'

            >>> # Use TimeSeriesSplit for temporal data
            >>> clf = ClassifierModel("rf", cv_strategy="timeseries", n_splits=5)
            >>> cv = clf.get_cv_splitter()
            >>> print(type(cv).__name__)
            'TimeSeriesSplit'

            >>> # Use with GridSearchCV
            >>> from sklearn.model_selection import GridSearchCV
            >>> clf = ClassifierModel("rf", cv_strategy="timeseries")
            >>> grid_search = GridSearchCV(clf.model, clf.grid, cv=clf.get_cv_splitter())
        """
        if self.cv_strategy == "timeseries":
            return TimeSeriesSplit(n_splits=self.n_splits)

        if self.cv_strategy == "stratified":
            return StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )

        return StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        )

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

        if self.classifier == "svm" or isinstance(self._model, SVC):
            return {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "kernel": ["poly", "rbf", "sigmoid"],
                "degree": [2, 3, 4, 5],
                "decision_function_shape": ["ovo", "ovr"],
            }

        if self.classifier == "knn" or isinstance(self._model, KNeighborsClassifier):
            return {
                "n_neighbors": [3, 6, 12, 24],
                "weights": ["uniform", "distance"],
                "p": [1, 2, 3],
            }

        if self.classifier == "dt" or isinstance(self._model, DecisionTreeClassifier):
            return {
                "max_depth": [3, 5, 7],
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", "log2", None],
            }

        if self.classifier == "rf" or isinstance(self._model, RandomForestClassifier):
            return {
                "n_estimators": [10, 30, 100],
                "max_depth": [3, 5, 7],
                "criterion": ["gini", "entropy"],
                "max_features": ["sqrt", "log2", None],
            }

        if self.classifier == "gb" or isinstance(
            self._model, GradientBoostingClassifier
        ):
            return {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
            }

        if self.classifier == "nn" or isinstance(self._model, MLPClassifier):
            return {
                "activation": ["identity", "logistic", "tanh", "relu"],
                "hidden_layer_sizes": [10, 100],
            }

        if self.classifier == "nb" or isinstance(self._model, GaussianNB):
            return {"var_smoothing": [1.0]}

        if self.classifier == "lr" or isinstance(self._model, LogisticRegression):
            return {
                "penalty": ["l2", "l1", "elasticnet"],
                "C": [0.001, 0.01, 0.1, 1, 10],
            }

        if self.classifier == "voting" or isinstance(self._model, VotingClassifier):
            # Grid for VotingClassifier ensemble
            # Parameters are prefixed with estimator name (e.g., rf__, gb__, etc.)
            return {
                "rf__n_estimators": [50, 100],
                "rf__max_depth": [5, 7],
                "gb__n_estimators": [50, 100],
                "gb__learning_rate": [0.1, 0.2],
                "lr__C": [0.1, 1, 10],
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
        | GradientBoostingClassifier
        | MLPClassifier
        | GaussianNB
        | LogisticRegression
        | VotingClassifier
    ):
        """Get the classifier model instance.

        Returns a new instance of the classifier with default settings
        if none has been set. Most classifiers use balanced class weights
        to handle the imbalanced eruption/non-eruption data.

        Returns:
            Classifier instance (SVC, KNeighborsClassifier, DecisionTreeClassifier,
            RandomForestClassifier, GradientBoostingClassifier, MLPClassifier,
            GaussianNB, LogisticRegression, or VotingClassifier).

        Example:
            >>> clf = ClassifierModel("rf")
            >>> model = clf.model
            >>> print(type(model).__name__)
            'RandomForestClassifier'

            >>> clf = ClassifierModel("voting")
            >>> model = clf.model
            >>> print(type(model).__name__)
            'VotingClassifier'
        """
        if self._model is not None:
            return self._model

        if self.classifier == "svm":
            return SVC(
                class_weight="balanced",
                random_state=self.random_state,
                probability=True,
            )

        if self.classifier == "knn":
            return KNeighborsClassifier()

        if self.classifier == "dt":
            return DecisionTreeClassifier(
                class_weight="balanced", random_state=self.random_state
            )

        if self.classifier == "rf":
            return RandomForestClassifier(
                class_weight="balanced", random_state=self.random_state
            )

        if self.classifier == "gb":
            return GradientBoostingClassifier(random_state=self.random_state)

        if self.classifier == "nn":
            return MLPClassifier(alpha=1, max_iter=1000, random_state=self.random_state)

        if self.classifier == "nb":
            return GaussianNB()

        if self.classifier == "lr":
            return LogisticRegression(
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=1000,
            )

        if self.classifier == "voting":
            # Ensemble VotingClassifier combining top-performing classifiers
            # Uses soft voting for probability-based predictions
            return VotingClassifier(
                estimators=[
                    (
                        "rf",
                        RandomForestClassifier(
                            class_weight="balanced", random_state=self.random_state
                        ),
                    ),
                    ("gb", GradientBoostingClassifier(random_state=self.random_state)),
                    (
                        "lr",
                        LogisticRegression(
                            class_weight="balanced",
                            random_state=self.random_state,
                            max_iter=1000,
                        ),
                    ),
                    (
                        "svm",
                        SVC(
                            class_weight="balanced",
                            random_state=self.random_state,
                            probability=True,
                        ),
                    ),
                ],
                voting="soft",
            )

        raise ValueError(f"Unknown classifier: {self.classifier}")

    @model.setter
    def model(
        self,
        model: (
            SVC
            | KNeighborsClassifier
            | DecisionTreeClassifier
            | RandomForestClassifier
            | GradientBoostingClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
            | VotingClassifier
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
            | GradientBoostingClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
            | VotingClassifier
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
                GradientBoostingClassifier,
                MLPClassifier,
                GaussianNB,
                LogisticRegression,
                VotingClassifier
            ]): Model classifier
            grid (dict[str, Any]): Grid parameters

        Returns:
            Self: ClassifierModel
        """
        self._model = model
        self._grid = grid
        return self
