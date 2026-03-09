from typing import Any, Self, Literal, ClassVar

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    StratifiedShuffleSplit,
)

from eruption_forecast.logger import logger
from eruption_forecast.model.constants import DEFAULT_GRID_PARAMS
from eruption_forecast.utils.formatting import slugify_class_name
from eruption_forecast.utils.validation import validate_random_state


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
        - xgb: XGBoost classifier (excellent for imbalanced data)
        - voting: Ensemble VotingClassifier combining rf and xgb
        - lite-rf: Random Forest but faster with more simple grid parameters

    Attributes:
        classifier (str): Classifier type identifier ("svm", "knn", "dt", etc.).
        random_state (int | None): Random seed for reproducibility.
        cv_strategy (str): Cross-validation strategy ("shuffle", "stratified", "timeseries").
        n_splits (int): Number of cross-validation splits.
        test_size (float): Proportion of data for test set (used with StratifiedShuffleSplit).
        verbose (bool): Enable verbose logging.
        class_weight (str | dict[int, float] | None): Class weight configuration.
        n_jobs (int): Number of parallel jobs for compatible classifiers.
        cv_name (str): Name of the cross-validation splitter class.
        _model: Internal cached model instance.
        _grid: Internal cached hyperparameter grid.

    Args:
        classifier (Literal[ "svm", "knn", "dt", "rf", "gb", "xgb", "nn", "nb", "lr", "voting", "lite-rf"]):
            Classifier type identifier.
        random_state (int | None, optional): Random seed for reproducibility. Applies to
            classifiers that support it (rf, gb, dt, nn, lr, svm). Defaults to None.
        cv_strategy (Literal["shuffle", "stratified", "timeseries"], optional):
            Cross-validation strategy. Defaults to "shuffle".

            - "shuffle": StratifiedShuffleSplit (randomized stratified folds)
            - "stratified": StratifiedKFold (preserves class distribution)
            - "timeseries": TimeSeriesSplit (for temporal data, prevents data leakage)
        n_splits (int, optional): Number of cross-validation splits. Defaults to 5.
        test_size (float, optional): Test set proportion for StratifiedShuffleSplit.
            Defaults to 0.2.
        verbose (bool, optional): Enable verbose logging. Defaults to False.
        class_weight (str | dict[int, float] | None, optional): Class weight for
            imbalanced data. Defaults to None (will use "balanced" for most classifiers).
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.

    Examples:
        >>> # Create a Random Forest classifier
        >>> clf = ClassifierModel("rf")
        >>> model, grid = clf.model_and_grid
        >>> print(grid)
        {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], ...}

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

    # Maps classifier key → hyperparameter grid; used by the ``grid`` property.
    _GRID_REGISTRY: ClassVar[dict[str, dict[str, Any]]] = DEFAULT_GRID_PARAMS

    def __init__(
        self,
        classifier: Literal[
            "svm", "knn", "dt", "rf", "gb", "xgb", "nn", "nb", "lr", "voting", "lite-rf"
        ],
        random_state: int | None = None,
        cv_strategy: Literal["shuffle", "stratified", "timeseries"] = "shuffle",
        n_splits: int = 5,
        test_size: float = 0.2,
        verbose: bool = False,
        class_weight: str | dict[int, float] | None = None,
        n_jobs: int = 1,
        use_gpu: bool = False,
        gpu_id: int = 0,
    ):
        """Initialize the ClassifierModel with a classifier type and cross-validation settings.

        Stores configuration and lazily initialises the underlying scikit-learn or
        XGBoost estimator and its hyperparameter grid via the model and grid properties.
        Also determines the CV splitter class name for logging purposes.

        Args:
            classifier (Literal["svm", "knn", "dt", "rf", "gb", "xgb", "nn", "nb",
                "lr", "voting", "lite-rf"]): Short identifier for the classifier to use.
            random_state (int | None, optional): Random seed passed to classifiers
                that support it. Defaults to None.
            cv_strategy (Literal["shuffle", "stratified", "timeseries"], optional):
                Cross-validation strategy. Defaults to "shuffle".
            n_splits (int, optional): Number of CV folds. Defaults to 5.
            test_size (float, optional): Fraction of data used for the test split
                when cv_strategy is "shuffle". Defaults to 0.2.
            verbose (bool, optional): Enable verbose logging. Defaults to False.
            class_weight (str | dict[int, float] | None, optional): Class weight
                scheme passed to classifiers that support it. Defaults to None.
            n_jobs (int, optional): Number of parallel jobs for supported classifiers.
                Defaults to 1.
            use_gpu (bool, optional): Enable GPU acceleration for XGBoost via
                ``device="cuda"``. Defaults to False (CPU only). Enable only when
                training on a single large batch outside cross-validation loops,
                as repeated CPU↔GPU transfers during CV can cause instability.
            gpu_id (int, optional): GPU device index to use when use_gpu is True
                (e.g. 0 for the first GPU, 1 for the second). Ignored when
                use_gpu is False. Defaults to 0.
        """
        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.classifier = classifier
        self.random_state = random_state
        self.cv_strategy = cv_strategy
        self.n_splits = n_splits
        self.test_size = test_size
        self.verbose = verbose
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.gpu_id = gpu_id

        if use_gpu and classifier not in {"xgb", "voting"}:
            logger.warning(
                f"use_gpu=True has no effect for classifier '{classifier}'. "
                "GPU acceleration is only supported for 'xgb' and 'voting'."
            )

        # ------------------------------------------------------------------
        # Set ADDITIONAL properties (derived values)
        # ------------------------------------------------------------------
        self._model: (
            SVC
            | KNeighborsClassifier
            | DecisionTreeClassifier
            | RandomForestClassifier
            | GradientBoostingClassifier
            | XGBClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
            | VotingClassifier
            | None
        ) = None
        self._grid: dict[str, Any] | None = None
        self.cv_name = type(self.get_cv_splitter()).__name__

    def set_random_state(self, random_state: int) -> Self:
        """Set the random seed for reproducibility.

        Args:
            random_state (int): Random seed value (must be >= 0). Applies to all
                classifiers that support random state (rf, gb, dt, nn, lr, svm).

        Returns:
            Self: The ClassifierModel instance for method chaining.

        Raises:
            ValueError: If random_state is negative.

        Example:
            >>> clf = ClassifierModel("rf")
            >>> clf.set_random_state(42)
            >>> model, grid = clf.model_and_grid
        """
        validate_random_state(random_state)
        self.random_state = random_state
        return self

    def get_cv_splitter(
        self,
        strategy: Literal["shuffle", "stratified", "timeseries"] | None = None,
    ) -> TimeSeriesSplit | StratifiedKFold | StratifiedShuffleSplit:
        """Get the cross-validation splitter based on cv_strategy.

        Returns the appropriate cross-validator for the configured strategy.
        Used with GridSearchCV for hyperparameter tuning.

        Args:
            strategy (Literal["shuffle", "stratified", "timeseries"] | None, optional):
                Cross-validation strategy. If None, uses ``self.cv_strategy``.
                Defaults to None.

                - "shuffle": StratifiedShuffleSplit (randomized stratified folds)
                - "stratified": StratifiedKFold (preserves class distribution)
                - "timeseries": TimeSeriesSplit (for temporal data, prevents data leakage)

        Returns:
            TimeSeriesSplit | StratifiedKFold | StratifiedShuffleSplit:
                Configured sklearn cross-validation splitter instance.

        Examples:
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
        strategy = strategy or self.cv_strategy

        if strategy == "timeseries":
            return TimeSeriesSplit(n_splits=self.n_splits)

        if strategy == "stratified":
            return StratifiedKFold(
                n_splits=self.n_splits, shuffle=True, random_state=self.random_state
            )

        return StratifiedShuffleSplit(
            n_splits=self.n_splits,
            test_size=self.test_size,
            random_state=self.random_state,
        )

    @property
    def slug_name(self) -> str:
        """Get the slugified classifier name.

        Converts the underlying classifier class name to lowercase kebab-case
        by delegating to ``slugify_class_name``.  Useful for building
        filesystem-safe output paths.

        Returns:
            str: Slugified version of the classifier class name (lowercase with hyphens).

        Examples:
            >>> clf = ClassifierModel("rf")
            >>> clf.slug_name
            'random-forest-classifier'
        """
        return slugify_class_name(self.name)

    @property
    def slug_cv_name(self) -> str:
        """Get the slugified cross-validation strategy name.

        Converts the CV splitter class name to lowercase kebab-case by
        delegating to ``slugify_class_name``.  Useful for building
        filesystem-safe output paths.

        Returns:
            str: Slugified version of the CV class name (lowercase with hyphens).

        Examples:
            >>> clf = ClassifierModel("rf", cv_strategy="stratified")
            >>> clf.slug_cv_name
            'stratified-k-fold'
        """
        return slugify_class_name(self.cv_name)

    @property
    def grid(self) -> dict[str, Any]:
        """Get the hyperparameter grid for GridSearchCV.

        Returns the default grid for the configured classifier if no custom
        grid has been set. Otherwise returns the custom grid.

        Returns:
            dict[str, Any]: Dictionary mapping parameter names to lists of values
                to search over during hyperparameter optimization.

        Raises:
            ValueError: If the classifier type is unknown.

        Examples:
            >>> clf = ClassifierModel("rf")
            >>> print(clf.grid["n_estimators"])
            [50, 100, 200]

            >>> clf = ClassifierModel("xgb")
            >>> print(clf.grid["learning_rate"])
            [0.01, 0.1, 0.2]
        """
        if self._grid is not None:
            return self._grid

        if self.classifier in self._GRID_REGISTRY:
            return self._GRID_REGISTRY[self.classifier]

        raise ValueError(f"Unknown classifier: {self.classifier}")

    @grid.setter
    def grid(self, grid: dict[str, Any]):
        """Set a custom hyperparameter grid, overriding the default.

        Stores the provided grid in ``_grid`` so that subsequent calls to
        ``self.grid`` return the custom values instead of the defaults.  Pass
        ``None`` to ``_grid`` directly to revert to the default grid.

        Args:
            grid (dict[str, Any]): Custom hyperparameter grid for GridSearchCV.

        Examples:
            >>> clf = ClassifierModel("rf")
            >>> clf.grid = {"n_estimators": [100, 200], "max_depth": [10, None]}
        """
        self._grid = grid
        if self.verbose:
            logger.info(f"Your grid parameters updated to: {grid}")

    @property
    def model(
        self,
    ) -> (
        SVC
        | KNeighborsClassifier
        | DecisionTreeClassifier
        | RandomForestClassifier
        | GradientBoostingClassifier
        | XGBClassifier
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
            SVC | KNeighborsClassifier | DecisionTreeClassifier | RandomForestClassifier |
            GradientBoostingClassifier | XGBClassifier | MLPClassifier | GaussianNB |
            LogisticRegression | VotingClassifier: Configured classifier instance.

        Raises:
            ValueError: If the classifier type is unknown.

        Examples:
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

        return self._build_model()

    def _xgb_device(self) -> str:
        """Return the XGBoost device string based on use_gpu and gpu_id.

        Returns "cuda:<gpu_id>" when use_gpu is True so that a specific GPU
        can be targeted on multi-GPU machines; otherwise returns "cpu".

        Returns:
            str: "cuda:<gpu_id>" if use_gpu is True, otherwise "cpu".
        """
        return f"cuda:{self.gpu_id}" if self.use_gpu else "cpu"

    def _build_model(
        self,
    ) -> (
        SVC
        | KNeighborsClassifier
        | DecisionTreeClassifier
        | RandomForestClassifier
        | GradientBoostingClassifier
        | XGBClassifier
        | MLPClassifier
        | GaussianNB
        | LogisticRegression
        | VotingClassifier
    ):
        """Construct and return a new classifier instance based on self.classifier.

        Called by the ``model`` property when no custom model has been set.
        Uses a dictionary of callables keyed by classifier name to eliminate
        repetitive if-elif chains.

        Returns:
            SVC | KNeighborsClassifier | DecisionTreeClassifier | RandomForestClassifier |
            GradientBoostingClassifier | XGBClassifier | MLPClassifier | GaussianNB |
            LogisticRegression | VotingClassifier: A freshly constructed classifier
            instance configured with the current random_state, class_weight, and n_jobs.

        Raises:
            ValueError: If self.classifier is not a recognised key.
        """
        _class_weight = (
            self.class_weight if self.class_weight is not None else "balanced"
        )

        # Factory dict mapping classifier key → zero-argument callable that constructs the estimator.
        _model_registry: dict[str, Any] = {
            "svm": lambda: SVC(
                class_weight=_class_weight,
                random_state=self.random_state,
                probability=True,
            ),
            "knn": lambda: KNeighborsClassifier(),
            "dt": lambda: DecisionTreeClassifier(
                class_weight=_class_weight, random_state=self.random_state
            ),
            "rf": lambda: RandomForestClassifier(
                class_weight=_class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "lite-rf": lambda: RandomForestClassifier(
                class_weight=_class_weight,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "gb": lambda: GradientBoostingClassifier(random_state=self.random_state),
            "xgb": lambda: XGBClassifier(
                device=self._xgb_device(),
                tree_method="hist",
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            "nn": lambda: MLPClassifier(
                alpha=1, max_iter=1000, random_state=self.random_state
            ),
            "nb": lambda: GaussianNB(),
            "lr": lambda: LogisticRegression(
                class_weight=_class_weight,
                random_state=self.random_state,
                max_iter=1000,
            ),
            "voting": lambda: VotingClassifier(
                estimators=[
                    (
                        "rf",
                        RandomForestClassifier(
                            class_weight=_class_weight,
                            random_state=self.random_state,
                            n_jobs=self.n_jobs,
                        ),
                    ),
                    (
                        "xgb",
                        XGBClassifier(
                            device=self._xgb_device(),
                            tree_method="hist",
                            eval_metric="logloss",
                            random_state=self.random_state,
                            n_jobs=self.n_jobs,
                        ),
                    ),
                ],
                voting="soft",
            ),
        }

        if self.classifier not in _model_registry:
            raise ValueError(f"Unknown classifier: {self.classifier}")

        return _model_registry[self.classifier]()

    @model.setter
    def model(
        self,
        model: (
            SVC
            | KNeighborsClassifier
            | DecisionTreeClassifier
            | RandomForestClassifier
            | GradientBoostingClassifier
            | XGBClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
            | VotingClassifier
        ),
    ):
        """Set a custom classifier instance, overriding the default.

        Stores the estimator in ``_model`` so that subsequent calls to
        ``self.model`` return the custom instance instead of constructing a new
        default one.

        Args:
            model (SVC | KNeighborsClassifier | DecisionTreeClassifier |
                RandomForestClassifier | GradientBoostingClassifier | XGBClassifier |
                MLPClassifier | GaussianNB | LogisticRegression | VotingClassifier):
                Custom classifier instance.

        Examples:
            >>> from sklearn.ensemble import RandomForestClassifier
            >>> clf = ClassifierModel("rf")
            >>> clf.model = RandomForestClassifier(n_estimators=500, max_depth=20)
        """
        self._model = model
        if self.verbose:
            logger.info(f"Your model to: {model.__class__.__name__}")

    @property
    def name(self) -> str:
        """Get the classifier class name.

        Returns ``type(self.model).__name__``, prefixed with ``"Lite"`` when
        the ``"lite-rf"`` classifier shorthand is active.

        Returns:
            str: The name of the classifier class (e.g., "RandomForestClassifier").

        Examples:
            >>> clf = ClassifierModel("rf")
            >>> clf.name
            'RandomForestClassifier'
        """
        class_type = type(self.model).__name__
        if self.classifier == "lite-rf":
            return f"Lite{class_type}"
        return class_type

    @property
    def model_and_grid(self) -> tuple:
        """Get a (model, grid) tuple for use with GridSearchCV.

        Convenience accessor that bundles ``self.model`` and ``self.grid`` into
        a single tuple, allowing it to be unpacked directly into
        ``GridSearchCV(*clf.model_and_grid, cv=...)``.

        Returns:
            tuple: A 2-tuple containing (classifier_instance, hyperparameter_grid).

        Examples:
            >>> clf = ClassifierModel("rf")
            >>> model, grid = clf.model_and_grid
            >>> from sklearn.model_selection import GridSearchCV
            >>> search = GridSearchCV(*clf.model_and_grid, cv=5)
        """
        return self.model, self.grid

    def update_model_and_grid(
        self,
        model: (
            SVC
            | KNeighborsClassifier
            | DecisionTreeClassifier
            | RandomForestClassifier
            | GradientBoostingClassifier
            | XGBClassifier
            | MLPClassifier
            | GaussianNB
            | LogisticRegression
            | VotingClassifier
        ),
        grid: dict[str, Any],
    ) -> Self:
        """Update both model classifier and hyperparameter grid.

        Convenience method to set both model and grid simultaneously with
        method chaining support.

        Args:
            model (SVC | KNeighborsClassifier | DecisionTreeClassifier |
                RandomForestClassifier | GradientBoostingClassifier | XGBClassifier |
                MLPClassifier | GaussianNB | LogisticRegression | VotingClassifier):
                Classifier instance to use. Accepts any supported sklearn or
                XGBoost estimator.
            grid (dict[str, Any]): Hyperparameter grid for GridSearchCV.

        Returns:
            Self: The ClassifierModel instance for method chaining.

        Examples:
            >>> from sklearn.svm import SVC
            >>> clf = ClassifierModel("svm")
            >>> clf.update_model_and_grid(
            ...     SVC(kernel="rbf", probability=True),
            ...     {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]}
            ... )

            >>> # Method chaining
            >>> clf = ClassifierModel("rf").update_model_and_grid(
            ...     RandomForestClassifier(n_estimators=500),
            ...     {"max_depth": [10, 20, None]}
            ... )
        """
        self._model = model
        self._grid = grid
        if self.verbose:
            logger.info(f"Your model updated to: {model.__class__.__name__}")
            logger.info(f"Your grid parameters updated to: {grid}")
        return self
