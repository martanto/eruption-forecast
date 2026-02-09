# Standard library imports
from typing import Literal

# Third party imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from tsfresh.feature_selection import select_features

# Project imports
from eruption_forecast.logger import logger


class FeatureSelector:
    """Two-stage feature selection combining tsfresh and RandomForest.

    Implements a robust two-stage pipeline:
    1. **Stage 1 (tsfresh)**: Statistical significance testing with FDR control
       - Fast filtering based on univariate statistical tests
       - Model-agnostic approach
       - Reduces features from 1000s → 100s

    2. **Stage 2 (RandomForest)**: Permutation importance analysis
       - Captures feature interactions
       - Model-specific refinement
       - Reduces features from 100s → final set (e.g., 20-50)

    This approach combines:
    - Speed and statistical rigor (tsfresh)
    - Interaction capture and model optimization (RandomForest)
    - Lower overfitting risk (statistical pre-filtering)
    - Better interpretability (p-values AND importance scores)

    Args:
        method (str): Feature selection method:
            - "tsfresh": Statistical significance only (Stage 1)
            - "random_forest": Permutation importance only
            - "combined": Two-stage (tsfresh → RandomForest) [RECOMMENDED]
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        verbose (bool, optional): Verbose logging. Defaults to False.

    Attributes:
        selected_features_ (pd.Series): Selected features with their scores
        p_values_ (pd.Series): P-values from tsfresh (if applicable)
        importance_scores_ (pd.Series): Permutation importance scores (if applicable)
        n_features_stage1_ (int): Number of features after stage 1
        n_features_stage2_ (int): Number of features after stage 2

    Example:
        >>> # Two-stage selection (recommended)
        >>> selector = FeatureSelector(method="combined", n_jobs=4)
        >>> selector.fit(X_train, y_train, top_n=30)
        >>> X_selected = selector.transform(X_train)
        >>> print(f"Reduced: {X_train.shape[1]} → {X_selected.shape[1]} features")

        >>> # Access selection results
        >>> print("Top features by p-value:", selector.p_values_.head(10))
        >>> print("Top features by importance:", selector.importance_scores_.head(10))

        >>> # Compare with tsfresh-only selection
        >>> selector_tsfresh = FeatureSelector(method="tsfresh")
        >>> selector_tsfresh.fit(X_train, y_train, fdr_level=0.05)
        >>> X_tsfresh = selector_tsfresh.transform(X_train)
    """

    def __init__(
        self,
        method: Literal["tsfresh", "random_forest", "combined"] = "combined",
        n_jobs: int = 1,
        random_state: int = 42,
        verbose: bool = False,
    ) -> None:
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Will be set after fit()
        self.selected_features_: pd.Series = pd.Series(dtype=float)
        self.p_values_: pd.Series = pd.Series(dtype=float)
        self.importance_scores_: pd.Series = pd.Series(dtype=float)
        self.n_features_stage1_: int = 0
        self.n_features_stage2_: int = 0
        self.feature_names_: list[str] = []

    def _select_tsfresh(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fdr_level: float = 0.05,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Stage 1: tsfresh statistical feature selection.

        Uses hypothesis testing to filter features based on statistical
        significance with FDR (False Discovery Rate) control.

        Args:
            X: Features DataFrame
            y: Target labels
            fdr_level: False discovery rate level (default: 0.05)

        Returns:
            Tuple of (filtered_features, p_values)
        """
        if self.verbose:
            logger.info("Stage 1: tsfresh statistical feature selection...")

        # Use tsfresh's select_features function
        X_filtered = select_features(
            X,
            y,
            fdr_level=fdr_level,
            n_jobs=self.n_jobs,
            ml_task="classification",
        )

        # Get p-values using get_significant_features utility
        # Project imports
        from eruption_forecast.utils import get_significant_features

        p_values = get_significant_features(X, y, n_jobs=self.n_jobs)

        # Filter p_values to only include selected features
        p_values = p_values[p_values.index.isin(X_filtered.columns)]

        self.n_features_stage1_ = X_filtered.shape[1]
        self.p_values_ = p_values

        if self.verbose:
            logger.info(
                f"tsfresh: {X.shape[1]} → {X_filtered.shape[1]} features "
                f"(FDR={fdr_level})"
            )

        return X_filtered, p_values

    def _select_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 30,
        n_estimators: int = 200,
        max_depth: int | None = 10,
        min_samples_leaf: int = 20,
        n_repeats: int = 10,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Stage 2: RandomForest permutation importance.

        Trains a RandomForest and uses permutation importance to rank features.
        This captures feature interactions and model-specific importance.

        Args:
            X: Features DataFrame
            y: Target labels
            top_n: Number of top features to select
            n_estimators: Number of trees in RandomForest
            max_depth: Maximum tree depth (None for unlimited)
            min_samples_leaf: Minimum samples per leaf (regularization)
            n_repeats: Number of permutation repeats

        Returns:
            Tuple of (selected_features, importance_scores)
        """
        if self.verbose:
            logger.info("Stage 2: RandomForest permutation importance...")

        # Train RandomForest with regularization to prevent overfitting
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        rf.fit(X, y)  # type: ignore

        # Use permutation importance (more reliable than Gini importance)
        perm_importance = permutation_importance(
            rf,
            X,  # type: ignore
            y,  # type: ignore
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # Create importance DataFrame
        # Note: permutation_importance returns a Bunch object with attributes
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm_importance.importances_mean,  # type: ignore[attr-defined]
                "importance_std": perm_importance.importances_std,  # type: ignore[attr-defined]
            }
        ).sort_values("importance_mean", ascending=False)

        # Select top N features
        top_features = importance_df.head(top_n)["feature"].tolist()
        X_selected = X[top_features]

        # Create importance series
        importance_scores = pd.Series(
            importance_df["importance_mean"].values,
            index=importance_df["feature"].values,
        )
        importance_scores.name = "importance"
        importance_scores.index.name = "features"

        self.n_features_stage2_ = X_selected.shape[1]
        self.importance_scores_ = importance_scores

        if self.verbose:
            logger.info(f"RandomForest: {X.shape[1]} → {X_selected.shape[1]} features")

        return X_selected, importance_scores

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fdr_level: float = 0.05,
        top_n: int = 30,
        **rf_kwargs,
    ) -> "FeatureSelector":
        """Fit the feature selector on training data.

        Args:
            X: Training features DataFrame
            y: Training labels Series
            fdr_level: FDR level for tsfresh (default: 0.05)
            top_n: Number of top features for final selection (default: 30)
            **rf_kwargs: Additional keyword arguments for RandomForest
                (n_estimators, max_depth, min_samples_leaf, n_repeats)

        Returns:
            self: Fitted FeatureSelector instance

        Raises:
            ValueError: If method is invalid or X/y are empty

        Example:
            >>> selector = FeatureSelector(method="combined")
            >>> selector.fit(X_train, y_train, fdr_level=0.05, top_n=20)
            >>> X_selected = selector.transform(X_train)
        """
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("X and y cannot be empty")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same length. X: {X.shape[0]}, y: {y.shape[0]}"
            )

        n_features_initial = X.shape[1]

        if self.method == "tsfresh":
            # Stage 1 only: tsfresh statistical selection
            X_filtered, p_values = self._select_tsfresh(X, y, fdr_level=fdr_level)
            self.selected_features_ = p_values
            self.feature_names_ = X_filtered.columns.tolist()

        elif self.method == "random_forest":
            # Stage 2 only: RandomForest permutation importance
            X_selected, importance_scores = self._select_random_forest(
                X, y, top_n=top_n, **rf_kwargs
            )
            self.selected_features_ = importance_scores
            self.feature_names_ = X_selected.columns.tolist()

        elif self.method == "combined":
            # Two-stage: tsfresh → RandomForest
            if self.verbose:
                logger.info("Running two-stage feature selection...")

            # Stage 1: tsfresh
            X_filtered, p_values = self._select_tsfresh(X, y, fdr_level=fdr_level)

            # Stage 2: RandomForest on filtered features
            X_selected, importance_scores = self._select_random_forest(
                X_filtered, y, top_n=top_n, **rf_kwargs
            )

            self.selected_features_ = importance_scores
            self.feature_names_ = X_selected.columns.tolist()

            if self.verbose:
                logger.info(
                    f"Two-stage selection: {n_features_initial} → "
                    f"{self.n_features_stage1_} → {self.n_features_stage2_} features"
                )

        else:
            raise ValueError(
                f"Invalid method: {self.method}. "
                "Must be 'tsfresh', 'random_forest', or 'combined'"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting only fitted features.

        Args:
            X: Features DataFrame to transform

        Returns:
            Transformed DataFrame with selected features only

        Raises:
            ValueError: If selector hasn't been fitted yet

        Example:
            >>> selector = FeatureSelector(method="combined")
            >>> selector.fit(X_train, y_train)
            >>> X_train_selected = selector.transform(X_train)
            >>> X_test_selected = selector.transform(X_test)
        """
        if len(self.feature_names_) == 0:
            raise ValueError(
                "FeatureSelector has not been fitted yet. Call fit() first."
            )

        # Check if all selected features exist in X
        missing_features = set(self.feature_names_) - set(X.columns)
        if missing_features:
            raise ValueError(
                f"Features missing in X: {missing_features}. "
                "Make sure to transform data from the same feature space."
            )

        return X[self.feature_names_]

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fdr_level: float = 0.05,
        top_n: int = 30,
        **rf_kwargs,
    ) -> pd.DataFrame:
        """Fit the selector and transform X in one step.

        Args:
            X: Training features DataFrame
            y: Training labels Series
            fdr_level: FDR level for tsfresh (default: 0.05)
            top_n: Number of top features (default: 30)
            **rf_kwargs: Additional kwargs for RandomForest

        Returns:
            Transformed DataFrame with selected features

        Example:
            >>> selector = FeatureSelector(method="combined")
            >>> X_selected = selector.fit_transform(X_train, y_train, top_n=20)
        """
        self.fit(X, y, fdr_level=fdr_level, top_n=top_n, **rf_kwargs)
        return self.transform(X)

    def get_feature_scores(self) -> pd.DataFrame:
        """Get comprehensive feature scores from selection.

        Returns:
            DataFrame with features, p-values (if available), and
            importance scores (if available)

        Example:
            >>> selector = FeatureSelector(method="combined")
            >>> selector.fit(X_train, y_train)
            >>> scores = selector.get_feature_scores()
            >>> print(scores.head(10))
        """
        df = pd.DataFrame({"feature": self.feature_names_})

        if not self.p_values_.empty:
            df = df.merge(
                self.p_values_.reset_index(),
                left_on="feature",
                right_on="features",
                how="left",
            ).drop("features", axis=1)

        if not self.importance_scores_.empty:
            df = df.merge(
                self.importance_scores_.reset_index(),
                left_on="feature",
                right_on="features",
                how="left",
            ).drop("features", axis=1)

        return df.set_index("feature")
