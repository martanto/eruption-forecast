import os
from typing import Self, Literal

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from eruption_forecast.logger import logger
from eruption_forecast.utils.ml import get_significant_features


class FeatureSelector:
    """Two-stage feature selection combining tsfresh and RandomForest.

    Implements a robust two-stage feature selection pipeline that combines
    statistical significance testing with machine learning importance analysis
    for optimal feature reduction and model performance.

    **Stage 1 (tsfresh)**: Statistical significance testing with FDR control
        - Fast filtering based on univariate statistical tests
        - Model-agnostic approach using hypothesis testing
        - Reduces features from thousands → hundreds
        - Controls False Discovery Rate (FDR) to avoid false positives

    **Stage 2 (RandomForest)**: Permutation importance analysis
        - Captures feature interactions and non-linear relationships
        - Model-specific refinement using RandomForest classifier
        - Reduces features from hundreds → final set (e.g., 20-50)
        - Uses permutation importance (more reliable than Gini impurity)

    This two-stage approach combines:
        - Speed and statistical rigor (tsfresh)
        - Interaction capture and model optimization (RandomForest)
        - Lower overfitting risk (statistical pre-filtering)
        - Better interpretability (p-values AND importance scores)

    Attributes:
        method (str): Feature selection method ("tsfresh", "random_forest", or "combined").
        n_jobs (int): Number of parallel jobs for computation.
        random_state (int): Random seed for reproducibility.
        output_dir (str): Directory for saving selected features.
        verbose (bool): Enable verbose logging.
        selected_features_ (pd.Series): Selected features with their scores (set after fit).
        p_values_ (pd.Series): P-values from tsfresh selection (set after fit, if applicable).
        importance_scores_ (pd.Series): Permutation importance scores (set after fit, if applicable).
        n_features_tsfresh (int): Number of features after tsfresh selection (set after fit).
        n_features_rf (int): Number of features after RandomForest selection (set after fit).
        feature_names_ (list[str]): List of selected feature names (set after fit).

    Args:
        method (Literal["tsfresh", "random_forest", "combined"], optional):
            Feature selection method. Options:
            - "tsfresh": Statistical significance only (Stage 1). Fast and model-agnostic.
            - "random_forest": Permutation importance only (Stage 2). Captures interactions.
            - "combined": Two-stage pipeline (tsfresh → RandomForest). **RECOMMENDED**.
            Defaults to "tsfresh".
        random_state (int, optional): Random seed for reproducibility.
            Applies to RandomForest classifier and permutation importance.
            Defaults to 42.
        output_dir (str | None, optional): Directory for saving selected features.
            If None, defaults to "output/features/selected" in current directory.
            Defaults to None.
        n_jobs (int, optional): Number of parallel jobs for computation.
            Used in both tsfresh and RandomForest stages. Defaults to 1.
        verbose (bool, optional): Enable verbose logging. Defaults to False.

    Raises:
        ValueError: If n_jobs is less than 1.

    Examples:
        >>> # Two-stage selection (recommended for best results)
        >>> selector = FeatureSelector(method="combined", n_jobs=4)
        >>> selector.fit(X_train, y_train, fdr_level=0.05, top_n=30)
        >>> X_selected = selector.transform(X_train)
        >>> print(f"Reduced: {X_train.shape[1]} → {X_selected.shape[1]} features")
        Reduced: 5000 → 30 features
        >>>
        >>> # Access selection results
        >>> print("Top features by p-value:")
        >>> print(selector.p_values_.head(10))
        >>> print("Top features by importance:")
        >>> print(selector.importance_scores_.head(10))
        >>>
        >>> # Compare with tsfresh-only selection
        >>> selector_tsfresh = FeatureSelector(method="tsfresh")
        >>> selector_tsfresh.fit(X_train, y_train, fdr_level=0.05)
        >>> X_tsfresh = selector_tsfresh.transform(X_train)
        >>>
        >>> # Use fit_transform for one-step operation
        >>> selector = FeatureSelector(method="combined")
        >>> X_selected = selector.fit_transform(X_train, y_train, top_n=20)
    """

    def __init__(
        self,
        method: Literal["tsfresh", "random_forest", "combined"] = "tsfresh",
        random_state: int = 42,
        output_dir: str | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> None:
        """Initialize the FeatureSelector with method, random state, and output settings.

        Sets up selection parameters and initialises result attributes to empty defaults.
        Calls validate() to check that n_jobs is at least 1.

        Args:
            method (Literal["tsfresh", "random_forest", "combined"], optional):
                Feature selection strategy. Defaults to "tsfresh".
            random_state (int, optional): Random seed for reproducibility.
                Defaults to 42.
            output_dir (str | None, optional): Directory for saving selected features.
                Defaults to ``os.getcwd()/output/features/selected``.
            n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
            verbose (bool, optional): Enable verbose logging. Defaults to False.

        Raises:
            ValueError: If n_jobs is less than 1.
        """
        # ------------------------------------------------------------------
        # Set DEFAULT parameter
        # ------------------------------------------------------------------
        output_dir = output_dir or os.path.join(
            os.getcwd(), "output", "features", "selected"
        )

        # ------------------------------------------------------------------
        # Set DEFAULT properties
        # ------------------------------------------------------------------
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.output_dir = output_dir
        self.verbose = verbose

        # ------------------------------------------------------------------
        # Will be set after fit() method called
        # ------------------------------------------------------------------
        self.selected_features_: pd.Series = pd.Series(dtype=float)
        self.p_values_: pd.Series = pd.Series(dtype=float)
        self.importance_scores_: pd.Series = pd.Series(dtype=float)
        self.n_features_tsfresh: int = 0
        self.n_features_rf: int = 0
        self.feature_names_: list[str] = []

        # ------------------------------------------------------------------
        # Validate and create directories
        # ------------------------------------------------------------------
        self.validate()

    def validate(self) -> None:
        """Validate the feature selector parameters.

        Checks that n_jobs is at least 1. Called automatically during __init__.

        Raises:
            ValueError: If n_jobs is less than 1.

        Returns:
            None
        """
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1. Your value is {self.n_jobs}")
        return None

    def set_random_state(self, random_state: int) -> Self:
        """Change the random state value for reproducibility.

        Updates the random seed used by the RandomForest classifier and
        permutation importance calculation.

        Args:
            random_state (int): Random seed for reproducibility. Must be >= 0.
                Applies to RandomForest classifier training and permutation
                importance estimation.

        Returns:
            Self: FeatureSelector instance for method chaining.

        Raises:
            ValueError: If random_state is less than 0.

        Examples:
            >>> selector = FeatureSelector(method="combined", random_state=42)
            >>> selector.set_random_state(123)
            >>> selector.fit(X_train, y_train)
        """
        if random_state < 0:
            raise ValueError(f"random_state must be >= 0. Your value is {random_state}")
        self.random_state = random_state
        return self

    def _select_tsfresh(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fdr_level: float = 0.05,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Stage 1: tsfresh statistical feature selection with FDR control.

        Uses hypothesis testing to filter features based on statistical
        significance with FDR (False Discovery Rate) control. This stage
        performs univariate tests between each feature and the target variable.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            y (pd.Series): Target labels with shape (n_samples,).
            fdr_level (float, optional): False Discovery Rate level for multiple
                testing correction. Controls the expected proportion of false
                positives among selected features. Defaults to 0.05.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - filtered_features (pd.DataFrame): Features that passed significance test
                - p_values (pd.Series): P-values for all features (indexed by feature name)

        Examples:
            >>> selector = FeatureSelector(method="tsfresh", verbose=True)
            >>> X_filtered, p_values = selector._select_tsfresh(X, y, fdr_level=0.05)
            >>> print(f"Features: {X.shape[1]} → {X_filtered.shape[1]}")
            Features: 5000 → 237
            >>> print(p_values.head())
        """
        if self.verbose:
            logger.info(
                f"{self.random_state:05d}: tsfresh statistical feature selection..."
            )

        # Get p-values using get_significant_features utility
        X_filtered, p_values = get_significant_features(
            X, y, fdr_level=fdr_level, n_jobs=self.n_jobs
        )

        self.n_features_tsfresh = X_filtered.shape[1]
        self.p_values_ = p_values

        if self.verbose:
            logger.info(
                f"{self.random_state:05d}: Features reduced: {X.shape[1]} → {self.n_features_tsfresh} "
                f"(FDR={fdr_level})"
            )

        return X_filtered, p_values

    def _select_random_forest(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 20,
        n_estimators: int = 200,
        max_depth: int | None = 10,
        min_samples_leaf: int = 20,
        n_repeats: int = 10,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Stage 2: RandomForest permutation importance selection.

        Trains a RandomForest classifier and uses permutation importance to rank
        features. This approach captures feature interactions and provides
        model-specific importance scores. Permutation importance is more reliable
        than Gini impurity as it directly measures impact on model performance.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            y (pd.Series): Target labels with shape (n_samples,).
            top_n (int, optional): Number of top features to select based on
                permutation importance ranking. Defaults to 20.
            n_estimators (int, optional): Number of trees in the RandomForest.
                More trees generally improve stability but increase computation time.
                Defaults to 200.
            max_depth (int | None, optional): Maximum tree depth. None means unlimited
                depth (trees grow until pure leaves). Limited depth acts as
                regularization to prevent overfitting. Defaults to 10.
            min_samples_leaf (int, optional): Minimum number of samples required at a
                leaf node. Acts as regularization to prevent overfitting by avoiding
                very specific splits. Defaults to 20.
            n_repeats (int, optional): Number of permutation repeats for importance
                estimation. Higher values give more stable and reliable importance
                scores but increase computation time. Defaults to 10.

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple containing:
                - selected_features (pd.DataFrame): Top-N features by importance
                - importance_scores (pd.Series): Mean permutation importance for all
                  features (indexed by feature name, sorted by importance descending)

        Examples:
            >>> selector = FeatureSelector(method="random_forest", verbose=True)
            >>> X_selected, importance = selector._select_random_forest(
            ...     X, y, top_n=30, n_estimators=200, n_repeats=10
            ... )
            >>> print(f"Features: {X.shape[1]} → {X_selected.shape[1]}")
            Features: 237 → 30
            >>> print("Top 5 features:")
            >>> print(importance.head())
        """
        if self.verbose:
            logger.info(f"{self.random_state:05d}: RandomForest features importance...")

        # Train RandomForest with regularization to prevent overfitting
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        rf.fit(X, y)

        # Use permutation importance (more reliable than Gini importance)
        perm_importance = permutation_importance(
            rf,
            X,
            y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        # Create importance DataFrame
        # Note: permutation_importance returns a Bunch object with attributes
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance_mean": perm_importance.importances_mean,
                "importance_std": perm_importance.importances_std,
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

        self.n_features_rf = X_selected.shape[1]
        self.importance_scores_ = importance_scores

        if self.verbose:
            logger.info(
                f"{self.random_state:05d}: Features reduced: {X.shape[1]} → {X_selected.shape[1]} features"
            )

        return X_selected, importance_scores

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        fdr_level: float = 0.05,
        top_n: int = 20,
        **rf_kwargs,
    ) -> Self:
        """Fit the feature selector on training data.

        Performs feature selection according to the specified method ("tsfresh",
        "random_forest", or "combined"). For "combined" method, runs two-stage
        selection: tsfresh statistical filtering followed by RandomForest
        permutation importance.

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            y (pd.Series): Target labels with shape (n_samples,). Must have same
                length as X.
            fdr_level (float, optional): False Discovery Rate level for tsfresh
                selection (Stage 1). Only used when method is "tsfresh" or "combined".
                Defaults to 0.05.
            top_n (int, optional): Number of top features for final selection.
                Used in RandomForest selection (Stage 2). Only used when method is
                "random_forest" or "combined". Defaults to 20.
            **rf_kwargs: Additional keyword arguments for RandomForest permutation
                importance. Supported arguments:
                - n_estimators (int): Number of trees (default: 200)
                - max_depth (int | None): Maximum tree depth (default: 10)
                - min_samples_leaf (int): Minimum samples per leaf (default: 20)
                - n_repeats (int): Permutation repeats (default: 10)

        Returns:
            Self: Fitted FeatureSelector instance for method chaining.

        Raises:
            ValueError: If X or y are empty.
            ValueError: If X and y have different lengths.
            ValueError: If method is not one of "tsfresh", "random_forest", or "combined".

        Examples:
            >>> # Combined two-stage selection
            >>> selector = FeatureSelector(method="combined", verbose=True)
            >>> selector.fit(X_train, y_train, fdr_level=0.05, top_n=20)
            >>> X_selected = selector.transform(X_train)
            >>>
            >>> # tsfresh-only selection
            >>> selector = FeatureSelector(method="tsfresh")
            >>> selector.fit(X_train, y_train, fdr_level=0.01)
            >>>
            >>> # RandomForest-only with custom parameters
            >>> selector = FeatureSelector(method="random_forest")
            >>> selector.fit(X_train, y_train, top_n=30, n_estimators=300, max_depth=15)
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
                    f"{self.n_features_tsfresh} → {self.n_features_rf} features"
                )

        else:
            raise ValueError(
                f"Invalid method: {self.method}. Must be 'tsfresh', 'random_forest', or 'combined'"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features by selecting only fitted features.

        Applies the feature selection determined by fit() to select only the
        chosen features from the input DataFrame.

        Args:
            X (pd.DataFrame): Extracted features DataFrame to transform.
                Must contain all features that were selected during fit().

        Returns:
            pd.DataFrame: Transformed DataFrame containing only the selected
                features (same row count as input, reduced column count).

        Raises:
            ValueError: If selector hasn't been fitted yet (call fit() first).
            ValueError: If selected features are missing from X (feature space
                mismatch between training and transform data).

        Examples:
            >>> selector = FeatureSelector(method="combined")
            >>> selector.fit(X_train, y_train)
            >>> X_train_selected = selector.transform(X_train)
            >>> X_test_selected = selector.transform(X_test)
            >>> print(f"Shape: {X_test.shape} → {X_test_selected.shape}")
            Shape: (50, 5000) → (50, 20)
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
        top_n: int = 20,
        **rf_kwargs,
    ) -> pd.DataFrame:
        """Fit the selector and transform X in one step.

        Convenience method that combines fit() and transform() into a single call.
        Equivalent to calling fit(X, y, ...) followed by transform(X).

        Args:
            X (pd.DataFrame): Extracted features DataFrame with shape (n_samples, n_features).
            y (pd.Series): Target labels with shape (n_samples,).
            fdr_level (float, optional): FDR level for tsfresh selection. Defaults to 0.05.
            top_n (int, optional): Number of top features to select. Defaults to 20.
            **rf_kwargs: Additional keyword arguments for RandomForest permutation importance
                (n_estimators, max_depth, min_samples_leaf, n_repeats).

        Returns:
            pd.DataFrame: Transformed DataFrame containing only the selected features.

        Examples:
            >>> selector = FeatureSelector(method="combined")
            >>> X_selected = selector.fit_transform(X_train, y_train, top_n=20)
            >>> print(X_selected.shape)
            (100, 20)
            >>>
            >>> # With custom RandomForest parameters
            >>> X_selected = selector.fit_transform(
            ...     X_train, y_train, top_n=30, n_estimators=300, n_repeats=15
            ... )
        """
        self.fit(X, y, fdr_level=fdr_level, top_n=top_n, **rf_kwargs)
        return self.transform(X)

    def get_feature_scores(self) -> pd.DataFrame:
        """Get comprehensive feature scores from selection process.

        Retrieves a DataFrame containing selected feature names along with their
        p-values (from tsfresh selection) and/or importance scores (from
        RandomForest selection), depending on the method used.

        Returns:
            pd.DataFrame: DataFrame with 'feature' as index and columns for
                available scores:
                - 'p_value': P-value from tsfresh (if method is "tsfresh" or "combined")
                - 'importance': Permutation importance (if method is "random_forest" or "combined")

        Raises:
            ValueError: If selector hasn't been fitted yet (call fit() first).

        Examples:
            >>> selector = FeatureSelector(method="combined")
            >>> selector.fit(X_train, y_train)
            >>> scores = selector.get_feature_scores()
            >>> print(scores.head(10))
                                       p_value  importance
            feature
            rsam_f2__quantile__q_0.9  0.000001    0.045123
            dsar_f3-f4__median        0.000002    0.038456
            ...
            >>>
            >>> # Sort by importance
            >>> print(scores.sort_values('importance', ascending=False).head())
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
