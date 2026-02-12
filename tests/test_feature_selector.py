"""
Unit tests for FeatureSelector class.

Tests all three feature selection methods (tsfresh, random_forest, combined)
using synthetic data.
"""

# Third party imports
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

# Project imports
from eruption_forecast.features import FeatureSelector


@pytest.fixture
def synthetic_data():
    """Create synthetic binary classification dataset."""
    X, y = make_classification(
        n_samples=500,
        n_features=100,
        n_informative=20,
        n_redundant=10,
        n_repeated=5,
        n_classes=2,
        random_state=42,
        shuffle=False,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    return X_df, y_series


def test_feature_selector_init():
    """Test FeatureSelector initialization with different methods."""
    # Test default (combined)
    selector = FeatureSelector()
    assert selector.method == "combined"
    assert selector.n_jobs == 1
    assert selector.random_state == 42

    # Test tsfresh method
    selector = FeatureSelector(method="tsfresh", n_jobs=2)
    assert selector.method == "tsfresh"
    assert selector.n_jobs == 2

    # Test random_forest method
    selector = FeatureSelector(method="random_forest", verbose=True)
    assert selector.method == "random_forest"
    assert selector.verbose is True


def test_tsfresh_selection(synthetic_data):
    """Test tsfresh-only feature selection."""
    X, y = synthetic_data

    selector = FeatureSelector(method="tsfresh", n_jobs=2, verbose=False)
    X_selected = selector.fit_transform(X, y, fdr_level=0.05)

    # Check that features were selected
    assert X_selected.shape[1] < X.shape[1], "Should reduce features"
    assert X_selected.shape[0] == X.shape[0], "Should keep all samples"

    # Check that p-values were computed
    assert not selector.p_values_.empty, "Should have p-values"
    assert len(selector.feature_names_) == X_selected.shape[1]


def test_random_forest_selection(synthetic_data):
    """Test RandomForest-only feature selection."""
    X, y = synthetic_data

    selector = FeatureSelector(method="random_forest", n_jobs=2, random_state=42)
    X_selected = selector.fit_transform(
        X, y, top_n=30, n_estimators=50, max_depth=5, n_repeats=5
    )

    # Check that correct number of features selected
    assert X_selected.shape[1] == 30, "Should select exactly 30 features"
    assert X_selected.shape[0] == X.shape[0], "Should keep all samples"

    # Check that importance scores were computed
    assert not selector.importance_scores_.empty, "Should have importance scores"
    assert len(selector.feature_names_) == 30


def test_combined_selection(synthetic_data):
    """Test combined two-stage feature selection."""
    X, y = synthetic_data

    selector = FeatureSelector(
        method="combined", n_jobs=2, random_state=42, verbose=False
    )
    X_selected = selector.fit_transform(
        X,
        y,
        fdr_level=0.05,
        top_n=20,
        n_estimators=50,
        max_depth=5,
        n_repeats=5,
    )

    # Check that features were selected in two stages
    # Note: top_n is a maximum - if tsfresh filters to fewer features, that's fine
    assert X_selected.shape[1] <= 20, "Should select at most 20 features"
    assert X_selected.shape[1] > 0, "Should select at least some features"
    assert X_selected.shape[0] == X.shape[0], "Should keep all samples"

    # Check that both p-values and importance scores were computed
    assert not selector.p_values_.empty, "Should have p-values from stage 1"
    assert not selector.importance_scores_.empty, "Should have importance from stage 2"

    # Check stage tracking
    assert selector.n_features_stage1_ > 0, "Stage 1 should reduce features"
    assert selector.n_features_stage2_ > 0, "Stage 2 should select features"
    assert selector.n_features_stage1_ >= selector.n_features_stage2_, (
        "Stage 1 should produce more/equal features than stage 2"
    )


def test_fit_transform_separate(synthetic_data):
    """Test fit() and transform() as separate operations."""
    X, y = synthetic_data
    X_train, y_train = X[:400], y[:400]
    X_test = X[400:]

    selector = FeatureSelector(method="tsfresh", n_jobs=2)
    selector.fit(X_train, y_train, fdr_level=0.05)

    # Transform both train and test
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)

    # Check that same features selected for both
    assert list(X_train_selected.columns) == list(X_test_selected.columns)
    assert X_train_selected.shape[1] == X_test_selected.shape[1]


def test_get_feature_scores(synthetic_data):
    """Test get_feature_scores() method."""
    X, y = synthetic_data

    selector = FeatureSelector(method="combined", n_jobs=2, random_state=42)
    selector.fit(X, y, fdr_level=0.05, top_n=15, n_estimators=50, n_repeats=3)

    scores = selector.get_feature_scores()

    # Check that scores DataFrame has correct structure
    assert isinstance(scores, pd.DataFrame), "Should return DataFrame"
    assert len(scores) > 0, "Should have feature scores"
    assert len(scores) <= 15, "Should have at most 15 feature scores"
    assert "p_values" in scores.columns, "Should have p-values column"
    assert "importance" in scores.columns, "Should have importance column"


def test_invalid_method():
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="Invalid method"):
        selector = FeatureSelector(method="invalid")
        X = pd.DataFrame(np.random.rand(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
        selector.fit(X, y)


def test_transform_before_fit():
    """Test that transform() before fit() raises ValueError."""
    selector = FeatureSelector(method="tsfresh")
    X = pd.DataFrame(np.random.rand(100, 10))

    with pytest.raises(ValueError, match="has not been fitted"):
        selector.transform(X)


def test_empty_data():
    """Test that empty data raises ValueError."""
    selector = FeatureSelector(method="tsfresh")
    X_empty = pd.DataFrame()
    y_empty = pd.Series(dtype=int)

    with pytest.raises(ValueError, match="cannot be empty"):
        selector.fit(X_empty, y_empty)


def test_mismatched_lengths():
    """Test that mismatched X and y lengths raise ValueError."""
    selector = FeatureSelector(method="tsfresh")
    X = pd.DataFrame(np.random.rand(100, 10))
    y = pd.Series(np.random.randint(0, 2, 50))  # Wrong length

    with pytest.raises(ValueError, match="must have same length"):
        selector.fit(X, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
