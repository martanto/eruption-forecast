"""Unit tests for SHAP plotting pipeline.

Covers:
- ``_extract_shap_array``: shape handling for 2D/3D ndarrays and Explanation objects.
- ``plot_aggregate_shap_summary``: identical and mixed feature sets, zero-filling,
  per-seed failure skipping, strict-zip mismatch, and ValueError on total failure.
- ``_build_aggregate_explanation``: zero-padding, None-skipping, feature-name preservation.
- ``MultiModelEvaluator._load_seed_data``: DataFrame return type, proba/decision-function
  branching, AttributeError on unsupported models, feature filtering.
- ``ModelEvaluator.plot_shap_summary``: smoke test for Figure return and title.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator
import shap

from eruption_forecast.plots.shap_plots import (
    _build_aggregate_explanation,
    _extract_shap_array,
    plot_aggregate_shap_summary,
)

# ---------------------------------------------------------------------------
# SHAP mock helpers
# ---------------------------------------------------------------------------
# SHAP 0.49.1 model-type detection breaks after importing project modules that
# call matplotlib.use("Agg") at import time (styles.py).  All tests that
# exercise aggregation or plotting functions patch shap.Explainer at its call
# site inside shap_plots.py so tests remain fast and deterministic.


class _FakeExplainer:
    """Minimal shap.Explainer stand-in that returns deterministic fake values.

    Raises TypeError when the model has no ``predict`` attribute (e.g. bare
    ``object()``) so tests for the per-seed failure path still work correctly.
    """

    def __init__(self, model, *args, **kwargs):
        """Store model reference for later callability check."""
        self._model = model

    def __call__(self, X, check_additivity: bool = True):
        """Return a SimpleNamespace Explanation with 2-D SHAP values."""
        # _compute_shap_explanation passes model.predict_proba (a bound method),
        # so check callable() rather than hasattr(model, "predict").
        if not callable(self._model):
            raise TypeError(
                "The passed model is not callable and cannot be analyzed directly "
                f"with the given masker! Model: {self._model!r}"
            )
        n, f = (X.shape[0], X.shape[1]) if hasattr(X, "shape") else (len(X), len(X[0]))
        rng = np.random.default_rng(42)
        # Return 2-D values to avoid subscript path in plot_shap_summary
        vals = rng.random((n, f))
        return types.SimpleNamespace(values=vals)


def _fake_beeswarm(*args, **kwargs):
    """No-op replacement for shap.plots.beeswarm during tests."""


@pytest.fixture
def mock_shap():
    """Patch shap.Explainer and shap.plots.beeswarm inside shap_plots."""
    with (
        patch("eruption_forecast.plots.shap_plots.shap.Explainer", _FakeExplainer),
        patch("eruption_forecast.plots.shap_plots.shap.plots.beeswarm", _fake_beeswarm),
    ):
        yield


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rf_and_data():
    """Return a fitted RandomForestClassifier with train/test DataFrames."""
    X, y = make_classification(
        n_samples=50,
        n_features=6,
        n_informative=4,
        n_classes=2,
        random_state=0,
    )
    cols = [f"feat_{i}" for i in range(6)]
    X_df = pd.DataFrame(X, columns=cols)
    y_series = pd.Series(y, name="label")
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.3, random_state=0
    )
    rf = RandomForestClassifier(n_estimators=5, random_state=0)
    rf.fit(X_train, y_train)
    return rf, X_test.reset_index(drop=True), y_test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Class 1: _extract_shap_array
# ---------------------------------------------------------------------------


class TestExtractShapArray:
    """Tests for the _extract_shap_array helper."""

    def test_2d_ndarray_passthrough(self):
        """2-D ndarray is returned unchanged."""
        arr = np.random.rand(10, 5)
        result = _extract_shap_array(arr)
        assert result.shape == (10, 5)
        assert np.array_equal(result, arr)

    def test_3d_ndarray_takes_class1_slice(self):
        """3-D ndarray (binary classifier output) returns class-1 slice."""
        arr = np.zeros((10, 5, 2))
        arr[:, :, 1] = 99.0  # Distinctive class-1 values
        result = _extract_shap_array(arr)
        assert result.shape == (10, 5)
        assert np.all(result == 99.0)

    def test_explanation_2d_values(self):
        """shap.Explanation with 2-D values extracts the values array."""
        vals = np.random.rand(10, 5)
        explanation = types.SimpleNamespace(values=vals)
        result = _extract_shap_array(explanation)
        assert result.shape == (10, 5)
        assert np.array_equal(result, vals)

    def test_explanation_3d_values_takes_class1_slice(self):
        """shap.Explanation with 3-D values takes the positive-class slice."""
        vals = np.zeros((10, 5, 2))
        vals[:, :, 1] = 7.0
        explanation = types.SimpleNamespace(values=vals)
        result = _extract_shap_array(explanation)
        assert result.shape == (10, 5)
        assert np.all(result == 7.0)

    def test_fallback_nested_list(self):
        """Plain Python list is converted to ndarray via np.array fallback."""
        nested = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = _extract_shap_array(nested)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# Class 2: plot_aggregate_shap_summary
# ---------------------------------------------------------------------------


class TestPlotAggregateShapSummary:
    """Tests for plot_aggregate_shap_summary with real tiny RF models."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_shap):
        """Apply SHAP mock and close figures around every test."""
        yield
        plt.close("all")

    def _make_rf(self, feature_cols: list[str]) -> tuple[RandomForestClassifier, pd.DataFrame]:
        """Return a fitted RF and test DataFrame for the given feature columns."""
        n = len(feature_cols)
        X, y = make_classification(
            n_samples=50,
            n_features=n,
            n_informative=max(1, n - 1),
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            random_state=0,
        )
        X_df = pd.DataFrame(X, columns=feature_cols)
        rf = RandomForestClassifier(n_estimators=3, random_state=0)
        rf.fit(X_df, y)
        return rf, X_df

    def test_identical_feature_sets_flat_names(self):
        """Flat feature_names list is broadcast to all seeds correctly."""
        cols = [f"feat_{i}" for i in range(5)]
        m1, X1 = self._make_rf(cols)
        m2, X2 = self._make_rf(cols)

        fig, exp = plot_aggregate_shap_summary(
            models=[m1, m2],
            X_tests=[X1, X2],
            feature_names=cols,
            max_display=5,
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(exp, shap.Explanation)
        assert exp.values.shape[1] == 5
        assert exp.feature_names == cols

    def test_different_feature_sets_union_length(self):
        """Mixed per-seed feature sets produce a union-sized explanation."""
        seed1_cols = ["feat_0", "feat_1", "feat_2"]
        seed2_cols = ["feat_2", "feat_3", "feat_4"]

        m1, X1 = self._make_rf(seed1_cols)
        m2, X2 = self._make_rf(seed2_cols)

        fig, exp = plot_aggregate_shap_summary(
            models=[m1, m2],
            X_tests=[X1, X2],
            feature_names=[seed1_cols, seed2_cols],
            max_display=5,
        )

        # Union: feat_0..feat_4 → 5 features
        assert exp.values.shape[1] == 5
        assert set(exp.feature_names) == {"feat_0", "feat_1", "feat_2", "feat_3", "feat_4"}

    def test_different_feature_sets_shared_feature_mean(self):
        """Feature present in both seeds appears in the merged explanation."""
        seed1_cols = ["a", "b", "c"]
        seed2_cols = ["c", "d", "e"]

        m1, X1 = self._make_rf(seed1_cols)
        m2, X2 = self._make_rf(seed2_cols)

        _, exp = plot_aggregate_shap_summary(
            models=[m1, m2],
            X_tests=[X1, X2],
            feature_names=[seed1_cols, seed2_cols],
        )

        assert "c" in exp.feature_names
        assert "a" in exp.feature_names
        assert "d" in exp.feature_names

    def test_raises_value_error_when_all_seeds_fail(self):
        """ValueError is raised when no seed produces valid SHAP values."""
        dummy = object()
        X = pd.DataFrame({"f0": [1.0, 2.0]})

        with pytest.raises(ValueError, match="No valid seeds"):
            plot_aggregate_shap_summary(
                models=[dummy, dummy],
                X_tests=[X, X],
                feature_names=[["f0"], ["f0"]],
            )

    def test_per_seed_failure_is_skipped(self):
        """A single bad seed is skipped; the function completes with remaining seeds."""
        cols = ["feat_0", "feat_1", "feat_2"]
        m1, X1 = self._make_rf(cols)
        dummy = object()
        X_dummy = pd.DataFrame({"feat_0": [1.0]})

        # Seed 2 will fail; seed 1 should still produce a result.
        fig, exp = plot_aggregate_shap_summary(
            models=[m1, dummy],
            X_tests=[X1, X_dummy],
            feature_names=[cols, ["feat_0"]],
        )

        assert isinstance(fig, plt.Figure)
        assert isinstance(exp, shap.Explanation)

    def test_strict_zip_mismatched_lengths_raises(self):
        """Mismatched models and X_tests raise ValueError (strict=True zip)."""
        cols = ["f0", "f1"]
        m1, X1 = self._make_rf(cols)

        with pytest.raises(ValueError):
            plot_aggregate_shap_summary(
                models=[m1],
                X_tests=[X1, X1],  # One extra X_test
                feature_names=[cols, cols],
            )


# ---------------------------------------------------------------------------
# Class 3: _build_aggregate_explanation
# ---------------------------------------------------------------------------


class TestBuildAggregateExplanation:
    """Tests for the _build_aggregate_explanation helper."""

    def _make_explanation(self, n_samples: int, features: list[str]) -> shap.Explanation:
        """Return a simple shap.Explanation with deterministic values."""
        rng = np.random.default_rng(0)
        n = len(features)
        vals = rng.random((n_samples, n))
        data = rng.random((n_samples, n))
        return shap.Explanation(values=vals, data=data, feature_names=features)

    def test_single_seed_passthrough(self):
        """Single seed with exact union set produces explanation without zero-padding."""
        feats = ["a", "b", "c"]
        exp = self._make_explanation(10, feats)
        result = _build_aggregate_explanation([exp], [feats], feats)
        assert result.values.shape == (10, 3)
        assert result.feature_names == feats

    def test_two_seeds_concatenated_sample_axis(self):
        """Two seeds with the same features concatenate along the sample axis."""
        feats = ["a", "b"]
        exp1 = self._make_explanation(5, feats)
        exp2 = self._make_explanation(7, feats)
        result = _build_aggregate_explanation([exp1, exp2], [feats, feats], feats)
        assert result.values.shape == (12, 2)

    def test_different_feature_sets_zero_padding(self):
        """Features absent in a seed are zero-padded to the union space."""
        seed1_feats = ["a", "b"]
        seed2_feats = ["b", "c"]
        all_feats = ["a", "b", "c"]
        exp1 = self._make_explanation(4, seed1_feats)
        exp2 = self._make_explanation(4, seed2_feats)

        result = _build_aggregate_explanation(
            [exp1, exp2], [seed1_feats, seed2_feats], all_feats
        )
        assert result.values.shape == (8, 3)
        # Samples from seed1 have zero for feature "c" (index 2)
        assert np.all(result.values[:4, 2] == 0.0)
        # Samples from seed2 have zero for feature "a" (index 0)
        assert np.all(result.values[4:, 0] == 0.0)

    def test_none_seeds_skipped(self):
        """None explanations are skipped; only valid seeds contribute."""
        feats = ["x", "y"]
        exp = self._make_explanation(3, feats)
        result = _build_aggregate_explanation([None, exp], [feats, feats], feats)
        assert result.values.shape == (3, 2)

    def test_all_none_raises_value_error(self):
        """ValueError raised when all explanations are None."""
        with pytest.raises(ValueError, match="No valid seeds"):
            _build_aggregate_explanation([None, None], [["a"], ["a"]], ["a"])

    def test_feature_names_preserved(self):
        """Returned Explanation carries the union feature names."""
        feats = ["f0", "f1"]
        exp = self._make_explanation(5, feats)
        result = _build_aggregate_explanation([exp], [feats], feats)
        assert result.feature_names == feats


# ---------------------------------------------------------------------------
# Class 4: MultiModelEvaluator._load_seed_data
# ---------------------------------------------------------------------------


class _NoProbaModel(BaseEstimator):
    """Minimal sklearn estimator without predict_proba or decision_function."""

    def fit(self, X, y):
        """Fit the model (no-op for testing)."""
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        """Return zeros as predictions."""
        return np.zeros(len(X), dtype=int)


def _write_seed_files(
    tmp_path,
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sig_features: list[str],
) -> pd.Series:
    """Write model, X_test, y_test, and features CSV to tmp_path; return a registry row."""
    model_path = str(tmp_path / "model.pkl")
    x_path = str(tmp_path / "X_test.csv")
    y_path = str(tmp_path / "y_test.csv")
    feat_path = str(tmp_path / "sig_features.csv")

    joblib.dump(model, model_path)
    X_test.to_csv(x_path)
    y_test.to_frame(name="label").to_csv(y_path)
    pd.DataFrame({"importance": 1.0}, index=sig_features).to_csv(feat_path)

    return pd.Series(
        {
            "trained_model_filepath": model_path,
            "X_test_filepath": x_path,
            "y_test_filepath": y_path,
            "significant_features_csv": feat_path,
        }
    )


class TestLoadSeedData:
    """Tests for MultiModelEvaluator._load_seed_data."""

    @pytest.fixture
    def seed_row(self, tmp_path, rf_and_data):
        """Write all seed files for the module-scoped RF fixture."""
        rf, X_test, y_test = rf_and_data
        sig = X_test.columns.tolist()
        return _write_seed_files(tmp_path, rf, X_test, y_test, sig)

    def test_returns_dataframe(self, seed_row):
        """X_test_filtered is a DataFrame, not a numpy array."""
        _, X_filtered, _, _ = MultiModelEvaluator._load_seed_data(seed_row)
        assert isinstance(X_filtered, pd.DataFrame)

    def test_y_true_is_ndarray(self, seed_row):
        """y_true is a numpy ndarray."""
        _, _, y_true, _ = MultiModelEvaluator._load_seed_data(seed_row)
        assert isinstance(y_true, np.ndarray)

    def test_proba_shape_matches_n_samples(self, seed_row, rf_and_data):
        """y_proba length equals the number of test samples."""
        _, _, y_test = rf_and_data
        _, _, y_true, y_proba = MultiModelEvaluator._load_seed_data(seed_row)
        assert y_proba.shape == y_true.shape

    def test_predict_proba_values_in_unit_interval(self, seed_row):
        """Probabilities from predict_proba are in [0, 1]."""
        _, _, _, y_proba = MultiModelEvaluator._load_seed_data(seed_row)
        assert y_proba.min() >= 0.0
        assert y_proba.max() <= 1.0

    def test_decision_function_fallback(self, tmp_path, rf_and_data):
        """SVC (no predict_proba) uses decision_function without error."""
        _, X_test, y_test = rf_and_data
        svc = SVC(kernel="linear", probability=False, random_state=0)
        svc.fit(X_test, y_test)
        row = _write_seed_files(tmp_path, svc, X_test, y_test, X_test.columns.tolist())

        _, _, y_true, y_proba = MultiModelEvaluator._load_seed_data(row)

        assert isinstance(y_proba, np.ndarray)
        assert y_proba.shape == y_true.shape

    def test_raises_attribute_error_for_unsupported_model(self, tmp_path, rf_and_data):
        """AttributeError raised when model has neither predict_proba nor decision_function."""
        _, X_test, y_test = rf_and_data
        bad_model = _NoProbaModel()
        bad_model.fit(X_test.values, y_test.values)
        row = _write_seed_files(
            tmp_path, bad_model, X_test, y_test, X_test.columns.tolist()
        )

        with pytest.raises(AttributeError, match="has neither predict_proba nor decision_function"):
            MultiModelEvaluator._load_seed_data(row)

    def test_significant_features_filtering(self, tmp_path):
        """Only the significant features are kept in X_test_filtered."""
        # Train on 2 features only so predict_proba matches sig_features.
        X, y = make_classification(
            n_samples=50, n_features=2, n_informative=1, n_redundant=0,
            n_repeated=0, n_classes=2, n_clusters_per_class=1, random_state=1,
        )
        sig = ["a", "b"]
        X_df = pd.DataFrame(X, columns=sig)
        y_s = pd.Series(y, name="label")
        rf2 = RandomForestClassifier(n_estimators=3, random_state=0).fit(X_df, y_s)
        # Write X_test with an extra column; sig_features lists only ["a", "b"]
        X_extra = X_df.copy()
        X_extra["c"] = 0.0
        row = _write_seed_files(tmp_path, rf2, X_extra, y_s, sig)

        _, X_filtered, _, _ = MultiModelEvaluator._load_seed_data(row)

        assert X_filtered.shape[1] == 2
        assert X_filtered.columns.tolist() == sig

    def test_column_names_preserved(self, seed_row, rf_and_data):
        """Column names on the returned DataFrame match the significant features."""
        _, X_test, _ = rf_and_data
        _, X_filtered, _, _ = MultiModelEvaluator._load_seed_data(seed_row)
        assert X_filtered.columns.tolist() == X_test.columns.tolist()


# ---------------------------------------------------------------------------
# Class 5: ModelEvaluator.plot_shap_summary (smoke tests)
# ---------------------------------------------------------------------------


class TestModelEvaluatorPlotShapSummary:
    """Smoke tests for ModelEvaluator.plot_shap_summary."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_shap):
        """Apply SHAP mock and close figures around every test."""
        yield
        plt.close("all")

    def test_returns_figure(self, tmp_path, rf_and_data):
        """plot_shap_summary returns a matplotlib Figure."""
        rf, X_test, y_test = rf_and_data
        evaluator = ModelEvaluator(
            model=rf,
            X_test=X_test,
            y_test=y_test,
            model_name="test_rf",
            output_dir=str(tmp_path),
        )
        fig = evaluator.plot_shap_summary(save=False)
        assert isinstance(fig, plt.Figure)

    def test_title_contains_model_name(self, tmp_path, rf_and_data):
        """Plot suptitle includes the model_name."""
        rf, X_test, y_test = rf_and_data
        evaluator = ModelEvaluator(
            model=rf,
            X_test=X_test,
            y_test=y_test,
            model_name="my_model",
            output_dir=str(tmp_path),
        )
        fig = evaluator.plot_shap_summary(save=False)

        # suptitle text is stored in fig.texts or accessed via _suptitle
        all_text = " ".join(t.get_text() for t in fig.texts)
        assert "my_model" in all_text
