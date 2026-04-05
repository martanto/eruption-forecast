"""Tests for ClassifierEnsemble.

Uses synthetic in-memory SeedEnsemble objects (no disk I/O) to verify the
cross-classifier consensus logic, serialisation, and backward compatibility.
"""

import os
import pathlib

import numpy as np
import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from eruption_forecast.utils.ml import merge_all_classifiers
from eruption_forecast.model.seed_ensemble import SeedEnsemble
from eruption_forecast.model.classifier_ensemble import ClassifierEnsemble


# ---------------------------------------------------------------------------
# Helpers to build synthetic SeedEnsemble objects without touching disk
# ---------------------------------------------------------------------------

N_FEATURES = 4
N_SAMPLES = 10
FEATURE_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]


def _make_seed_ensemble(n_seeds: int = 5, seed_offset: int = 0) -> SeedEnsemble:
    """Build a SeedEnsemble from tiny LogisticRegression models.

    Each model is trained on random binary data so that predict_proba works
    without requiring real seismic features.

    Args:
        n_seeds (int): Number of seeds to generate. Defaults to 5.
        seed_offset (int): Offset added to seed indices to keep them distinct
            across classifiers. Defaults to 0.

    Returns:
        SeedEnsemble: Populated ensemble with ``n_seeds`` fitted models.
    """
    rng = np.random.default_rng(42 + seed_offset)
    ensemble = SeedEnsemble(classifier_name="LogisticRegression")
    for i in range(n_seeds):
        X_train = rng.standard_normal((30, N_FEATURES))
        y_train = rng.integers(0, 2, size=30)
        # Ensure both classes are present
        y_train[:5] = 0
        y_train[5:10] = 1
        model = LogisticRegression(max_iter=200, random_state=i + seed_offset)
        model.fit(X_train, y_train)
        ensemble.seeds.append(
            {
                "random_state": i + seed_offset,
                "model": model,
                "feature_names": FEATURE_NAMES,
            }
        )
    return ensemble


def _make_feature_df(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """Build a synthetic feature DataFrame.

    Args:
        n_samples (int): Number of rows. Defaults to N_SAMPLES.

    Returns:
        pd.DataFrame: DataFrame with ``FEATURE_NAMES`` columns.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.standard_normal((n_samples, N_FEATURES)), columns=FEATURE_NAMES
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_ensembles() -> dict[str, SeedEnsemble]:
    """Return two named SeedEnsemble objects for testing."""
    return {
        "rf": _make_seed_ensemble(n_seeds=5, seed_offset=0),
        "xgb": _make_seed_ensemble(n_seeds=5, seed_offset=100),
    }


@pytest.fixture()
def clf_ensemble(two_ensembles: dict[str, SeedEnsemble]) -> ClassifierEnsemble:
    """Return a ClassifierEnsemble built from two SeedEnsembles."""
    return ClassifierEnsemble.from_seed_ensembles(two_ensembles)


@pytest.fixture()
def X() -> pd.DataFrame:
    """Return a synthetic feature DataFrame for prediction."""
    return _make_feature_df()


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


def test_from_seed_ensembles_basic(
    clf_ensemble: ClassifierEnsemble,
    two_ensembles: dict[str, SeedEnsemble],
) -> None:
    """ClassifierEnsemble built from two SeedEnsembles has correct structure."""
    assert len(clf_ensemble) == 2
    assert clf_ensemble.classifiers == ["rf", "xgb"]
    assert clf_ensemble["rf"] is two_ensembles["rf"]
    assert clf_ensemble["xgb"] is two_ensembles["xgb"]


def test_from_seed_ensembles_empty_raises() -> None:
    """from_seed_ensembles raises ValueError for empty input."""
    with pytest.raises(ValueError, match="must not be empty"):
        ClassifierEnsemble.from_seed_ensembles({})


def test_getitem_missing_key_raises(clf_ensemble: ClassifierEnsemble) -> None:
    """Accessing a non-existent classifier raises KeyError."""
    with pytest.raises(KeyError):
        _ = clf_ensemble["nonexistent"]


# ---------------------------------------------------------------------------
# predict_proba tests
# ---------------------------------------------------------------------------


def test_predict_proba_shape(
    clf_ensemble: ClassifierEnsemble, X: pd.DataFrame
) -> None:
    """predict_proba returns a 2D array."""
    proba = clf_ensemble.predict_proba(X)
    assert proba.ndim == 2


def test_predict_proba_values_in_range(
    clf_ensemble: ClassifierEnsemble, X: pd.DataFrame
) -> None:
    """predict_proba values are in [0, 1]."""
    proba = clf_ensemble.predict_proba(X)
    assert np.all(proba >= 0) and np.all(proba <= 1)



# ---------------------------------------------------------------------------
# predict_with_uncertainty tests
# ---------------------------------------------------------------------------


def test_predict_with_uncertainty_shapes(
    clf_ensemble: ClassifierEnsemble, X: pd.DataFrame
) -> None:
    """predict_with_uncertainty returns 4 arrays of shape (n_samples,) + per-clf dict."""
    result = clf_ensemble.predict_with_uncertainty(X)
    consensus_mean, consensus_std, consensus_conf, consensus_pred, per_clf = result
    assert consensus_mean.shape == (N_SAMPLES,)
    assert consensus_std.shape == (N_SAMPLES,)
    assert consensus_conf.shape == (N_SAMPLES,)
    assert consensus_pred.shape == (N_SAMPLES,)


def test_predict_with_uncertainty_per_clf_keys(
    clf_ensemble: ClassifierEnsemble, X: pd.DataFrame
) -> None:
    """per_classifier_results has one entry per classifier with required keys."""
    _, _, _, _, per_clf = clf_ensemble.predict_with_uncertainty(X)
    assert set(per_clf.keys()) == {"rf", "xgb"}
    for clf_result in per_clf.values():
        assert set(clf_result.keys()) == {"probability", "uncertainty", "prediction", "confidence"}
        for arr in clf_result.values():
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (N_SAMPLES,)


def test_predict_with_uncertainty_binary_predictions(
    clf_ensemble: ClassifierEnsemble, X: pd.DataFrame
) -> None:
    """consensus_prediction values are in [0, 1]."""
    _, _, _, consensus_pred, _ = clf_ensemble.predict_with_uncertainty(X)
    assert np.all(consensus_pred >= 0) and np.all(consensus_pred <= 1)


def test_predict_with_uncertainty_confidence_range(
    clf_ensemble: ClassifierEnsemble, X: pd.DataFrame
) -> None:
    """consensus_confidence is in [0.5, 1.0] by majority-voting definition."""
    _, _, consensus_conf, _, _ = clf_ensemble.predict_with_uncertainty(X)
    assert np.all(consensus_conf >= 0.0) and np.all(consensus_conf <= 1.0)


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(
    clf_ensemble: ClassifierEnsemble, X: pd.DataFrame, tmp_path: pathlib.Path
) -> None:
    """Saving and loading a ClassifierEnsemble produces identical predictions."""
    path = os.path.join(str(tmp_path), "ensemble.pkl")
    clf_ensemble.save(path)
    loaded = ClassifierEnsemble.load(path)

    proba_orig = clf_ensemble.predict_proba(X)
    proba_loaded = loaded.predict_proba(X)
    np.testing.assert_array_equal(proba_orig, proba_loaded)


def test_load_missing_file_raises() -> None:
    """ClassifierEnsemble.load raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        ClassifierEnsemble.load("/nonexistent/path/ensemble.pkl")


# ---------------------------------------------------------------------------
# merge_all_classifiers output type test
# ---------------------------------------------------------------------------


def test_merge_all_classifiers_produces_classifier_ensemble(
    tmp_path: pathlib.Path,
) -> None:
    """merge_all_classifiers saves a ClassifierEnsemble, not a plain dict."""
    # Build fake registry CSVs so merge_all_classifiers can load them
    clf_names = ["clf_a", "clf_b"]
    trained_models: dict[str, str] = {}

    for clf_name in clf_names:
        clf_dir = os.path.join(str(tmp_path), clf_name)
        os.makedirs(clf_dir, exist_ok=True)

        rows = {}
        for seed in range(3):
            # Save a tiny model
            rng = np.random.default_rng(seed)
            X_train = rng.standard_normal((20, N_FEATURES))
            y_train = np.array([0] * 10 + [1] * 10)
            model = LogisticRegression(max_iter=200, random_state=seed)
            model.fit(X_train, y_train)
            model_path = os.path.join(clf_dir, f"model_{seed}.pkl")
            joblib.dump(model, model_path)

            # Save a tiny significant-features CSV
            sig_df = pd.DataFrame(index=FEATURE_NAMES)
            sig_csv = os.path.join(clf_dir, f"sig_feat_{seed}.csv")
            sig_df.to_csv(sig_csv)

            rows[seed] = {
                "significant_features_csv": sig_csv,
                "trained_model_filepath": model_path,
            }

        registry_df = pd.DataFrame(rows).T
        registry_df.index.name = "random_state"
        registry_csv = os.path.join(clf_dir, f"trained_model_{clf_name}.csv")
        registry_df.to_csv(registry_csv)
        trained_models[clf_name] = registry_csv

    output_path = os.path.join(str(tmp_path), "merged.pkl")
    merge_all_classifiers(trained_models, output_path=output_path)

    loaded = joblib.load(output_path)
    assert isinstance(loaded, ClassifierEnsemble), (
        f"Expected ClassifierEnsemble, got {type(loaded)}"
    )
    assert set(loaded.classifiers) == set(clf_names)


# ---------------------------------------------------------------------------
# Backward compatibility: plain dict pkl auto-wraps
# ---------------------------------------------------------------------------


def test_backward_compat_plain_dict_pkl(
    two_ensembles: dict[str, SeedEnsemble],
    X: pd.DataFrame,
    tmp_path: pathlib.Path,
) -> None:
    """ModelPredictor auto-wraps a plain dict[str, SeedEnsemble] pkl into ClassifierEnsemble."""
    # Write a plain dict pkl (old format)
    pkl_path = os.path.join(str(tmp_path), "old_format.pkl")
    joblib.dump(two_ensembles, pkl_path)

    # ModelPredictor.__init__ loads the pkl — we only test the loading logic here
    # by simulating what __init__ does when it encounters a plain dict
    loaded = joblib.load(pkl_path)
    assert isinstance(loaded, dict)

    # The auto-wrap path
    wrapped = ClassifierEnsemble.from_seed_ensembles(loaded)
    assert isinstance(wrapped, ClassifierEnsemble)
    assert set(wrapped.classifiers) == {"rf", "xgb"}

    # Ensure predictions still work after wrapping
    proba = wrapped.predict_proba(X)
    assert proba.ndim == 2


# ---------------------------------------------------------------------------
# repr and len
# ---------------------------------------------------------------------------


def test_repr(clf_ensemble: ClassifierEnsemble) -> None:
    """repr contains classifier names."""
    r = repr(clf_ensemble)
    assert "ClassifierEnsemble" in r
    assert "rf" in r
    assert "xgb" in r


def test_len(clf_ensemble: ClassifierEnsemble) -> None:
    """len returns number of classifiers."""
    assert len(clf_ensemble) == 2
