"""Tests for SeedEnsemble merge functionality and prediction methods."""

import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from eruption_forecast.model.seed_ensemble import SeedEnsemble
from eruption_forecast.utils.ml import merge_seed_models, merge_all_classifiers

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SAMPLES = 200
N_FEATURES = 10
N_SEEDS = 5

FEATURE_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]

_rng = np.random.default_rng(42)
_X_all = pd.DataFrame(
    _rng.standard_normal((N_SAMPLES, N_FEATURES)), columns=FEATURE_NAMES
)
_y_all = pd.Series(
    (_rng.random(N_SAMPLES) < 0.1).astype(int), name="is_erupted"
)


def _save_seed(seed: int, feature_subset: list[str], base_dir: str) -> tuple[str, str]:
    """Train and save one RF model and its significant-features CSV."""
    models_dir = os.path.join(base_dir, "models")
    sig_dir = os.path.join(base_dir, "features", "significant_features")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(sig_dir, exist_ok=True)

    clf = RandomForestClassifier(n_estimators=10, random_state=seed)
    clf.fit(_X_all[feature_subset], _y_all)

    model_path = os.path.join(models_dir, f"{seed:05d}.pkl")
    joblib.dump(clf, model_path)

    sig_df = pd.DataFrame(
        {"p_values": [0.01 * (i + 1) for i in range(len(feature_subset))]},
        index=pd.Index(feature_subset, name="features"),
    )
    sig_path = os.path.join(sig_dir, f"{seed:05d}.csv")
    sig_df.to_csv(sig_path)

    return sig_path, model_path


def _build_registry(records: list[dict], path: str) -> str:
    """Write a mock registry CSV and return its path."""
    registry_df = pd.DataFrame(records).set_index("random_state")
    registry_df.to_csv(path)
    return path


@pytest.fixture(scope="module")
def ensemble_dir(tmp_path_factory):
    """Create all per-seed models and return registry CSV path + base dir."""
    base = str(tmp_path_factory.mktemp("seed_ensemble"))
    records = []
    for seed in range(N_SEEDS):
        subset = FEATURE_NAMES[seed % 3 : seed % 3 + 7]
        sig_path, model_path = _save_seed(seed, subset, base)
        records.append(
            {
                "random_state": seed,
                "significant_features_csv": sig_path,
                "trained_model_filepath": model_path,
            }
        )
    registry_csv = os.path.join(base, "trained_model_rf.csv")
    _build_registry(records, registry_csv)
    return base, registry_csv


@pytest.fixture(scope="module")
def merged_ensemble(ensemble_dir):
    """Return a loaded SeedEnsemble from merge_seed_models."""
    base, registry_csv = ensemble_dir
    merged_path = merge_seed_models(registry_csv)
    return SeedEnsemble.load(merged_path), merged_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMergeAndLoad:
    def test_merged_file_exists(self, ensemble_dir):
        _, registry_csv = ensemble_dir
        merged_path = merge_seed_models(registry_csv)
        assert os.path.isfile(merged_path)

    def test_loads_as_seed_ensemble(self, merged_ensemble):
        ensemble, _ = merged_ensemble
        assert isinstance(ensemble, SeedEnsemble)

    def test_correct_seed_count(self, merged_ensemble):
        ensemble, _ = merged_ensemble
        assert len(ensemble) == N_SEEDS

    def test_custom_output_dir(self, ensemble_dir, tmp_path):
        _, registry_csv = ensemble_dir
        path = merge_seed_models(registry_csv, output_dir=str(tmp_path))
        assert os.path.isfile(path)
        assert os.path.dirname(os.path.abspath(path)) == str(tmp_path)


class TestPredictionShapes:
    def test_predict_with_uncertainty_shapes(self, merged_ensemble):
        ensemble, _ = merged_ensemble
        n = len(_X_all)
        mean_p, std_p, conf, pred = ensemble.predict_with_uncertainty(_X_all)
        assert mean_p.shape == (n,)
        assert std_p.shape == (n,)
        assert conf.shape == (n,)
        assert pred.shape == (n,)

    def test_mean_prediction_in_range(self, merged_ensemble):
        """mean_prediction is the mean of per-seed binary votes, in [0, 1]."""
        ensemble, _ = merged_ensemble
        _, _, mean_pred, _ = ensemble.predict_with_uncertainty(_X_all)
        assert (mean_pred >= 0.0).all() and (mean_pred <= 1.0).all()

    def test_confidence_non_negative(self, merged_ensemble):
        ensemble, _ = merged_ensemble
        _, _, _, conf = ensemble.predict_with_uncertainty(_X_all)
        assert (conf >= 0.0).all()

    def test_predict_proba_shape(self, merged_ensemble):
        ensemble, _ = merged_ensemble
        proba = ensemble.predict_proba(_X_all)
        assert proba.shape == (_X_all.shape[0], 2)

    def test_predict_proba_sums_to_one(self, merged_ensemble):
        ensemble, _ = merged_ensemble
        proba = ensemble.predict_proba(_X_all)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_col1_equals_mean_probability(self, merged_ensemble):
        ensemble, _ = merged_ensemble
        proba = ensemble.predict_proba(_X_all)
        mean_p, _, _, _ = ensemble.predict_with_uncertainty(_X_all)
        assert np.allclose(proba[:, 1], mean_p)

    def test_predict_binary_1d(self, merged_ensemble):
        ensemble, _ = merged_ensemble
        preds = ensemble.predict(_X_all)
        assert preds.shape == (_X_all.shape[0],)
        unique_vals = set(preds.astype(int).tolist())
        assert unique_vals.issubset({0, 1})


class TestMergeAllClassifiers:
    def test_bundle_created(self, ensemble_dir, tmp_path):
        base, rf_csv = ensemble_dir
        # Build a second registry for "xgb"
        records_xgb = []
        for seed in range(N_SEEDS):
            subset = FEATURE_NAMES[1:8]
            sig_path, model_path = _save_seed(seed + 100, subset, str(tmp_path))
            records_xgb.append(
                {
                    "random_state": seed + 100,
                    "significant_features_csv": sig_path,
                    "trained_model_filepath": model_path,
                }
            )
        xgb_csv = os.path.join(str(tmp_path), "trained_model_xgb.csv")
        _build_registry(records_xgb, xgb_csv)

        bundle_path = merge_all_classifiers({"rf": rf_csv, "xgb": xgb_csv})
        assert os.path.isfile(bundle_path)

    def test_bundle_is_dict_with_seed_ensembles(self, ensemble_dir, tmp_path):
        base, rf_csv = ensemble_dir
        records_xgb = []
        for seed in range(N_SEEDS):
            subset = FEATURE_NAMES[1:8]
            sig_path, model_path = _save_seed(seed + 200, subset, str(tmp_path))
            records_xgb.append(
                {
                    "random_state": seed + 200,
                    "significant_features_csv": sig_path,
                    "trained_model_filepath": model_path,
                }
            )
        xgb_csv = os.path.join(str(tmp_path), "trained_model_xgb2.csv")
        _build_registry(records_xgb, xgb_csv)

        from eruption_forecast.model.classifier_ensemble import ClassifierEnsemble

        bundle_path = merge_all_classifiers({"rf": rf_csv, "xgb": xgb_csv})
        loaded = joblib.load(bundle_path)
        assert isinstance(loaded, ClassifierEnsemble)
        assert set(loaded.classifiers) == {"rf", "xgb"}
        for name in loaded.classifiers:
            assert isinstance(loaded[name], SeedEnsemble)
