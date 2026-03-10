"""End-to-end test for SeedEnsemble merge functionality.

Uses synthetic data to:
1. Train 5 tiny RandomForest models (one per seed).
2. Save each model + significant-features CSV to mimic ModelTrainer output.
3. Write a mock registry CSV.
4. Verify merge_seed_models() and SeedEnsemble.load() work correctly.
5. Verify predict_proba(), predict(), predict_with_uncertainty() shapes.
6. Verify merge_all_classifiers() bundles multiple classifiers.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from eruption_forecast.model.seed_ensemble import SeedEnsemble
from eruption_forecast.utils.ml import merge_seed_models, merge_all_classifiers

# ---------------------------------------------------------------------------
# Output directory (inside tests/ — never committed)
# ---------------------------------------------------------------------------
TESTS_OUT = os.path.join(
    os.path.dirname(__file__), "output", "seed_ensemble_test"
)
MODELS_DIR = os.path.join(TESTS_OUT, "models")
SIG_DIR = os.path.join(TESTS_OUT, "features", "significant_features")

for d in (MODELS_DIR, SIG_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
N_SAMPLES = 200
N_FEATURES = 10
N_SEEDS = 5
RNG = np.random.default_rng(42)

FEATURE_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]
X_all = pd.DataFrame(
    RNG.standard_normal((N_SAMPLES, N_FEATURES)), columns=FEATURE_NAMES
)
# Imbalanced: 10% eruptions
y_all = pd.Series(
    (RNG.random(N_SAMPLES) < 0.1).astype(int), name="is_erupted"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_seed(seed: int, feature_subset: list[str]) -> tuple[str, str]:
    """Train and save one RF model and its significant-features CSV."""
    clf = RandomForestClassifier(n_estimators=10, random_state=seed)
    clf.fit(X_all[feature_subset], y_all)

    model_path = os.path.join(MODELS_DIR, f"{seed:05d}.pkl")
    joblib.dump(clf, model_path)

    sig_df = pd.DataFrame(
        {"p_values": [0.01 * (i + 1) for i in range(len(feature_subset))]},
        index=pd.Index(feature_subset, name="features"),
    )
    sig_path = os.path.join(SIG_DIR, f"{seed:05d}.csv")
    sig_df.to_csv(sig_path)

    return sig_path, model_path


def _build_registry(records: list[dict], suffix: str) -> str:
    """Write a mock registry CSV and return its path."""
    registry_df = pd.DataFrame(records).set_index("random_state")
    csv_path = os.path.join(TESTS_OUT, f"trained_model_{suffix}.csv")
    registry_df.to_csv(csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_merge_and_load() -> str:
    """Test merge_seed_models() and SeedEnsemble.load()."""
    # Each seed uses a (possibly different) subset of features
    records = []
    for seed in range(N_SEEDS):
        subset = FEATURE_NAMES[seed % 3 : seed % 3 + 7]  # 7-feature window
        sig_path, model_path = _save_seed(seed, subset)
        records.append(
            {
                "random_state": seed,
                "significant_features_csv": sig_path,
                "trained_model_filepath": model_path,
            }
        )

    registry_csv = _build_registry(records, "rf_test")
    print(f"[test] Registry CSV: {registry_csv}")

    merged_path = merge_seed_models(registry_csv)
    print(f"[test] Merged pkl:   {merged_path}")

    assert os.path.isfile(merged_path), f"Merged file not created: {merged_path}"

    ensemble = SeedEnsemble.load(merged_path)
    assert isinstance(ensemble, SeedEnsemble), "Loaded object is not SeedEnsemble"
    assert len(ensemble) == N_SEEDS, f"Expected {N_SEEDS} seeds, got {len(ensemble)}"
    print(f"[PASS] merge_and_load: {ensemble}")
    return merged_path


def test_predict_shapes(merged_path: str) -> None:
    """Test that all prediction methods return correct shapes."""
    ensemble = SeedEnsemble.load(merged_path)
    n = len(X_all)

    # predict_with_uncertainty
    mean_p, std_p, conf, pred = ensemble.predict_with_uncertainty(X_all)
    assert mean_p.shape == (n,), f"mean_p shape: {mean_p.shape}"
    assert std_p.shape == (n,), f"std_p shape: {std_p.shape}"
    assert conf.shape == (n,), f"conf shape: {conf.shape}"
    assert pred.shape == (n,), f"pred shape: {pred.shape}"
    assert ((pred == 0) | (pred == 1)).all(), "pred must be binary"
    assert (conf >= 0.0).all() and (conf <= 1.0).all(), "conf must be in [0, 1]"
    print(
        f"[PASS] predict_with_uncertainty: shape=({n},), "
        f"mean_P={mean_p.mean():.4f}, mean_conf={conf.mean():.4f}"
    )

    # predict_proba — sklearn (n_samples, 2)
    proba = ensemble.predict_proba(X_all)
    assert proba.shape == (n, 2), f"predict_proba shape: {proba.shape}"
    assert np.allclose(proba.sum(axis=1), 1.0), "proba rows must sum to 1"
    assert (proba[:, 1] == mean_p).all(), "proba[:,1] must equal mean_p"
    print(f"[PASS] predict_proba: shape={proba.shape}")

    # predict — 1-D binary
    preds = ensemble.predict(X_all)
    assert preds.shape == (n,), f"predict shape: {preds.shape}"
    expected = (proba[:, 1] >= 0.5).astype(int)
    assert (preds == expected).all(), "predict must match proba[:,1] >= 0.5"
    print(f"[PASS] predict: shape={preds.shape}")


def test_merge_all_classifiers() -> None:
    """Test merge_all_classifiers() with two mock classifier registries."""
    # Build a second fake registry for "xgb"
    records_xgb = []
    for seed in range(N_SEEDS):
        subset = FEATURE_NAMES[1:8]
        sig_path, model_path = _save_seed(seed + 100, subset)
        records_xgb.append(
            {
                "random_state": seed + 100,
                "significant_features_csv": sig_path,
                "trained_model_filepath": model_path,
            }
        )
    rf_csv = os.path.join(TESTS_OUT, "trained_model_rf_test.csv")  # existing
    xgb_csv = _build_registry(records_xgb, "xgb_test")

    bundle_path = merge_all_classifiers({"rf": rf_csv, "xgb": xgb_csv})
    assert os.path.isfile(bundle_path), f"Bundle not found: {bundle_path}"

    loaded = joblib.load(bundle_path)
    assert isinstance(loaded, dict), "Bundle should be a dict"
    assert set(loaded.keys()) == {"rf", "xgb"}, f"Keys: {loaded.keys()}"
    for k, v in loaded.items():
        assert isinstance(v, SeedEnsemble), f"{k} value should be SeedEnsemble"
    print(f"[PASS] merge_all_classifiers: keys={list(loaded.keys())}")


def test_custom_output_dir() -> None:
    """Test merge_seed_models() with an explicit output directory."""
    registry_csv = os.path.join(TESTS_OUT, "trained_model_rf_test.csv")
    custom_dir = os.path.join(TESTS_OUT, "custom_output")
    os.makedirs(custom_dir, exist_ok=True)
    path = merge_seed_models(registry_csv, output_dir=custom_dir)
    assert os.path.isfile(path), f"Merged file not created: {path}"
    assert os.path.dirname(os.path.abspath(path)) == os.path.abspath(custom_dir)
    print(f"[PASS] custom output dir: {path}")


def main() -> None:
    """Run all SeedEnsemble tests."""
    print("=" * 60)
    print("SeedEnsemble Unit Tests (synthetic data)")
    print("=" * 60)

    print("\n--- Test 1: merge_seed_models + SeedEnsemble.load ---")
    merged_path = test_merge_and_load()

    print("\n--- Test 2: prediction method shapes ---")
    test_predict_shapes(merged_path)

    print("\n--- Test 3: merge_all_classifiers ---")
    test_merge_all_classifiers()

    print("\n--- Test 4: custom output directory ---")
    test_custom_output_dir()

    print("\n" + "=" * 60)
    print("All tests PASSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
