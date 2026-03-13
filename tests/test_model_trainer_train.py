"""Integration tests for ModelTrainer.train().

Covers the two-phase parallel dispatch (Phase 1 per-seed feature selection +
Phase 2 per (seed, classifier) training), including:

1. Basic ``train()`` run: verifies that significant-feature CSVs, model pkl
   files, and the registry CSV are all written to disk.
2. Resume / skip behaviour: a second ``train()`` call with ``overwrite=False``
   must leave the existing files untouched (mtime unchanged) and not raise.
3. Overwrite behaviour: a ``train()`` call with ``overwrite=True`` must
   regenerate all outputs (mtime updated).
"""

import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import pytest

from eruption_forecast.model.model_trainer import ModelTrainer

# ---------------------------------------------------------------------------
# Shared test configuration
# ---------------------------------------------------------------------------

TESTS_OUT = os.path.join(os.path.dirname(__file__), "output", "model_trainer_train")

N_SAMPLES = 300
N_FEATURES = 20
N_SEEDS = 2
RANDOM_STATE = 0
# ~15 % eruption rate keeps the minority class large enough for stratified
# cross-validation with only 2 folds yet still imbalanced enough to exercise
# the under-sampling path.
ERUPTION_RATE = 0.15

_RNG = np.random.default_rng(42)

FEATURE_NAMES = [f"feat_{i:02d}" for i in range(N_FEATURES)]

# Features CSV: row index is a window ID, columns are feature names
_X = pd.DataFrame(
    _RNG.standard_normal((N_SAMPLES, N_FEATURES)),
    index=pd.RangeIndex(N_SAMPLES, name="id"),
    columns=FEATURE_NAMES,
)
# Labels CSV: must have 'id' and 'is_erupted' columns
_y_values = (_RNG.random(N_SAMPLES) < ERUPTION_RATE).astype(int)
_LABELS = pd.DataFrame({"id": _X.index, "is_erupted": _y_values})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def csv_paths(tmp_path_factory):
    """Write synthetic feature / label CSVs and return their paths."""
    tmp = tmp_path_factory.mktemp("data")
    features_csv = str(tmp / "features.csv")
    labels_csv = str(tmp / "labels.csv")
    _X.to_csv(features_csv, index=True)
    _LABELS.to_csv(labels_csv, index=False)
    return features_csv, labels_csv


@pytest.fixture()
def output_dir():
    """Return (and clean up) a fresh output directory for each test."""
    if os.path.exists(TESTS_OUT):
        shutil.rmtree(TESTS_OUT)
    os.makedirs(TESTS_OUT, exist_ok=True)
    yield TESTS_OUT
    # Leave artefacts in place so failures are easy to inspect; CI will clean.


def _make_trainer(features_csv: str, labels_csv: str, out: str, **kwargs) -> ModelTrainer:
    """Create a fast ModelTrainer with a minimal grid for integration testing."""
    trainer = ModelTrainer(
        extracted_features_csv=features_csv,
        label_features_csv=labels_csv,
        output_dir=out,
        classifiers="lite-rf",
        feature_selection_method="random_forest",
        cv_strategy="shuffle",
        cv_splits=2,
        number_of_significant_features=5,
        n_jobs=1,
        **kwargs,
    )
    # Narrow the hyperparameter grid to a single combination so GridSearchCV
    # finishes quickly without sacrificing correctness of the pipeline test.
    trainer.update_grid_params(
        trainer.classifier_models[0],
        {"n_estimators": [10], "max_depth": [3]},
    )
    return trainer


# ---------------------------------------------------------------------------
# Helper: collect expected output paths after train()
# ---------------------------------------------------------------------------


def _expected_paths(trainer: ModelTrainer, random_state: int, n_seeds: int):
    """Return (significant_csvs, model_pkls, registry_csvs) expected on disk."""
    random_states = [random_state + s for s in range(n_seeds)]
    slug = trainer.classifier_models[0].slug_name

    sig_csvs = [
        os.path.join(trainer.shared_significant_dir, f"{rs:05d}.csv")
        for rs in random_states
    ]
    model_pkls = [
        trainer._get_model_filepath(rs, slug)
        for rs in random_states
    ]
    # registry CSV name includes classifier id, rs, ts, top-N
    registry_csvs = list(trainer.csv.values())

    return sig_csvs, model_pkls, registry_csvs


# ---------------------------------------------------------------------------
# Test 1: basic train() run
# ---------------------------------------------------------------------------


def test_train_creates_outputs(csv_paths, output_dir):
    """train() writes significant-feature CSVs, model pkl files, and a registry."""
    features_csv, labels_csv = csv_paths

    trainer = _make_trainer(features_csv, labels_csv, output_dir)
    trainer.train(random_state=RANDOM_STATE, total_seed=N_SEEDS, sampling_strategy=0.75)

    slug = trainer.classifier_models[0].slug_name
    random_states = [RANDOM_STATE + s for s in range(N_SEEDS)]

    # --- significant-feature CSVs exist and are non-empty --------------------
    for rs in random_states:
        sig_path = os.path.join(trainer.shared_significant_dir, f"{rs:05d}.csv")
        assert os.path.isfile(sig_path), f"Missing significant features CSV: {sig_path}"
        df_sig = pd.read_csv(sig_path, index_col=0)
        assert not df_sig.empty, f"Significant features CSV is empty: {sig_path}"

    # --- trained model pkl files exist and can be loaded ---------------------
    for rs in random_states:
        model_path = trainer._get_model_filepath(rs, slug)
        assert os.path.isfile(model_path), f"Missing model file: {model_path}"
        model = joblib.load(model_path)
        assert hasattr(model, "predict"), "Loaded model has no predict() method"

    # --- registry CSV written to self.csv ------------------------------------
    assert slug in trainer.csv, f"Registry CSV path not set for classifier '{slug}'"
    registry_csv = trainer.csv[slug]
    assert os.path.isfile(registry_csv), f"Registry CSV missing: {registry_csv}"
    df_registry = pd.read_csv(registry_csv, index_col=0)
    assert len(df_registry) == N_SEEDS, (
        f"Registry has {len(df_registry)} rows; expected {N_SEEDS}"
    )
    assert "significant_features_csv" in df_registry.columns
    assert "trained_model_filepath" in df_registry.columns

    # --- aggregated significant features file exists -------------------------
    agg_dir = trainer.shared_features_dir
    agg_csvs = [f for f in os.listdir(agg_dir) if f.endswith(".csv")]
    assert agg_csvs, "No aggregated significant-features CSV in shared features dir"


# ---------------------------------------------------------------------------
# Test 2: resume / skip behaviour (overwrite=False)
# ---------------------------------------------------------------------------


def test_train_skips_existing_outputs(csv_paths, output_dir):
    """Second train() call with overwrite=False must not modify existing files."""
    features_csv, labels_csv = csv_paths

    # First run — produce all outputs.
    trainer1 = _make_trainer(features_csv, labels_csv, output_dir)
    trainer1.train(random_state=RANDOM_STATE, total_seed=N_SEEDS)

    slug = trainer1.classifier_models[0].slug_name
    random_states = [RANDOM_STATE + s for s in range(N_SEEDS)]

    # Capture modification times of the significant-feature CSVs and models.
    def _mtimes():
        sig_mtimes = {
            rs: os.path.getmtime(
                os.path.join(trainer1.shared_significant_dir, f"{rs:05d}.csv")
            )
            for rs in random_states
        }
        model_mtimes = {
            rs: os.path.getmtime(trainer1._get_model_filepath(rs, slug))
            for rs in random_states
        }
        return sig_mtimes, model_mtimes

    sig_before, model_before = _mtimes()

    # Ensure at least 1 second elapses so mtime resolution is sufficient.
    time.sleep(1.1)

    # Second run — must skip all work.
    trainer2 = _make_trainer(features_csv, labels_csv, output_dir, overwrite=False)
    trainer2.train(random_state=RANDOM_STATE, total_seed=N_SEEDS)

    sig_after, model_after = _mtimes()

    for rs in random_states:
        assert sig_after[rs] == sig_before[rs], (
            f"Significant features CSV was rewritten for seed {rs} despite overwrite=False"
        )
        assert model_after[rs] == model_before[rs], (
            f"Model pkl was rewritten for seed {rs} despite overwrite=False"
        )


# ---------------------------------------------------------------------------
# Test 3: overwrite behaviour (overwrite=True)
# ---------------------------------------------------------------------------


def test_train_overwrites_existing_outputs(csv_paths, output_dir):
    """Third train() call with overwrite=True must regenerate all outputs."""
    features_csv, labels_csv = csv_paths

    # First run — produce all outputs.
    trainer1 = _make_trainer(features_csv, labels_csv, output_dir)
    trainer1.train(random_state=RANDOM_STATE, total_seed=N_SEEDS)

    slug = trainer1.classifier_models[0].slug_name
    random_states = [RANDOM_STATE + s for s in range(N_SEEDS)]

    def _sig_mtimes():
        return {
            rs: os.path.getmtime(
                os.path.join(trainer1.shared_significant_dir, f"{rs:05d}.csv")
            )
            for rs in random_states
        }

    sig_before = _sig_mtimes()

    # Ensure at least 1 second elapses.
    time.sleep(1.1)

    # Second run with overwrite=True — everything should be regenerated.
    trainer2 = _make_trainer(features_csv, labels_csv, output_dir, overwrite=True)
    trainer2.train(random_state=RANDOM_STATE, total_seed=N_SEEDS)

    sig_after = _sig_mtimes()

    for rs in random_states:
        assert sig_after[rs] > sig_before[rs], (
            f"Significant features CSV was NOT rewritten for seed {rs} with overwrite=True"
        )
