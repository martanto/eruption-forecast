"""Tests for ``SeedEnsemble`` and the path-based ``ClassifierEnsemble`` factories.

``test_classifier_ensemble.py`` covers in-memory consensus logic. This module
covers what it does not: loading from JSON / legacy CSV registries,
``from_any`` dispatch for every supported extension, per-seed inference, and
matrix persistence.
"""

import os
import json

import numpy as np
import joblib
import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from tests.conftest import FEATURE_NAMES
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _write_registry(ensemble: SeedEnsemble, root, fmt: str = "json") -> str:
    """Dump every seed model to disk and write a JSON or legacy CSV registry."""
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    records: list[dict] = []
    for seed in ensemble.seeds:
        random_state = seed["random_state"]
        model_path = models_dir / f"{random_state:05d}.pkl"
        joblib.dump(seed["model"], model_path)
        record = {
            "random_state": random_state,
            "features": seed["feature_names"],
            "model_filepath": str(model_path),
        }
        if fmt == "csv":
            features_csv = models_dir / f"{random_state:05d}_features.csv"
            pd.DataFrame(index=pd.Index(seed["feature_names"], name="features")).to_csv(
                features_csv
            )
            record["features_csv"] = str(features_csv)
        records.append(record)

    if fmt == "json":
        path = root / "trained-model__LogisticRegression.json"
        path.write_text(json.dumps(records), encoding="utf-8")
    else:
        path = root / "trained-model__LogisticRegression.csv"
        pd.DataFrame(records).set_index("random_state")[
            ["features_csv", "model_filepath"]
        ].to_csv(path)
    return str(path)


# ---------------------------------------------------------------------------
# SeedEnsemble factories
# ---------------------------------------------------------------------------


class TestSeedEnsembleFactories:
    @pytest.mark.parametrize("fmt", ["json", "csv"])
    def test_from_any_round_trip(self, make_seed_ensemble, tmp_path, fmt) -> None:
        original = make_seed_ensemble(n_seeds=3)
        path = _write_registry(original, tmp_path, fmt=fmt)

        loaded = SeedEnsemble.from_any(path)

        assert loaded.classifier_name == "LogisticRegression"
        assert [s["random_state"] for s in loaded.seeds] == [0, 1, 2]
        assert loaded.features == sorted(FEATURE_NAMES)

    def test_classifier_name_override(self, make_seed_ensemble, tmp_path) -> None:
        path = _write_registry(make_seed_ensemble(n_seeds=1), tmp_path)
        assert SeedEnsemble.from_json(path, classifier_name="lr").classifier_name == "lr"

    def test_features_is_union_of_seed_features(self, make_seed_ensemble, tmp_path) -> None:
        ensemble = make_seed_ensemble(n_seeds=2)
        ensemble.seeds[1]["feature_names"] = ["feat_0", "feat_1"]
        ensemble.seeds[0]["feature_names"] = ["feat_3"]
        path = _write_registry(ensemble, tmp_path)
        assert SeedEnsemble.from_json(path).features == ["feat_0", "feat_1", "feat_3"]

    def test_unsupported_extension_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            SeedEnsemble.from_any(str(tmp_path / "registry.txt"))

    @pytest.mark.parametrize("method", ["from_json", "from_registry"])
    def test_missing_registry_raises(self, tmp_path, method) -> None:
        with pytest.raises(FileNotFoundError):
            getattr(SeedEnsemble, method)(str(tmp_path / "missing"))

    @pytest.mark.parametrize(
        "payload",
        [[], {"a": 1}, [{"random_state": 0, "features": ["x"]}]],
        ids=["empty", "not-a-list", "missing-key"],
    )
    def test_invalid_json_payload_raises(self, tmp_path, payload) -> None:
        path = tmp_path / "registry.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            SeedEnsemble.from_json(str(path))

    def test_missing_model_pickle_raises_runtime_error(self, tmp_path) -> None:
        path = tmp_path / "registry.json"
        record = {"random_state": 0, "features": ["x"], "model_filepath": "gone.pkl"}
        path.write_text(json.dumps([record]), encoding="utf-8")
        with pytest.raises(RuntimeError, match="partial ensembles are not allowed"):
            SeedEnsemble.from_json(str(path))

    def test_legacy_csv_missing_column_raises(self, tmp_path) -> None:
        path = tmp_path / "registry.csv"
        pd.DataFrame({"model_filepath": ["a.pkl"]}, index=[0]).to_csv(path)
        with pytest.raises(ValueError):
            SeedEnsemble.from_registry(str(path))


# ---------------------------------------------------------------------------
# SeedEnsemble inference
# ---------------------------------------------------------------------------


class TestSeedEnsembleInference:
    def test_probability_and_prediction_matrices(
        self, make_seed_ensemble, features_and_labels
    ) -> None:
        X, _ = features_and_labels
        probabilities, predictions = make_seed_ensemble(
            n_seeds=3
        ).compute_probabilities_and_predictions(X)
        assert probabilities.shape == predictions.shape == (len(X), 3)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert set(np.unique(predictions)) <= {0, 1}

    def test_predict_proba_is_seed_mean(self, make_seed_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        ensemble = make_seed_ensemble(n_seeds=3)
        probabilities, _ = ensemble.compute_probabilities_and_predictions(X)

        proba = ensemble.predict_proba(X)

        np.testing.assert_allclose(proba[:, 1], probabilities.mean(axis=1))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_predict_threshold(self, make_seed_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        ensemble = make_seed_ensemble(n_seeds=2)
        assert ensemble.predict(X, threshold=0.0).sum() == len(X)
        assert ensemble.predict(X, threshold=1.01).sum() == 0

    def test_separable_data_is_learned(self, make_seed_ensemble, features_and_labels) -> None:
        X, y = features_and_labels
        accuracy = (make_seed_ensemble(n_seeds=2).predict(X) == y.to_numpy()).mean()
        assert accuracy > 0.9

    def test_decision_function_fallback(self, features_and_labels) -> None:
        X, y = features_and_labels
        ensemble = SeedEnsemble("LinearSVC")
        model = LinearSVC(random_state=0).fit(X, y)
        ensemble.seeds.append({"random_state": 0, "model": model, "feature_names": FEATURE_NAMES})

        probabilities, predictions = ensemble.compute_probabilities_and_predictions(X)

        expected = 1.0 / (1.0 + np.exp(-model.decision_function(X)))
        np.testing.assert_allclose(probabilities[:, 0], expected)
        np.testing.assert_array_equal(predictions[:, 0], (expected >= 0.5).astype(int))

    def test_estimator_without_scores_raises(self, features_and_labels) -> None:
        X, _ = features_and_labels
        ensemble = SeedEnsemble("Bad")
        ensemble.seeds.append({"random_state": 0, "model": object(), "feature_names": FEATURE_NAMES})
        with pytest.raises(RuntimeError):
            ensemble.compute_probabilities_and_predictions(X)

    def test_uses_each_seeds_feature_subset(self, make_seed_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        ensemble = make_seed_ensemble(n_seeds=1, feature_names=["feat_0", "feat_2"])
        # Columns the seed never saw must not matter.
        ensemble.compute_probabilities_and_predictions(X[["feat_0", "feat_2"]])


class TestSeedEnsembleMatrices:
    def test_predict_with_uncertainty_writes_matrices(
        self, make_seed_ensemble, features_and_labels, tmp_path
    ) -> None:
        X, _ = features_and_labels
        ensemble = make_seed_ensemble(n_seeds=3)

        probability, uncertainty, prediction, confidence = ensemble.predict_with_uncertainty(
            X, save=True, output_dir=str(tmp_path)
        )

        assert probability.shape == uncertainty.shape == prediction.shape == confidence.shape
        assert ensemble.probabilities is not None
        assert ensemble.probabilities.columns.tolist() == [
            "seed_00000",
            "seed_00001",
            "seed_00002",
        ]
        assert (tmp_path / "LogisticRegression_seed_probabilities.parquet").is_file()
        assert (tmp_path / "LogisticRegression_seed_predictions.parquet").is_file()
        assert sorted(os.listdir(tmp_path / "seeds")) == [
            "p_00000.csv",
            "p_00001.csv",
            "p_00002.csv",
        ]

    def test_save_load_matrices_round_trip(self, make_seed_ensemble, tmp_path) -> None:
        ensemble = make_seed_ensemble(n_seeds=2)
        probabilities = np.array([[0.1, 0.2], [0.7, 0.9]])
        predictions = np.array([[0, 0], [1, 1]])
        index = pd.Index([10, 11], name="id")

        proba_path, pred_path = ensemble.save_matrices(
            str(tmp_path), probabilities, predictions, index=index
        )
        restored = make_seed_ensemble(n_seeds=2).load_matrices(proba_path, pred_path)

        pd.testing.assert_frame_equal(restored.probabilities, ensemble.probabilities)
        assert restored.predictions is not None
        assert restored.predictions.dtypes.unique().tolist() == [np.int8]
        assert restored.predictions.index.tolist() == [10, 11]

    def test_load_matrices_missing_file_raises(self, make_seed_ensemble, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            make_seed_ensemble(n_seeds=1).load_matrices(
                str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
            )

    def test_pickle_round_trip(self, make_seed_ensemble, features_and_labels, tmp_path) -> None:
        X, _ = features_and_labels
        ensemble = make_seed_ensemble(n_seeds=2)
        path = tmp_path / "SeedEnsemble.pkl"
        ensemble.save(str(path))

        restored = SeedEnsemble.load(str(path))

        np.testing.assert_allclose(restored.predict_proba(X), ensemble.predict_proba(X))


# ---------------------------------------------------------------------------
# ClassifierEnsemble path factories
# ---------------------------------------------------------------------------


class TestClassifierEnsembleFromPaths:
    def test_from_any_seed_ensemble_instance(self, make_seed_ensemble) -> None:
        ensemble = ClassifierEnsemble.from_any(make_seed_ensemble(n_seeds=1))
        assert ensemble.classifiers == ["LogisticRegression"]

    @pytest.mark.parametrize("fmt", ["json", "csv"])
    def test_from_any_registry(self, make_seed_ensemble, tmp_path, fmt) -> None:
        path = _write_registry(make_seed_ensemble(n_seeds=2), tmp_path, fmt=fmt)
        ensemble = ClassifierEnsemble.from_any(path)
        assert ensemble.classifiers == ["LogisticRegression"]
        assert len(ensemble["LogisticRegression"]) == 2

    def test_from_any_results_map_json(self, make_seed_ensemble, tmp_path) -> None:
        lr_dir = tmp_path / "lr"
        rf_dir = tmp_path / "rf"
        lr_dir.mkdir()
        rf_dir.mkdir()
        results = {
            "lr": _write_registry(make_seed_ensemble(n_seeds=1), lr_dir),
            "rf": _write_registry(make_seed_ensemble(n_seeds=1, kind="rf"), rf_dir),
        }
        path = tmp_path / "ClassifierEnsemble.json"
        path.write_text(json.dumps(results), encoding="utf-8")

        ensemble = ClassifierEnsemble.from_any(str(path))

        assert ensemble.classifiers == ["lr", "rf"]
        assert ensemble.features == sorted(FEATURE_NAMES)

    @pytest.mark.parametrize("wrap", [True, False], ids=["classifier-pkl", "seed-pkl"])
    def test_from_any_pickle(self, make_seed_ensemble, tmp_path, wrap) -> None:
        seed_ensemble = make_seed_ensemble(n_seeds=1)
        obj = ClassifierEnsemble.from_seed_ensembles({"lr": seed_ensemble}) if wrap else seed_ensemble
        path = tmp_path / "ensemble.pkl"
        obj.save(str(path))

        ensemble = ClassifierEnsemble.from_any(str(path))

        assert isinstance(ensemble, ClassifierEnsemble)
        assert len(ensemble) == 1

    def test_from_any_unsupported_extension(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            ClassifierEnsemble.from_any(str(tmp_path / "ensemble.txt"))

    def test_from_any_missing_json(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            ClassifierEnsemble.from_any(str(tmp_path / "missing.json"))

    def test_from_dict_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            ClassifierEnsemble.from_dict({})

    def test_from_json_rejects_list_payload(self, tmp_path) -> None:
        path = tmp_path / "list.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError):
            ClassifierEnsemble.from_json(str(path))
