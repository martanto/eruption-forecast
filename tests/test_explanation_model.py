"""Smoke tests for ExplanationModel — the SHAP orchestrator stage.

Builds a lightweight TrainingModel-like duck (just the attributes
ExplanationModel actually reads) so the tests stay free of disk-bound
seismic CSVs while still exercising the full ``.explain()`` path.
"""

import os
import tempfile
from typing import Literal
from datetime import datetime
from dataclasses import field, dataclass

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.model.explanation_model import ExplanationModel
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


# ---------------------------------------------------------------------------
# Lightweight upstream-model duck
# ---------------------------------------------------------------------------


@dataclass
class _UpstreamModelDuck:
    """Minimal stand-in for TrainingModel / PredictionModel.

    Carries only the attributes ``ExplanationModel.__init__`` actually
    reads — avoids the full BaseModel construction machinery (tremor CSV,
    window grid validation, etc.) which is irrelevant for the SHAP path.
    """

    _tremor_data: pd.DataFrame
    start_date: datetime
    end_date: datetime
    window_size: int
    output_dir: str
    ClassifierEnsemble: ClassifierEnsemble | None
    features_df: pd.DataFrame
    kind: Literal["training", "prediction"] = "prediction"
    window_step: int = 12
    window_step_unit: Literal["minutes", "hours"] = "hours"
    labels: pd.Series = field(default_factory=pd.Series)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_TRAIN = 60
N_TEST = 12
N_FEATURES = 4
FEATURE_NAMES = [f"feat_{i}" for i in range(N_FEATURES)]


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Synthetic train + test data with a DatetimeIndex on the test rows."""
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(
        rng.standard_normal((N_TRAIN, N_FEATURES)), columns=FEATURE_NAMES
    )
    y_train = (
        X_train["feat_0"] + 0.5 * X_train["feat_1"] > 0
    ).astype(int).to_numpy()

    test_index = pd.date_range("2025-03-16", periods=N_TEST, freq="h")
    X_test = pd.DataFrame(
        rng.standard_normal((N_TEST, N_FEATURES)),
        columns=FEATURE_NAMES,
        index=test_index,
    )
    return X_train, y_train, X_test


@pytest.fixture
def rf_ensemble(synthetic_data) -> ClassifierEnsemble:
    """ClassifierEnsemble with two RandomForest seeds."""
    X_train, y_train, _ = synthetic_data
    se = SeedEnsemble(classifier_name="RandomForestClassifier")
    for rs in range(2):
        model = RandomForestClassifier(random_state=rs, n_estimators=10)
        model.fit(X_train.to_numpy(), y_train)
        se.seeds.append(
            {
                "random_state": rs,
                "model": model,
                "feature_names": FEATURE_NAMES,
            }
        )
    return ClassifierEnsemble.from_seed_ensembles(
        {"RandomForestClassifier": se}
    )


@pytest.fixture
def mixed_ensemble(synthetic_data) -> ClassifierEnsemble:
    """ClassifierEnsemble mixing RF (tree) and LR (non-tree)."""
    X_train, y_train, _ = synthetic_data
    rf = SeedEnsemble(classifier_name="RandomForestClassifier")
    lr = SeedEnsemble(classifier_name="LogisticRegression")
    for rs in range(2):
        rf_model = RandomForestClassifier(random_state=rs, n_estimators=10)
        rf_model.fit(X_train.to_numpy(), y_train)
        rf.seeds.append(
            {
                "random_state": rs,
                "model": rf_model,
                "feature_names": FEATURE_NAMES,
            }
        )
        lr_model = LogisticRegression(max_iter=200, random_state=rs)
        lr_model.fit(X_train.to_numpy(), y_train)
        lr.seeds.append(
            {
                "random_state": rs,
                "model": lr_model,
                "feature_names": FEATURE_NAMES,
            }
        )
    return ClassifierEnsemble.from_seed_ensembles(
        {"RandomForestClassifier": rf, "LogisticRegression": lr}
    )


def _build_duck(
    classifier_ensemble: ClassifierEnsemble,
    features_df: pd.DataFrame,
    output_dir: str,
    kind: Literal["training", "prediction"] = "prediction",
) -> _UpstreamModelDuck:
    """Build a TrainingModel/PredictionModel duck around a real ensemble."""
    tremor = pd.DataFrame(
        {"rsam_f0": [0.0, 1.0]},
        index=pd.date_range("2025-03-16", periods=2, freq="D"),
    )
    return _UpstreamModelDuck(
        _tremor_data=tremor,
        start_date=datetime(2025, 3, 16),
        end_date=datetime(2025, 3, 22),
        window_size=2,
        output_dir=output_dir,
        ClassifierEnsemble=classifier_ensemble,
        features_df=features_df,
        kind=kind,
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructor:
    """Construction-time validation and attribute wiring."""

    def test_raises_when_classifier_ensemble_missing(self, synthetic_data) -> None:
        """ExplanationModel raises if the upstream model has no ensemble."""
        _, _, X_test = synthetic_data
        with tempfile.TemporaryDirectory() as tmp:
            duck = _build_duck(
                classifier_ensemble=None,  # type: ignore[arg-type]
                features_df=X_test,
                output_dir=tmp,
            )
            duck.ClassifierEnsemble = None
            with pytest.raises(ValueError, match="ClassifierEnsemble"):
                ExplanationModel(model=duck)  # type: ignore[arg-type]

    def test_attributes_wired_through_from_upstream(
        self, rf_ensemble, synthetic_data
    ) -> None:
        """ExplanationModel inherits dates/window/features from the upstream model."""
        _, _, X_test = synthetic_data
        with tempfile.TemporaryDirectory() as tmp:
            duck = _build_duck(rf_ensemble, X_test, tmp)
            em = ExplanationModel(model=duck)  # type: ignore[arg-type]

            assert em.kind == "explanation"
            assert em.model_kind == "prediction"
            assert em.start_date == duck.start_date
            assert em.end_date == duck.end_date
            assert em.window_size == 2
            assert em.features_df is X_test
            assert em.ClassifierEnsemble is rf_ensemble


# ---------------------------------------------------------------------------
# Output directory namespacing
# ---------------------------------------------------------------------------


class TestOutputDirectories:
    """Output paths are namespaced by ``model_kind``."""

    @pytest.mark.parametrize("kind", ["training", "prediction"])
    def test_explanation_dir_uses_model_kind(
        self, rf_ensemble, synthetic_data, kind
    ) -> None:
        """``explanation_dir`` lands under ``explanation/{kind}/``."""
        _, _, X_test = synthetic_data
        with tempfile.TemporaryDirectory() as tmp:
            duck = _build_duck(rf_ensemble, X_test, tmp, kind=kind)
            em = ExplanationModel(model=duck)  # type: ignore[arg-type]
            assert em.explanation_dir == os.path.join(tmp, "explanation", kind)
            assert em.classifiers_dir == os.path.join(
                tmp, "explanation", kind, "classifiers"
            )


# ---------------------------------------------------------------------------
# explain() smoke test
# ---------------------------------------------------------------------------


class TestExplainSmoke:
    """Full ``.explain()`` path produces the expected artefacts."""

    def test_explain_produces_summary_csv(
        self, rf_ensemble, synthetic_data
    ) -> None:
        """Top-level summary CSV lands under ``explanation/{kind}/``."""
        _, _, X_test = synthetic_data
        with tempfile.TemporaryDirectory() as tmp:
            duck = _build_duck(rf_ensemble, X_test, tmp, kind="training")
            em = ExplanationModel(
                model=duck,  # type: ignore[arg-type]
                n_observations_to_explain=3,
            ).explain(plot_aggregate=False)

            assert em.ExplainerEnsemble is not None
            assert "RandomForestClassifier" in em.ExplainerEnsemble.results
            summary_csv = os.path.join(
                tmp,
                "explanation",
                "training",
                "result_all_model_explanations_"
                f"{em.start_date_str}_{em.end_date_str}.csv",
            )
            assert os.path.isfile(summary_csv)

    def test_explain_writes_aggregate_csv_per_classifier(
        self, rf_ensemble, synthetic_data
    ) -> None:
        """Per-classifier ``aggregate_importance.csv`` is written."""
        _, _, X_test = synthetic_data
        with tempfile.TemporaryDirectory() as tmp:
            duck = _build_duck(rf_ensemble, X_test, tmp, kind="training")
            ExplanationModel(
                model=duck,  # type: ignore[arg-type]
                n_observations_to_explain=3,
            ).explain(plot_aggregate=False)

            agg_csv = os.path.join(
                tmp,
                "explanation",
                "training",
                "classifiers",
                "RandomForestClassifier",
                "shap",
                "aggregate_importance.csv",
            )
            assert os.path.isfile(agg_csv)
            df = pd.read_csv(agg_csv)
            assert list(df.columns) == [
                "feature",
                "mean_abs_shap",
                "selection_frequency",
                "n_seeds_selected",
            ]

    def test_explain_skips_non_tree_classifiers(
        self, mixed_ensemble, synthetic_data
    ) -> None:
        """LR seeds in the upstream ensemble are skipped silently."""
        _, _, X_test = synthetic_data
        with tempfile.TemporaryDirectory() as tmp:
            duck = _build_duck(mixed_ensemble, X_test, tmp, kind="training")
            em = ExplanationModel(
                model=duck,  # type: ignore[arg-type]
                n_observations_to_explain=3,
                verbose=True,
            ).explain(plot_aggregate=False)

            assert set(em.ExplainerEnsemble.results.keys()) == {
                "RandomForestClassifier"
            }


# ---------------------------------------------------------------------------
# selection strategies
# ---------------------------------------------------------------------------


class TestSelectionStrategies:
    """``selection`` knob picks the right observations per classifier."""

    def test_top_proba_picks_highest_probability(
        self, rf_ensemble, synthetic_data
    ) -> None:
        """``selection='top_proba'`` orders by descending positive probability."""
        _, _, X_test = synthetic_data
        with tempfile.TemporaryDirectory() as tmp:
            duck = _build_duck(rf_ensemble, X_test, tmp)
            em = ExplanationModel(
                model=duck,  # type: ignore[arg-type]
                n_observations_to_explain=3,
                selection="top_proba",
            ).explain(plot_aggregate=False)

            arrays = rf_ensemble.predict_per_classifier(X_test)[
                "RandomForestClassifier"
            ]
            probs = arrays["probability"]
            expected = X_test.index[np.argsort(-probs)[:3]].tolist()
            picked = em.ExplainerEnsemble.results["RandomForestClassifier"][
                "observation_ids"
            ]
            assert picked == expected

    def test_near_threshold_picks_closest_to_half(
        self, rf_ensemble, synthetic_data
    ) -> None:
        """``selection='near_threshold'`` orders by ``|p - 0.5|`` ascending."""
        _, _, X_test = synthetic_data
        with tempfile.TemporaryDirectory() as tmp:
            duck = _build_duck(rf_ensemble, X_test, tmp)
            em = ExplanationModel(
                model=duck,  # type: ignore[arg-type]
                n_observations_to_explain=3,
                selection="near_threshold",
            ).explain(plot_aggregate=False)

            arrays = rf_ensemble.predict_per_classifier(X_test)[
                "RandomForestClassifier"
            ]
            probs = arrays["probability"]
            expected = X_test.index[np.argsort(np.abs(probs - 0.5))[:3]].tolist()
            picked = em.ExplainerEnsemble.results["RandomForestClassifier"][
                "observation_ids"
            ]
            assert picked == expected


# ---------------------------------------------------------------------------
# from_file
# ---------------------------------------------------------------------------


class TestFromFile:
    """``ExplanationModel.from_file`` round-trips through a real ``.pkl``."""

    def test_from_file_rejects_wrong_type(self) -> None:
        """A pickle of a non-TrainingModel/PredictionModel object is rejected."""
        import joblib

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bogus.pkl")
            joblib.dump({"not": "a model"}, path)
            with pytest.raises(TypeError, match="TrainingModel or PredictionModel"):
                ExplanationModel.from_file(path)

    def test_from_file_missing_path_raises(self) -> None:
        """``from_file`` raises ``FileNotFoundError`` on a non-existent path."""
        with pytest.raises(FileNotFoundError):
            ExplanationModel.from_file("/nonexistent/PredictionModel.pkl")
