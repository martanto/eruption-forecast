"""Tests for ExplainerEnsemble (SHAP-based explanation worker).

Uses synthetic in-memory SeedEnsemble objects (no disk I/O on the seismic
side) to verify the per-classifier × per-seed SHAP loop, the binary-class
shape normalisation across backends, the isinstance-based filter, the
``model_output`` / ``check_additivity`` knobs, the frequency-weighted
aggregate, and the joblib save/load round-trip.
"""

import os
import tempfile

import shap
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.ensemble.explainer_ensemble import (
    TREE_CLASSIFIERS,
    ExplainerEnsemble,
    _is_tree_classifier,
    _normalise_shap_values,
)
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

N_FEATURES = 6
N_TRAIN = 60
N_TEST = 12


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """Module-scoped RNG for the synthetic data fixtures."""
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def train_data(rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic training data — linearly separable enough to train trees."""
    feature_names = [f"feat_{i}" for i in range(N_FEATURES)]
    X = pd.DataFrame(rng.standard_normal((N_TRAIN, N_FEATURES)), columns=feature_names)
    y = (X["feat_0"] + 0.5 * X["feat_1"] - 0.3 * X["feat_2"] > 0).astype(int).to_numpy()
    return X, y


@pytest.fixture(scope="module")
def feature_df(rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic feature DataFrame for the test rows (held-out window grid)."""
    feature_names = [f"feat_{i}" for i in range(N_FEATURES)]
    return pd.DataFrame(
        rng.standard_normal((N_TEST, N_FEATURES)),
        columns=feature_names,
    )


def _build_seed_ensemble(
    classifier_cls: type,
    classifier_name: str,
    train_data: tuple[pd.DataFrame, np.ndarray],
    n_seeds: int = 3,
    feature_subset: list[str] | None = None,
    **kwargs,
) -> SeedEnsemble:
    """Fit ``n_seeds`` instances of ``classifier_cls`` and bundle them.

    Each seed uses its own subset of features (or the full set if not
    provided) so the test exercises the per-seed ``feature_names`` slicing
    inside ExplainerEnsemble.
    """
    X_train, y_train = train_data
    ensemble = SeedEnsemble(classifier_name=classifier_name)
    feature_names = feature_subset or list(X_train.columns)

    for rs in range(n_seeds):
        try:
            model = classifier_cls(random_state=rs, **kwargs)
        except TypeError:
            model = classifier_cls(**kwargs)
        model.fit(X_train[feature_names].to_numpy(), y_train)
        ensemble.seeds.append(
            {
                "random_state": rs,
                "model": model,
                "feature_names": feature_names,
            }
        )
    return ensemble


@pytest.fixture(scope="module")
def rf_ensemble(train_data) -> SeedEnsemble:
    """RandomForest SeedEnsemble — first three features only, per seed."""
    return _build_seed_ensemble(
        RandomForestClassifier,
        "RandomForestClassifier",
        train_data,
        n_seeds=3,
        feature_subset=[f"feat_{i}" for i in range(3)],
        n_estimators=10,
    )


@pytest.fixture(scope="module")
def gb_ensemble(train_data) -> SeedEnsemble:
    """GradientBoosting SeedEnsemble — first four features per seed."""
    return _build_seed_ensemble(
        GradientBoostingClassifier,
        "GradientBoostingClassifier",
        train_data,
        n_seeds=3,
        feature_subset=[f"feat_{i}" for i in range(4)],
        n_estimators=10,
    )


@pytest.fixture(scope="module")
def xgb_ensemble(train_data) -> SeedEnsemble:
    """XGBClassifier SeedEnsemble — features 0..4 per seed (heterogeneous union)."""
    return _build_seed_ensemble(
        XGBClassifier,
        "XGBClassifier",
        train_data,
        n_seeds=3,
        feature_subset=[f"feat_{i}" for i in range(5)],
        n_estimators=10,
        verbosity=0,
    )


@pytest.fixture(scope="module")
def lr_ensemble(train_data) -> SeedEnsemble:
    """LogisticRegression SeedEnsemble — used to test the non-tree skip path."""
    return _build_seed_ensemble(
        LogisticRegression,
        "LogisticRegression",
        train_data,
        n_seeds=2,
        feature_subset=[f"feat_{i}" for i in range(3)],
        max_iter=200,
    )


@pytest.fixture
def tree_classifier_ensemble(
    rf_ensemble, gb_ensemble, xgb_ensemble
) -> ClassifierEnsemble:
    """ClassifierEnsemble containing only tree-based classifiers."""
    return ClassifierEnsemble.from_seed_ensembles(
        {
            "RandomForestClassifier": rf_ensemble,
            "GradientBoostingClassifier": gb_ensemble,
            "XGBClassifier": xgb_ensemble,
        }
    )


@pytest.fixture
def mixed_classifier_ensemble(
    rf_ensemble, lr_ensemble
) -> ClassifierEnsemble:
    """ClassifierEnsemble mixing RF (tree) and LR (non-tree)."""
    return ClassifierEnsemble.from_seed_ensembles(
        {
            "RandomForestClassifier": rf_ensemble,
            "LogisticRegression": lr_ensemble,
        }
    )


# ---------------------------------------------------------------------------
# TREE_CLASSIFIERS tuple + isinstance helper
# ---------------------------------------------------------------------------


class TestTreeClassifierFilter:
    """Tree-classifier filter behaviour."""

    def test_tree_classifiers_tuple_of_types(self) -> None:
        """``TREE_CLASSIFIERS`` is a tuple of class objects (not name strings)."""
        assert isinstance(TREE_CLASSIFIERS, tuple)
        assert all(isinstance(c, type) for c in TREE_CLASSIFIERS)

    def test_is_tree_classifier_accepts_rf(self, rf_ensemble) -> None:
        """RandomForest seed ensembles are recognised as tree-based."""
        assert _is_tree_classifier(rf_ensemble) is True

    def test_is_tree_classifier_accepts_gb(self, gb_ensemble) -> None:
        """GradientBoosting seed ensembles are recognised as tree-based."""
        assert _is_tree_classifier(gb_ensemble) is True

    def test_is_tree_classifier_accepts_xgb(self, xgb_ensemble) -> None:
        """XGB seed ensembles are recognised as tree-based."""
        assert _is_tree_classifier(xgb_ensemble) is True

    def test_is_tree_classifier_rejects_lr(self, lr_ensemble) -> None:
        """LogisticRegression seed ensembles are rejected."""
        assert _is_tree_classifier(lr_ensemble) is False

    def test_is_tree_classifier_rejects_empty(self) -> None:
        """An empty SeedEnsemble returns False without IndexError."""
        empty = SeedEnsemble(classifier_name="Empty")
        assert _is_tree_classifier(empty) is False


# ---------------------------------------------------------------------------
# Shape normalisation
# ---------------------------------------------------------------------------


class TestShapeNormalisation:
    """Binary-classifier shape normalisation across backends."""

    def test_3d_values_collapse_to_positive_class(self) -> None:
        """sklearn 3-D ``(n_obs, n_features, 2)`` reduces to 2-D positive class."""
        n_obs, n_feat = 4, 3
        vals = np.zeros((n_obs, n_feat, 2))
        vals[..., 1] = np.arange(n_obs * n_feat).reshape(n_obs, n_feat)
        base = np.zeros((n_obs, 2))
        base[..., 1] = 0.25
        explanation = shap.Explanation(values=vals, base_values=base)
        shap_values, base_value = _normalise_shap_values(explanation)
        assert shap_values.shape == (n_obs, n_feat)
        assert shap_values.ndim == 2
        np.testing.assert_array_equal(shap_values, vals[..., 1])
        assert base_value == pytest.approx(0.25)

    def test_2d_values_pass_through(self) -> None:
        """XGB 2-D ``(n_obs, n_features)`` is returned unchanged."""
        n_obs, n_feat = 4, 3
        vals = np.arange(n_obs * n_feat).reshape(n_obs, n_feat).astype(float)
        base = np.full(n_obs, 0.7)
        explanation = shap.Explanation(values=vals, base_values=base)
        shap_values, base_value = _normalise_shap_values(explanation)
        assert shap_values.shape == (n_obs, n_feat)
        assert shap_values.ndim == 2
        np.testing.assert_array_equal(shap_values, vals)
        assert base_value == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Compute path
# ---------------------------------------------------------------------------


class TestComputeTreeOnly:
    """Compute path on a tree-only ensemble."""

    def test_compute_populates_results_for_all_classifiers(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """``compute`` materialises a result entry for each tree classifier."""
        obs_ids = feature_df.index[:4].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            ee = ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={
                    "RandomForestClassifier": obs_ids,
                    "GradientBoostingClassifier": obs_ids,
                    "XGBClassifier": obs_ids,
                },
                output_dir=tmp,
            ).compute()

            assert set(ee.results.keys()) == {
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
            }

    def test_compute_shap_values_are_2d_across_all_backends(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """Per-seed ``shap_values`` is 2-D for sklearn RF/GB and XGB alike.

        Catches the case where sklearn returns 3-D and the positive-class
        slice was missed.
        """
        obs_ids = feature_df.index[:4].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            ee = ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={
                    "RandomForestClassifier": obs_ids,
                    "GradientBoostingClassifier": obs_ids,
                    "XGBClassifier": obs_ids,
                },
                output_dir=tmp,
            ).compute()

            for classifier_name, payload in ee.results.items():
                for seed_data in payload["seeds"].values():
                    sv = seed_data["shap_values"]
                    assert sv.ndim == 2, (
                        f"{classifier_name} shap_values has rank "
                        f"{sv.ndim}, expected 2 after positive-class slice"
                    )
                    assert sv.shape == (
                        len(obs_ids),
                        len(seed_data["feature_names"]),
                    )

    def test_compute_persists_explanations_pkl_per_classifier(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """Each classifier gets a ``shap/explanations.pkl`` artefact."""
        obs_ids = feature_df.index[:4].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={
                    "RandomForestClassifier": obs_ids,
                    "GradientBoostingClassifier": obs_ids,
                    "XGBClassifier": obs_ids,
                },
                output_dir=tmp,
            ).compute()

            for clf in (
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "XGBClassifier",
            ):
                path = os.path.join(
                    tmp, "classifiers", clf, "shap", "explanations.pkl"
                )
                assert os.path.isfile(path)

    def test_compute_is_idempotent(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """Calling ``compute`` twice does not re-run the SHAP loop."""
        obs_ids = feature_df.index[:4].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            ee = ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={
                    "RandomForestClassifier": obs_ids,
                    "GradientBoostingClassifier": obs_ids,
                    "XGBClassifier": obs_ids,
                },
                output_dir=tmp,
                verbose=True,
            )
            ee.compute()
            results_snapshot = ee.results
            ee.compute()
            assert ee.results is results_snapshot


class TestComputeMixedEnsemble:
    """Non-tree classifiers in a mixed ensemble are skipped silently."""

    def test_non_tree_classifier_skipped(
        self, mixed_classifier_ensemble, feature_df, caplog
    ) -> None:
        """LR seeds are skipped and only the RF classifier produces results."""
        obs_ids = feature_df.index[:4].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            ee = ExplainerEnsemble(
                classifier_ensemble=mixed_classifier_ensemble,
                features_df=feature_df,
                observation_ids={
                    "RandomForestClassifier": obs_ids,
                    "LogisticRegression": obs_ids,
                },
                output_dir=tmp,
            ).compute()

            assert set(ee.results.keys()) == {"RandomForestClassifier"}
            assert "LogisticRegression" not in ee.results


# ---------------------------------------------------------------------------
# Frequency-weighted aggregate
# ---------------------------------------------------------------------------


class TestAggregateImportance:
    """``aggregate_importance.csv`` schema + frequency-weighted mean semantics."""

    def test_aggregate_csv_columns(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """Aggregate CSV has the four documented columns in order."""
        obs_ids = feature_df.index[:4].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={
                    "RandomForestClassifier": obs_ids,
                    "GradientBoostingClassifier": obs_ids,
                    "XGBClassifier": obs_ids,
                },
                output_dir=tmp,
            ).compute()

            csv_path = os.path.join(
                tmp,
                "classifiers",
                "RandomForestClassifier",
                "shap",
                "aggregate_importance.csv",
            )
            df = pd.read_csv(csv_path)
            assert list(df.columns) == [
                "feature",
                "mean_abs_shap",
                "selection_frequency",
                "n_seeds_selected",
            ]

    def test_frequency_weighted_mean_excludes_unselected_seeds(
        self, train_data, feature_df
    ) -> None:
        """``mean_abs_shap`` averages only over seeds that selected the feature.

        Three seeds: seeds 0 and 1 use ``[feat_0, feat_1]``; seed 2 uses
        ``[feat_1, feat_2]``. After aggregation, ``feat_0`` should have
        ``selection_frequency`` == 2/3 and ``n_seeds_selected`` == 2, *not*
        be deflated by a zero contribution from seed 2.
        """
        X_train, y_train = train_data
        ensemble = SeedEnsemble(classifier_name="RandomForestClassifier")
        per_seed_features = [
            ["feat_0", "feat_1"],
            ["feat_0", "feat_1"],
            ["feat_1", "feat_2"],
        ]
        for rs, feats in enumerate(per_seed_features):
            model = RandomForestClassifier(random_state=rs, n_estimators=10)
            model.fit(X_train[feats].to_numpy(), y_train)
            ensemble.seeds.append(
                {"random_state": rs, "model": model, "feature_names": feats}
            )

        ce = ClassifierEnsemble.from_seed_ensembles(
            {"RandomForestClassifier": ensemble}
        )
        obs_ids = feature_df.index[:4].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            ee = ExplainerEnsemble(
                classifier_ensemble=ce,
                features_df=feature_df,
                observation_ids={"RandomForestClassifier": obs_ids},
                output_dir=tmp,
            ).compute()

            agg = ee._aggregate_importance("RandomForestClassifier")
            agg_indexed = agg.set_index("feature")

            assert agg_indexed.loc["feat_0", "n_seeds_selected"] == 2
            assert agg_indexed.loc["feat_0", "selection_frequency"] == pytest.approx(
                2 / 3
            )
            assert agg_indexed.loc["feat_1", "n_seeds_selected"] == 3
            assert agg_indexed.loc["feat_1", "selection_frequency"] == pytest.approx(
                1.0
            )
            assert agg_indexed.loc["feat_2", "n_seeds_selected"] == 1
            assert agg_indexed.loc["feat_2", "selection_frequency"] == pytest.approx(
                1 / 3
            )


# ---------------------------------------------------------------------------
# model_output + feature_perturbation validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Constructor-time validation of ``model_output`` vs ``feature_perturbation``."""

    def test_probability_requires_interventional(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """``model_output='probability'`` with ``tree_path_dependent`` raises."""
        with pytest.raises(ValueError, match="probability"):
            ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={"RandomForestClassifier": [0, 1]},
                feature_perturbation="tree_path_dependent",
                model_output="probability",
            )

    def test_log_loss_requires_interventional(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """``model_output='log_loss'`` with ``tree_path_dependent`` raises."""
        with pytest.raises(ValueError, match="log_loss"):
            ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={"RandomForestClassifier": [0, 1]},
                feature_perturbation="tree_path_dependent",
                model_output="log_loss",
            )

    def test_interventional_probability_constructs(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """The valid ``interventional + probability`` combination does not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            ee = ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={"RandomForestClassifier": [0, 1]},
                feature_perturbation="interventional",
                model_output="probability",
                background_size=10,
                output_dir=tmp,
            )
            assert ee.model_output == "probability"
            assert ee.feature_perturbation == "interventional"


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Joblib save/load round-trips the populated ``results`` dict."""

    def test_save_load_round_trips_results(
        self, tree_classifier_ensemble, feature_df
    ) -> None:
        """``load(save())`` restores the same ``results`` content."""
        obs_ids = feature_df.index[:3].tolist()
        with tempfile.TemporaryDirectory() as tmp:
            ee = ExplainerEnsemble(
                classifier_ensemble=tree_classifier_ensemble,
                features_df=feature_df,
                observation_ids={
                    "RandomForestClassifier": obs_ids,
                    "GradientBoostingClassifier": obs_ids,
                    "XGBClassifier": obs_ids,
                },
                output_dir=tmp,
            ).compute()
            path = ee.save()

            restored = ExplainerEnsemble.load(path)
            assert set(restored.results.keys()) == set(ee.results.keys())
            for clf, payload in ee.results.items():
                for rs, seed_data in payload["seeds"].items():
                    np.testing.assert_array_equal(
                        restored.results[clf]["seeds"][rs]["shap_values"],
                        seed_data["shap_values"],
                    )

    def test_load_missing_file_raises(self) -> None:
        """``load`` raises ``FileNotFoundError`` on a non-existent path."""
        with pytest.raises(FileNotFoundError):
            ExplainerEnsemble.load("/nonexistent/path/ExplainerEnsemble.pkl")
