"""Tests for ``ExplainerEnsemble``.

Tiny random forests (5 trees, depth 3) keep ``shap.TreeExplainer`` fast. A
``LogisticRegression`` ensemble is mixed in to check that non-tree
classifiers are skipped instead of failing the run.
"""

import os

import shap
import numpy as np
import pandas as pd
import pytest

from eruption_forecast.ensemble.explainer_ensemble import ExplainerEnsemble
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.dataclass.classifier_explanation import (
    SeedExplanation,
    ClassifierExplanation,
)


N_SEEDS = 2


@pytest.fixture()
def mixed_ensemble(make_seed_ensemble) -> ClassifierEnsemble:
    return ClassifierEnsemble.from_seed_ensembles(
        {
            "RandomForestClassifier": make_seed_ensemble(n_seeds=N_SEEDS, kind="rf"),
            "LogisticRegression": make_seed_ensemble(n_seeds=N_SEEDS),
        }
    )


@pytest.fixture()
def explainer(mixed_ensemble, features_and_labels, tmp_path) -> ExplainerEnsemble:
    X, _ = features_and_labels
    return ExplainerEnsemble(
        classifier_ensemble=mixed_ensemble,
        features_df=X,
        kind="training",
        output_dir=str(tmp_path / "explanation" / "classifiers"),
    )


class TestNormaliseShapValues:
    def test_three_dimensional_takes_positive_class(self) -> None:
        values = np.stack([np.zeros((3, 2)), np.ones((3, 2))], axis=-1)
        explanation = shap.Explanation(values=values, base_values=np.array([[0.4, 0.6]] * 3))

        shap_values, base_values = ExplainerEnsemble.normalise_shap_values(explanation)

        assert shap_values.shape == (3, 2)
        assert (shap_values == 1).all()
        np.testing.assert_allclose(base_values, 0.6)

    def test_two_dimensional_passthrough(self) -> None:
        explanation = shap.Explanation(values=np.ones((3, 2)), base_values=np.zeros(3))
        shap_values, base_values = ExplainerEnsemble.normalise_shap_values(explanation)
        assert shap_values.shape == (3, 2)
        assert base_values.shape == (3,)


class TestExplainSeed:
    def test_shapes_and_short_names(self, make_seed_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        seed = make_seed_ensemble(n_seeds=1, kind="rf").seeds[0]

        explanation = ExplainerEnsemble.explain_seed(seed, X)

        assert explanation.values.shape == (len(X), X.shape[1])
        assert explanation.feature_names == list(X.columns)

    def test_shap_values_are_additive(self, make_seed_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        seed = make_seed_ensemble(n_seeds=1, kind="rf").seeds[0]

        explanation = ExplainerEnsemble.explain_seed(seed, X)

        reconstructed = explanation.values.sum(axis=1) + explanation.base_values
        np.testing.assert_allclose(
            reconstructed, seed["model"].predict_proba(X)[:, 1], atol=1e-6
        )

    def test_saves_when_requested(self, make_seed_ensemble, features_and_labels, tmp_path) -> None:
        X, _ = features_and_labels
        path = tmp_path / "00000.pkl"
        ExplainerEnsemble.explain_seed(
            make_seed_ensemble(n_seeds=1, kind="rf").seeds[0],
            X,
            save_per_seed=True,
            seed_explanation_filepath=str(path),
        )
        assert path.is_file()

    def test_failure_is_wrapped_with_seed_id(self, make_seed_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        seed = make_seed_ensemble(n_seeds=1).seeds[0]  # LogisticRegression is not a tree.
        with pytest.raises(ValueError, match="Explaining Seed 0"):
            ExplainerEnsemble.explain_seed(seed, X)


class TestExplain:
    def test_skips_non_tree_classifiers(self, explainer: ExplainerEnsemble) -> None:
        explainer.explain()

        assert [e.classifier_name for e in explainer.explanations] == ["RandomForestClassifier"]
        explanation = explainer.explanations[0]
        assert isinstance(explanation, ClassifierExplanation)
        assert [s.random_state for s in explanation.seeds] == [0, 1]
        assert all(isinstance(s, SeedExplanation) for s in explanation.seeds)

    def test_writes_classifier_and_seed_artefacts(self, explainer: ExplainerEnsemble) -> None:
        explainer.explain(save_per_seed=True)
        clf_dir = os.path.join(explainer.output_dir, "RandomForestClassifier")

        assert os.path.isfile(
            os.path.join(clf_dir, "ClassifierExplanation_RandomForestClassifier.pkl")
        )
        assert sorted(os.listdir(os.path.join(clf_dir, "shap_values"))) == [
            "00000.pkl",
            "00001.pkl",
        ]

    def test_cached_classifier_explanation_is_reused(
        self, mixed_ensemble, features_and_labels, explainer: ExplainerEnsemble
    ) -> None:
        explainer.explain()
        X, _ = features_and_labels
        second = ExplainerEnsemble(
            mixed_ensemble, X, kind="training", output_dir=explainer.output_dir
        )
        # Drop the seeds: only a cache hit can still produce an explanation.
        second.ClassifierEnsemble["RandomForestClassifier"].seeds = []

        second.explain()

        assert len(second.explanations[0].seeds) == N_SEEDS

    def test_parallel_matches_serial(
        self, mixed_ensemble, features_and_labels, tmp_path
    ) -> None:
        X, _ = features_and_labels
        seed_ensemble = mixed_ensemble["RandomForestClassifier"]
        serial = ExplainerEnsemble(mixed_ensemble, X, n_jobs=1).explain_classifier(
            seed_ensemble, X, output_dir=str(tmp_path / "serial")
        )
        parallel = ExplainerEnsemble(mixed_ensemble, X, n_jobs=2).explain_classifier(
            seed_ensemble, X, output_dir=str(tmp_path / "parallel")
        )
        for a, b in zip(serial.seeds, parallel.seeds, strict=True):
            np.testing.assert_allclose(a.shap_values.values, b.shap_values.values)


class TestPlots:
    def test_plots_require_explain(self, explainer: ExplainerEnsemble) -> None:
        for method in (explainer.plot_seed, explainer.plot_aggregate):
            with pytest.raises(ValueError):
                method()
        with pytest.raises(ValueError):
            explainer.plot_waterfall(labels=pd.Series(dtype=int), eruption_dates=[])

    def test_plot_seed_writes_bar_and_beeswarm(self, explainer: ExplainerEnsemble) -> None:
        explainer.explain().plot_seed(max_display=4, dpi=50)
        figures = os.path.join(explainer.output_dir, "RandomForestClassifier", "figures")
        for kind in ("bar", "beeswarm"):
            assert sorted(os.listdir(os.path.join(figures, kind))) == [
                "00000.png",
                "00001.png",
            ]

    def test_plot_aggregate_writes_sidecars(self, explainer: ExplainerEnsemble) -> None:
        explainer.explain().plot_aggregate(max_display=4, top_n=4, dpi=50)
        aggregate = os.path.join(
            explainer.output_dir, "RandomForestClassifier", "figures", "aggregate"
        )
        for name in ("bar.png", "bar.csv", "beeswarm.png", "beeswarm.parquet"):
            assert os.path.isfile(os.path.join(aggregate, name)), name

    def test_plot_waterfall_per_eruption(
        self, explainer: ExplainerEnsemble, features_and_labels, tmp_path
    ) -> None:
        X, _ = features_and_labels
        seed_ensemble = explainer.ClassifierEnsemble["RandomForestClassifier"]
        seed_ensemble.predict_with_uncertainty(X, output_dir=str(tmp_path / "matrices"))
        index = pd.date_range("2025-03-18", periods=len(X), freq="2h", name="datetime")
        labels = pd.DataFrame({"id": X.index, "is_erupted": 0}, index=index)

        explainer.explain().plot_waterfall(labels, eruption_dates=["2025-03-20"], dpi=50)

        eruption_dir = os.path.join(explainer.explanation_dir, "eruptions", "2025-03-20")
        assert os.path.isdir(eruption_dir)
        assert any(name.endswith(".png") for name in os.listdir(eruption_dir))
