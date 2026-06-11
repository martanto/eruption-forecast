"""Integration smoke tests for the DALEX explanation layer.

Builds a hand-rolled ``ClassifierEnsemble`` containing one
:class:`~eruption_forecast.ensemble.seed_ensemble.SeedEnsemble` per tree
classifier (``RandomForestClassifier``, ``XGBClassifier``,
``GradientBoostingClassifier``) plus one non-tree classifier
(``LogisticRegression``) using ``sklearn.datasets.make_classification``,
then exercises :class:`~eruption_forecast.ensemble.dalex_explainer_ensemble.DalexExplainerEnsemble`
end to end. The non-tree classifier must be skipped without error and the
tree classifiers must produce CSV + HTML artefacts under the expected
directory layout.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.ensemble.dalex_explainer_ensemble import (
    TREE_CLASSIFIERS,
    DalexExplainerEnsemble,
)
from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.utils.formatting import slugify


def _make_dataset() -> tuple[pd.DataFrame, np.ndarray]:
    """Tiny binary-classification dataset with named features.

    Returns:
        tuple[pd.DataFrame, np.ndarray]: ``(features_df, y_true)``. ``features_df``
            has 60 rows and 8 features named ``feat_0``..``feat_7``.
    """
    X, y = make_classification(
        n_samples=60,
        n_features=8,
        n_informative=4,
        n_redundant=2,
        random_state=0,
    )
    columns = [f"feat_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), y.astype(int)


def _seed_ensemble_for(
    classifier_name: str,
    estimator_cls,
    features_df: pd.DataFrame,
    y_true: np.ndarray,
    n_seeds: int = 2,
) -> SeedEnsemble:
    """Fit ``n_seeds`` instances of ``estimator_cls`` and bundle them into a SeedEnsemble.

    Each seed is fit on a different random subset of the feature columns so
    that the per-seed ``feature_names`` list — which DALEX must respect —
    actually varies between seeds. Mirrors what ``TrainingModel.fit()``
    produces but without the surrounding pipeline.

    Args:
        classifier_name (str): Name the SeedEnsemble will be stored under
            (must match what ``ClassifierEnsemble`` indexes by).
        estimator_cls: sklearn-compatible classifier class.
        features_df (pd.DataFrame): Feature matrix.
        y_true (np.ndarray): Binary labels.
        n_seeds (int): Number of seed estimators to fit. Defaults to ``2``.

    Returns:
        SeedEnsemble: Populated ensemble ready for DALEX consumption.
    """
    rng = np.random.default_rng(42)
    ensemble = SeedEnsemble(classifier_name=classifier_name)
    all_cols = features_df.columns.tolist()
    for i in range(n_seeds):
        random_state = i + 1
        n_pick = max(4, len(all_cols) - i)
        cols = rng.choice(all_cols, size=n_pick, replace=False).tolist()
        if estimator_cls is XGBClassifier:
            est = estimator_cls(
                n_estimators=20,
                max_depth=3,
                random_state=random_state,
                eval_metric="logloss",
                verbosity=0,
            )
        elif estimator_cls is GradientBoostingClassifier:
            est = estimator_cls(
                n_estimators=20, max_depth=3, random_state=random_state
            )
        elif estimator_cls is RandomForestClassifier:
            est = estimator_cls(
                n_estimators=20,
                max_depth=4,
                random_state=random_state,
                n_jobs=1,
            )
        else:
            est = estimator_cls(random_state=random_state, max_iter=200)
        est.fit(features_df[cols], y_true)
        ensemble.seeds.append(
            {
                "random_state": random_state,
                "model": est,
                "feature_names": cols,
            }
        )
    return ensemble


@pytest.fixture(scope="module")
def fitted_ensemble() -> tuple[ClassifierEnsemble, pd.DataFrame, np.ndarray]:
    """Hand-built ensemble with three tree classifiers + one non-tree classifier."""
    features_df, y_true = _make_dataset()
    ensemble = ClassifierEnsemble.from_seed_ensembles(
        {
            "RandomForestClassifier": _seed_ensemble_for(
                "RandomForestClassifier",
                RandomForestClassifier,
                features_df,
                y_true,
            ),
            "XGBClassifier": _seed_ensemble_for(
                "XGBClassifier",
                XGBClassifier,
                features_df,
                y_true,
            ),
            "GradientBoostingClassifier": _seed_ensemble_for(
                "GradientBoostingClassifier",
                GradientBoostingClassifier,
                features_df,
                y_true,
            ),
            "LogisticRegression": _seed_ensemble_for(
                "LogisticRegression",
                LogisticRegression,
                features_df,
                y_true,
            ),
        },
    )
    return ensemble, features_df, y_true


class TestTreeClassifierFilter:
    """The whitelist defined by :data:`TREE_CLASSIFIERS`."""

    def test_whitelist_contains_three_classifiers(self) -> None:
        """RF, XGB, and GB are the three explicitly supported tree classifiers."""
        assert TREE_CLASSIFIERS == {
            "RandomForestClassifier",
            "XGBClassifier",
            "GradientBoostingClassifier",
        }


class TestFeatureNameSlug:
    """``slugify`` strips Windows-illegal characters from tsfresh feature names."""

    def test_strips_double_quotes_and_special_chars(self) -> None:
        """A canonical tsfresh feature name becomes a path-safe slug."""
        slug = slugify(
            'entropy__fft_coefficient__attr_"abs"__coeff_44', hyphen="_"
        )
        for ch in '<>:"/\\|?*':
            assert ch not in slug
        assert "entropy" in slug
        assert "abs" in slug
        assert "44" in slug

    def test_collapses_double_underscores(self) -> None:
        """Tsfresh's ``__`` separators collapse to single ``_`` in the slug."""
        assert "__" not in slugify("foo__bar__baz", hyphen="_")


class TestDalexExplainerEnsemble:
    """End-to-end exercise of the DALEX layer on a hand-built ensemble."""

    def test_skipped_classifiers_excludes_tree_models(
        self,
        fitted_ensemble: tuple[ClassifierEnsemble, pd.DataFrame, np.ndarray],
    ) -> None:
        """Only the non-tree classifier should appear in :attr:`skipped_classifiers`."""
        ensemble, features_df, y_true = fitted_ensemble
        de = DalexExplainerEnsemble(
            classifier_ensemble=ensemble,
            features_df=features_df,
            y_true=y_true,
        )
        assert de.skipped_classifiers == ["LogisticRegression"]
        assert set(de.tree_classifiers) == {
            "RandomForestClassifier",
            "XGBClassifier",
            "GradientBoostingClassifier",
        }

    def test_compute_writes_per_seed_and_aggregate_artefacts(
        self,
        fitted_ensemble: tuple[ClassifierEnsemble, pd.DataFrame, np.ndarray],
    ) -> None:
        """``compute()`` materialises CSV + HTML per seed and an aggregate CSV per classifier."""
        ensemble, features_df, y_true = fitted_ensemble
        with tempfile.TemporaryDirectory() as tmp:
            de = DalexExplainerEnsemble(
                classifier_ensemble=ensemble,
                features_df=features_df,
                y_true=y_true,
                output_dir=tmp,
                n_seeds_to_explain=2,
                n_observations_to_explain=2,
                top_k_features=3,
                permutation_B=2,
                shap_B=5,
                verbose=False,
            ).compute()

            # Per-classifier per-seed outputs for tree models only.
            for clf in (
                "RandomForestClassifier",
                "XGBClassifier",
                "GradientBoostingClassifier",
            ):
                clf_dir = os.path.join(tmp, "classifiers", clf)
                shap_seeds = os.path.join(clf_dir, "shap", "seeds")
                vi_seeds = os.path.join(
                    clf_dir, "variable_importance", "seeds"
                )
                shap_agg = os.path.join(clf_dir, "shap", "aggregate.csv")
                vi_agg = os.path.join(
                    clf_dir, "variable_importance", "aggregate.csv"
                )
                assert os.path.isdir(shap_seeds), f"missing {shap_seeds}"
                assert os.path.isdir(vi_seeds), f"missing {vi_seeds}"
                assert any(
                    f.endswith(".csv") for f in os.listdir(shap_seeds)
                ), f"no SHAP CSV under {shap_seeds}"
                assert any(
                    f.endswith(".html") for f in os.listdir(shap_seeds)
                ), f"no SHAP HTML under {shap_seeds}"
                assert os.path.isfile(shap_agg), f"missing {shap_agg}"
                assert os.path.isfile(vi_agg), f"missing {vi_agg}"

            # Non-tree classifier must not produce a directory.
            assert not os.path.exists(
                os.path.join(tmp, "classifiers", "LogisticRegression")
            )

            # Per-classifier in-memory results populated for tree models only.
            assert set(de.shap_results.keys()) == set(de.vi_results.keys())
            assert "LogisticRegression" not in de.shap_results

    def test_save_load_round_trip(
        self,
        fitted_ensemble: tuple[ClassifierEnsemble, pd.DataFrame, np.ndarray],
    ) -> None:
        """``save()`` → ``load()`` preserves the populated result dicts."""
        ensemble, features_df, y_true = fitted_ensemble
        with tempfile.TemporaryDirectory() as tmp:
            de = DalexExplainerEnsemble(
                classifier_ensemble=ensemble,
                features_df=features_df,
                y_true=y_true,
                output_dir=tmp,
                n_seeds_to_explain=2,
                n_observations_to_explain=2,
                top_k_features=3,
                permutation_B=2,
                shap_B=5,
            ).compute(plot_local=False, plot_global=False, plot_profile=False)

            pkl = de.save()
            restored = DalexExplainerEnsemble.load(pkl)
            assert set(restored.shap_results.keys()) == set(de.shap_results.keys())
            assert set(restored.vi_results.keys()) == set(de.vi_results.keys())

    def test_load_missing_file_raises(self) -> None:
        """``load()`` raises ``FileNotFoundError`` for a missing path."""
        with pytest.raises(FileNotFoundError):
            DalexExplainerEnsemble.load("/nonexistent/path.pkl")
