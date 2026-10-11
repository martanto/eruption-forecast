"""Tests for ``MetricsEnsemble``.

Uses an in-memory two-classifier ``ClassifierEnsemble`` built from the shared
``make_seed_ensemble`` factory, so ``compute()`` and the plot loops run on a
60-row synthetic matrix with ``n_jobs=1``.
"""

import os

import numpy as np
import pandas as pd
import pytest

from eruption_forecast.ensemble.metrics_ensemble import MetricsEnsemble
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


N_SEEDS = 2

METRIC_COLUMNS = {
    "accuracy",
    "balanced_accuracy",
    "precision",
    "average_precision",
    "pr_auc",
    "recall",
    "specificity",
    "f1_score",
    "roc_auc",
    "mcc",
    "g_mean",
    "true_positives",
    "false_positives",
    "true_negatives",
    "false_negatives",
}


@pytest.fixture()
def classifier_ensemble(make_seed_ensemble) -> ClassifierEnsemble:
    return ClassifierEnsemble.from_seed_ensembles(
        {
            "LogisticRegression": make_seed_ensemble(n_seeds=N_SEEDS),
            "RandomForestClassifier": make_seed_ensemble(n_seeds=N_SEEDS, kind="rf"),
        }
    )


@pytest.fixture()
def metrics_ensemble(classifier_ensemble, features_and_labels, tmp_path) -> MetricsEnsemble:
    X, y = features_and_labels
    return MetricsEnsemble(
        classifier_ensemble=classifier_ensemble,
        features_df=X,
        y_true=y,
        kind="training",
        output_dir=str(tmp_path / "evaluation"),
    )


class TestInit:
    def test_series_y_true_converted(self, metrics_ensemble) -> None:
        assert isinstance(metrics_ensemble.y_true, np.ndarray)
        assert metrics_ensemble.metrics == {}
        assert metrics_ensemble.classifiers_dir.endswith("classifiers")

    def test_default_output_dir_namespaced_by_kind(
        self, classifier_ensemble, features_and_labels, tmp_path
    ) -> None:
        X, y = features_and_labels
        ensemble = MetricsEnsemble(classifier_ensemble, X, y, kind="prediction", root_dir=str(tmp_path))
        assert ensemble.output_dir == os.path.join(str(tmp_path), "output", "evaluation", "prediction")


class TestCompute:
    def test_populates_per_classifier_results(self, metrics_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        metrics_ensemble.compute()

        assert set(metrics_ensemble.metrics) == {"LogisticRegression", "RandomForestClassifier"}
        for name, df in metrics_ensemble.metrics.items():
            assert set(df.columns) == METRIC_COLUMNS
            assert df.index.tolist() == list(range(N_SEEDS))
            assert metrics_ensemble.y_probas[name].shape == (len(X), N_SEEDS)
            assert metrics_ensemble.y_preds[name].shape == (len(X), N_SEEDS)

    def test_confusion_counts_sum_to_samples(self, metrics_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        metrics_ensemble.compute()
        df = metrics_ensemble.metrics["LogisticRegression"]
        counts = df[["true_positives", "false_positives", "true_negatives", "false_negatives"]]
        assert (counts.sum(axis=1) == len(X)).all()

    def test_learns_separable_data(self, metrics_ensemble) -> None:
        metrics_ensemble.compute()
        assert (metrics_ensemble.metrics["LogisticRegression"]["roc_auc"] > 0.9).all()

    def test_writes_prediction_matrices(self, metrics_ensemble, features_and_labels) -> None:
        X, _ = features_and_labels
        metrics_ensemble.compute()
        predictions_dir = os.path.join(
            metrics_ensemble.classifiers_dir, "LogisticRegression", "predictions"
        )
        y_proba = pd.read_csv(os.path.join(predictions_dir, "y_proba.csv"), index_col=0)
        y_pred = pd.read_csv(os.path.join(predictions_dir, "y_pred.csv"), index_col=0)

        assert y_proba.columns.tolist() == ["seed_00000", "seed_00001"]
        assert y_proba.index.tolist() == X.index.tolist()
        assert set(np.unique(y_pred.to_numpy())) <= {0, 1}

    def test_writes_metrics_csv(self, metrics_ensemble) -> None:
        metrics_ensemble.compute()
        path = os.path.join(
            metrics_ensemble.classifiers_dir,
            "LogisticRegression",
            "metrics",
            f"LogisticRegression_metrics_seeds-{N_SEEDS}.csv",
        )
        assert os.path.isfile(path)

    def test_idempotent(self, metrics_ensemble) -> None:
        metrics_ensemble.compute()
        first = metrics_ensemble.y_probas
        assert metrics_ensemble.compute().y_probas is first

    def test_y_true_length_mismatch_raises(self, classifier_ensemble, features_and_labels, tmp_path) -> None:
        X, y = features_and_labels
        ensemble = MetricsEnsemble(classifier_ensemble, X, y.iloc[:-1], output_dir=str(tmp_path))
        with pytest.raises(ValueError):
            ensemble.compute()


class TestPlots:
    def test_plot_aggregate_writes_png_and_csv(self, metrics_ensemble) -> None:
        stems = metrics_ensemble.plot_aggregate(include_plots=["roc_curve", "mcc_curve"])

        assert len(stems) == 2 * 2  # Two classifiers x two plots.
        for stem in stems:
            assert os.path.isfile(f"{stem}.png")
            assert os.path.isfile(f"{stem}.csv")

    def test_plot_aggregate_exclude(self, metrics_ensemble) -> None:
        stems = metrics_ensemble.plot_aggregate(
            include_plots=["roc_curve", "mcc_curve"], exclude_plots=["mcc_curve"]
        )
        assert all(stem.endswith("roc_curve") for stem in stems)

    def test_plot_seed_one_file_per_seed(self, metrics_ensemble) -> None:
        stems = metrics_ensemble.plot_seed(include_plots=["confusion_matrix"])
        assert len(stems) == 2 * N_SEEDS
        assert all(os.path.isfile(f"{stem}.png") for stem in stems)
        assert any(stem.endswith(os.path.join("confusion_matrix", "00001")) for stem in stems)

    def test_existing_plot_not_rerendered(self, metrics_ensemble) -> None:
        (stem,) = metrics_ensemble.plot_seed(include_plots=["roc_curve"])[:1]
        mtime = os.path.getmtime(f"{stem}.png")
        metrics_ensemble.plot_seed(include_plots=["roc_curve"])
        assert os.path.getmtime(f"{stem}.png") == mtime


class TestPersistence:
    def test_save_load_round_trip(self, metrics_ensemble) -> None:
        metrics_ensemble.compute()
        path = metrics_ensemble.save()

        restored = MetricsEnsemble.load(path)

        assert path.endswith("MetricsEnsemble.pkl")
        assert set(restored.metrics) == set(metrics_ensemble.metrics)
        np.testing.assert_allclose(
            restored.y_probas["LogisticRegression"],
            metrics_ensemble.y_probas["LogisticRegression"],
        )

    def test_save_custom_path(self, metrics_ensemble, tmp_path) -> None:
        path = metrics_ensemble.save(str(tmp_path / "nested" / "me.pkl"))
        assert os.path.isfile(path)

    def test_load_wrong_type_raises(self, tmp_path) -> None:
        path = tmp_path / "not-me.pkl"
        pd.to_pickle({"a": 1}, path)
        with pytest.raises(TypeError):
            MetricsEnsemble.load(str(path))


class TestFromFile:
    def test_builds_from_disk(self, make_seed_ensemble, features_and_labels, tmp_path) -> None:
        X, _ = features_and_labels
        model_path = tmp_path / "ClassifierEnsemble.pkl"
        ClassifierEnsemble.from_seed_ensembles(
            {"LogisticRegression": make_seed_ensemble(n_seeds=1)}
        ).save(str(model_path))
        features_path = tmp_path / "features.parquet"
        X.to_parquet(features_path)
        index = pd.date_range("2025-03-19", periods=len(X), freq="2h", name="datetime")
        label_csv = tmp_path / "label.csv"
        pd.DataFrame({"id": X.index, "is_erupted": 0}, index=index).to_csv(label_csv)

        ensemble = MetricsEnsemble.from_file(
            str(model_path),
            str(features_path),
            str(label_csv),
            eruption_dates=["2025-03-20"],
            output_dir=str(tmp_path / "evaluation"),
        )

        # 2025-03-20 covers 12 two-hour windows.
        assert ensemble.y_true.sum() == 12
        assert ensemble.ClassifierEnsemble.classifiers == ["LogisticRegression"]

    def test_missing_artefact_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            MetricsEnsemble.from_file(
                str(tmp_path / "m.pkl"),
                str(tmp_path / "f.parquet"),
                str(tmp_path / "l.csv"),
                eruption_dates=["2025-01-01"],
            )

