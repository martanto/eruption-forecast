"""Tests for ``eruption_forecast.utils.ml``.

Covers ground-truth construction, the leakage-free temporal split, threshold
sweeps, class resampling, per-seed resampled reloads, and classifier-model
factories. ``grid_search_cv`` / ``compute_seed`` are exercised by the
stage-model tests and are not repeated here.
"""

import numpy as np
import pandas as pd
import pytest

from eruption_forecast.utils.ml import (
    resample,
    build_y_true,
    random_under_sampler,
    get_classifier_models,
    load_features_resampled,
    compute_threshold_metrics,
    temporal_train_test_split,
    build_classifier_ensemble_summary,
    compute_aggregate_threshold_metrics,
)


# ---------------------------------------------------------------------------
# build_y_true
# ---------------------------------------------------------------------------


@pytest.fixture()
def features_label_csv(tmp_path) -> str:
    """Six-hourly features-label CSV over three days with no positives."""
    index = pd.date_range("2025-03-19", "2025-03-21 18:00", freq="6h", name="datetime")
    df = pd.DataFrame({"id": range(len(index)), "is_erupted": 0}, index=index)
    path = tmp_path / "features-label.csv"
    df.to_csv(path)
    return str(path)


class TestBuildYTrue:
    def test_marks_eruption_day(self, features_label_csv: str) -> None:
        y_true = build_y_true(features_label_csv, eruption_dates=["2025-03-20"])
        assert y_true.name == "is_erupted"
        assert y_true.index.name == "id"
        # Windows 4-7 fall on 2025-03-20.
        assert y_true.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]

    def test_eruption_outside_range_is_noop(self, features_label_csv: str) -> None:
        y_true = build_y_true(features_label_csv, eruption_dates=["2024-01-01"])
        assert y_true.sum() == 0

    def test_missing_column_raises(self, tmp_path) -> None:
        path = tmp_path / "bad.csv"
        pd.DataFrame(
            {"datetime": ["2025-01-01"], "id": [0]},
        ).to_csv(path, index=False)
        with pytest.raises(KeyError):
            build_y_true(str(path), eruption_dates=["2025-01-01"])


# ---------------------------------------------------------------------------
# temporal_train_test_split
# ---------------------------------------------------------------------------


class TestTemporalTrainTestSplit:
    def test_docstring_example(self) -> None:
        train, test = temporal_train_test_split(
            ["2024-06-15", "2025-03-20", "2024-12-01", "2025-08-10"], test_size=0.25
        )
        assert train == ["2024-06-15", "2024-12-01", "2025-03-20"]
        assert test == ["2025-08-10"]

    def test_no_future_leakage(self) -> None:
        dates = [f"2025-{m:02d}-01" for m in (5, 1, 9, 3, 7, 11, 2)]
        train, test = temporal_train_test_split(dates, test_size=0.3)
        assert max(train) < min(test)
        assert sorted(train + test) == sorted(dates)

    @pytest.mark.parametrize("test_size", [0.0, 1.0])
    def test_never_returns_empty_split(self, test_size: float) -> None:
        train, test = temporal_train_test_split(
            ["2025-01-01", "2025-02-01", "2025-03-01"], test_size=test_size
        )
        assert train and test

    @pytest.mark.parametrize("test_size", [-0.1, 1.1])
    def test_out_of_range_test_size_raises(self, test_size: float) -> None:
        with pytest.raises(ValueError):
            temporal_train_test_split(["2025-01-01", "2025-02-01"], test_size)


# ---------------------------------------------------------------------------
# Threshold metrics
# ---------------------------------------------------------------------------


class TestComputeThresholdMetrics:
    def test_perfect_classifier(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        thresholds, metrics = compute_threshold_metrics(y_true, y_proba, resolution=11)

        assert len(thresholds) == 11
        assert set(metrics) == {
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "specificity",
            "mcc",
            "g_mean",
        }
        mid = 5  # Threshold 0.5 separates the classes perfectly.
        assert thresholds[mid] == pytest.approx(0.5)
        for name in ("precision", "recall", "f1", "balanced_accuracy", "mcc", "g_mean"):
            assert metrics[name][mid] == pytest.approx(1.0), name

    def test_zero_threshold_predicts_everything_positive(self) -> None:
        _, metrics = compute_threshold_metrics(
            np.array([0, 1]), np.array([0.3, 0.6]), resolution=3
        )
        assert metrics["recall"][0] == 1.0
        assert metrics["specificity"][0] == 0.0

    def test_aggregate_shapes_and_mean(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_probas = [np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.6, 0.2, 0.8, 0.9])]
        thresholds, mean, std, per_seed = compute_aggregate_threshold_metrics(
            y_true, y_probas, resolution=11
        )
        assert thresholds.shape == (11,)
        assert per_seed["f1"].shape == (2, 11)
        np.testing.assert_allclose(mean["f1"], per_seed["f1"].mean(axis=0))
        np.testing.assert_allclose(std["f1"], per_seed["f1"].std(axis=0))

    def test_aggregate_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_aggregate_threshold_metrics(np.array([0, 1]), [])

    def test_aggregate_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_aggregate_threshold_metrics(
                [np.array([0, 1])], [np.array([0.1, 0.9])] * 2
            )


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResample:
    def test_under_balances_classes(self, features_and_labels) -> None:
        X, y = features_and_labels
        X_res, y_res = resample(X, y, method="under")
        counts = y_res.value_counts()
        assert counts[0] == counts[1] == y.sum()
        assert len(X_res) == len(y_res)

    def test_over_balances_classes(self, features_and_labels) -> None:
        X, y = features_and_labels
        _, y_res = resample(X, y, method="over")
        counts = y_res.value_counts()
        assert counts[0] == counts[1] == (y == 0).sum()

    def test_none_returns_input(self, features_and_labels) -> None:
        X, y = features_and_labels
        X_res, y_res = resample(X, y, method=None)
        assert X_res is X
        assert y_res is y

    def test_float_strategy(self, features_and_labels) -> None:
        X, y = features_and_labels
        _, y_res = random_under_sampler(X, y, sampling_strategy=0.5)
        counts = y_res.value_counts()
        assert counts[1] / counts[0] == pytest.approx(0.5)

    def test_reproducible_with_random_state(self, features_and_labels) -> None:
        X, y = features_and_labels
        first, _ = resample(X, y, random_state=3)
        second, _ = resample(X, y, random_state=3)
        pd.testing.assert_frame_equal(first, second)

    def test_keeps_id_index(self, features_and_labels) -> None:
        X, y = features_and_labels
        X_res, _ = resample(X, y, method="under")
        assert set(X_res.index) <= set(X.index)


class TestLoadFeaturesResampled:
    def test_from_series(self, features_and_labels) -> None:
        X, y = features_and_labels
        subset = y.iloc[[0, 5, 50]]
        X_res, y_res = load_features_resampled(X, subset)
        assert X_res.index.tolist() == [0, 5, 50]
        assert y_res is subset

    def test_preserves_duplicate_ids(self, features_and_labels) -> None:
        X, y = features_and_labels
        X_res, _ = load_features_resampled(X, y.iloc[[50, 50, 1]])
        assert X_res.index.tolist() == [50, 50, 1]

    def test_column_projection(self, features_and_labels) -> None:
        X, y = features_and_labels
        X_res, _ = load_features_resampled(X, y.iloc[:3], columns=["feat_1"])
        assert X_res.columns.tolist() == ["feat_1"]

    def test_from_paths(self, features_and_labels, tmp_path) -> None:
        X, y = features_and_labels
        features_path = tmp_path / "features.parquet"
        resampled_path = tmp_path / "00042.csv"
        columns_path = tmp_path / "seed.csv"
        X.to_parquet(features_path)
        y.iloc[:4].to_frame().to_csv(resampled_path)
        pd.DataFrame({"score": [1, 2]}, index=["feat_2", "feat_0"]).to_csv(
            columns_path
        )

        X_res, y_res = load_features_resampled(
            str(features_path), str(resampled_path), columns=str(columns_path)
        )

        assert X_res.columns.tolist() == ["feat_2", "feat_0"]
        assert y_res.tolist() == y.iloc[:4].tolist()


# ---------------------------------------------------------------------------
# Classifier factories / summaries
# ---------------------------------------------------------------------------


def test_get_classifier_models() -> None:
    models = get_classifier_models(["rf", "lr"], cv_strategy="stratified", cv_splits=3)
    assert [m.name for m in models] == ["RandomForestClassifier", "LogisticRegression"]
    assert all(m.n_splits == 3 for m in models)


class TestBuildClassifierEnsembleSummary:
    @pytest.fixture()
    def labels(self) -> pd.DataFrame:
        index = pd.date_range("2025-03-19", periods=8, freq="12h", name="datetime")
        return pd.DataFrame({"id": range(8), "is_erupted": 0}, index=index)

    def test_requires_probabilities(self, make_seed_ensemble, labels) -> None:
        with pytest.raises(RuntimeError):
            build_classifier_ensemble_summary(
                make_seed_ensemble(n_seeds=2), labels, ["2025-03-20"]
            )

    def test_picks_extremes_per_window(self, make_seed_ensemble, labels) -> None:
        ensemble = make_seed_ensemble(n_seeds=2)
        probabilities = np.full((8, 2), 0.5)
        # 2025-03-20 covers ids 2 and 3.
        probabilities[2] = [0.9, 0.3]
        probabilities[3] = [0.1, 0.7]
        ensemble.probabilities = pd.DataFrame(
            probabilities, columns=["seed_00000", "seed_00001"]
        )

        summary = build_classifier_ensemble_summary(
            ensemble, labels, ["2025-03-20", "2030-01-01"]
        )

        assert summary.classifier_name == "LogisticRegression"
        assert len(summary.eruption_windows) == 1  # 2030 is off-grid.
        window = summary.eruption_windows[0]
        assert window.eruption_date == "2025-03-20"
        assert (window.highest.value, window.highest.random_state) == (0.9, 0)
        assert (window.lowest.value, window.lowest.random_state) == (0.1, 0)
        assert [s.random_state for s in window.seeds] == [0, 1]
        assert window.seeds[1].highest.value == 0.7
        assert summary.highest == window.highest
        assert summary.lowest == window.lowest
