"""Tests for MetricsComputer."""

import numpy as np
import pytest

from eruption_forecast.model.metrics_computer import MetricsComputer


@pytest.fixture
def sample_data():
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15, 0.95, 0.25])
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_proba, y_pred


@pytest.fixture
def computer(sample_data):
    y_true, y_proba, y_pred = sample_data
    return MetricsComputer(y_true, y_proba, y_pred)


class TestMetricsComputerInit:
    def test_stores_arrays(self, sample_data):
        y_true, y_proba, y_pred = sample_data
        mc = MetricsComputer(y_true, y_proba, y_pred)
        assert mc.y_true is not None
        assert mc.y_proba is not None
        assert mc.y_pred is not None

    def test_accepts_lists(self):
        mc = MetricsComputer([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1])
        metrics = mc.compute_all_metrics()
        assert "accuracy" in metrics


class TestComputeAllMetrics:
    EXPECTED_KEYS = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "sensitivity",
        "specificity",
        "roc_auc",
        "pr_auc",
        "average_precision",
        "mcc",
        "optimal_threshold",
        "f1_at_optimal",
        "recall_at_optimal",
        "precision_at_optimal",
    ]

    def test_all_expected_keys_present(self, computer):
        metrics = computer.compute_all_metrics()
        for key in self.EXPECTED_KEYS:
            assert key in metrics, f"Missing key: {key}"

    def test_accuracy_in_range(self, computer):
        metrics = computer.compute_all_metrics()
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_roc_auc_in_range(self, computer):
        metrics = computer.compute_all_metrics()
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_perfect_classifier_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.1, 0.9, 0.9])
        y_pred = np.array([0, 0, 1, 1])
        mc = MetricsComputer(y_true, y_proba, y_pred)
        metrics = mc.compute_all_metrics()
        assert metrics["accuracy"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_optimal_threshold_in_0_1(self, computer):
        metrics = computer.compute_all_metrics()
        assert 0.0 <= metrics["optimal_threshold"] <= 1.0

    def test_sensitivity_equals_recall(self, computer):
        metrics = computer.compute_all_metrics()
        assert abs(metrics["sensitivity"] - metrics["recall"]) < 1e-9

    def test_average_precision_in_range(self, computer):
        metrics = computer.compute_all_metrics()
        assert 0.0 <= metrics["average_precision"] <= 1.0

    def test_mcc_in_range(self, computer):
        metrics = computer.compute_all_metrics()
        assert -1.0 <= metrics["mcc"] <= 1.0
