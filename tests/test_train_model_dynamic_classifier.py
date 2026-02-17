"""Test dynamic classifier functionality in ModelTrainer.

Tests that ModelTrainer can use different classifiers (RF, GB, LR, etc.)
through the ClassifierModel integration.
"""

# Standard library imports
import os
import json
import shutil

# Third party imports
# Project imports
from eruption_forecast.model.model_trainer import ModelTrainer


def test_random_forest_classifier():
    """Test training with Random Forest classifier (default)."""
    output_dir = "tests/output/trainings_rf"

    # Clean up previous test runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create trainer with Random Forest
    trainer = ModelTrainer(
        features_csv="D:/Projects/eruption-forecast/output/VG.OJN.00.EHZ/features/relevant_features_2025-01-03-2025-07-24.csv",
        label_csv="D:/Projects/eruption-forecast/output/VG.OJN.00.EHZ/features/label_features_2025-01-03-2025-07-24.csv",
        output_dir=output_dir,
        classifier="rf",
        cv_strategy="shuffle",
        cv_splits=3,
        n_jobs=1,
        verbose=True,
    )

    # Run with 1 seed for quick testing
    trainer.train_and_evaluate(
        random_state=42,
        total_seed=1,
        number_of_significant_features=5,
        sampling_strategy=0.75,
        overwrite=True,
    )

    # Verify outputs
    metrics_file = os.path.join(trainer.metrics_dir, "00042.json")
    assert os.path.exists(metrics_file), "Metrics file not found"

    with open(metrics_file) as f:
        metrics = json.load(f)

    assert metrics["classifier"] == "rf", "Classifier type should be 'rf'"
    assert metrics["cv_strategy"] == "shuffle", "CV strategy should be 'shuffle'"
    assert metrics["cv_splits"] == 3, "CV splits should be 3"
    assert "balanced_accuracy" in metrics, "Missing balanced_accuracy metric"

    print("\nPASSED: Random Forest test")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Classifier: {metrics['classifier']}")
    print(f"  CV Strategy: {metrics['cv_strategy']}")


def test_gradient_boosting_classifier():
    """Test training with Gradient Boosting classifier."""
    output_dir = "tests/output/trainings_gb"

    # Clean up previous test runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create trainer with Gradient Boosting
    trainer = ModelTrainer(
        features_csv="D:/Projects/eruption-forecast/output/VG.OJN.00.EHZ/features/relevant_features_2025-01-03-2025-07-24.csv",
        label_csv="D:/Projects/eruption-forecast/output/VG.OJN.00.EHZ/features/label_features_2025-01-03-2025-07-24.csv",
        output_dir=output_dir,
        classifier="gb",
        cv_strategy="stratified",
        cv_splits=3,
        n_jobs=1,
        verbose=True,
    )

    # Run with 1 seed for quick testing
    trainer.train_and_evaluate(
        random_state=42,
        total_seed=1,
        number_of_significant_features=5,
        sampling_strategy=0.75,
        overwrite=True,
    )

    # Verify outputs
    metrics_file = os.path.join(trainer.metrics_dir, "00042.json")
    assert os.path.exists(metrics_file), "Metrics file not found"

    with open(metrics_file) as f:
        metrics = json.load(f)

    assert metrics["classifier"] == "gb", "Classifier type should be 'gb'"
    assert metrics["cv_strategy"] == "stratified", "CV strategy should be 'stratified'"
    assert "balanced_accuracy" in metrics, "Missing balanced_accuracy metric"

    print("\nPASSED: Gradient Boosting test")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Classifier: {metrics['classifier']}")
    print(f"  CV Strategy: {metrics['cv_strategy']}")


def test_logistic_regression_classifier():
    """Test training with Logistic Regression classifier."""
    output_dir = "tests/output/trainings_lr"

    # Clean up previous test runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create trainer with Logistic Regression
    trainer = ModelTrainer(
        features_csv="D:/Projects/eruption-forecast/output/VG.OJN.00.EHZ/features/relevant_features_2025-01-03-2025-07-24.csv",
        label_csv="D:/Projects/eruption-forecast/output/VG.OJN.00.EHZ/features/label_features_2025-01-03-2025-07-24.csv",
        output_dir=output_dir,
        classifier="lr",
        cv_strategy="timeseries",
        cv_splits=3,
        n_jobs=1,
        verbose=True,
    )

    # Run with 1 seed for quick testing
    trainer.train_and_evaluate(
        random_state=42,
        total_seed=1,
        number_of_significant_features=5,
        sampling_strategy=0.75,
        overwrite=True,
    )

    # Verify outputs
    metrics_file = os.path.join(trainer.metrics_dir, "00042.json")
    assert os.path.exists(metrics_file), "Metrics file not found"

    with open(metrics_file) as f:
        metrics = json.load(f)

    assert metrics["classifier"] == "lr", "Classifier type should be 'lr'"
    assert metrics["cv_strategy"] == "timeseries", "CV strategy should be 'timeseries'"
    assert "balanced_accuracy" in metrics, "Missing balanced_accuracy metric"

    print("\nPASSED: Logistic Regression test")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Classifier: {metrics['classifier']}")
    print(f"  CV Strategy: {metrics['cv_strategy']}")


def run_all_tests():
    """Run all classifier tests."""
    print("=" * 60)
    print("Testing Dynamic Classifier Functionality")
    print("=" * 60)

    test_random_forest_classifier()
    test_gradient_boosting_classifier()
    test_logistic_regression_classifier()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
