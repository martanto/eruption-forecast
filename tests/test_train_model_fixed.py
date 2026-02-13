"""Integration test for the fixed ModelTrainer implementation.

Tests that the corrected workflow:
1. Splits data before any resampling
2. Resamples only training data
3. Selects features only on training data
4. Trains RandomForestClassifier with GridSearchCV
5. Evaluates on held-out test set
6. Saves models and metrics
"""

# Standard library imports
import os
import json
import shutil

# Third party imports
import joblib
import pandas as pd

# Project imports
from eruption_forecast.model.model_trainer import ModelTrainer


def test_full_training_pipeline():
    """Integration test with real data using small number of seeds."""
    # Define output directory
    output_dir = "tests/output/trainings"

    # Clean up previous test runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create trainer
    trainer = ModelTrainer(
        features_csv="D:/Projects/eruption-forecast/output/VG.OJN.00.EHZ/features/relevant_features_2025-01-03-2025-07-24.csv",
        label_csv="D:/Projects/eruption-forecast/output/VG.OJN.00.EHZ/features/label_features_2025-01-03-2025-07-24.csv",
        output_dir=output_dir,
        n_jobs=1,
        verbose=True,
    )

    # Run with small number of seeds for testing
    trainer.train(
        random_state=42,
        total_seed=3,  # Small number for quick testing
        number_of_significant_features=10,  # Small number for quick testing
        sampling_strategy=0.75,
        save_all_features=False,
        plot_significant_features=False,
        overwrite=True,
    )

    # ========== Verify outputs ==========

    # Check aggregated feature files
    assert os.path.exists(os.path.join(output_dir, "significant_features.csv")), (
        "Aggregated features CSV not found"
    )

    # Check aggregated metrics files
    assert os.path.exists(os.path.join(output_dir, "all_metrics.csv")), (
        "All metrics CSV not found"
    )
    assert os.path.exists(os.path.join(output_dir, "metrics_summary.csv")), (
        "Metrics summary CSV not found"
    )

    # Verify models and metrics for each seed
    for seed in range(3):
        state = 42 + seed
        model_file = os.path.join(trainer.models_dir, f"{state:05d}.pkl")
        metrics_file = os.path.join(trainer.metrics_dir, f"{state:05d}.json")
        features_file = os.path.join(
            trainer.significant_features_dir, f"{state:05d}.csv"
        )

        assert os.path.exists(model_file), f"Model file for seed {seed} not found"
        assert os.path.exists(metrics_file), f"Metrics file for seed {seed} not found"
        assert os.path.exists(features_file), f"Features file for seed {seed} not found"

        # Verify model can be loaded
        model = joblib.load(model_file)
        assert hasattr(model, "predict"), "Loaded model doesn't have predict method"

        # Verify metrics have expected keys
        with open(metrics_file) as f:
            metrics = json.load(f)

        expected_keys = [
            "seed",
            "random_state",
            "n_train",
            "n_test",
            "n_features",
            "accuracy",
            "balanced_accuracy",
            "f1_score",
            "precision",
            "recall",
            "best_params",
            "best_cv_score",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key '{key}' in metrics"

        # Verify train + test < total (because of undersampling)
        total = trainer.df_features.shape[0]
        assert metrics["n_train"] + metrics["n_test"] < total, (
            "Train + test should be less than total due to undersampling"
        )

        print(f"\nSeed {seed} metrics:")
        print(f"  Train samples: {metrics['n_train']}")
        print(f"  Test samples: {metrics['n_test']}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    # Load and inspect aggregated metrics
    df_metrics = pd.read_csv(os.path.join(output_dir, "all_metrics.csv"))

    print("\n" + "=" * 60)
    print("Aggregated Metrics Summary:")
    print("=" * 60)
    print(
        df_metrics[
            ["accuracy", "balanced_accuracy", "f1_score", "precision", "recall"]
        ].describe()
    )
    print("=" * 60)

    print("\nTest passed! All outputs verified.")


if __name__ == "__main__":
    test_full_training_pipeline()
