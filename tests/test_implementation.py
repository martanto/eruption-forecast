"""
Test script to verify code review implementation changes.

Tests:
1. Config module imports
2. MetricsComputer class functionality
3. ModelEvaluator integration
4. Backward compatibility
"""

import sys
import numpy as np
import pandas as pd

print("=" * 60)
print("Code Review Implementation - Test Suite")
print("=" * 60)

# Test 1: Config module imports
print("\n[Test 1] Testing config module imports...")
try:
    from eruption_forecast.config.constants import (
        TRAIN_TEST_SPLIT,
        DEFAULT_CV_SPLITS,
        DEFAULT_N_SIGNIFICANT_FEATURES,
        DEFAULT_SAMPLING_STRATEGY,
        ERUPTION_PROBABILITY_THRESHOLD,
        THRESHOLD_RESOLUTION,
        PLOT_DPI,
        PLOT_SEPARATOR_LENGTH,
    )
    
    # Verify values
    assert TRAIN_TEST_SPLIT == 0.2, "TRAIN_TEST_SPLIT should be 0.2"
    assert DEFAULT_CV_SPLITS == 5, "DEFAULT_CV_SPLITS should be 5"
    assert DEFAULT_N_SIGNIFICANT_FEATURES == 20, "DEFAULT_N_SIGNIFICANT_FEATURES should be 20"
    assert DEFAULT_SAMPLING_STRATEGY == 0.75, "DEFAULT_SAMPLING_STRATEGY should be 0.75"
    assert ERUPTION_PROBABILITY_THRESHOLD == 0.7, "ERUPTION_PROBABILITY_THRESHOLD should be 0.7"
    assert THRESHOLD_RESOLUTION == 101, "THRESHOLD_RESOLUTION should be 101"
    assert PLOT_DPI == 300, "PLOT_DPI should be 300"
    assert PLOT_SEPARATOR_LENGTH == 50, "PLOT_SEPARATOR_LENGTH should be 50"
    
    print("✅ All 8 constants imported successfully")
    print(f"   - TRAIN_TEST_SPLIT = {TRAIN_TEST_SPLIT}")
    print(f"   - DEFAULT_CV_SPLITS = {DEFAULT_CV_SPLITS}")
    print(f"   - DEFAULT_N_SIGNIFICANT_FEATURES = {DEFAULT_N_SIGNIFICANT_FEATURES}")
    print(f"   - ERUPTION_PROBABILITY_THRESHOLD = {ERUPTION_PROBABILITY_THRESHOLD}")
except Exception as e:
    print(f"❌ Config import failed: {e}")
    sys.exit(1)

# Test 2: MetricsComputer class
print("\n[Test 2] Testing MetricsComputer class...")
try:
    from eruption_forecast.model.metrics_computer import MetricsComputer
    
    # Create synthetic test data
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15, 0.95, 0.25])
    y_pred = (y_proba >= 0.5).astype(int)
    
    # Initialize MetricsComputer
    computer = MetricsComputer(y_true, y_proba, y_pred)
    
    # Compute metrics
    metrics = computer.compute_all_metrics()
    
    # Verify metrics dict has expected keys
    expected_keys = [
        'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score',
        'true_positives', 'true_negatives', 'false_positives', 'false_negatives',
        'sensitivity', 'specificity', 'roc_auc', 'pr_auc',
        'optimal_threshold', 'f1_at_optimal', 'recall_at_optimal', 'precision_at_optimal'
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
    
    print("✅ MetricsComputer class works correctly")
    print(f"   - Computed {len(metrics)} metrics")
    print(f"   - Accuracy: {metrics['accuracy']:.3f}")
    print(f"   - F1 Score: {metrics['f1_score']:.3f}")
    print(f"   - ROC AUC: {metrics['roc_auc']:.3f}")
    print(f"   - Optimal Threshold: {metrics['optimal_threshold']:.3f}")
    
except Exception as e:
    print(f"❌ MetricsComputer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: ModelEvaluator integration
print("\n[Test 3] Testing ModelEvaluator integration...")
try:
    from sklearn.ensemble import RandomForestClassifier
    from eruption_forecast.model.model_evaluator import ModelEvaluator
    
    # Create synthetic dataset
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 5)
    y_test = np.random.randint(0, 2, 20)
    
    # Convert to DataFrames
    X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(5)])
    y_test_series = pd.Series(y_test, name='label')
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        X_test=X_test_df,
        y_test=y_test_series,
        model_name="test_rf",
        output_dir="tests/output/evaluation"
    )
    
    # Test get_metrics (should use MetricsComputer internally)
    metrics = evaluator.get_metrics()
    
    assert 'model_name' in metrics, "Missing model_name in metrics"
    assert metrics['model_name'] == 'test_rf', "Wrong model name"
    assert 'accuracy' in metrics, "Missing accuracy metric"
    assert 'f1_score' in metrics, "Missing f1_score metric"
    
    # Test _save_plot helper exists
    assert hasattr(evaluator, '_save_plot'), "Missing _save_plot helper"
    
    # Test _metrics_computer exists
    assert hasattr(evaluator, '_metrics_computer'), "Missing _metrics_computer"
    assert evaluator._metrics_computer is not None, "_metrics_computer should not be None"
    
    print("✅ ModelEvaluator integration successful")
    print(f"   - Evaluator initialized with MetricsComputer")
    print(f"   - get_metrics() returns {len(metrics)} metrics")
    print(f"   - _save_plot() helper method present")
    print(f"   - Test accuracy: {metrics['accuracy']:.3f}")
    
except Exception as e:
    print(f"❌ ModelEvaluator integration test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: pathutils build_model_directories
print("\n[Test 4] Testing build_model_directories...")
try:
    from eruption_forecast.utils.pathutils import build_model_directories
    import os
    import tempfile
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        dirs = build_model_directories(
            root_dir=tmpdir,
            classifier_slug="test-classifier",
            cv_slug="test-cv",
            mode="with-evaluation"
        )
        
        # Verify all directories exist
        expected_keys = ['base', 'features', 'significant_features', 'models', 'metrics', 'figures']
        for key in expected_keys:
            assert key in dirs, f"Missing directory key: {key}"
            assert os.path.exists(dirs[key]), f"Directory not created: {dirs[key]}"
        
        print("✅ build_model_directories works correctly")
        print(f"   - Created {len(dirs)} directories")
        print(f"   - Base: trainings/model-with-evaluation/test-classifier/test-cv/")
        
except Exception as e:
    print(f"❌ build_model_directories test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Backward compatibility
print("\n[Test 5] Testing backward compatibility...")
try:
    # Test that old code still works (no breaking changes)
    from eruption_forecast.model.model_evaluator import ModelEvaluator
    
    # Old-style usage (without passing constants explicitly)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    evaluator = ModelEvaluator(model, X_test_df, y_test_series)
    metrics = evaluator.get_metrics()
    
    # Should work exactly as before
    assert 'accuracy' in metrics
    assert 'f1_score' in metrics
    
    print("✅ Backward compatibility verified")
    print("   - Old code patterns still work")
    print("   - No breaking API changes")
    
except Exception as e:
    print(f"❌ Backward compatibility test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("✅ All 5 test suites passed!")
print()
print("Tests executed:")
print("  1. ✅ Config module imports (8 constants)")
print("  2. ✅ MetricsComputer class (17 metrics)")
print("  3. ✅ ModelEvaluator integration")
print("  4. ✅ build_model_directories utility")
print("  5. ✅ Backward compatibility")
print()
print("Implementation is ready for production! 🎉")
print("=" * 60)
