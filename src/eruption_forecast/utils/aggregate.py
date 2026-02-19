"""Aggregate metric computation utilities for multi-seed model evaluation.

This module provides functions to load per-seed model artifacts from a
registry CSV and compute ensemble-level evaluation statistics across all
seeds — without rendering any matplotlib figures.
"""

import os
from typing import Any

import numpy as np
import joblib
import pandas as pd

from eruption_forecast.logger import logger
from eruption_forecast.plots.evaluation_plots import (
    compute_aggregate_pr_data,
    compute_aggregate_roc_data,
    compute_aggregate_threshold_data,
    compute_aggregate_calibration_data,
    compute_aggregate_confusion_matrix_data,
    compute_aggregate_feature_importance_data,
    compute_aggregate_prediction_distribution_data,
)


def load_seed_data(
    row: pd.Series,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Load model, filtered test features, true labels, and predicted probabilities for one seed.

    Reads the trained model, X_test, y_test, and significant-features list
    from paths stored in a registry row, filters the test set to the selected
    features, and runs ``predict_proba`` to produce probability estimates.

    Args:
        row (pd.Series): A single row from the model registry DataFrame with
            at least the keys ``trained_model_filepath``,
            ``X_test_filepath``, ``y_test_filepath``, and
            ``significant_features_csv``.

    Returns:
        tuple[Any, np.ndarray, np.ndarray, np.ndarray]: A 4-tuple of
            ``(model, X_test_filtered, y_true, y_proba)`` where
            ``X_test_filtered`` contains only the significant features
            selected for this seed.
    """
    model = joblib.load(row["trained_model_filepath"])
    X_test = pd.read_csv(row["X_test_filepath"], index_col=0)
    y_true = pd.read_csv(row["y_test_filepath"], index_col=0).iloc[:, 0].to_numpy()

    sig_features = pd.read_csv(
        row["significant_features_csv"], index_col=0
    ).index.tolist()
    X_test_filtered = X_test[sig_features]

    proba: np.ndarray = model.predict_proba(X_test_filtered)[:, 1]
    return model, X_test_filtered.to_numpy(), y_true, proba


def compute_aggregate_metrics(
    registry_csv: str,
    output_dir: str | None = None,
    save: bool = True,
    n_bins: int = 10,
) -> dict[str, pd.DataFrame | None]:
    """Compute aggregate evaluation metrics across all seeds without generating plots.

    Loads every seed's model and held-out test data from the registry,
    computes the same statistics as ``plot_all_aggregate()`` but skips
    matplotlib rendering entirely. Use this when you only need the underlying
    CSV data — for further analysis, custom plotting, or headless environments.

    Each returned DataFrame has exactly the same column structure as the CSV
    file saved alongside the corresponding aggregate plot.

    Args:
        registry_csv (str): Path to the model registry CSV produced by
            ``ModelTrainer.train_and_evaluate()``.
        output_dir (str | None, optional): Directory for saved CSVs. Defaults
            to ``<registry_dir>/plots/``.
        save (bool, optional): Whether to save each DataFrame as a CSV file.
            Defaults to True.
        n_bins (int, optional): Number of calibration bins. Defaults to 10.

    Returns:
        dict[str, pd.DataFrame | None]: Mapping of metric name to DataFrame.
            Keys and DataFrame column structures:

            - ``"roc_curve"``: ``fpr``, ``mean_tpr``, ``std_tpr``
            - ``"pr_curve"``: ``recall``, ``mean_precision``, ``std_precision``
            - ``"calibration"``: ``prob_bin``, ``mean_frac_positives``,
              ``std_frac_positives``
            - ``"prediction_distribution"``: ``y_proba``, ``y_true``
              (pooled across all seeds)
            - ``"confusion_matrix"``: 2×2 DataFrame with index and columns
              ``["not_erupted", "erupted"]``
            - ``"threshold_analysis"``: ``threshold``, mean/std for
              ``precision``, ``recall``, ``f1``, ``balanced_accuracy``
            - ``"feature_importance"``: ``feature``, ``mean_importance``,
              ``std_importance`` sorted by descending mean importance
              (``None`` when model lacks ``feature_importances_``)

    Examples:
        >>> from eruption_forecast.utils.aggregate import compute_aggregate_metrics
        >>> results = compute_aggregate_metrics(
        ...     registry_csv="output/.../trained_model_*.csv",
        ...     output_dir="output/eval/aggregate",
        ...     save=True,
        ... )
        >>> roc_df = results["roc_curve"]
        >>> # Find the threshold that maximises mean F1
        >>> df = results["threshold_analysis"]
        >>> best = df.loc[df["mean_f1"].idxmax(), "threshold"]
        >>> print(f"Optimal threshold: {best:.3f}")
    """
    registry = pd.read_csv(registry_csv, index_col=0)
    y_trues: list[np.ndarray] = []
    y_probas: list[np.ndarray] = []
    y_preds: list[np.ndarray] = []
    models: list[Any] = []
    feature_names: list[str] = []

    for _, row in registry.iterrows():
        model, X_test, y_true, y_proba = load_seed_data(row)
        y_trues.append(y_true)
        y_probas.append(y_proba)
        y_preds.append(model.predict(X_test))
        models.append(model)
        if not feature_names:
            feature_names = pd.read_csv(
                row["significant_features_csv"], index_col=0
            ).index.tolist()

    results: dict[str, pd.DataFrame | None] = {
        "roc_curve": compute_aggregate_roc_data(y_trues, y_probas),
        "pr_curve": compute_aggregate_pr_data(y_trues, y_probas),
        "calibration": compute_aggregate_calibration_data(
            y_trues, y_probas, n_bins=n_bins
        ),
        "prediction_distribution": compute_aggregate_prediction_distribution_data(
            y_trues, y_probas
        ),
        "confusion_matrix": compute_aggregate_confusion_matrix_data(y_trues, y_preds),
        "threshold_analysis": compute_aggregate_threshold_data(y_trues, y_probas),
        "feature_importance": compute_aggregate_feature_importance_data(
            models, feature_names
        ),
    }

    if save:
        out = output_dir or os.path.join(os.path.dirname(registry_csv), "plots")
        os.makedirs(out, exist_ok=True)
        filenames: dict[str, str] = {
            "roc_curve": "aggregate_roc_curve.csv",
            "pr_curve": "aggregate_pr_curve.csv",
            "calibration": "aggregate_calibration.csv",
            "prediction_distribution": "aggregate_prediction_distribution.csv",
            "confusion_matrix": "aggregate_confusion_matrix.csv",
            "threshold_analysis": "aggregate_threshold_analysis.csv",
            "feature_importance": "aggregate_feature_importance.csv",
        }
        for key, df in results.items():
            if df is not None:
                path = os.path.join(out, filenames[key])
                df.to_csv(path)
                logger.info(f"Saved aggregate metrics: {path}")

    return results
