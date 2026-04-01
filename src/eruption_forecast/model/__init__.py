"""Model package for volcanic eruption forecasting.

Exposes the primary model classes used to train, evaluate, and
run inference with classifier ensembles built from seismic tremor features.

Public API:
    - ``SeedEnsemble``: Bundles all seed models for a single classifier into a
      serialisable sklearn-compatible estimator.
    - ``ClassifierEnsemble``: Wraps multiple ``SeedEnsemble`` objects to produce
      cross-classifier consensus probabilities and uncertainty estimates.
    - ``ModelEvaluator``: Evaluates a single fitted model — computes metrics,
      plots ROC/calibration/learning curves, and optionally saves outputs.
    - ``MultiModelEvaluator``: Aggregates per-seed evaluation results from JSON
      metrics files or a registry CSV and generates ensemble-level statistics
      and plots (SHAP, ROC, calibration, learning curves, confusion matrix).
    - ``ClassifierComparator``: Accepts a mapping of classifier names to registry
      CSV paths and produces side-by-side comparison tables and plots.

All classifiers use the standard scikit-learn implementations.
GridSearchCV is always run under the ``loky`` parallel backend for nested-parallelism safety.
"""

from eruption_forecast.model.seed_ensemble import SeedEnsemble
from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.model.classifier_ensemble import ClassifierEnsemble
from eruption_forecast.model.classifier_comparator import ClassifierComparator
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator


__all__ = [
    "ClassifierComparator",
    "ClassifierEnsemble",
    "ModelEvaluator",
    "MultiModelEvaluator",
    "SeedEnsemble",
]
