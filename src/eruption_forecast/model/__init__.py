"""Model module for volcanic eruption forecasting.

This module provides components for training, evaluating, and deploying
machine learning models for eruption prediction.

Classes:
    ClassifierModel: Manages ML classifiers with hyperparameter grids.
    ModelTrainer: Trains models with multi-seed cross-validation.
    ModelEvaluator: Evaluates a single fitted model with metrics and plots.
    MultiModelEvaluator: Aggregates metrics and plots across multiple seeds.
    ClassifierComparator: Compares multiple classifiers side-by-side.
    ModelPredictor: Runs inference in evaluation or forecast mode.
    ForecastModel: Orchestrates the complete forecasting pipeline.
"""

import logging


logging.getLogger("sklearnex").setLevel(logging.WARNING)

try:
    from sklearnex import patch_sklearn  # type: ignore[import-untyped]
    patch_sklearn(verbose=False)
except ImportError:
    pass

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
