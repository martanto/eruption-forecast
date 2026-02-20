"""Model module for volcanic eruption forecasting.

This module provides components for training, evaluating, and deploying
machine learning models for eruption prediction.

Classes:
    ClassifierModel: Manages ML classifiers with hyperparameter grids.
    ModelTrainer: Trains models with multi-seed cross-validation.
    ModelEvaluator: Evaluates a single fitted model with metrics and plots.
    MultiModelEvaluator: Aggregates metrics and plots across multiple seeds.
    ModelPredictor: Runs inference in evaluation or forecast mode.
    ForecastModel: Orchestrates the complete forecasting pipeline.
"""

from eruption_forecast.model.model_evaluator import ModelEvaluator
from eruption_forecast.model.multi_model_evaluator import MultiModelEvaluator


__all__ = [
    "ModelEvaluator",
    "MultiModelEvaluator",
]
