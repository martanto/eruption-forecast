"""Ensemble package for volcanic eruption forecasting.

Exposes the serialisable ensemble classes that aggregate predictions across
random seeds and classifier types.

Public API:
    - ``BaseEnsemble``: Joblib-based save/load mixin shared by
      :class:`SeedEnsemble` and :class:`ClassifierEnsemble`.
    - ``SeedEnsemble``: Bundles all seed models for a single classifier into a
      serialisable sklearn-compatible estimator.
    - ``ClassifierEnsemble``: Wraps multiple ``SeedEnsemble`` objects to produce
      cross-classifier consensus probabilities and uncertainty estimates.

:class:`~eruption_forecast.ensemble.metrics_ensemble.MetricsEnsemble` lives in
this package too but is **not** re-exported here — importing it would pull
``MetricsComputer`` from ``model/`` which in turn imports ``utils.ml``, and
``utils.ml`` imports ``SeedEnsemble`` from this package, completing a cycle.
Import it directly from ``eruption_forecast.ensemble.metrics_ensemble``.
"""

from eruption_forecast.ensemble.seed_ensemble import SeedEnsemble
from eruption_forecast.ensemble.classifier_ensemble import ClassifierEnsemble


__all__ = [
    "ClassifierEnsemble",
    "SeedEnsemble",
]
