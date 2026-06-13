Ensemble layer: sklearn-compatible bundles that wrap many seed-trained estimators into objects that can be saved, reloaded, and used as drop-in classifiers downstream.

Files:
- base_ensemble.py — `BaseEnsemble`: joblib `save()`/`load()` mixin shared by every concrete ensemble.
- seed_ensemble.py — `SeedEnsemble(BaseEstimator, ClassifierMixin)`: all seeds for one classifier, with per-seed `feature_names`.
- classifier_ensemble.py — `ClassifierEnsemble`: maps classifier name → `SeedEnsemble`; owns cross-classifier consensus aggregation.
- metrics_ensemble.py — `MetricsEnsemble`: per-seed metric engine; materialises `(n_samples, n_seeds)` `y_proba`/`y_pred` matrices and runs the aggregate + per-seed plot dispatchers.
- explainer_ensemble.py — `ExplainerEnsemble`: parallel of `SeedEnsemble` for SHAP TreeExplainer instances.
