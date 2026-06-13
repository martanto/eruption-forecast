Model stack: orchestrator + per-stage models + ABCs + cross-classifier comparator.

Files:
- forecast_model.py — `ForecastModel`: pipeline orchestrator (calculate → train → predict → evaluate); captures stage kwargs into `ForecastConfig` for `save_config()` / `from_config()` / `run()`.
- base_model.py — `BaseModel(ABC)`: tremor loading, `output_dir`/`n_jobs` resolution, joblib `save()`/`load()`.
- cache_model.py — `CacheModel`: content-addressable cache (`{output_dir}/cache/{ClassName}/{hash}.pkl` + `.params.json`).
- training_model.py — `TrainingModel`: build_label → extract_features → multi-seed GridSearchCV fit; auto-merges seeds into a `ClassifierEnsemble`.
- prediction_model.py — `PredictionModel`: forecast over unlabelled window grid. Cache identity ties to `training_hash`.
- evaluation_model.py — `EvaluationModel`: pre-trained ensemble vs ground truth; "training reuse" or "prediction reuse" mode; never re-fits.
- explanation_model.py — `ExplanationModel`: SHAP TreeExplainer over a fitted `ClassifierEnsemble`.
- classifier_model.py — `ClassifierModel`: sklearn-compatible estimator wrapper with default hyper-grid + CV strategy.
- classifier_comparator.py — `ClassifierComparator`: cross-classifier ranking + plots from a `MetricsEnsemble`.
