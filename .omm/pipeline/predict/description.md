`PredictionModel(BaseModel, CacheModel)` runs inference over an unlabelled forecast-window grid.

1. `build_label()` constructs the forecast window grid (no eruption marking — only the grid).
2. `extract_features()` runs tsfresh in prediction mode (relevance filtering disabled).
3. `forecast()` calls `ClassifierEnsemble.predict_proba(X)` per classifier; persists per-seed probabilities (if `save_seed_result=True`) plus aggregated consensus columns to `prediction/results/{clf-slug}/`; renders the forecast plot via `plots.forecast_plots.plot_forecast`.

`self.results` columns: `{clf}_eruption_probability` / `_uncertainty` / `_confidence` / `_prediction` per classifier, plus `consensus_*` cross-classifier columns. Cache identity includes `training_hash` so re-training invalidates the prediction cache.
