`TrainingModel(BaseModel, CacheModel)` orchestrates the train stage:

1. `build_label()` — `LabelBuilder` or `DynamicLabelBuilder` constructs the labeled window grid.
2. `extract_features()` — `TremorMatrixBuilder` aligns tremor to windows; `FeaturesBuilder` runs tsfresh per tremor column; the result is concatenated.
3. `fit()` — per seed: `FeatureSelector` picks top-N features, `random_under_sampler` (or `auto`) balances classes if needed, `grid_search_cv` fits a `ClassifierModel`. Runs in a `joblib.Parallel(backend="loky")` over `n_jobs` outer workers, with `n_grids` inner workers inside `GridSearchCV`/`FeatureSelector`.
4. Auto-merge — `merge_seed_models` writes one `SeedEnsemble_*.pkl` per classifier; `merge_all_classifiers` bundles them into a `ClassifierEnsemble` (path stored on `self.classifier_ensemble_path`).

Cache identity covers tremor frame + label/feature/fit params + classifier list.
