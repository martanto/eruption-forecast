For each random seed (default 25, up to 500), in a `joblib.Parallel` outer worker:

1. `FeatureSelector` selects top-N features (`tsfresh`, `random_forest`, or `combined`).
2. Optional resample (`utils.ml.random_under_sampler`; `auto` triggers when minority share < `minority_threshold`).
3. `grid_search_cv` runs `GridSearchCV` for the `ClassifierModel`, using `n_grids` inner workers and the chosen CV strategy.

Per-seed estimator + selected feature list are saved to `classifiers/{clf-slug}/{cv-slug}/models/{seed:05d}.pkl` and registered in `trained_model_{suffix}.csv`.
