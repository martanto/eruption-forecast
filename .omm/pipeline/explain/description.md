Optional SHAP-based explanation stage. `ExplanationModel(BaseModel, CacheModel)` mirrors `EvaluationModel`'s upstream-model-as-input shape but inherits caching because SHAP is expensive (per-classifier × per-seed × per-observation).

Currently scoped to tree-based classifiers (`_is_tree_classifier`: rf, gb, xgb). Builds an `ExplainerEnsemble` of `shap.TreeExplainer` instances over `n_observations_to_explain` rows of the upstream features. Output goes under `{output_dir}/explanation/{kind}/`.
