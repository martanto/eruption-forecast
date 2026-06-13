End-to-end ML pipeline orchestrated by `ForecastModel`. Five stages in canonical order: calculate → train → predict → evaluate, plus an optional `explain` SHAP stage that can hang off either training or prediction.

Each stage method on `ForecastModel` returns `Self`, captures its kwargs into the matching `ForecastConfig` sub-config, and writes artefacts under a single station directory keyed by `{network}.{station}.{location}.{channel}`. `train` and `predict` opt into content-addressable caching (`use_cache=True`) so re-runs that share the same inputs skip the expensive work.

Stage handoffs:
- `calculate → train`: tremor CSV (DateTime-indexed `rsam_f0..f4`, `dsar_f0-f1..f3-f4`, `entropy`).
- `train → predict`: a `ClassifierEnsemble` made of per-classifier `SeedEnsemble`s.
- `train → evaluate`: in-sample features + truth labels (training reuse).
- `predict → evaluate`: forecast-grid features + ensemble; `eruption_dates` required to rebuild ground truth on the forecast grid (prediction reuse).
- `train|predict → explain`: same features + ensemble, fed into `shap.TreeExplainer` per (classifier, seed).
