The pipeline is intentionally linear and stateful per stage: each stage class holds the artefacts of the previous stage as attributes (`fm.TrainingModel.classifier_ensemble`, `fm.PredictionModel.results`, ...), so downstream stages don't re-load from disk during a single `ForecastModel` run.

This is what makes "training reuse" and "prediction reuse" possible in `EvaluationModel` — the same in-memory features + ensemble are reused; no tsfresh re-run, no model re-fit.

`day_to_forecast` (default 2) propagates through every stage as the forecast lead time and the window size, so changing it in the constructor automatically lines up label horizons, feature windows, and forecast grids.
