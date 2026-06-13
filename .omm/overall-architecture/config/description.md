Configuration layer: dataclasses that mirror every stage method's kwargs, enabling YAML/JSON round-trip via `ForecastModel.save_config()` / `from_config()` / `run()`.

Files:
- forecast_config.py — `ForecastConfig` plus `BaseForecastConfig` and per-stage sub-configs: `ForecastCalculateConfig`, `ForecastTrainConfig`, `ForecastPredictConfig`, `ForecastEvaluateConfig`, `ForecastExplainConfig`.
- training_config.py — `TrainingConfig`: snapshot of `TrainingModel.__init__` signature. Kept in sync with the constructor (rule 16 in CLAUDE.md).
- base_config.py — common YAML/JSON serialisation + path-resolution helpers.
- constants.py — calculate-stage defaults (`DEFAULT_FREQUENCY_BANDS`, `BANDPASS_FILTER_CORNERS`, `DEFAULT_SAMPLING_FREQUENCY`, `DEFAULT_MINIMUM_COMPLETION_RATIO`, `CALCULATE_METHODS`).
