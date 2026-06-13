The package targets volcano observatory researchers who want to forecast eruptions a few days in advance from continuous broadband seismic tremor.

Key design choices that shape this diagram:
- Single orchestrator (ForecastModel) with fluent method chaining captures stage kwargs into ForecastConfig sub-configs so a full run can be replayed from a YAML config.
- Multi-seed training (typically 25–500 random seeds per classifier) is the unit of model robustness — every model output is per-seed, then aggregated into a consensus.
- Content-addressable caching on TrainingModel/PredictionModel means expensive feature extraction and fitting can be re-used across re-runs.
- ObsPy + tsfresh + scikit-learn/XGBoost/LightGBM define the science stack. Everything else (cache, ensemble, comparator) is hand-written glue around it.
