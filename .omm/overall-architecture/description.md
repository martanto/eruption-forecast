High-level architecture of the eruption-forecast package — a Python ML pipeline that turns raw seismic waveforms into volcanic-eruption probability forecasts.

The package is organised around the ForecastModel orchestrator. Domain layers (sources → tremor → label/features → model → ensemble) are independent and composable; cross-cutting layers (config, plots, utils) are reused by every domain layer.

Data flow at this level:
- Seismic Sources (SDS/FDSN) provide per-day ObsPy Streams
- Tremor Calculation extracts RSAM/DSAR/Shannon entropy CSVs from those streams
- Label Builder marks sliding windows around known eruption dates
- Feature Engineering turns each window into a tsfresh feature row
- Model Stack trains classifiers across many seeds, predicts on unlabelled windows, and evaluates against truth
- Ensemble Layer wraps seed-level estimators into sklearn-compatible bundles consumable by the model stack
