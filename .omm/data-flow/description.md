End-to-end data flow from raw seismic miniSEED to evaluation/forecast artefacts. Every persisted artefact lives under a single station directory `{output_dir}/{network}.{station}.{location}.{channel}/`, so a station's full lifecycle is co-located.

Data shapes by stop:
- **Raw seismic** — daily ObsPy `Stream` per (net, sta, loc, cha, date) from SDS or FDSN.
- **Tremor CSV** — DateTime-indexed at 10-minute sampling; columns `rsam_f0..f4`, `dsar_f0-f1..f3-f4`, `entropy`.
- **Label CSV** — DateTime-indexed; columns `id` (window int), `is_erupted` (0/1). Filename encodes window params.
- **Feature matrix** — one row per window id, columns are tsfresh feature names (one column family per tremor metric), joined to labels by window id.
- **Seed artefacts** — per-seed `{seed:05d}.pkl` fitted estimators + per-classifier `trained_model_*.csv` registry of selected features.
- **Ensemble bundle** — `ClassifierEnsemble.pkl` (or `.json` registry) containing one `SeedEnsemble` per classifier.
- **Cache store** — content-addressable `{hash}.pkl + .params.json` per stage; identity covers all inputs (tremor frame, params, classifier list, upstream `training_hash` for prediction).
- **Forecast results** — per-classifier `_eruption_probability`, `_uncertainty`, `_confidence`, `_prediction` + `consensus_*`; plus a forecast plot.
- **Evaluation artefacts** — `(n_samples, n_seeds)` `y_proba`/`y_pred` matrices per classifier + aggregate plots + cross-classifier comparison.
- **SHAP artefacts** — SHAP values for a small `n_observations_to_explain` slice of the feature matrix.
