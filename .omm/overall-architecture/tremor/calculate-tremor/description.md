Top-level orchestrator for the tremor stage. Pulls a `SeismicDataSource` (via `from_sds()` or `from_fdsn()`), walks the date range, dispatches per-day work to a `multiprocessing.Pool(n_jobs)`, merges per-day CSVs into one tremor matrix, and optionally renders daily plots.

Per-day worker calls into `RSAM`, `DSAR`, and `ShannonEntropy` per the user-selected `methods` list. Supports outlier removal, gap interpolation, value scaling, anomaly removal, and an optional `cleanup_daily_dir` that deletes the per-day staging directory once the merged CSV is written.
