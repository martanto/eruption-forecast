Reads raw seismic Streams day-by-day from a `SeismicDataSource` (SDS or FDSN) and computes RSAM, DSAR, and Shannon entropy across five frequency bands at a 10-minute sampling interval.

Outputs a merged tremor CSV at `{station_dir}/tremor/{nslc}_{start}_{end}.csv` (default columns `rsam_f0..f4`, `dsar_f0-f1..f3-f4`, `entropy`). Per-day staging directory `tremor/daily/` can be cleaned up after the merge.

Implemented by `CalculateTremor`. `n_jobs` controls the size of the `multiprocessing.Pool` used to fan out daily work. `ForecastModel.calculate()` adjusts `start_date` backward by `day_to_forecast` days so downstream label/feature windows have full lead-in coverage.
