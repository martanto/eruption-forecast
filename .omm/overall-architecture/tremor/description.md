Tremor calculation domain. Turns per-day ObsPy Streams (from a `SeismicDataSource`) into a single time-series CSV with three families of metrics computed over five frequency bands.

Files:
- calculate_tremor.py — `CalculateTremor` orchestrator (`from_sds()` / `from_fdsn()` → `run()`); parallelises per-day work with `multiprocessing.Pool`.
- rsam.py — `RSAM`: bandpassed mean absolute amplitude per band.
- dsar.py — `DSAR`: per-band displacement ratios between consecutive bands.
- shannon_entropy.py — `ShannonEntropy`: broadband signal-complexity column.
- tremor_data.py — `TremorData`: thin wrapper over the saved tremor CSV that exposes start/end, sampling rate, and `@cached_property` accessors.

Default bands: 0.01–0.1, 0.1–2, 2–5, 4.5–8, 8–16 Hz at 10-minute sampling.
