"""Tremor calculation and metrics package for volcanic seismic data analysis.

Processes raw seismic waveform data into tremor metrics used for
eruption forecasting. It supports reading data from SDS archives or FDSN web
services, computing metrics across multiple frequency bands at 10-minute intervals,
and writing time-series CSV files for downstream feature extraction.

Key classes:
    - ``CalculateTremor``: Main orchestrator — reads seismic data, computes RSAM,
      DSAR, and Shannon entropy, and saves per-day and merged CSV files.
    - ``RSAM``: Computes Real Seismic Amplitude Measurement (mean absolute amplitude)
      per frequency band over fixed-duration time windows.
    - ``DSAR``: Computes Displacement Seismic Amplitude Ratio (ratio of mean absolute
      amplitudes between two consecutive frequency bands).
    - ``ShannonEntropy``: Computes Shannon entropy over a bandpass-filtered seismic
      stream as a measure of signal complexity.
    - ``TremorData``: Loads and validates merged tremor CSV files; exposes start/end
      dates and sampling-rate consistency checks.

Default frequency bands: (0.01–0.1), (0.1–2), (2–5), (4.5–8), (8–16) Hz.
Output CSV columns follow the naming convention ``rsam_f0``, ``dsar_f0-f1``, etc.
"""
