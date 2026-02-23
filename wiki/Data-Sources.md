# Data Sources

`eruption-forecast` supports two live seismic data sources — SDS and FDSN — as well as loading pre-calculated tremor data from a previous run.

---

## SDS — SeisComP Data Structure

SDS (SeisComP Data Structure) is a standardised directory and file layout defined by SeisComP for storing waveform data in a portable, archive-friendly format. Each file holds one channel's data for one day, making it straightforward to locate and read any segment without scanning the entire archive.

Full specification: https://www.seiscomp.de/seiscomp3/doc/applications/slarchive/SDS.html

### Directory layout

```
<sds_dir>/
└── YEAR/
    └── NET/
        └── STA/
            └── CHAN.TYPE/
                └── NET.STA.LOC.CHAN.TYPE.YEAR.DAY
```

### File naming example

For network `VG`, station `OJN`, channel `EHZ`, location `00`, day 075 of 2025:

```
/data/sds/2025/VG/OJN/EHZ.D/VG.OJN.00.EHZ.D.2025.075
```

### Field reference

| Field | Description | Constraints | Example |
|-------|-------------|-------------|---------|
| YEAR | Four-digit year | — | 2025 |
| NET | Network code | ≤ 8 chars, no spaces | VG |
| STA | Station code | ≤ 8 chars, no spaces | OJN |
| CHAN | Channel code | ≤ 8 chars, no spaces | EHZ |
| LOC | Location code | ≤ 8 chars, may be empty | 00 |
| TYPE | Data type character | See table below | D |
| DAY | Day of year | 3 digits, zero-padded | 075 |

### Data type codes

| Code | Meaning |
|------|---------|
| D | Waveform data (most common for seismic monitoring) |
| E | Detection data |
| L | Log data |
| T | Timing data |
| C | Calibration data |
| R | Response data |
| O | Opaque data |

> **Note:** Periods (`.`) in filenames are always present even when neighbouring fields (e.g. location) are empty. Files are in miniSEED format.

### Python usage

```python
from eruption_forecast import CalculateTremor

tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-31",
    n_jobs=4,
).from_sds(sds_dir="/data/sds").run()
```

`network` defaults to `"VG"` and `location` defaults to `"00"` in both `ForecastModel` and `CalculateTremor`. Pass explicit values when your data uses a different network or location code.

---

## FDSN — Web Service

FDSN support allows `CalculateTremor` to download waveform data from any FDSN-compatible web service — IRIS, GEOFON, Raspberry Shake, and others. Downloaded files are cached locally as SDS miniSEED, so subsequent runs for the same time range skip the network entirely and read from the local cache instead.

The `download_dir` used for caching is created automatically if it does not already exist.

### Python usage

```python
from eruption_forecast import CalculateTremor

tremor = CalculateTremor(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-01-31",
).from_fdsn(client_url="https://service.iris.edu").run()
```

> **Tip:** Downloaded miniSEED files are cached locally so subsequent runs skip the network entirely. Re-running the same date range costs no additional bandwidth.

---

## Using Pre-Calculated Tremor Data

If you have already calculated tremor data from a previous run, you can skip the `calculate()` step entirely by loading the existing CSV directly with `load_tremor_data()`.

```python
from eruption_forecast import ForecastModel

fm = ForecastModel(
    station="OJN",
    channel="EHZ",
    start_date="2025-01-01",
    end_date="2025-12-31",
    window_size=2,
    volcano_id="Lewotobi",
)

fm.load_tremor_data(
    tremor_csv="output/VG.OJN.00.EHZ/tremor/tremor_2025-01-01_2025-12-31.csv"
).build_label(...).extract_features(...).train(...)
```

This is useful when experimenting with different label or feature configurations while keeping the (often time-consuming) tremor calculation fixed.

---

## Output Format

After `CalculateTremor.run()`, the resulting tremor CSV has a DateTime index at 10-minute intervals with the following columns:

| Column | Metric | Description |
|--------|--------|-------------|
| rsam_f0 … rsam_f4 | RSAM | Real Seismic Amplitude Measurement per frequency band |
| dsar_f0-f1 … dsar_f3-f4 | DSAR | Displacement Seismic Amplitude Ratio between consecutive bands |
| entropy | Shannon Entropy | Signal complexity (single broadband column) |

Default frequency bands: (0.01–0.1), (0.1–2), (2–5), (4.5–8), (8–16) Hz.

Custom bands can be set with `.change_freq_bands([(0.1, 1.0), (1.0, 5.0), ...])` before calling `.from_sds()` or `.from_fdsn()`.
