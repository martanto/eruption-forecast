# Data Sources

`eruption-forecast` accepts seismic waveform data via two adapters defined in `src/eruption_forecast/sources/`:

| Adapter | Module | Role |
|---------|--------|------|
| `SDS` | `sources/sds.py` | Reads a **local** SeisComP Data Structure archive |
| `FDSN` | `sources/fdsn.py` | Pulls from any FDSN web service and **caches locally as SDS** |

Both adapters subclass `SeismicDataSource` (`sources/base.py`) and expose the same `get(date) -> obspy.Stream` contract used by `CalculateTremor`.

---

## The `SeismicDataSource` ABC

```python
class SeismicDataSource(ABC):
    nslc: str   # "{network}.{station}.{location}.{channel}"

    @abstractmethod
    def get(self, date: datetime) -> Stream: ...

    def _make_log_prefix(self, date: datetime) -> str:
        return f"{date.strftime('%Y-%m-%d')} :: {self.nslc}"
```

Every concrete adapter must implement `get()` and may rely on the shared `_make_log_prefix()` helper for consistent log lines.

---

## SDS — Local SeisComP Archive

The SDS layout is the de-facto standard for ObsPy / SeisComP archives:

```
{sds_dir}/{year}/{network}/{station}/{channel}.{type}/{network}.{station}.{location}.{channel}.{type}.{year}.{julian_day}

Example:
D:\Data\OJN\2025\VG\OJN\EHZ.D\VG.OJN.00.EHZ.D.2025.015
                                                   ^^^ Julian day 015 = Jan 15
```

`SDS` reads exactly **one calendar day** per `get(date)` call, optionally interpolating gaps when `interpolate=True`.

### Use from `ForecastModel`

```python
fm.calculate(
    start_date="2025-01-01",
    end_date="2025-12-31",
    source="sds",
    sds_dir=r"D:\Data\OJN",       # required when source="sds"
    methods=["rsam", "dsar", "entropy"],
    interpolate=True,
)
```

### Use directly

```python
from datetime import datetime
from eruption_forecast.sources.sds import SDS

sds = SDS(
    sds_dir=r"D:\Data\OJN",
    station="OJN", channel="EHZ", network="VG", location="00",
    interpolate=True,
)
stream = sds.get(datetime(2025, 3, 20))     # ObsPy Stream
trace  = sds.get_trace(datetime(2025, 3, 20))  # merged single Trace or None
```

### Key methods

| Method | Returns | Purpose |
|--------|---------|---------|
| `get(date)` | `Stream` | Read one day's miniSEED, merge + interpolate gaps |
| `get_trace(date)` | `Trace \| None` | Merged single trace (raises if more than one survives) |
| `get_filepath(date)` | `str` | Construct the canonical SDS path without reading |
| `save(stream, date)` | `str \| None` | Write a stream to its SDS-canonical path (used by `FDSN`) |

---

## FDSN — Web Service with Local Cache

`FDSN` wraps `obspy.clients.fdsn.Client` and writes every successful download into an **SDS cache** at `download_dir`. The next call for the same `(date, nslc)` is served from disk — no network round-trip.

### Read path

```
                   ┌──────────────────────────┐
                   │   FDSN.get(date)         │
                   └────────────┬─────────────┘
                                │
              ┌─────────────────┴─────────────────┐
              │  Local SDS cache file exists?     │
              └─────────┬─────────────────┬───────┘
                    yes │                 │ no / overwrite=True
                        ▼                 ▼
              ┌──────────────────┐  ┌─────────────────────────────┐
              │ obspy.read(file) │  │ client.get_waveforms(...)   │
              └────────┬─────────┘  │ network round-trip via      │
                       │            │ ObsPy FDSN client           │
                       │            └──────────────┬──────────────┘
                       │                           │
                       │                  ┌────────▼────────┐
                       │                  │ SDS.save(stream)│  miniSEED → SDS layout
                       │                  └────────┬────────┘
                       │                           │
                       └────────────┬──────────────┘
                                    ▼
                            return obspy.Stream
```

### Use from `ForecastModel`

```python
fm.calculate(
    start_date="2025-01-01",
    end_date="2025-12-31",
    source="fdsn",
    client_url="https://service.iris.edu",   # any FDSN endpoint
    methods=["rsam", "dsar", "entropy"],
)
```

The downloaded miniSEED is cached under `./downloads/{year}/{network}/...` by default. To pin the cache location, instantiate `FDSN` directly:

```python
from eruption_forecast.sources.fdsn import FDSN

fdsn = FDSN(
    station="OJN", channel="EHZ", network="VG", location="00",
    client_url="https://service.iris.edu",
    download_dir=r"D:\Data\OJN-cache",
    overwrite=False,
)
```

### Constructor parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `station`, `channel`, `network`, `location` | — | Standard NSLC components. `location=None` and `""` are equivalent |
| `channel_type` | `"D"` | SDS channel-type suffix |
| `client_url` | `"https://service.iris.edu"` | Any FDSN endpoint accepted by `obspy.clients.fdsn.Client` |
| `download_dir` | `<cwd>/downloads` | Cache root — created automatically if absent |
| `overwrite` | `False` | When `True`, re-download even when cached |
| `verbose`, `debug` | `False` | Forwarded to loguru |

---

## SDS vs FDSN at a Glance

|                       | `SDS` | `FDSN` |
|-----------------------|-------|--------|
| Source of truth | Local archive | Remote web service |
| Network required | No | First time only — subsequent calls hit the SDS cache |
| Where data ends up | Already on disk | Written to `download_dir` in SDS layout |
| Failure mode | Empty `Stream` if file missing | Empty `Stream` if FDSN raises `FDSNException` |
| Best for | Offline pipelines, large pre-existing archives | Ad-hoc fetches, missing days in a partial SDS |

Because `FDSN` caches into an SDS directory, you can start a pipeline against FDSN and later switch to `source="sds"` pointing at the same `download_dir` once the archive is complete.

---

## Frequency Bands Read by `CalculateTremor`

`CalculateTremor` applies five default band-pass filters per day:

| Alias | Range (Hz) | Used by |
|-------|------------|---------|
| `f0`  | 0.01 – 0.1 | RSAM |
| `f1`  | 0.1 – 2    | RSAM, DSAR |
| `f2`  | 2 – 5      | RSAM, DSAR |
| `f3`  | 4.5 – 8    | RSAM, DSAR |
| `f4`  | 8 – 16     | RSAM, DSAR |

The resulting CSV columns are `rsam_f0..f4`, `dsar_f0-f1..f3-f4`, and `entropy`. Override the band list with `CalculateTremor(...).change_freq_bands([(0.1, 1.0), (1.0, 5.0)])`.
