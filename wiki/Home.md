# eruption-forecast Wiki

A Python package for volcanic eruption forecasting from continuous seismic tremor and machine-learning ensembles.

---

## ⚠️ Research Use Only

`eruption-forecast` produces **probabilistic** eruption likelihoods, not deterministic warnings. It is a research tool and must not be used as the sole basis for public safety decisions. Always consult qualified volcanologists and official observatory bulletins.

---

## Navigation

| # | Page | Description |
|---|------|-------------|
| 1 | [Getting Started](Getting-Started) | Prerequisites, installation, dev commands |
| 2 | [Data Sources](Data-Sources) | SDS archive layout, FDSN web service, local caching |
| 3 | [Usage](Usage) | Quick Start + annotated end-to-end example |
| 4 | [Pipeline Walkthrough](Pipeline-Walkthrough) | Research Workflow (`main.py`) + Scenarios Workflow (`scenarios.py`) |
| 5a | [Training Workflow](Training-Workflow) | `TrainingModel`, classifiers, CV, imbalance, parallelism |
| 5b | [Prediction Workflow](Prediction-Workflow) | `PredictionModel`, forecast outputs, consensus |
| 5c | [Evaluation Workflow](Evaluation-Workflow) | `EvaluationModel`, `MetricsEnsemble`, `ClassifierComparator` |
| 6 | [Visualization](Visualization) | Plot catalog + output paths |
| 7 | [Configuration](Configuration) | `ForecastConfig`, YAML save/replay, Telegram, logging |
| 8 | [Output Structure](Output-Structure) | Full directory tree + slug conventions |
| 9 | [Architecture](Architecture) | Package layout, class relationships, data flow |
| 10 | [API Reference](API-Reference) | Constructor + method parameter tables |

---

## What This Package Does

```
                Raw Seismic (SDS / FDSN)
                        │
                        ▼
            ┌───────────────────────┐
            │   CalculateTremor     │  RSAM + DSAR + Shannon Entropy
            └──────────┬────────────┘  per frequency band
                       │
                       ▼
            ┌───────────────────────┐
            │     TrainingModel     │  LabelBuilder → FeaturesBuilder → fit
            │  (BaseModel +         │  multi-seed GridSearchCV
            │   CacheModel)         │  produces ClassifierEnsemble
            └──────────┬────────────┘
                       │  ClassifierEnsemble  (N classifiers × M seeds)
                       │
              ┌────────┴────────┐
              ▼                 ▼
   ┌───────────────────┐  ┌──────────────────────┐
   │  PredictionModel  │  │   EvaluationModel    │  per-seed metrics JSON
   │  forecast grid →  │  │  MetricsEnsemble +   │  aggregate CSV + plots
   │  probabilities    │  │  ClassifierComparator│
   └───────────────────┘  └──────────────────────┘
```

The high-level `ForecastModel` class chains every stage with a fluent API:

```python
from eruption_forecast import ForecastModel

(
    ForecastModel(station="OJN", channel="EHZ", network="VG", location="00",
                  day_to_forecast=2, n_jobs=4)
    .calculate(start_date="2025-01-01", end_date="2025-12-31",
               source="sds", sds_dir="/data/sds")
    .train(start_date="2025-01-01", end_date="2025-07-26",
           eruption_dates=["2025-03-20", "2025-04-22", "..."],
           window_step=6, window_step_unit="hours",
           classifiers=["rf", "xgb"], seeds=25)
    .predict(start_date="2025-07-27", end_date="2025-08-22",
             window_step=10, window_step_unit="minutes",
             plot_threshold=0.7)
    .evaluate(model="prediction")
)
```

---

## Repository Map

```
eruption-forecast/
├── src/eruption_forecast/      Package source (64 .py files)
├── wiki/                       This wiki (Markdown sources)
├── tests/                      Unit tests
├── main.py                     Research Workflow — single train + predict
├── scenarios.py                Scenarios Workflow — loop over date-split scenarios
├── config.example.yaml         Annotated ForecastConfig template
├── CLAUDE.md                   Project rules and architecture cheatsheet
└── WIKI.md                     Local wiki-rewrite progress tracker
```

---

## Key Links

- [README](https://github.com/martanto/eruption-forecast/blob/master/README.md)
- [`main.py` — Research Workflow](https://github.com/martanto/eruption-forecast/blob/master/main.py)
- [`scenarios.py` — Scenarios Workflow](https://github.com/martanto/eruption-forecast/blob/master/scenarios.py)
- [`config.example.yaml` — config template](https://github.com/martanto/eruption-forecast/blob/master/config.example.yaml)
