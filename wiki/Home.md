# eruption-forecast Wiki

A Python package for volcanic eruption forecasting using seismic data analysis.

---

## ⚠️ Research Use Only

This package produces **probabilistic predictions**, not deterministic guarantees. It is a research tool and must not be used as the sole basis for public safety decisions. Always consult qualified volcanologists and official observatory warnings.

---

## Navigation

| Page | Description |
|------|-------------|
| [Installation](Installation) | Setup, environment, development commands |
| [Data Sources](Data-Sources) | SDS format, FDSN web service, data loading |
| [Quick Start](Quick-Start) | End-to-end pipeline example |
| [Pipeline Walkthrough](Pipeline-Walkthrough) | Per-stage guide (tremor → labels → features → training → forecast) |
| [Training Workflows](Training-Workflows) | `evaluate()` vs `train()`, multi-seeding, feature selection |
| [Classifiers and CV](Classifiers-and-CV) | Supported classifiers, hyperparameter grids, cross-validation strategies |
| [Evaluation and Forecasting](Evaluation-and-Forecasting) | ModelEvaluator, MultiModelEvaluator, ModelPredictor |
| [Visualization](Visualization) | All plot types with code examples |
| [Configuration](Configuration) | Telegram notify decorator + direct send function, pipeline config save/replay, logging |
| [Output Structure](Output-Structure) | Full output directory tree |
| [Architecture](Architecture) | Package design, module responsibilities, key classes |
| [API Reference](API-Reference) | Constructor and method parameter tables |

---

## What This Package Does

`eruption-forecast` processes raw seismic data through a six-stage pipeline:

```
Raw Seismic Data (SDS / FDSN)
        ↓
  CalculateTremor    →  RSAM + DSAR + Shannon Entropy per frequency band
        ↓
   LabelBuilder      →  Binary labels from known eruption dates
        ↓
TremorMatrixBuilder  →  Windowed time-series aligned to labels
        ↓
 FeaturesBuilder     →  700+ tsfresh statistical features
        ↓
  ModelTrainer       →  Multi-seed classifier training (10 classifiers)
        ↓
 ModelPredictor      →  Probabilistic eruption forecasts
```

The high-level `ForecastModel` class chains all stages with a single fluent API.

---

## Repository Structure

```
eruption-forecast/
├── src/eruption_forecast/   # Package source
│   ├── data_container.py    # BaseDataContainer — shared ABC for data wrappers
│   ├── sources/             # SeismicDataSource ABC + SDS/FDSN adapters
│   └── ...                  # tremor/, label/, features/, model/, utils/, plots/
├── wiki/                    # Reference documentation (GitHub Wiki)
├── tests/                   # Unit tests
├── main.py                  # Complete working example
└── config.example.yaml      # Annotated pipeline config template
```

---

## Key Links

- [README](https://github.com/martanto/eruption-forecast/blob/master/README.md)
- [main.py — Full working example](https://github.com/martanto/eruption-forecast/blob/master/main.py)
- [config.example.yaml — Annotated config template](https://github.com/martanto/eruption-forecast/blob/master/config.example.yaml)
