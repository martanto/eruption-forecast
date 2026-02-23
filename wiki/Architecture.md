# Architecture

## Package Layout

```
src/eruption_forecast/
├── tremor/              # Seismic tremor processing
│   ├── calculate_tremor.py    # CalculateTremor — main orchestrator
│   ├── rsam.py                # Real Seismic Amplitude Measurement
│   ├── dsar.py                # Displacement Seismic Amplitude Ratio
│   ├── shanon_entropy.py      # Shannon Entropy metric
│   └── tremor_data.py         # TremorData — wraps tremor CSV
├── label/               # Training label generation
│   ├── label_builder.py       # LabelBuilder — sliding window labelling
│   └── label_data.py          # LabelData — wraps label CSV
├── features/            # Feature extraction & selection
│   ├── features_builder.py    # FeaturesBuilder — tsfresh extraction
│   ├── feature_selector.py    # FeatureSelector — 3-method selection
│   └── tremor_matrix_builder.py  # TremorMatrixBuilder — windowed alignment
├── model/               # ML model training & prediction
│   ├── forecast_model.py      # ForecastModel — full pipeline orchestrator
│   ├── model_trainer.py       # ModelTrainer — multi-seed training
│   ├── model_predictor.py     # ModelPredictor — inference & forecasting
│   ├── model_evaluator.py     # ModelEvaluator — single-seed evaluation
│   ├── multi_model_evaluator.py  # MultiModelEvaluator — aggregate evaluation
│   └── classifier_model.py   # ClassifierModel — classifier + grid management
├── sources/             # Seismic data source adapters
│   ├── sds.py                 # SDS reader (SeisComP Data Structure)
│   └── fdsn.py                # FDSN web service client with local caching
├── config/              # Pipeline configuration
│   └── pipeline_config.py     # PipelineConfig + sub-config dataclasses
├── plots/               # Visualization utilities
│   ├── tremor_plots.py
│   ├── feature_plots.py
│   ├── evaluation_plots.py
│   └── shap_plots.py
├── utils/               # Focused utility modules
│   ├── array.py               # Z-score outlier detection
│   ├── window.py              # Sliding window construction
│   ├── date_utils.py          # Date validation and conversion
│   ├── dataframe.py           # DataFrame validation
│   ├── ml.py                  # Resampling and feature utilities
│   ├── pathutils.py           # Path resolution relative to root_dir
│   └── formatting.py          # Text formatting
└── decorators/          # Function decorators
    └── notify.py              # Telegram notification decorator
```

---

## Design Principles

| Principle | What it means in practice |
|-----------|--------------------------|
| **Single Responsibility** | Each module has one purpose — `rsam.py` only computes RSAM, `date_utils.py` only handles dates |
| **Explicit Imports** | No hidden re-exports; import exactly what you need: `from eruption_forecast.utils.date_utils import to_datetime` |
| **Minimal Dependencies** | Each utility module imports only its own direct dependencies |
| **Fluent API** | All pipeline classes support method chaining via `return self` |
| **Data Leakage Prevention** | Train/test split always happens before resampling and feature selection |
| **Cached Properties** | `TremorData` and `LabelData` use `@cached_property` so attributes are computed once |

---

## Key Class Relationships

```
ForecastModel
    ├── uses CalculateTremor  (or load_tremor_data)
    ├── uses LabelBuilder
    ├── uses TremorMatrixBuilder
    ├── uses FeaturesBuilder
    │       └── uses FeatureSelector
    ├── uses ModelTrainer
    │       └── uses ClassifierModel (grid + CV)
    └── uses ModelPredictor
            ├── uses ModelEvaluator     (evaluation mode)
            └── aggregates predictions  (forecast mode)

MultiModelEvaluator
    ├── reads trained_model_*.csv  (registry from ModelTrainer)
    └── reads metrics/*.json       (per-seed metrics from ModelEvaluator)
```

---

## Pipeline Data Flow

| Stage | Input | Output |
|-------|-------|--------|
| `CalculateTremor` | Raw SDS/FDSN waveforms | `tremor_*.csv` — DateTime index, RSAM/DSAR/entropy columns |
| `LabelBuilder` | Date range + eruption dates | `label_*.csv` — DateTime index, `id`, `is_erupted` columns |
| `TremorMatrixBuilder` | tremor DataFrame + label DataFrame | `tremor_matrix_*.csv` — long-format with `id`, `datetime`, tremor columns |
| `FeaturesBuilder` | tremor matrix + labels | `all_extracted_features_*.csv`, `label_features_*.csv` |
| `ModelTrainer` | features CSV + labels CSV | `models/*.pkl`, `trained_model_*.csv`, metrics files |
| `ModelPredictor` | `trained_model_*.csv` + new tremor | `predictions.csv`, `eruption_forecast.png` |

---

## Utility Modules

| Module | Key Functions |
|--------|--------------|
| `utils/array.py` | `detect_maximum_outlier()`, `remove_outliers()` — Z-score based |
| `utils/window.py` | `construct_windows()`, `calculate_window_metrics()` |
| `utils/date_utils.py` | `to_datetime()`, `validate_date_ranges()`, `validate_window_step()` |
| `utils/ml.py` | `random_under_sampler()`, `get_significant_features()` |
| `utils/pathutils.py` | `resolve_output_dir()` — resolves relative paths against `root_dir` |
| `utils/dataframe.py` | DataFrame shape/column validation helpers |
| `utils/formatting.py` | Human-readable text formatting (elapsed time, file sizes, etc.) |

---

## Configuration Dataclasses

`PipelineConfig` holds sub-configs for each pipeline stage:

| Dataclass | Stage it covers |
|-----------|----------------|
| `ModelConfig` | `ForecastModel` constructor parameters |
| `CalculateConfig` | `calculate()` parameters |
| `BuildLabelConfig` | `build_label()` parameters |
| `ExtractFeaturesConfig` | `extract_features()` parameters |
| `TrainConfig` | `train()` parameters |
| `ForecastConfig` | `forecast()` parameters |

Configs are serialised to YAML or JSON via `PipelineConfig.save()` and loaded via `ForecastModel.from_config()`. See the [Configuration](Configuration) wiki page.
