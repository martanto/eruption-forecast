# Configuration

> Back to [README](../README.md)

- [notify — Telegram Notifications](#notify--telegram-notifications)
- [Pipeline Configuration — Save & Replay](#pipeline-configuration--save--replay)
- [Logging](#logging)

## Decorators

The `decorators` package provides utilities for wrapping functions with cross-cutting behaviour such as timing, parameter persistence, and remote notifications.

### notify — Telegram Notifications

The package provides two Telegram notification APIs:

- `notify`: decorator for function success/error notifications.
- `send_telegram_notification`: direct function call for one-off messages (with optional file attachments).

Both are useful for long-running pipeline steps such as tremor calculation or multi-seed training.

#### Setup

1. Copy `.env.example` to `.env` and fill in your credentials:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

2. Obtain a bot token from [@BotFather](https://t.me/BotFather) and your chat ID from [@userinfobot](https://t.me/userinfobot).

#### Basic Usage

```python
from eruption_forecast.decorators import notify, send_telegram_notification

# Credentials read automatically from .env
@notify(name="Training Run")
def train_model():
    ...

# Or pass credentials explicitly
@notify(bot_token="TOKEN", chat_id="CHAT_ID", name="Training Run")
def train_model():
    ...

# Direct function-style call (no decorator)
send_telegram_notification(
    message="Training run finished successfully.",
    files=["output/forecast.png", "output/metrics.csv"],
    file_caption="Training artifacts",
)
```

#### Message Format

**Success:**
```
🖥 Host:
`DESKTOP-ABC`

📋 Task:
`train_model`

🕐 Time:
`2026-02-23 14:05:00`

✅ Status:
finished successfully.

💬 Message:
Function completed without errors.

⏱ Elapsed:
`00h 02m 35s`
```

**Error:**
```
🖥 Host:
`DESKTOP-ABC`

📋 Task:
`train_model`

🕐 Time:
`2026-02-23 14:05:12`

❌ Status:
raised `ValueError`

💬 Message:
`Something went wrong`

⏱ Elapsed:
`00h 00m 12s`
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bot_token` | `str \| None` | `None` | Telegram bot token. Falls back to `TELEGRAM_BOT_TOKEN` env var |
| `chat_id` | `str \| int \| None` | `None` | Telegram chat ID. Falls back to `TELEGRAM_CHAT_ID` env var |
| `name` | `str \| None` | `None` | Display name in messages. Defaults to `func.__name__` |
| `on_success` | `bool` | `True` | Send notification on successful completion |
| `on_error` | `bool` | `True` | Send notification when an exception is raised |
| `include_elapsed` | `bool` | `True` | Include elapsed time in the message |
| `files` | `list[str \| Path] \| Callable \| None` | `None` | Files to attach after a success notification. May be a static list or a callable receiving the return value |

#### File Attachments

```python
# Static list of files
@notify(files=["output/plot.png", "output/metrics.csv"])
def generate_report():
    ...

# Dynamic — receive the function's return value
@notify(files=lambda result: [result["plot_path"]])
def run_pipeline():
    return {"plot_path": "output/forecast.png"}
```

PNG/JPG files are sent via `sendPhoto`; all other files via `sendDocument`.

#### Suppressing Notifications

```python
# Only notify on error (skip success messages)
@notify(on_success=False)
def long_running_job():
    ...

# Only notify on success (skip error messages)
@notify(on_error=False)
def experimental_step():
    ...
```

> **Note:** Exceptions are always re-raised after the error notification — `notify` never swallows errors. Network errors during notification are logged via loguru and never propagate.

---

## Configuration

### Pipeline Configuration — Save & Replay

After any pipeline run, every parameter is recorded in a `PipelineConfig` object. You can persist it as YAML or JSON, then reload it to replay the exact same run or resume from a saved model.

> **Important — output files are required for replay.**
> The config file (or pickled model) stores *parameters*, not data. To replay or resume a pipeline, the output files produced by each stage must still exist on disk at the same paths:
>
> | Stage | Required output files |
> |---|---|
> | `calculate()` | `tremor/tremor_*.csv` |
> | `build_label()` | `label_*.csv` |
> | `extract_features()` | `features/all_extracted_features_*.csv`, `features/label_features_*.csv`, `features/tremor_matrix_*.csv` |
> | `train()` | `trainings/.../trained_model_*.csv`, `trainings/.../models/*.pkl` |
>
> If any of these files have been moved or deleted, the corresponding stage must be re-run with `overwrite=True`.
> To avoid this, keep the entire `output/VG.OJN.00.EHZ/` directory intact alongside the config and model files.

#### How it works

`ForecastModel` accumulates parameters automatically as each stage completes:

- `train(save_model=True)` *(default)*: serialises the full instance to `{station_dir}/forecast_model.pkl`
- `forecast()`: saves `{station_dir}/config.yaml` after each call

You can also save explicitly at any point:

```python
fm.save_config()                          # YAML → {station_dir}/config.yaml
fm.save_config("my_run.yaml")             # custom path
fm.save_config("my_run.json", fmt="json") # JSON format
fm.save_model()                           # joblib → {station_dir}/forecast_model.pkl
fm.save_model("my_model.pkl")             # custom path
```

#### Saved YAML format

A fully annotated template with all fields and inline comments is available at [`config.example.yaml`](config.example.yaml) in the project root. Copy it, edit the values for your run, and load it with `ForecastModel.from_config("config.yaml")`.

```yaml
# eruption-forecast pipeline configuration
version: '1.0'
saved_at: '2026-02-17T12:00:00'

model:
  station: OJN
  channel: EHZ
  start_date: '2025-03-16'
  end_date: '2025-03-22'
  window_size: 1
  volcano_id: MERAPI
  network: VG
  location: '00'
  output_dir: null
  root_dir: null
  overwrite: false
  n_jobs: 4
  verbose: false
  debug: false

calculate:
  source: sds
  sds_dir: D:/Data/OJN
  methods: null
  filename_prefix: null
  remove_outlier_method: maximum
  remove_tremor_anomalies: false
  interpolate: true
  value_multiplier: null
  cleanup_daily_dir: false
  plot_daily: false
  save_plot: false
  overwrite_plot: false
  client_url: https://service.iris.edu
  n_jobs: null
  verbose: false
  debug: false

build_label:
  window_step: 12
  window_step_unit: hours
  day_to_forecast: 2
  eruption_dates:
  - '2025-03-20'
  start_date: null
  end_date: null
  tremor_columns: null
  verbose: false
  debug: false

extract_features:
  select_tremor_columns:
  - rsam_f2
  - rsam_f3
  save_tremor_matrix_per_method: true
  save_tremor_matrix_per_id: false
  exclude_features: null
  use_relevant_features: false
  overwrite: false
  n_jobs: null
  verbose: null

train:
  classifiers:
  - xgb
  cv_strategy: stratified
  random_state: 0
  total_seed: 500
  number_of_significant_features: 20
  sampling_strategy: 0.75
  save_all_features: false
  plot_significant_features: false
  n_jobs: null
  grid_search_n_jobs: 1
  overwrite: false
  verbose: false

forecast:
  start_date: '2025-03-23'
  end_date: '2025-03-30'
  window_step: 12
  window_step_unit: hours
  save_predictions: true
  save_plot: true
  n_jobs: null
  overwrite: false
  verbose: false
```

Sections that were not called are omitted from the file. A partial run produces a partial YAML that can still be loaded.

#### Replay the full pipeline from a config file

`from_config` + `run()` re-executes every stage using the saved parameters. Stages that already produced output files are skipped automatically (because `overwrite` defaults to `False`); stages whose config section is absent are silently skipped too.

```python
from eruption_forecast import ForecastModel

# All stage output files must still exist in the output directory
fm = ForecastModel.from_config("output/VG.OJN.00.EHZ/config.yaml")
fm.run()  # calculate() → build_label() → extract_features() → train() → forecast()
```

To force a specific stage to re-run from scratch, set `overwrite=True` in the corresponding config section before calling `run()`:

```python
fm = ForecastModel.from_config("output/VG.OJN.00.EHZ/config.yaml")
fm._loaded_config.train.overwrite = True  # force re-train only
fm.run()
```

#### Resume from a saved model (skip re-training)

After `train()` saves `forecast_model.pkl`, you can load it later and call `forecast()` directly. The pickle file embeds all in-memory state — the tremor DataFrame, labels, feature DataFrame, trained model paths — so no intermediate CSV files need to be re-read. However, the **trained model `.pkl` files** referenced in `trained_models` must still be accessible on disk, because `forecast()` / `ModelPredictor` loads them at inference time.

```python
fm = ForecastModel.load_model("output/VG.OJN.00.EHZ/forecast_model.pkl")
fm.forecast(
    start_date="2025-04-01",
    end_date="2025-04-07",
    window_step=12,
    window_step_unit="hours",
)
```

#### Build a config manually and save it without running

`PipelineConfig` can also be used standalone — useful for preparing a run before executing it:

```python
from eruption_forecast import PipelineConfig
from eruption_forecast.config import (
    ModelConfig, CalculateConfig, BuildLabelConfig,
    ExtractFeaturesConfig, TrainConfig, ForecastConfig,
)

config = PipelineConfig(
    model=ModelConfig(
        station="OJN", channel="EHZ",
        start_date="2025-03-16", end_date="2025-03-22",
        window_size=1, volcano_id="MERAPI", n_jobs=4,
    ),
    calculate=CalculateConfig(source="sds", sds_dir="D:/Data/OJN"),
    build_label=BuildLabelConfig(
        window_step=12, window_step_unit="hours",
        day_to_forecast=2, eruption_dates=["2025-03-20"],
    ),
    extract_features=ExtractFeaturesConfig(
        select_tremor_columns=["rsam_f2", "rsam_f3"],
    ),
    train=TrainConfig(classifier="xgb", cv_strategy="stratified", total_seed=500),
    forecast=ForecastConfig(
        start_date="2025-03-23", end_date="2025-03-30",
        window_step=12, window_step_unit="hours",
    ),
)

config.save("my_run.yaml")           # YAML
config.save("my_run.json", fmt="json")  # JSON

# Load and run
fm = ForecastModel.from_config("my_run.yaml")
fm.run()
```

#### API summary

| Method / Classmethod | Description |
|---|---|
| `fm.save_config(path=None, fmt="yaml")` | Saves accumulated config; defaults to `{station_dir}/config.yaml` |
| `fm.save_model(path=None)` | joblib-dumps the full instance; defaults to `{station_dir}/forecast_model.pkl` |
| `ForecastModel.from_config(path)` | Loads config, constructs instance, attaches `_loaded_config` |
| `ForecastModel.load_model(path)` | Restores a joblib-pickled instance with all pipeline state |
| `fm.run()` | Replays all stages from `_loaded_config`; only valid after `from_config()` |
| `PipelineConfig.save(path, fmt)` | Standalone save to YAML or JSON |
| `PipelineConfig.load(path)` | Loads YAML or JSON; format detected from extension |

### Logging

```python
from eruption_forecast.logger import set_log_level, set_log_directory

set_log_level("DEBUG")  # Options: DEBUG, INFO, WARNING, ERROR
set_log_directory("/custom/logs")
```

---
