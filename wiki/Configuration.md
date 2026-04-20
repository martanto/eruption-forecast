# Configuration

## notify — Telegram Notifications

The package provides two Telegram notification APIs:

- `notify`: decorator for function success/error notifications.
- `send_telegram_notification`: direct function call for one-off messages (with optional file attachments).

Both are useful for long-running steps like multi-seed training.

### Setup

1. Create a bot via [@BotFather](https://t.me/BotFather) and get your chat ID from [@userinfobot](https://t.me/userinfobot).
2. Copy `.env.example` to `.env` and fill in your credentials:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Credentials are loaded automatically from `.env` via `python-dotenv`.

### Basic Usage

```python
from eruption_forecast.decorators import notify, send_telegram_notification

# Credentials read from .env automatically
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

### Message Format

**Success:**
```
🖥 Host:       DESKTOP-ABC
📋 Task:       train_model
🕐 Time:       2026-02-23 14:05:00
✅ Status:     finished successfully.
💬 Message:    Function completed without errors.
⏱ Elapsed:    00h 02m 35s
```

**Error:**
```
🖥 Host:       DESKTOP-ABC
📋 Task:       train_model
🕐 Time:       2026-02-23 14:05:12
❌ Status:     raised ValueError
💬 Message:    Something went wrong
⏱ Elapsed:    00h 00m 12s
```

> Exceptions are always re-raised after the error notification — `notify` never swallows errors. Network errors during notification are logged and never propagate.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bot_token` | `str \| None` | `None` | Telegram bot token. Falls back to `TELEGRAM_BOT_TOKEN` env var |
| `chat_id` | `str \| int \| None` | `None` | Chat ID. Falls back to `TELEGRAM_CHAT_ID` env var |
| `name` | `str \| None` | `None` | Display name in messages. Defaults to `func.__name__` |
| `on_success` | `bool` | `True` | Send notification on success |
| `on_error` | `bool` | `True` | Send notification on error |
| `include_elapsed` | `bool` | `True` | Include elapsed time in the message |
| `files` | `list \| Callable \| None` | `None` | Files to attach on success (static list or callable receiving return value) |

### File Attachments

```python
# Static list of files
@notify(files=["output/plot.png", "output/metrics.csv"])
def generate_report():
    ...

# Dynamic — callable receives the function's return value
@notify(files=lambda result: [result["plot_path"]])
def run_pipeline():
    return {"plot_path": "output/forecast.png"}
```

PNG/JPG files are sent via `sendPhoto`; all others via `sendDocument`.

### Suppressing Notifications

```python
@notify(on_success=False)   # Only notify on error
def long_running_job(): ...

@notify(on_error=False)     # Only notify on success
def experimental_step(): ...
```

---

## Pipeline Configuration — Save & Replay

`ForecastModel` accumulates all pipeline parameters into a `PipelineConfig` object as each stage completes. You can save it to YAML/JSON, then reload it later to replay the run or resume from a saved model.

> **Important:** The config stores *parameters*, not data. All output files from each stage must still exist on disk when you replay. If files have been moved or deleted, re-run the corresponding stage with `overwrite=True`.

### Saving Configuration

`ForecastModel` saves automatically after `train()` and `forecast()`. You can also save manually at any point:

```python
fm.save_config()                           # YAML → {station_dir}/config.yaml
fm.save_config("my_run.yaml")              # Custom path
fm.save_config("my_run.json", fmt="json")  # JSON format
fm.save_model()                            # Serialise full instance → forecast_model.pkl
fm.save_model("my_model.pkl")             # Custom path
```

### YAML Config Format

A fully annotated template is at [`config.example.yaml`](https://github.com/martanto/eruption-forecast/blob/master/config.example.yaml).

```yaml
version: '1.0'
saved_at: '2026-02-23T12:00:00'

model:
  station: OJN
  channel: EHZ
  start_date: '2025-03-16'
  end_date: '2025-03-22'
  window_size: 1
  volcano_id: MERAPI
  network: VG
  location: '00'
  n_jobs: 4

calculate:
  source: sds
  sds_dir: D:/Data/OJN
  remove_outlier_method: maximum

build_label:
  window_step: 12
  window_step_unit: hours
  day_to_forecast: 2
  eruption_dates: ['2025-03-20']

extract_features:
  select_tremor_columns: [rsam_f2, rsam_f3]
  use_relevant_features: false

train:
  classifiers: [xgb]
  cv_strategy: stratified
  total_seed: 500
  number_of_significant_features: 20

forecast:
  start_date: '2025-03-23'
  end_date: '2025-03-30'
  window_step: 12
  window_step_unit: hours
```

Sections for stages that were not called are omitted.

### Replay the Full Pipeline

```python
from eruption_forecast import ForecastModel

fm = ForecastModel.from_config("output/VG.OJN.00.EHZ/config.yaml")
fm.run()   # calculate() → build_label() → extract_features() → train() → forecast()
```

Stages that already have output files are skipped automatically (`overwrite` defaults to `False`). To force a specific stage to re-run:

```python
fm = ForecastModel.from_config("output/VG.OJN.00.EHZ/config.yaml")
fm._loaded_config.train.overwrite = True  # Force re-train only
fm.run()
```

### Resume from a Saved Model

The pickle file embeds all in-memory state (tremor DataFrame, labels, feature DataFrame, trained model paths). The trained `.pkl` model files must still be accessible on disk.

```python
fm = ForecastModel.load_model("output/VG.OJN.00.EHZ/forecast_model.pkl")
fm.forecast(
    start_date="2025-04-01",
    end_date="2025-04-07",
    window_step=12,
    window_step_unit="hours",
)
```

### Build a Config Manually

Useful for preparing a run before executing it:

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

config.save("my_run.yaml")
fm = ForecastModel.from_config("my_run.yaml")
fm.run()
```

### API Summary

| Method | Description |
|--------|-------------|
| `fm.save_config(path=None, fmt="yaml")` | Save to YAML (default) or JSON |
| `fm.save_model(path=None)` | joblib-serialise the full instance |
| `ForecastModel.from_config(path)` | Load config, construct instance |
| `ForecastModel.load_model(path)` | Restore serialised instance |
| `fm.run()` | Replay all stages (only after `from_config()`) |
| `PipelineConfig.save(path, fmt)` | Save standalone config |
| `PipelineConfig.load(path)` | Load YAML or JSON (format from extension) |

---

## Logging

The package uses [loguru](https://loguru.readthedocs.io/) for structured logging.

```python
from eruption_forecast.logger import set_log_level, set_log_directory

set_log_level("DEBUG")            # DEBUG, INFO, WARNING, ERROR
set_log_directory("/custom/logs") # Write logs to a custom directory
```

Most pipeline classes accept `verbose=True` for progress messages and `debug=True` for debug-level logging without changing the global log level.
