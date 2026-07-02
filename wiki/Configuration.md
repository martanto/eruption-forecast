# Configuration

Every `ForecastModel` stage auto-captures its kwargs into a `ForecastConfig` object, 
which can be saved to YAML/JSON and replayed via `from_config() → run()`. 
This page covers config persistence, the dataclass layout, Telegram notifications, and runtime logging.

---

## ForecastConfig Lifecycle

```
              ┌────────────────────────────────────────┐
              │            ForecastModel               │
              │   _config: ForecastConfig              │
              └────────────┬───────────────────────────┘
                           │ stage methods auto-capture kwargs
                           ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  ForecastConfig                                              │
   │   ├── version, saved_at                                      │
   │   ├── model:      BaseForecastConfig                         │
   │   ├── calculate:  ForecastCalculateConfig | None             │
   │   ├── train:      ForecastTrainConfig     | None             │
   │   ├── predict:    ForecastPredictConfig   | None             │
   │   ├── evaluate:   ForecastEvaluateConfig  | None             │
   │   └── explain:    ForecastExplainConfig   | None             │
   └────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
         fm.save_config()         ForecastModel.from_config(path)
         → forecast.config.yaml   → new ForecastModel
                                  → fm.run() replays each non-None section
```

- A stage that hasn't run yet is `None` in the YAML - the produced config is "partial" and can be loaded + continued.
- `fm.evaluate(...)` calls `save_config()` automatically before returning. Call it manually at earlier points to checkpoint a partial pipeline.

### Default path

```
{station_dir}/forecast.config.yaml      # fm.save_config()
{station_dir}/forecast.config.json      # fm.save_config(fmt="json")
```

`{station_dir} = {output_dir}/{network}.{station}.{location}.{channel}` - sibling of the per-stage `cache/` directories.

### Round-trip

```python
fm.save_config("output/config.yaml")
fm2 = ForecastModel.from_config("output/config.yaml")
fm2.run()    # idempotent - replays every captured stage
```

---

## ForecastConfig Schema (YAML)

```yaml
# eruption-forecast ForecastModel configuration
version: "1.0"
saved_at: "2026-06-10T11:23:45"

model:
  station: OJN
  channel: EHZ
  network: VG
  location: "00"
  day_to_forecast: 2
  output_dir: null
  root_dir: null
  overwrite: false
  n_jobs: 8
  verbose: true

calculate:
  start_date: "2025-01-01"
  end_date: "2025-12-31"
  source: sds
  methods: [rsam, dsar, entropy]
  remove_outlier_method: maximum
  remove_tremor_anomalies: false
  interpolate: true
  plot_daily: true
  save_plot: true
  plot_overwrite: true
  sds_dir: "D:/Data/OJN"
  client_url: "https://service.iris.edu"
  minimum_completion_ratio: 0.3
  overwrite: false
  n_jobs: null               # null → inherit from model.n_jobs at replay
  verbose: null

train:
  start_date: "2025-01-01"
  end_date: "2025-07-26"
  eruption_dates:
    - "2025-03-20"
    - "2025-04-22"
  window_step: 6
  window_step_unit: hours
  label_builder: standard
  classifiers: [lite-rf, rf, gb, xgb]
  cv_strategy: shuffle-stratified
  cv_splits: 5
  scoring: recall
  top_n_features: 20
  include_eruption_date: true
  select_tremor_columns: [rsam_f2, rsam_f3, rsam_f4, dsar_f3-f4, entropy]
  save_tremor_matrix_per_method: true
  exclude_features: [agg_linear_trend, linear_trend_timewise, length]
  seeds: 25
  resample_method: under
  sampling_strategy: 0.75
  plot_features: true
  n_jobs: 4
  n_grids: 4
  use_cache: true

predict:
  start_date: "2025-07-27"
  end_date: "2025-08-22"
  window_step: 10
  window_step_unit: minutes
  save_seed_result: true
  plot_threshold: 0.7
  plot_pdf: true
  use_cache: false

evaluate:
  model: prediction
  plot_per_seed: true
  plot_aggregate: true

explain:
  model: prediction              # "prediction" | "training"
  eruption_dates: null           # null → reuse the dates captured during train()
  save_per_seed: true            # persist each per-seed shap.Explanation
  plot_per_seed: true            # bar + beeswarm per seed
  figsize: null                  # null → auto-size from max_display
  max_display: 20
  group_remaining_features: false
  dpi: 150
  check_additivity: false        # forwarded to shap.TreeExplainer
  overwrite_classifier_explanation: false
  output_dir: null
  overwrite: null
  n_jobs: null
  verbose: null
```

The keys mirror the kwargs accepted by each method 1:1 - see [API Reference](API-Reference) for the per-stage signatures.

### ForecastExplainConfig fields

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `model` | `Literal["training", "prediction"]` | `"prediction"` | Which upstream stage to explain |
| `eruption_dates` | `list[str] \| None` | `None` | Falls back to `train()` dates at replay |
| `save_per_seed` | `bool` | `True` | Persist `shap_values/{seed:05d}.pkl` per seed |
| `plot_per_seed` | `bool` | `True` | Bar + beeswarm per seed under `classifiers/{Clf}/figures/` |
| `figsize` | `tuple[float, float] \| None` | `None` | Auto-sized when `None` |
| `max_display` | `int` | `20` | tsfresh labels truncated to this many in plots |
| `group_remaining_features` | `bool` | `False` | Forwarded to `shap.plots.beeswarm` |
| `dpi` | `int` | `150` | Figure resolution |
| `check_additivity` | `bool` | `False` | Forwarded to `shap.TreeExplainer` |
| `overwrite_classifier_explanation` | `bool` | `False` | Overwrite cached `ClassifierExplanation_*.pkl` |
| `output_dir` | `str \| None` | `None` | Inherits from `ForecastModel` |
| `overwrite` | `bool \| None` | `None` | Inherits from `ForecastModel` |
| `n_jobs` | `int \| None` | `None` | Inherits from `ForecastModel` |
| `verbose` | `bool \| None` | `None` | Inherits from `ForecastModel` |

### `None`-as-inherit

For `overwrite`, `n_jobs`, and `verbose`, a YAML value of `null` means "inherit the 
value `ForecastModel.__init__` was constructed with". This is the same semantics 
applied at runtime when the kwarg is omitted, so a replay behaves identically.

---

## Per-stage configs (Standalone)

Every stage model (`TrainingModel`, `PredictionModel`, `EvaluationModel`, 
`ExplanationModel`) captures its own `__init__` surface into a matching dataclass 
under `config/` and exposes `save_config(path=None, fmt="yaml")`. Each main run 
method auto-calls `save_config()` once its primary artefacts are written, so a 
standalone run always leaves a YAML snapshot behind without any extra wiring.

| Model | Config dataclass | Auto-save trigger | Default path |
|-------|------------------|--------------------|--------------|
| `TrainingModel` | `config/training_config.py` | end of `fit()` | `{training_dir}/training.config.yaml` |
| `PredictionModel` | `config/prediction_config.py` | end of `forecast()` | `{prediction_dir}/prediction.config.yaml` |
| `EvaluationModel` | `config/evaluation_config.py` | end of `evaluate()` | `{evaluation_dir}/evaluation.config.yaml` |
| `ExplanationModel` | `config/explanation_config.py` | end of `explain()` | `{explanation_dir}/explanation.config.yaml` |

`{evaluation_dir}` and `{explanation_dir}` are already mode-namespaced (`evaluation/training/` 
vs `evaluation/prediction/`, same for `explanation/`), so training-reuse and 
prediction-reuse configs never collide.

```python
tm.save_config()      # → {training_dir}/training.config.yaml
pm.save_config()      # → {prediction_dir}/prediction.config.yaml
em.save_config()      # → {evaluation_dir}/evaluation.config.yaml
xm.save_config()      # → {explanation_dir}/explanation.config.yaml
```

Each call wraps the YAML write in a `try/except` and only logs a warning if 
the dump fails — a read-only output directory can never regress the underlying 
`fit() / forecast() / evaluate() / explain()` run itself.

**Non-serializable inputs are reduced to string handles**: `tremor_data` is 
emitted as `null` when a pre-loaded `pd.DataFrame` was passed (and as the CSV 
path otherwise); the upstream `model` parameter on `EvaluationConfig` / 
`ExplanationConfig` is **intentionally omitted** since it is always a live 
`TrainingModel` / `PredictionModel` instance. `PredictionConfig.model` keeps 
the path when the user passed one and `null` otherwise.

See [Training Workflow](Training-Workflow#standalone-use), 
[Prediction Workflow](Prediction-Workflow), 
[Evaluation Workflow](Evaluation-Workflow), and 
[Explanation Workflow](Explanation-Workflow) for the per-stage signatures.

---

## `config.example.yaml`

A fully annotated example config ships at the repo root: [`config.example.yaml`](https://github.com/martanto/eruption-forecast/blob/master/config.example.yaml). 
Project Rule 11 keeps it in sync with `forecast_config.py` - when any `ForecastConfig` 
field is added, renamed, or has its default changed, the example YAML is updated in the same commit.

---

## Telegram Notifications

`eruption_forecast.decorators` exposes two complementary primitives.

### `notify` decorator

Wraps a function to send a Telegram message on success or failure:

```python
from eruption_forecast import notify
import dotenv; dotenv.load_dotenv()

@notify("Run Forecasting")
def main():
    fm = ForecastModel(...)
    fm.calculate(...).train(...).predict(...).evaluate(...)

main()    # Telegram chat receives start, finish, and error messages
```

### `send_telegram_notification(...)` helper

Used by `scenarios.py` to ship the per-scenario forecast plot:

```python
from eruption_forecast import send_telegram_notification

send_telegram_notification(
    message=f"{name}: {description}",
    files=[fm.PredictionModel.forecast_plot_path],
    file_caption=f"{name}: {description}",
    send_as_document=True,        # preserves DPI - Telegram does not re-encode
)
```

### Credentials (`.env`)

```dotenv
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

- Bot token from [@BotFather](https://t.me/BotFather)
- Chat ID from [@userinfobot](https://t.me/userinfobot)

Both primitives degrade gracefully when the env vars are absent - they emit a warning and skip the network call instead of raising.

---

## Logging

The package wraps [`loguru`](https://github.com/Delgan/loguru) behind `eruption_forecast.logger`.

| Function | Purpose |
|----------|---------|
| `enable_logging()` | Restore console + file handlers using the current log directory |
| `disable_logging()` | Remove every active loguru handler - no console, no file |
| `set_log_level(level)` | Change the console handler level (`"DEBUG"` / `"INFO"` / `"WARNING"` / `"ERROR"` / `"CRITICAL"`) |
| `set_log_directory(dir)` | Move the log file to a new directory - created if missing |

```python
from eruption_forecast import enable_logging, disable_logging
from eruption_forecast.logger import set_log_level, set_log_directory

set_log_directory("logs/2026-06-10")
set_log_level("DEBUG")        # console only - file handlers keep their level

disable_logging()
fm.calculate(...)             # silent - useful during tests
enable_logging()              # restore handlers
```

`enable_logging`, `disable_logging`, `notify`, and `send_telegram_notification` are exported from the package root.

---

## Where Configuration Lives in the Filesystem

```
{station_dir}/
├── forecast.config.yaml                                # fm.save_config()  - full pipeline
├── training/training.config.yaml                       # tm.save_config()  - standalone TrainingModel
├── prediction/prediction.config.yaml                   # pm.save_config()  - standalone PredictionModel
├── evaluation/{training|prediction}/evaluation.config.yaml   # em.save_config()  - standalone EvaluationModel
├── explanation/{training|prediction}/explanation.config.yaml # xm.save_config()  - standalone ExplanationModel
│   # Cache identity dumps (diff-able JSON sidecars) live next to each
│   # stage's cache pickle — no central cache/ subtree:
│   #   training/{hash}.TrainingModel.params.json
│   #   prediction/{hash}.PredictionModel.params.json
│   #   explanation/{kind}/{hash}.ExplanationModel.params.json
└── ...
```

The `*.params.json` files next to each stage's cache pickle capture **exactly** what went into the cache hash. 
They are handy when debugging a cache miss - `diff` two of them to see which kwarg differed.
