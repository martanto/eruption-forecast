# Getting Started

This page covers everything needed to install `eruption-forecast` and run the bundled examples. 
For per-stage detail jump to [Pipeline Walkthrough](Pipeline-Walkthrough); for an end-to-end snippet see [Usage](Usage).

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python ≥ 3.11** | Required by `pandas >= 3.0.0` and modern type-syntax used across the package |
| **[`uv`](https://docs.astral.sh/uv/)** | The project's only supported package manager. `pip` workflows are not tested |
| **Git** | Required for cloning the repository |
| **Seismic data archive** | An [SDS-formatted](Data-Sources#sds) local directory, **or** access to an FDSN web service (default: `https://service.iris.edu`) |
| **Eruption dates** | Known absolute eruption timestamps for the target volcano - supervised labelling is keyed off these dates |
| **Telegram bot** *(optional)* | Push notifications when long-running stages finish - see [Configuration](Configuration#telegram-notifications) |

---

## Installation

```bash
# 1. Clone
git clone https://github.com/martanto/eruption-forecast.git
cd eruption-forecast

# 2. Install runtime dependencies
uv sync

# 3. Install development dependencies (lint, type-check, tests)
uv sync --group dev
```

The first `uv sync` resolves and downloads every transitive dependency into `.venv/`. Subsequent runs are incremental.

### Verify the install

```bash
uv run python -c "from eruption_forecast import ForecastModel; print(ForecastModel)"
```

A working install prints `<class 'eruption_forecast.model.forecast_model.ForecastModel'>` and exits cleanly.

---

## Runtime Dependencies

| Package | Role |
|---------|------|
| `obspy` | Seismic-stream IO (`Stream`, `Trace`) |
| `pandas` (≥ 3.0.0) | Time-series manipulation and CSV IO |
| `numpy` | Numerical kernels for tremor metrics |
| `scipy` | Signal processing (filtering, FFT) |
| `tsfresh` | Automated extraction of 700+ time-series features |
| `scikit-learn` | Classifiers, CV splitters, GridSearchCV |
| `imbalanced-learn` | `RandomUnderSampler` for class imbalance |
| `xgboost` (≥ 3.x) | Gradient boosting |
| `shap` (≥ 0.46) | Model interpretability (used by evaluation plots) |
| `joblib` | Parallel workers + ensemble serialisation |
| `matplotlib`, `seaborn` | Plotting backends |
| `loguru` | Structured logging - wrapped by `eruption_forecast.logger` |
| `python-dotenv` | `.env` loading for Telegram credentials |

## Development Dependencies (`--group dev`)

| Package | Role |
|---------|------|
| `ruff` | Linting + autofix (`uv run ruff check --fix src/`) |
| `ty` | Type checking (`uvx ty check src/` - note the forward slash) |
| `pytest` | Test runner (`uv run pytest tests/`) |

---

## Development Commands

| Command | Purpose |
|---------|---------|
| `uv sync` | Refresh runtime dependencies |
| `uv sync --group dev` | Add dev tooling |
| `uv run ruff check --fix src/` | Lint and autofix |
| `uvx ty check src/` | Type check (always forward slashes on Windows) |
| `uv run pytest tests/` | Run the test suite |
| `uv run pytest tests/test_imports.py -v` | Confirm no circular imports |
| `uv run python main.py` | Run the bundled Research Workflow |
| `uv run python scenarios.py` | Run the bundled Scenarios Workflow |

All `uv`, `uvx`, and `python` commands are pre-approved in the project hooks - no permission prompt will be shown.

---

## Telegram Notifications (Optional)

The `notify` / `timer` decorators and the `TelegramNotification` client push a Telegram message when long stages finish or fail. Enable them by copying `.env.example` to `.env`:

```dotenv
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

- Obtain a bot token from [@BotFather](https://t.me/BotFather).
- Obtain your chat ID from [@userinfobot](https://t.me/userinfobot).

Full usage is documented in [Configuration → Telegram Notifications](Configuration#telegram-notifications).

---

## Next Steps

1. Confirm a seismic data archive is reachable - see [Data Sources](Data-Sources).
2. Run the [Usage](Usage) Quick Start to produce your first forecast.
3. Dive into [Pipeline Walkthrough](Pipeline-Walkthrough) for the annotated `main.py` / `scenarios.py` tours.
