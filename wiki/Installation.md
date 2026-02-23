# Installation

## Requirements

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
# Clone the repository
git clone https://github.com/martanto/eruption-forecast.git
cd eruption-forecast

# Install runtime dependencies
uv sync

# Install with development dependencies
uv sync --group dev
```

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `pandas >= 3.0.0` | Time-series data manipulation |
| `numpy` | Numerical computations |
| `scipy` | Signal processing |
| `obspy` | Seismic data reading (SDS/FDSN) |
| `tsfresh` | Automated time-series feature extraction (700+ features) |
| `scikit-learn` | Machine learning models and CV |
| `imbalanced-learn` | `RandomUnderSampler` for class imbalance |
| `xgboost` | XGBoost classifier |
| `joblib` | Parallel processing and model serialisation |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisation |
| `loguru` | Structured logging |
| `python-dotenv` | Load `.env` credentials for Telegram notify decorator |

### Development

| Package | Purpose |
|---------|---------|
| `ruff` | Linting and auto-fix (`uv run ruff check --fix src/`) |
| `ty` | Type checking (`uvx ty check src/`) |
| `pytest` | Testing (`uv run pytest tests/`) |

## Development Commands

```bash
# Lint and auto-fix
uv run ruff check --fix src/

# Type checking
uvx ty check src/

# Run all tests
uv run pytest tests/

# Run tests with coverage
uv run pytest --cov=src/eruption_forecast tests/

# Run the full pipeline example
uv run python main.py
```

## Telegram Notifications (Optional)

The `notify` decorator can send Telegram messages when long-running pipeline steps finish. To enable it, copy `.env.example` to `.env` and fill in your credentials:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Obtain a bot token from [@BotFather](https://t.me/BotFather) and your chat ID from [@userinfobot](https://t.me/userinfobot). See the [Configuration](Configuration) wiki page for full details.
