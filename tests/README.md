# Tests

Unit and component tests for `eruption-forecast`, run with `pytest`.

Every test is self-contained. Inputs are synthetic and built in memory or written under pytest's `tmp_path`. No test reads a real SDS archive or needs network access, except the tests marked `integration`.

## Running

```bash
uv sync --group dev

# Full suite
uv run pytest tests/

# Fast suite: skip the end-to-end pipeline and real external services
uv run pytest tests/ -m "not slow and not integration"

# End-to-end pipeline tests only (a few minutes)
uv run pytest tests/ -m slow

# Coverage report with missing lines (configured in pyproject.toml)
uv run pytest tests/ -m "not integration" --cov

# Circular-import check (run after any module move)
uv run pytest tests/test_imports.py -v
```

## Markers

| Marker | Meaning |
|--------|---------|
| `integration` | Sends real external requests (e.g. the Telegram API). Needs credentials in `.env`. |
| `slow` | End-to-end tests that run tsfresh and fit models (`test_pipeline.py`, the scenario sweep). They take a few minutes. |

`--strict-markers` is on, so register any new marker in `pyproject.toml` before using it.

## Shared fixtures (`conftest.py`)

| Fixture / helper | Provides |
|------------------|----------|
| `synthetic_tremor_df` | Ten days of 10-minute tremor data with every default column (`rsam_f0..f4`, `dsar_*`, `entropy`). |
| `synthetic_tremor_csv` | The same frame written as a tremor CSV under `tmp_path`. |
| `synthetic_stream` / `build_stream(...)` | A one-day `VG.OJN.00.EHZ` ObsPy stream carrying a known sinusoid (default 3 Hz, 1000 counts, 20 Hz). |
| `features_and_labels` | A separable 60-row `(X, y)` pair indexed by window `id`. |
| `make_seed_ensemble(n_seeds, kind, ...)` | A fitted `SeedEnsemble` built from `LogisticRegression` (`kind="lr"`) or tiny `RandomForestClassifier` (`kind="rf"`) seeds. |
| `tiny_grids` | Module-scoped patch that collapses the `rf`, `lite-rf` and `nb` hyperparameter grids to one point, so `GridSearchCV` runs a single fit. |
| `eruption_ramp_tremor_df(...)` | Tremor whose first column ramps up over the three days before an eruption (default 2025-01-20), giving the models a real signal. |
| `seed_tremor_csv(output_dir, df)` | Writes the merged `tremor-interpolated_{nslc}_{start}-{end}.csv` that `ForecastModel.calculate()` reuses, so pipeline tests skip waveform processing. |

`conftest.py` also forces the matplotlib `Agg` backend, so plot code never opens a window.

## Layout

| Area | Files |
|------|-------|
| Tremor | `test_tremor_metrics.py` (RSAM / DSAR / entropy), `test_calculate_tremor.py`, `test_window_utils.py` |
| Data sources | `test_sources.py` (SDS round-trips; FDSN with a fake client) |
| Labels and features | `test_label_builder.py`, `test_dynamic_label_builder.py`, `test_label_plots.py`, `test_tremor_matrix_builder.py`, `test_features_builder.py`, `test_feature_selector.py`, `test_feature_count_sweep.py` |
| Models and cache | `test_base_model_cache.py`, `test_classifier_model.py`, `test_classifier_comparator.py` |
| Pipeline (`slow`) | `test_pipeline.py` (`ForecastModel` plus the four stage models, cache and config replay), `test_forecast_model_scenario.py` |
| Ensembles | `test_seed_ensemble.py`, `test_classifier_ensemble.py`, `test_metrics_ensemble.py`, `test_explainer_ensemble.py` |
| Configs | `test_forecast_config.py`, `test_stage_configs.py` (also guards config ↔ `__init__` drift) |
| Utilities | `test_utils.py`, `test_utils_array.py`, `test_utils_ml.py`, `test_utils_features.py`, `test_utils_misc.py` |
| Containers | `test_data_container.py`, `test_station_data.py` |
| Infrastructure | `test_imports.py`, `test_logger.py`, `test_notify_decorator.py`, `test_send_telegram_notification.py` |

## Recording known bugs

When a test exposes a bug that won't be fixed in the same change, mark it `@pytest.mark.xfail(strict=True, reason=...)`. Once the bug is fixed, the strict `xfail` turns into an `XPASS` failure. Then remove the marker.

## Adding tests

1. Name files `test_*.py` and keep all imports at the top of the module.
2. Build inputs from the `conftest.py` fixtures. Write every output under `tmp_path`, never under the working directory.
3. Fake external services with `monkeypatch` (see `test_sources.py` and `test_notify_decorator.py`). Tests that must reach the real service get `@pytest.mark.integration`.
4. Use the `tiny_grids` fixture for anything that calls `fit()`. The production grids take minutes per seed.
5. Ad-hoc scripts that touch tsfresh or joblib need an `if __name__ == "__main__":` guard on Windows. Without it, every spawned worker re-runs the script.
6. Lint before committing. `ruff.toml` excludes `tests/`, so `ruff check tests/` is a silent no-op. Pass the files you changed explicitly instead, e.g. `uv run ruff check --fix tests/conftest.py tests/test_pipeline.py`.
