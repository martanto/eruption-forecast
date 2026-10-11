# AGENTS.md

_Last updated: 2026-09-17_

This file provides guidance when working with code in this repository.

## Python Package Manager
This package is using UV (https://docs.astral.sh/uv/) as package manager.

## Code Guidelines

- Always use available skills whenever possible when executing commands (e.g., use the scikit-learn skill for ML tasks, matplotlib/seaborn skills for plotting, etc.)

## Rules

1. **Log every completed task in the daily changelog.** Any finished task â€” bug fix, refactor, new feature, test, documentation change â€” must have its outcome recorded in `changelogs/YYYY-mm-dd.md` (using today's date) before moving on. Create the file if it does not exist. Append new entries; never overwrite previous entries in the same file. The `changelogs/` directory is git-ignored (local only).
2. **Type checker is `ty`.** Use `uvx ty check src/` for type checking. Always use forward slashes: `uvx ty check src/` (not `.\src`).
3. **Lint with ruff.** Use `uv run ruff check --fix src/` for linting.
4. **All `uv`, `uvx`, and `python` commands are permitted.** `uv sync`, `uv run`, `uv pip install/uninstall`, `uv lock`, `uvx ty check`, `python main.py`, etc. â€” no need to ask. User has granted permission to run these commands without approval.
5. **ALWAYS create a new branch before any commits or modifications.** Use `git checkout -b <prefix>/<branch-name>` to create a new branch before making ANY commits or code modifications. Choose the prefix based on the type of work: `fix/` for bug fixes (e.g., `fix/docstring-errors`), `ft/` for new features (e.g., `ft/add-fdsn-source`), and `dev/` as the default for everything else (e.g., `dev/refactor-utils`). Never work directly on main or master branches.
6. **Always use `tests` directory when running testing.** Create the tests output inside `tests` directory.
7. **Do not commit temporary test files.** Files starting with `test` in the root directory (e.g., `test_*.py`, `test.py`) should not be committed. These are temporary test scripts and are excluded via `.gitignore`.
8. **All module imports must be at the top of the file.** Never place `import` statements inside functions, methods, or conditional blocks. All stdlib, third-party, and local imports belong at the module level, grouped and sorted by ruff.
9. **No signatures.** Never add author attributions, or any name to any file.
10. **Run circular import test after every module change.** After adding, removing, or reorganising any module or import, run `uv run pytest tests/test_imports.py -v` to confirm no circular imports were introduced.
11. **Keep `config.example.yaml` in sync.** Whenever `src/eruption_forecast/model/forecast_model.py` or `src/eruption_forecast/config/forecast_config.py` is modified â€” new parameter added, removed, renamed, or default changed â€” update `config.example.yaml` to reflect the change. The YAML keys, defaults, and inline comments must match the current dataclass fields and `forecast_model.py` constructor calls exactly.
12. **Documentation updates are comprehensive.** When asked to add/update docs, update ALL of: `wiki/*.md`, `README.md`. Also update `AGENTS.md` for architecture/design changes. The `docs/` directory has been removed â€” `wiki/` is the single source of truth.
13. **Always `git checkout <branch>` before creating or working on a branch.** Never create a branch from the wrong base â€” verify current branch first with `git status` or `git branch`.
14. **Ask when unsure.** If the intent, scope, or correct approach is unclear, ask before proceeding.
15. **Keep `forecast_config.py` in sync with the new `ForecastModel`.** Whenever `src/eruption_forecast/model/forecast_model.py` or `src/eruption_forecast/config/forecast_config.py` is modified â€” new parameter added, removed, renamed, default changed â€” run `uvx ty check tests/test_forecast_config.py` to confirm the test file (which asserts every dataclass field and default) still type-checks against the current API. If `ty` flags drift, update both `tests/test_forecast_config.py` and `config.example.yaml` so the example YAML, the dataclass, and the constructor surface all match.
16. **Keep per-stage `*_config.py` in sync with each stage model's `__init__`.** Whenever a stage model constructor signature changes â€” parameter added, removed, renamed, default changed â€” update the matching dataclass and snapshot:
    - `training_model.py` â†” `config/training_config.py` (`TrainingConfig`) â€” snapshot lives in `TrainingModel._init_config(...)`.
    - `prediction_model.py` â†” `config/prediction_config.py` (`PredictionConfig`) â€” snapshot lives in `PredictionModel._init_config(...)`.
    - `evaluation_model.py` â†” `config/evaluation_config.py` (`EvaluationConfig`) â€” snapshot built inline at the end of `EvaluationModel.__init__`; upstream `model` parameter intentionally omitted.
    - `explanation_model.py` â†” `config/explanation_config.py` (`ExplanationConfig`) â€” snapshot built inline at the end of `ExplanationModel.__init__`; upstream `model` parameter intentionally omitted.

## Testing

Optional test data:

| Field | Value |
|-------|-------|
| Input directory (SDS) | `D:\Data\OJN` |
| Start / End | `2025-03-16` / `2025-03-22` |
| Eruption date | `2025-03-20` |

See `main.py` for a complete usage example of the pipeline.

## Project Overview

`eruption-forecast` is a Python package for volcanic eruption forecasting using seismic data analysis. The package processes seismic tremor data, extracts features, builds labels, and creates forecast models to predict volcanic eruptions based on time-series seismic measurements.

## Development Commands

```bash
uv sync                          # Install deps
uv sync --group dev              # + dev deps
uv run ruff check --fix src/     # Lint
uvx ty check src/                # Type check
uv run pytest tests/             # Run tests
uv run python main.py            # Run pipeline
```

## Architecture Overview

### Pipeline

`ForecastModel` orchestrates the full pipeline with method chaining:

```
CalculateTremor â†’ TrainingModel (label + features + multi-seed fit) â†’ PredictionModel â†’ EvaluationModel â†’ ExplanationModel
```

### Package Layout

All paths under `src/eruption_forecast/`.

| Module | File(s) | Purpose |
|--------|---------|---------|
| _root_ | `data_container.py` | `BaseDataContainer` â€” ABC for `TremorData`/`LabelData` |
| | `logger.py` | Loguru wrapper (enable/disable/set_level/set_directory) + per-category error files (`register_error_category`, `get_category_logger`; `telegram` category ships pre-registered) |
| `tremor/` | `calculate_tremor.py` | `CalculateTremor` â€” main orchestrator |
| | `rsam.py`, `dsar.py`, `shannon_entropy.py` | Tremor metrics |
| | `tremor_data.py` | `TremorData` â€” wraps tremor CSV |
| `label/` | `label_builder.py` | `LabelBuilder` â€” global sliding-window labelling |
| | `dynamic_label_builder.py` | `DynamicLabelBuilder` â€” per-eruption windows |
| | `label_data.py` | `LabelData` â€” wraps label CSV |
| `features/` | `features_builder.py` | `FeaturesBuilder` â€” tsfresh extraction |
| | `feature_selector.py` | `FeatureSelector` â€” 3-method selection |
| | `tremor_matrix_builder.py` | `TremorMatrixBuilder` â€” windowed alignment |
| `model/` | `forecast_model.py` | `ForecastModel` â€” pipeline orchestrator |
| | `forecast_model_scenario.py` | `ForecastModelScenario` â€” multi-scenario orchestrator; wraps one `ForecastModel`, shares one `CalculateTremor` + `shared_*` param dicts across N `Scenario` dicts |
| | `base_model.py` | `BaseModel` â€” ABC with cache-aware `save()`/`load()` + content-addressable identity helpers (`build_identity`, `compute_hash`, `_canonicalize`, `tremor_fingerprint`) |
| | `training_model.py` | `TrainingModel(BaseModel)` |
| | `prediction_model.py` | `PredictionModel(BaseModel)` |
| | `evaluation_model.py` | `EvaluationModel(BaseModel)` |
| | `explanation_model.py` | `ExplanationModel(BaseModel)` |
| | `classifier_comparator.py` | `ClassifierComparator` â€” cross-classifier rank |
| | `classifier_model.py` | `ClassifierModel` â€” estimator + grid |
| `ensemble/` | `base_ensemble.py` | `BaseEnsemble` â€” joblib save/load mixin |
| | `seed_ensemble.py` | `SeedEnsemble` â€” all seeds, 1 classifier |
| | `classifier_ensemble.py` | `ClassifierEnsemble` â€” N `SeedEnsemble`s + factories |
| | `metrics_ensemble.py` | `MetricsEnsemble` â€” per-seed metric engine |
| | `explainer_ensemble.py` | `ExplainerEnsemble` â€” per-seed SHAP engine |
| `sources/` | `base.py` | `SeismicDataSource` â€” ABC with `get(date)` |
| | `sds.py` | SDS (SeisComP) reader |
| | `fdsn.py` | FDSN client with local SDS caching |
| `config/` | `forecast_config.py` | `ForecastConfig` + per-stage sub-configs |
| | `training_config.py`, `prediction_config.py`, `evaluation_config.py`, `explanation_config.py` | Per-stage standalone `*Config` dataclasses backing each stage model's `save_config()` |
| `dataclass/` | `station_data.py` | `StationData` â€” immutable station identity |
| | `classifier_ensemble_summary.py` | `ClassifierEnsembleSummary`, `EruptionWindow`, `SeedSummary`, `ProbabilityPick` |
| | `classifier_explanation.py` | `SeedExplanation`, `ClassifierExplanation` â€” SHAP payloads |
| `plots/` | `styles.py`, `tremor_plots.py`, `feature_plots.py`, `evaluation_plots.py`, `forecast_plots.py`, `explanation_plots.py`, `label_plots.py` | Visualisation utilities (label distribution plots now live here â€” moved from `label/`) |
| `decorators/` | `notify.py` | Decorators (incl. Telegram `notify` and `timer`) |
| `utils/` | `array.py`, `window.py`, `date_utils.py`, `dataframe.py`, `ml.py`, `validation.py`, `pathutils.py`, `formatting.py` | Focused utility modules |

### Model Stack

| Class | Role | Inherits | Cache | Entry point |
|-------|------|----------|-------|-------------|
| `BaseModel` | Tremor loading, `output_dir`/`n_jobs` resolution, content-addressable cache identity helpers (`build_identity`, `compute_hash`, `_canonicalize`, `tremor_fingerprint`, `cache_path`) plus dual-mode joblib `save(identity=None, path=None)` / cache-only `load(stage_dir, identity)`. Cache writes land at `{stage_dir}/{hash}.{ClassName}.pkl` + `.params.json`; legacy mode (no identity) keeps the `{output_dir}/{ClassName}_{basename}.pkl` joblib dump for manual saves. | â€” | self | â€” |
| `TrainingModel` | Build labels â†’ extract features â†’ multi-seed GridSearchCV fit. Auto-merges seeds into `ClassifierEnsemble`. `fit()` builds its identity via `type(self).build_identity(...)` (kwargs from `self`) and calls `self.save(identity)`. | `BaseModel` | âœ“ | `fit()` |
| `PredictionModel` | Forecast over unlabelled window grid. Cache identity ties to `training_hash` (constructor param). `forecast()` builds its identity via `type(self).build_identity(...)` and calls `self.save(identity)`. | `BaseModel` | âœ“ | `forecast()` |
| `EvaluationModel` | Pre-trained ensemble vs ground truth. **training reuse** (pass `TrainingModel`) or **prediction reuse** (pass `PredictionModel`; `eruption_dates` required). Output namespaced to `evaluation/{training\|prediction}/`. `evaluate()` builds its identity via `type(self).build_identity(...)` and calls `self.save(identity)` when `save_model=True`; on a cache hit the metrics / MetricsEnsemble / comparator are restored without re-running the per-classifier `predict_proba` pass. | `BaseModel` | âœ“ | `evaluate()`, `compare()` |
| `ExplanationModel` | Per-seed SHAP explanations over a fitted `ClassifierEnsemble`. Reuses `TrainingModel` or `PredictionModel`; dispatches on `model.kind`. Tree classifiers only (RF / `lite-rf` / GB / XGB); non-tree classifiers are skipped with a warning. Output namespaced to `explanation/{training\|prediction}/`. `explain()` builds the identity, looks up the cache via `BaseModel.load`, and calls `self.save(identity)`. | `BaseModel` | âœ“ | `explain()`, `plot()` |
| `ForecastModel` | Orchestrator â€” `calculate` â†’ `train` â†’ `predict` â†’ `evaluate` â†’ `explain`. Captures stage kwargs into `_config` for `save_config()` / `from_config()` / `run()`. Threads `nslc` into `TrainingModel`/`PredictionModel` constructors and `training_hash` into `PredictionModel` so their cache identities match the external lookup `train()`/`predict()` use. | â€” | â€” | â€” |
| `ForecastModelScenario` | Multi-scenario orchestrator that wraps one `ForecastModel` and loops `train â†’ predict â†’ evaluate â†’ explain` over N `Scenario` `TypedDict`s. `calculate()` runs once; the shared tremor frame is reused by every scenario. `shared_training()` / `shared_prediction()` / `shared_evaluation()` / `shared_explanation()` capture kwargs into per-stage dicts; `run()` validates, writes `scenarios.json`, and drives the loop. Ships a Telegram document per scenario. | â€” | â€” | `run()` |

The legacy `model_trainer.py` / `model_predictor.py` / `base_model_trainer.py` / `evaluation_trainer.py` / `model_evaluator.py` / `multi_model_evaluator.py` / `metrics_computer.py` / `plots/shap_plots.py` stack has been removed.

### Ensemble Package

| Class | Purpose |
|-------|---------|
| `BaseEnsemble` | joblib `save(path)` / `load(path)` mixin |
| `SeedEnsemble` | All seed models + per-seed feature lists for one classifier. sklearn `BaseEstimator + ClassifierMixin`. `from_registry(registry_csv)`, `predict_proba(X)`, `predict_with_uncertainty(X, threshold)`. Exposes `self.features: list[str]` â€” sorted union of per-seed `feature_names` â€” populated by every factory (`from_any`, `from_json`, `from_registry`); empty on the bare constructor. Also caches per-seed `(n_samples, n_seeds)` matrices on `self.probabilities` / `self.predictions` â€” populated on-demand by `predict_with_uncertainty` via `save_matrices(output_dir, ...)`, which also writes `{classifier_name}_seed_probabilities.parquet` / `_seed_predictions.parquet`. Both are `None` on a bare load (from `from_any` / joblib pickle), because they are prediction-time artefacts â€” rehydrate them from those Parquet files via `load_matrices(probabilities_path, predictions_path)`. |
| `ClassifierEnsemble` | Wraps multiple `SeedEnsemble`s. Factories: `from_any()`, `from_json()`, `from_dict()`, `from_seed_ensembles()`, `from_registry_dict()`. Exposes `self.features: list[str]` â€” sorted union of `SeedEnsemble.features` across every registered classifier â€” populated by every factory (all delegate through `from_seed_ensembles`); empty on the bare constructor. |
| `MetricsEnsemble` | Per-seed metric engine over a fitted `ClassifierEnsemble`. `compute()` writes only `(n_samples, n_seeds)` `y_proba.csv` / `y_pred.csv` matrices under `classifiers/{clf}/predictions/`; `metrics` / `y_probas` / `y_preds` stay in memory. `save()` joblib-dumps the full instance to `{output_dir}/MetricsEnsemble.pkl` so `load()` round-trips without recomputation. Factory: `from_file(model_filepath, features_path, features_label_csv, eruption_dates, kind="prediction", ...)`. |
| `ExplainerEnsemble` | Per-seed SHAP engine driven by `shap.TreeExplainer`. `explain()` produces a `ClassifierExplanation` per tree classifier (RF / `lite-rf` / GB / XGB); non-tree classifiers are skipped with a warning. `plot_seed()` renders per-seed bar + beeswarm under `classifiers/{Clf}/figures/{bar,beeswarm}/{seed:05d}.png`; `plot_aggregate()` collapses every seed into the NaN-padded union feature space and renders per-classifier bar + beeswarm under `classifiers/{Clf}/figures/aggregate/` â€” `bar.{png,csv}` (frequency-weighted importance table) and `beeswarm.{png,parquet}` (tidy non-NaN cell list, Snappy-compressed Parquet because the tidy stack grows to millions of rows for realistic ensembles); `plot_waterfall(labels, eruption_dates)` renders per-eruption highest-probability waterfalls under the sibling `eruptions/{date}/` directory. |

`ensemble/__init__.py` re-exports `SeedEnsemble` and `ClassifierEnsemble`. Import `BaseEnsemble` from `eruption_forecast.ensemble.base_ensemble`, `MetricsEnsemble` from `eruption_forecast.ensemble.metrics_ensemble`, and `ExplainerEnsemble` from `eruption_forecast.ensemble.explainer_ensemble` â€” they are intentionally not re-exported to avoid cycles (`MetricsEnsemble â†’ utils.ml â†’ SeedEnsemble`; `ExplainerEnsemble â†’ plots.explanation_plots â†’ dataclass â†’ SeedEnsemble`). Never import from `eruption_forecast.model.metrics_computer` / `.model_evaluator` / `.multi_model_evaluator` (old paths removed).

### Tremor Calculation

`CalculateTremor` reads seismic data (SDS or FDSN) and computes three metrics across multiple frequency bands in parallel:

| Metric | Description |
|--------|-------------|
| RSAM | Real Seismic Amplitude Measurement â€” mean amplitude per band |
| DSAR | Displacement Seismic Amplitude Ratio â€” ratio between consecutive bands |
| Shannon Entropy | Single broadband signal-complexity column (`entropy`) |

Default frequency bands: `(0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz`. Supports `n_jobs` multiprocessing; outputs 10-minute interval CSVs. Optional params: `remove_outlier_method`, `remove_tremor_anomalies`, `interpolate`, `value_multiplier`, `cleanup_daily_dir`, `plot_daily`, `save_plot`, `filename_prefix`, `methods`.

```python
# SDS
CalculateTremor(station="OJN", channel="EHZ",
                start_date="2025-01-01", end_date="2025-01-03",
                n_jobs=4).from_sds(sds_dir="/path/to/sds").run()
# FDSN
CalculateTremor(...).from_fdsn(client_url="https://service.iris.edu").run()
# Columns: rsam_f0..f4, dsar_f0-f1..f3-f4, entropy
```

### Label Building

- **`LabelBuilder`** â€” sliding time-window binary labels (erupted=1/0). Uses `day_to_forecast` to look ahead N days. `include_eruption_date=False` (default) additionally marks the eruption day positive (`day_to_forecast + 1` total positive days). Label filenames: `label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv`.
- **`DynamicLabelBuilder`** (extends `LabelBuilder`) â€” three-phase per-eruption build: (1) zero frames per eruption, (2) concat + deduplicate datetimes, (3) mark positives per eruption.

Window config: `window_size` (days), `window_step` + `window_step_unit` (`minutes`/`hours`), `day_to_forecast`.

### Feature Extraction & Selection

- **`TremorMatrixBuilder`** â€” slices tremor time-series into windows aligned with labels. Validates sample counts per window; concatenates into a unified matrix with `id`, `datetime`, and tremor columns. `save_tremor_matrix_per_method=True` saves per-column matrices to `per_method/`.
- **`FeaturesBuilder`** â€” tsfresh extraction. Training mode (labels) filters windows + saves aligned labels; prediction mode disables relevance filtering. Per-column independent extraction. Every features artefact (per-column parquets under `extracted/` and the merged `features-matrix-dt_*.parquet`) is written with a `DatetimeIndex` derived from the labels' canonical window-start timestamps (fallback: the tremor matrix's first sample per window when the builder runs without a label frame). The sibling `features-label_*.csv` is still written as the label registry but is no longer required to attach datetimes downstream.
- **`FeatureSelector`** â€” single-stage. Methods: `"tsfresh"` (FDR p-value filter, default) or `"random_forest"` (permutation importance from an RF probe).

### `TrainingModel` Specifics

- Classifiers (11): `rf`, `gb`, `xgb`, `svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting`, `lite-rf`. `classifiers` accepts `str` or `list[str]`; one `SeedEnsemble` per classifier, bundled into a `ClassifierEnsemble`. After `fit()`, each `SeedEnsemble` exposes `self.features` â€” the sorted union of its per-seed `feature_names` (features actually used by at least one seed of that classifier) â€” and the bundling `ClassifierEnsemble` exposes `self.features` as the sorted union of every registered `SeedEnsemble.features` (features used by at least one seed of any classifier). Both are derived eagerly by every ensemble factory; distinct from `TrainingModel.features_selected_df.index`, which is the *pre-cut* ranked pool of all significant features (a superset of what any seed actually trained on).
- CV strategies: `shuffle`, `stratified`, `shuffle-stratified` (default). No `timeseries`.
- Imbalance: `resample_method: Literal["under", "over", "auto"] | None = "auto"`. `"auto"` picks `"under"` (`RandomUnderSampler`) when minority share < `minority_threshold` (default 0.15).
- Parallelism: `n_jobs` = outer seed workers; `n_grids` = inner `GridSearchCV` workers. `BaseModel.total_cpu` is `max(1, cpu_count() - 2)` â€” the 2-core OS reservation lives at the source, so every downstream clamp inherits it. `BaseModel._validate()` clamps `n_jobs` to `total_cpu` when the caller exceeds it. `TrainingModel.validate()` clamps `n_grids` so `n_jobs Ã— n_grids â‰¤ total_cpu`, boosts `n_grids` to `total_cpu` when both default to `1`, and pins `FeatureSelector.n_jobs = 1` whenever `n_jobs > 1` (tsfresh's raw `multiprocessing.Pool` has no nested-parallelism fallback of its own, whereas nested inner `GridSearchCV` is handled by joblib's own lokyâ†’threading fallback). `joblib.Parallel(backend="loky")`.
- Pipeline (each step returns `Self`): `build_label() â†’ extract_features() â†’ fit()`. `fit()` is the single training entry point and calls `self.save(self.build_identity())` once the ensemble is built; the resulting cache file lives at `{training_dir}/{hash}.TrainingModel.pkl` with a `.params.json` sidecar.
- Cache keyed off tremor DataFrame, label/feature/fit params, classifier list, and the optional `nslc` constructor param. `ForecastModel.train()` checks `TrainingModel.load(training_dir, identity)` first; on a hit the build is short-circuited entirely.
- Once `fit()` completes, the run's content-addressable hash is exposed on `self.id` regardless of `save_model` â€” consumers (including `ForecastModel._training_cache_hash`) read it directly instead of re-hashing the identity dict.

### `PredictionModel` Specifics

- Constructor `model` accepts: `ClassifierEnsemble` / `SeedEnsemble` instance, `ClassifierEnsemble.json` / `.pkl`, `SeedEnsemble_*.pkl`, or trained-model registry CSV â€” resolved via `ClassifierEnsemble.from_any()`.
- Constructor `nslc` and `training_hash` are threaded by `ForecastModel.predict()` so the prediction cache identity matches the external lookup. Standalone callers can omit both (the identity simply hashes with `None`).
- Pipeline: `build_label() â†’ extract_features() â†’ forecast()`. Writes per-seed + consensus probabilities under `prediction/results/` + forecast plot under `prediction/figures/`. `forecast()` calls `self.save(self.build_identity())` so the cache pickle lives at `{prediction_dir}/{hash}.PredictionModel.pkl` with a `.params.json` sidecar.
- `self.results` columns: `{classifier}_eruption_probability` / `_uncertainty` / `_confidence` / `_prediction` + `consensus_*`.
- Cache identity includes `training_hash`, so a re-trained ensemble invalidates downstream forecasts.

### `EvaluationModel` Specifics

- Dispatched on `model.kind`:
  - **training reuse** â€” `y_true` from `TrainingModel.labels`; `eruption_dates` optional.
  - **prediction reuse** â€” `eruption_dates` **required**; fresh `LabelBuilder` over the forecast window grid joined onto `PredictionModel.labels`.
- Output namespaced to `{output_dir}/evaluation/{training|prediction}/`.
- `EvaluationModel.from_file(filepath, ...)` (classmethod) accepts a `.pkl` from `TrainingModel.save()` / `PredictionModel.save()` â€” dispatches on loaded `kind`.
- Models are **never re-fit**. `evaluate()` builds a `MetricsEnsemble` once and caches it on `self.MetricsEnsemble`; the per-classifier `predict_proba` pass is amortised across `evaluate()` + any number of `compare()` calls. There is no per-seed JSON cache anymore â€” re-runs are gated by `MetricsEnsemble.compute()`'s in-memory idempotency fast-path (`overwrite` only controls plot regeneration).
- Cache identity mirrors `ExplanationModel`: an `_upstream_hash()` fingerprint (model kind, classifier names, features shape and columns, evaluation window, and `eruption_dates`) is combined with the `evaluate()` knobs (`plot_aggregate`, `plot_per_seed`, `compare_classifiers`) via `build_identity()`. `evaluate()` calls `EvaluationModel.load(evaluation_dir, identity)` before running work; on a hit `self.metrics` / `self.MetricsEnsemble` / `self.comparator` are restored and the per-classifier `predict_proba` pass is skipped entirely. With `save_model=True` the pickle plus `.params.json` sidecar land at `{evaluation_dir}/{hash}.EvaluationModel.pkl` (already mode-namespaced under `evaluation/{training|prediction}/`). The load path is gated on `use_cache and not self.overwrite`: passing `use_cache=False` (from either the standalone `evaluate(..., use_cache=False)` call or the `ForecastModel.evaluate(..., use_cache=False)` wrapper) skips the load short-circuit even when a cached pickle exists on disk, independent of `overwrite`.
- Prediction-reuse mode also writes `{evaluation_dir}/labels/y_true.csv` via `build_label()` (a fresh `LabelBuilder` joined onto `self.model.labels` by window id).
- `evaluate(plot_aggregate=True, plot_per_seed=False, compare_classifiers=True, use_cache=True) -> dict[str, pd.DataFrame]`: delegates the per-seed metric loop to `MetricsEnsemble.compute()` which persists per-classifier `predictions/{y_proba,y_pred}.csv` (matrices, one column per seed). Returns the per-classifier per-seed metrics dict and assigns it to `self.metrics`. `plot_aggregate=True` calls `MetricsEnsemble.plot_aggregate()` (writes `figures/aggregate/{plot_name}.{png,csv}` per classifier). `compare_classifiers=True` builds and retains a `ClassifierComparator` on `self.comparator`. `plot_per_seed` calls `MetricsEnsemble.plot_seed()` (writes `figures/{plot_name}/{seed:05d}.png` per classifier). `use_cache=False` skips the top-level `EvaluationModel.load(...)` short-circuit; SHAP rendering lives in the dedicated `ExplanationModel` stage.
- `compare(metrics=None) -> ClassifierComparator`: reuses the cached `MetricsEnsemble` from `evaluate()` (lazily builds it on first call) and writes ranking + cross-classifier plots under `{evaluation_dir}/comparison/`.

### `ExplanationModel` Specifics

- Reuses the upstream `TrainingModel` or `PredictionModel` and dispatches on `model.kind`. Output namespaced to `{output_dir}/explanation/{training|prediction}/` so the two modes never share a directory.
- TreeExplainer-only: tree classifiers (`rf`, `lite-rf`, `gb`, `xgb`) are explained per seed; non-tree classifiers (`svm`, `lr`, `nn`, `dt`, `knn`, `nb`, `voting`) are skipped at the `ExplainerEnsemble.explain()` loop with a warning so mixed ensembles still produce output for the qualifying classifiers.
- `ExplanationModel.from_file(filepath, eruption_dates=None, ...)` (classmethod) accepts a `.pkl` from `TrainingModel.save()` or `PredictionModel.save()`. Prediction-reuse waterfalls require `eruption_dates`; training-reuse works without.
- Cache identity is built from the upstream model fingerprint (`model_kind`, classifier names, features shape and columns, date range) plus `save_per_seed`. Cache hits restore `self.explanations` without re-running SHAP. Cache files land at `{explanation_dir}/{hash}.ExplanationModel.pkl` with a matching `.params.json` sidecar â€” and because `explanation_dir` is already mode-namespaced (`explanation/{training\|prediction}/`), training-reuse and prediction-reuse caches never collide. The load path is gated on `use_cache and not self.overwrite`: passing `use_cache=False` (from either the standalone `explain(..., use_cache=False)` call or the `ForecastModel.explain(..., use_cache=False)` wrapper) skips the load short-circuit even when a cached pickle exists on disk, independent of `overwrite`.
- `explain(save_per_seed=True, check_additivity=False, overwrite_classifier_explanation=False, use_cache=True) -> Self`: delegates to `ExplainerEnsemble.explain()`. Produces `ClassifierExplanation_*.pkl` per classifier under `classifiers/{ClfName}/` plus `shap_values/{seed:05d}.pkl` per seed when `save_per_seed=True`. `use_cache=False` skips the top-level `ExplanationModel.load(...)` short-circuit.
- `plot(figsize=None, max_display=20, group_remaining_features=False, dpi=150, plot_per_seed=True, plot_aggregate=True)`: renders per-eruption waterfalls (only when `eruption_dates` is set), per-seed bar + beeswarm plots under `classifiers/{ClfName}/figures/{bar,beeswarm}/{seed:05d}.png` when `plot_per_seed=True`, and per-classifier aggregate plots under `classifiers/{ClfName}/figures/aggregate/` â€” `bar.{png,csv}` and `beeswarm.{png,parquet}` â€” when `plot_aggregate=True`. The aggregate path stacks every seed into the NaN-padded union feature space; bar reports frequency-weighted mean |SHAP| with `selection_frequency` annotated, beeswarm shows the full sample Ã— seed distribution per feature.

### Helper Classes

| Class | File | Role |
|-------|------|------|
| `ClassifierModel` | `classifier_model.py` | sklearn-compatible estimator + hyperparameter grid; `slug_name`, `slug_cv_name` drive output paths. |
| `MetricsEnsemble` | `ensemble/metrics_ensemble.py` | Per-seed metric engine. `compute()` materialises `(n_samples, n_seeds)` `y_proba` / `y_pred` matrices and writes them as `classifiers/{clf}/predictions/{y_proba,y_pred}.csv` â€” no JSON, no per-seed metric files. `compute()` is idempotent once `y_probas` is populated. `plot_aggregate()` renders `figures/aggregate/{plot_name}.{png,csv}` per classifier (ROC, PR, threshold analysis, g-mean, MCC); `plot_seed()` renders per-seed plots under `figures/{plot_name}/{seed:05d}.png`. `save()` / `load()` round-trip the full instance via joblib (`MetricsEnsemble.pkl`). |
| `ClassifierComparator` | `classifier_comparator.py` | Cross-classifier rankings + plots. Constructor: `ClassifierComparator(metrics_ensemble, metrics=None, output_dir=None)`. Factory: `from_classifier_ensemble(classifier_ensemble, features_df, y_true, kind="training", ...)` (used by `EvaluationModel.compare()`). |

### Persistence & Cache

- **`BaseModel.save(identity=None, path=None) -> str`** â€” dual mode. With `identity` (cache mode): joblib-dumps to `{stage_dir}/{hash}.{ClassName}.pkl` plus a `{hash}.{ClassName}.params.json` sidecar. Without `identity` (legacy mode): joblib-dumps to `{output_dir}/{ClassName}_{basename}.pkl` for standalone manual saves; `path` overrides the default destination. Each cache-using subclass overrides the `stage_dir` property so artefacts land under `training/`, `prediction/`, or `explanation/{kind}/`.
- **`BaseModel.load(stage_dir, identity) -> Self | None`** â€” cache-only classmethod. Resolves the hash, returns the restored instance on a cache hit or `None` on a miss. Path-based restores live outside this API â€” use `eruption_forecast.utils.pathutils.load_pickle(path)` directly when you know a `.pkl` location (the existing `EvaluationModel.from_file()` and `ExplanationModel.from_file()` already do this).
- **Cache identity helpers** live on `BaseModel`: `build_identity(cls, **kwargs)` (override per subclass), `compute_hash(identity, length=12)`, `_canonicalize(value)`, and `tremor_fingerprint(df)`. `build_identity` is a single classmethod with explicit kwargs â€” there is no separate instance wrapper. Both `ForecastModel` (no instance yet) and the stage methods (state populated on `self`) call it with the same kwargs shape.
- Stage-internal save call sites: `TrainingModel.fit()`, `PredictionModel.forecast()`, `EvaluationModel.evaluate()`, and `ExplanationModel.explain()` each call `type(self).build_identity(...)` with kwargs pulled from instance attributes, hash it into `self.id` unconditionally, and then call `self.save(identity)` only when `save_model=True`. `ForecastModel.train()` / `.predict()` still handle the cache **lookup** externally via `TrainingModel.load(training_dir, identity)` / `PredictionModel.load(prediction_dir, identity)` so the expensive `build_label` / `extract_features` work is skipped on a hit, but the **write** is delegated to the stage method. The orchestrator threads `nslc` into both constructors and `training_hash` into `PredictionModel` so the externally-computed identity matches the one the stage method rebuilds for the save.
- **Uniform `self.id` attribute**: every cache-using stage exposes its content-addressable 12-char hex hash on `self.id` after its main method runs. Declared as `None` on `BaseModel` and populated by `fit()` / `forecast()` / `evaluate()` / `explain()` regardless of `save_model`, so downstream code can always inspect the identity without re-hashing the identity dict.
- **`BaseModel.save_config(path=None, fmt="yaml") -> str`** â€” abstract; every stage model implements it by delegating to `self._config.save(path, fmt)`. Defaults are stage-scoped: `{training_dir}/training.config.{fmt}`, `{prediction_dir}/prediction.config.{fmt}`, `{evaluation_dir}/evaluation.config.{fmt}` (already mode-namespaced under `evaluation/{kind}/`), `{explanation_dir}/explanation.config.{fmt}` (same for `explanation/{kind}/`). Each stage's main run method auto-calls `save_config()` once primary artefacts are written (`TrainingModel.fit()`, `PredictionModel.forecast()`, `EvaluationModel.evaluate()`, `ExplanationModel.explain()`) inside a `try/except` so a YAML-write failure cannot regress the run. `TrainingModel` and `PredictionModel` extract the snapshot logic into a private `_init_config(...)` static method; `EvaluationConfig` and `ExplanationConfig` omit the upstream `model` parameter (live instances are not serializable).
- **`merge_seed_models(registry_csv)`** in `utils/ml.py` writes `SeedEnsemble_{suffix}.pkl`; **`merge_all_classifiers(trained_models)`** produces a `ClassifierEnsemble`. `TrainingModel.fit()` invokes both internally; path stored on `self.classifier_ensemble_path`.

### `ForecastModel` Orchestrator

- Constructor: `(station, channel, network, location="", day_to_forecast=2, output_dir=None, root_dir=None, overwrite=False, n_jobs=1, verbose=False)`. Date ranges live on stage methods. `day_to_forecast` doubles as `window_size` threaded into `TrainingModel`/`PredictionModel`.
- Stage methods (each returns `Self`):
  - `calculate(start_date, end_date, source, ...)` â€” runs `CalculateTremor` (SDS or FDSN). Adjusts `start_date` backward by `day_to_forecast` days for full lead-in coverage.
  - `train(start_date, end_date, eruption_dates, window_step, window_step_unit, classifiers="rf", cv_strategy, cv_splits, scoring, top_n_features, seeds, resample_method, select_tremor_columns, exclude_features, ...)` â€” builds `TrainingModel` (threading `nslc=self.nslc`), runs `build_label â†’ extract_features â†’ fit`. The cache **write** is done inside `fit()` via `self.save(self.build_identity())`; the orchestrator only handles the cache **lookup** through `TrainingModel.load(training_dir, identity)`. Captures `ClassifierEnsemble` on `self.ClassifierEnsemble` and exposes `self._training_cache_hash` for downstream `predict()`.
  - `predict(start_date, end_date, window_step, window_step_unit, save_seed_result, plot_threshold, plot_title, plot_pdf, ..., **plot_kwargs)` â€” builds `PredictionModel` (threading `nslc=self.nslc` and `training_hash=self._training_cache_hash`), runs `build_label â†’ extract_features â†’ forecast`. The cache **write** happens inside `forecast()`; the orchestrator only checks the cache via `PredictionModel.load(prediction_dir, identity)`. The `training_hash` ties the prediction cache to its upstream training run so a re-trained ensemble automatically invalidates downstream forecasts.
  - `evaluate(model="prediction", eruption_dates=None, plot_per_seed=False, plot_aggregate=True, use_cache=True, ...)` â€” spins up an `EvaluationModel` in the chosen mode (passing `save_model=use_cache`) and calls `evaluation_model.evaluate(..., use_cache=use_cache)` so `use_cache` gates both the internal load and the write. `eruption_dates` falls back to `train()` dates when omitted. Auto-calls `save_config()` at the end.
  - `explain(model="prediction", eruption_dates=None, save_per_seed=True, plot_per_seed=True, plot_aggregate=True, figsize=None, max_display=20, group_remaining_features=False, dpi=150, check_additivity=False, overwrite_classifier_explanation=False, use_cache=True, ...)` â€” spins up an `ExplanationModel` (passing `save_model=use_cache`) and calls `.explain(..., use_cache=use_cache).plot(...)`, captures everything into `ForecastExplainConfig`. `eruption_dates` falls back to `train()` dates when omitted. Sets `self.ExplanationModel`.
- Cross-classifier comparison: `fm.EvaluationModel.compare()` after `fm.evaluate(...)`. `ForecastModel` does not wrap this on its own surface.
- Persistence: `save_config(path=None, fmt="yaml") -> str` (defaults to `{station_dir}/forecast.config.{fmt}`), `from_config(path) -> Self`, `run() -> Self` (replays captured stages; idempotent).
- Attributes by stage: `CalculateTremor` / `tremor_df` (calculate); `TrainingModel` / `ClassifierEnsemble` (train); `PredictionModel` / `results` (predict); `EvaluationModel` / `evaluation_results` (evaluate); `ExplanationModel` / `ExplanationModel.explanations` (explain).

**`ForecastConfig`** sub-configs (`config/forecast_config.py`): `BaseForecastConfig`, `ForecastCalculateConfig`, `ForecastTrainConfig`, `ForecastPredictConfig`, `ForecastEvaluateConfig`, `ForecastExplainConfig`. Each stage method auto-captures kwargs into the matching slot before default resolution, so `save_config()` round-trips through `from_config()` + `run()`.

### `ForecastModelScenario` Specifics

- Sibling orchestrator to `ForecastModel` (`model/forecast_model_scenario.py`). Wraps **one** `ForecastModel` instance and loops `train â†’ predict â†’ evaluate â†’ explain` over N `Scenario` `TypedDict`s. Not a `BaseModel` subclass; no cache identity of its own â€” it inherits every per-stage cache from the wrapped `ForecastModel`.
- Constructor: `(station, channel, network, eruption_dates, scenarios, location="", day_to_forecast=2, output_dir=None, scenarios_dir="scenarios", root_dir=None, overwrite=False, n_jobs=1, verbose=False)`. `scenarios_dir` is a child directory under `{station_dir}/`; per-scenario artefacts land at `{station_dir}/{scenarios_dir}/{slugify(name)}/`.
- Setup surface (each returns `Self`):
  - `calculate(...)` â€” delegates to `ForecastModel.calculate(...)` and auto-fills `plot_eruption_dates=self.eruption_dates`. Runs **once**; the tremor frame is reused by every scenario.
  - `shared_training(...)`, `shared_prediction(...)`, `shared_evaluation(...)`, `shared_explanation(...)` â€” capture kwargs into `self.shared_{stage}_params` dicts. The per-scenario `start_date` / `end_date` / `output_dir` are injected inside `run()` and are deliberately absent from these signatures.
  - `shared_training(...)` and `shared_prediction(...)` are **always required**; `validate()` raises `RuntimeError` for either empty dict.
  - `shared_evaluation(...)` and `shared_explanation(...)` are **conditionally required**: `run()` raises `RuntimeError` only when the matching flag (`evaluate_model=True` / `explain_model=True`, both defaults) is set and the params dict is empty. Sweeps that pass `evaluate_model=False` / `explain_model=False` do not need to call these methods at all.
- Loop entry point: `run(evaluate_model=True, explain_model=True, notification_message=None) -> None`. Two conditional preconditions fire first (raise `RuntimeError` if `evaluate_model=True` / `explain_model=True` but the matching `shared_*_params` dict is empty), then `validate()` enforces the always-required preconditions (unique slugs, `calculate()` ran, `shared_training` / `shared_prediction` populated). Order per scenario: `train â†’ predict â†’ Telegram send_document â†’ evaluate â†’ explain`; the `evaluate` / `explain` calls are gated on the same two flags, so `evaluate_model=False` / `explain_model=False` also act as per-sweep off-switches.
- **`use_features_from="files"` first-scenario coercion.** `run()` captures the caller's original `use_features_from` up-front, then force-sets the dict entry to `"all"` for `index == 0` so scenario 1 always extracts a fresh features matrix. When the caller *asked* for `"files"`, the loop captures scenario 1's `features_matrix_path` / `label_features_csv` from `fm.PredictionModel` and threads them into every later `fm.predict(...)` call â€” so scenarios 2..N reuse them instead of re-extracting.
- **Coverage constraint.** Because scenarios 2..N reuse scenario 1's features matrix, **scenario 1's prediction window must span every datetime any later scenario predicts over**. Order the sweep so the widest prediction window comes first, or leave `use_features_from="all"` and let every scenario re-extract.
- Manifest: `_save_scenarios()` writes `{scenarios_dir}/scenarios.json` inside `run()` after `validate()` succeeds â€” an auditable record of the sweep parameters that lives next to the per-scenario output dirs.

### Data Classes

| Class | File | Notes |
|-------|------|-------|
| `BaseDataContainer` | `data_container.py` | ABC for `TremorData`/`LabelData`. Provides `filename`, `basename`, `filetype` cached properties. |
| `TremorData` | `tremor/tremor_data.py` | Wraps tremor CSV. Start/end dates, sampling-rate validation. `@cached_property` throughout. |
| `LabelData` | `label/label_data.py` | Wraps label CSV. Parses params from filename. `@cached_property`. |
| `StationData` | `dataclass/station_data.py` | Immutable station identity. Fields: `station`, `channel`, `network`, `location` (str, default `""`), `channel_type` (default `"D"`). Derived: `nslc`, `nslct`. All uppercased in `__post_init__`. |
| `SeedExplanation` | `dataclass/classifier_explanation.py` | Frozen, slotted SHAP payload for one seed. Fields: `random_state` (int), `shap_values` (`shap.Explanation`). Produced per seed by `ExplainerEnsemble.explain_seed`. |
| `ClassifierExplanation` | `dataclass/classifier_explanation.py` | Mutable, slotted aggregate of `SeedExplanation`s for one classifier. Fields: `classifier_name` (str), `seeds` (`list[SeedExplanation]`). Mutable so `ExplainerEnsemble.explain_classifier()` can append seeds incrementally; consumed by every helper in `plots/explanation_plots.py`. |
| `ProbabilityPick` | `dataclass/classifier_ensemble_summary.py` | Frozen `(seed, sample)` pick from a per-seed probability matrix. Fields: `random_state` (int), `index` (int; row position aligning with the SHAP explanation for the same seed), `datetime` (`pd.Timestamp`), `value` (float). |
| `SeedSummary` | `dataclass/classifier_ensemble_summary.py` | Frozen top / bottom probability picks for one seed inside one eruption window. Fields: `random_state` (int), `highest` (`ProbabilityPick`), `lowest` (`ProbabilityPick`). |
| `EruptionWindow` | `dataclass/classifier_ensemble_summary.py` | Frozen per-eruption rollup. Fields: `eruption_date` (`YYYY-MM-DD` str), `highest` / `lowest` (`ProbabilityPick` across all seeds â€” guaranteed set by the builder), `seeds` (`list[SeedSummary]`). Custom `__repr__` reports seed count instead of the full list. |
| `ClassifierEnsembleSummary` | `dataclass/classifier_ensemble_summary.py` | Mutable per-classifier rollup of a `SeedEnsemble`'s probability matrix (one instance per classifier â€” no cross-classifier collapse). Fields: `classifier_name` (str), `eruption_windows` (`list[EruptionWindow]`, empty windows skipped), `highest` / `lowest` (`ProbabilityPick \| None` across seeds Ã— windows; `None` when nothing intersected). Built by `eruption_forecast.utils.ml.build_classifier_ensemble_summary`; consumed by `plots.explanation_plots.plot_classifier_waterfall`. |

### Utility Functions (`src/eruption_forecast/utils/`)

| Module | Key Functions |
|--------|---------------|
| `array.py` | `detect_maximum_outlier()`, `remove_outliers()`, `detect_anomalies_zscore()`, `aggregate_seed_probabilities()`, `predict_proba_from_estimator()` |
| `window.py` | `construct_windows()`, `calculate_window_metrics()` |
| `date_utils.py` | `to_datetime()`, `normalize_dates()`, `sort_dates()`, `parse_label_filename()`, `set_datetime_index()` |
| `ml.py` | `random_under_sampler()`, `get_significant_features()`, `load_labels_from_csv()`, `merge_seed_models()`, `merge_all_classifiers()`, `compute_seed_eruption_probability()`, `compute_model_probabilities()`, `get_classifier_models()`, `compute_g_mean()` |
| `validation.py` | `validate_random_state()`, `validate_date_ranges()`, `validate_window_step()`, `validate_columns()`, `check_sampling_consistency()` |
| `pathutils.py` | `resolve_output_dir()`, `ensure_dir()`, `save_figure()`, `save_data()`, `load_json()`, `load_pickle()`, `pdf_metadata()` |
| `dataframe.py` | `load_label_csv()`, `load_datetime_indexed()`, `load_features_matrix()`, `get_envelope_values()`, `remove_anomalies()`, `to_series()` |
| `feature_utils.py` | `load_select_features()`, `load_feature_aliases()`, `migrate_score_column()`, `update_top_features_csv()`, `concat_features()`, `concat_significant_features()`, `find_common_features()`, `merge_features_matrix()` |
| `formatting.py` | `slugify()`, `slugify_class_name()`, `shorten_feature_name()`, `humanize_feature_name()`, `get_classifier_label()` |

### Decorators (`src/eruption_forecast/decorators/`)

- `notify(task, message=None, to="telegram", on_success=True, on_error=True, timeout=3.0, verbose=False)` â€” Telegram success/error notifications; MarkdownV2 body includes hostname, timestamp, elapsed time, and (on error) exception details.
- `timer(name=None, send_to=None)` â€” logs the wrapped function's elapsed wall-clock time via `loguru`; when `send_to="telegram"` the same message is mirrored to Telegram.

Both decorators delegate to `TelegramNotification` (see `notification/` below). Credentials via `.env` (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`) or explicit constructor args.

### Notification (`src/eruption_forecast/notification/`)

- `TelegramNotification(token=None, chat_id=None, verbose=False)` â€” fluent-chain Telegram Bot API client. Every send method returns `self` so calls can be chained (`tn.send_message(...).send_document(...)`). Network failures are logged and swallowed.
    - `send_message(message, timeout=3.0)` â€” MarkdownV2 text via `sendMessage`.
    - `send_document(file, timeout=30.0, **kwargs)` â€” single file via `sendDocument`; no re-encoding, so DPI is preserved.
    - `send_photo(file, timeout=30.0, **kwargs)` â€” single image via `sendPhoto`; non-photo suffixes transparently fall back to `send_document`.
    - `send_media_group(files, kind="photo"|"document", caption=None, timeout=30.0, disable_notification=False)` â€” 2â€“10 items per album, larger inputs auto-chunked; caption attaches to the first item of the first album only. When `kind="photo"` and any input isn't a supported photo format, the whole batch downgrades to `kind="document"`.

## Important Patterns

### Method Chaining
Most builder classes support fluent interfaces:
```python
CalculateTremor(...).from_sds(sds_dir).change_freq_bands([(0.1, 1.0), (1.0, 5.0)]).run()
```

### Multiprocessing
`CalculateTremor` supports parallel processing via `n_jobs`. Each day's calculation runs independently.

### Data Leakage Prevention
Train/test split MUST happen before resampling and feature selection. The split is always performed inside the trainer, never externally.

### Data Format Standards

| Field | Format |
|-------|--------|
| Tremor CSV | DateTime index; cols `rsam_f0..f4`, `dsar_f0-f1..f3-f4`, `entropy` |
| Label CSV | DateTime index; cols `id` (int), `is_erupted` (0/1) |
| Features matrix parquet | DateTime index (named `datetime`); one column per tsfresh feature. Filename stem `features-matrix-dt_*.parquet` â€” the `-dt` suffix marks the DatetimeIndex-first format, so old integer-`id`-indexed caches (`features-matrix_*.parquet`) are ignored automatically |
| Per-seed resampled CSV | DateTime index (matching `features_df`); single `is_erupted` column â€” features rehydrated via `features_df.loc[datetimes]` in `load_features_resampled` |
| Date format | `YYYY-MM-DD` |
| Sampling interval | 10 minutes (default) |

### Frequency Band Naming

| Alias | Band (Hz) |
|-------|-----------|
| `f0`  | 0.01 â€“ 0.1 |
| `f1`  | 0.1 â€“ 2    |
| `f2`  | 2 â€“ 5      |
| `f3`  | 4.5 â€“ 8    |
| `f4`  | 8 â€“ 16     |

Columns: `rsam_f0`â€“`rsam_f4`, `dsar_f0-f1`â€“`dsar_f3-f4`, `entropy`.

### Feature Name Rendering

Canonical tsfresh names (`dsar_f3-f4__fft_coefficient__attr_"abs"__coeff_91`) are unreadable on plot axes. Three rendering options are available â€” reach for whichever fits the display surface:

| Style | Helper | Output shape | Best for |
|-------|--------|--------------|----------|
| Compact tsfresh | `shorten_feature_name(name)` in `utils/formatting.py` | `dsar_f3-f4 \| fft_coef(abs, 91)` | Bar-chart y-labels, SHAP plots (inherited via `ExplainerEnsemble.explain_seed` at ingest) |
| Plain English | `humanize_feature_name(name, freq_bands=None)` in `utils/formatting.py` | `Fourier coefficient (attr=abs, coeff=91) of DSAR ratio 4.5/8 Hz` | CSV `description` column, prose reports; accepts a `freq_bands` override when tremor bands differ from `DEFAULT_FREQUENCY_BANDS` |
| Rank-based alias | `alias` column in `top_features.csv` / `top_{N}_features.csv` / `common_top_features.csv`; look up via `load_feature_aliases(source)` | `ft_1`, `ft_2`, â€¦ | Crowded heatmaps and axes where even the shortened form doesn't fit; both cross-scenario heatmaps expose `label_style="alias"` |

Both alias and description are populated automatically by `concat_significant_features` and `find_common_features` when they write the ranked CSVs. For legacy CSVs written before that rollout, backfill them with `update_top_features_csv(csv_path)` â€” idempotent by default, `overwrite=True` to refresh (e.g. after changing `freq_bands`). Wiki: [Feature alias utilities](wiki/API-Reference.md#feature-alias-utilities).

### Output Directory Structure

Each pipeline stage writes under a single station directory (`{output_dir}/{nslc}/`) named from `network.station.location.channel`.

```
output/
â””â”€â”€ {network}.{station}.{location}.{channel}/         e.g., VG.OJN.00.EHZ/
    â”œâ”€â”€ tremor/                                        # CalculateTremor
    â”‚   â”œâ”€â”€ daily/                                     # Per-day CSVs (removed if cleanup_daily_dir=True)
    â”‚   â”œâ”€â”€ figures/                                   # Daily plots (if plot_daily=True)
    â”‚   â””â”€â”€ {nslc}_{start}_{end}.csv                   # Final merged tremor CSV
    â”‚
    â”œâ”€â”€ training/                                      # TrainingModel
    â”‚   â”œâ”€â”€ training.config.yaml                       # Auto-written at end of fit()
    â”‚   â”œâ”€â”€ features/{cv-slug}/
    â”‚   â”‚   â”œâ”€â”€ seed/                                  # Per-seed selected-feature CSVs
    â”‚   â”‚   â”‚   â””â”€â”€ figures/                           # Per-seed importance plots (if plot_features=True)
    â”‚   â”‚   â””â”€â”€ resampled/                             # Per-seed (datetime + is_erupted) â€” features recovered via features_df.loc[datetimes]
    â”‚   â””â”€â”€ classifiers/{clf-slug}/{cv-slug}/
    â”‚       â”œâ”€â”€ models/{seed:05d}.pkl                  # One pickle per seed
    â”‚       â”œâ”€â”€ trained_model_{suffix}.csv             # Per-classifier registry
    â”‚       â””â”€â”€ SeedEnsemble_{suffix}.pkl              # Auto-merged by fit()
    â”‚
    â”œâ”€â”€ prediction/                                    # PredictionModel
    â”‚   â”œâ”€â”€ prediction.config.yaml                     # Auto-written at end of forecast()
    â”‚   â”œâ”€â”€ features/                                  # Forecast-grid features
    â”‚   â”œâ”€â”€ results/{clf-slug}/                        # Per-seed probability CSVs (save_seed_result=True)
    â”‚   â””â”€â”€ figures/forecast_{start}_{end}.{png,pdf}   # Forecast plot
    â”‚
    â”œâ”€â”€ evaluation/                                    # EvaluationModel
    â”‚   â”œâ”€â”€ training/                                  # When model.kind == "training"
    â”‚   â”‚   â”œâ”€â”€ evaluation.config.yaml                 # Auto-written at end of evaluate()
    â”‚   â”‚   â”œâ”€â”€ classifiers/{clf-name}/
    â”‚   â”‚   â”‚   â”œâ”€â”€ predictions/                       # MetricsEnsemble.compute()
    â”‚   â”‚   â”‚   â”‚   â”œâ”€â”€ y_proba.csv                    # (n_samples, n_seeds) matrix
    â”‚   â”‚   â”‚   â”‚   â””â”€â”€ y_pred.csv                     # (n_samples, n_seeds) matrix
    â”‚   â”‚   â”‚   â””â”€â”€ figures/
    â”‚   â”‚   â”‚       â”œâ”€â”€ aggregate/{plot_name}.{png,csv}  # plot_aggregate=True
    â”‚   â”‚   â”‚       â””â”€â”€ {plot_name}/{seed:05d}.png     # plot_per_seed=True
    â”‚   â”‚   â”œâ”€â”€ comparison/                            # ClassifierComparator (ranking + plots)
    â”‚   â”‚   â””â”€â”€ MetricsEnsemble.pkl                    # Optional, via MetricsEnsemble.save()
    â”‚   â””â”€â”€ prediction/                                # When model.kind == "prediction"
    â”‚       â”œâ”€â”€ evaluation.config.yaml                 # Auto-written at end of evaluate()
    â”‚       â”œâ”€â”€ labels/y_true.csv                      # Built by EvaluationModel.build_label()
    â”‚       â”œâ”€â”€ classifiers/{clf-name}/...             # Same layout as training/
    â”‚       â””â”€â”€ comparison/
    â”‚
    â”œâ”€â”€ explanation/                                   # ExplanationModel
    â”‚   â”œâ”€â”€ training/                                  # When upstream model.kind == "training"
    â”‚   â”‚   â”œâ”€â”€ explanation.config.yaml                # Auto-written at end of explain()
    â”‚   â”‚   â”œâ”€â”€ classifiers/{clf-name}/
    â”‚   â”‚   â”‚   â”œâ”€â”€ ClassifierExplanation_{clf-name}.pkl   # Bundled SHAP payload
    â”‚   â”‚   â”‚   â”œâ”€â”€ shap_values/{seed:05d}.pkl             # Per-seed shap.Explanation (save_per_seed=True)
    â”‚   â”‚   â”‚   â””â”€â”€ figures/
    â”‚   â”‚   â”‚       â”œâ”€â”€ bar/{seed:05d}.png                 # plot_per_seed=True
    â”‚   â”‚   â”‚       â”œâ”€â”€ beeswarm/{seed:05d}.png            # plot_per_seed=True
    â”‚   â”‚   â”‚       â””â”€â”€ aggregate/                         # plot_aggregate=True
    â”‚   â”‚   â”‚           â”œâ”€â”€ bar.{png,csv}                  # frequency-weighted importance
    â”‚   â”‚   â”‚           â””â”€â”€ beeswarm.{png,parquet}         # NaN-padded union beeswarm (Snappy Parquet)
    â”‚   â”‚   â””â”€â”€ eruptions/{YYYY-MM-DD}/                    # Per-eruption waterfall sibling
    â”‚   â”‚       â””â”€â”€ {clf-name}_{datetime}_seed={i}_index={j}.png
    â”‚   â””â”€â”€ prediction/                                # When upstream model.kind == "prediction"
    â”‚       â”œâ”€â”€ explanation.config.yaml                # Auto-written at end of explain()
    â”‚       â””â”€â”€ (identical sub-tree)
    â”‚
    â”‚   # Cache pickles for the four cache-using stages now live next to
    â”‚   # each stage's other outputs â€” no separate cache/ subtree:
    â”‚   #   training/{hash}.TrainingModel.pkl + {hash}.TrainingModel.params.json
    â”‚   #   prediction/{hash}.PredictionModel.pkl + {hash}.PredictionModel.params.json
    â”‚   #   evaluation/{kind}/{hash}.EvaluationModel.pkl + {hash}.EvaluationModel.params.json
    â”‚   #   explanation/{kind}/{hash}.ExplanationModel.pkl + {hash}.ExplanationModel.params.json
    â”‚
    â”œâ”€â”€ forecast.config.yaml                           # ForecastModel.save_config()
    â”œâ”€â”€ forecast-results_{start}_{end}.csv # PredictionModel.forecast() top-level dump
    â”œâ”€â”€ TrainingModel_*.pkl                            # Optional legacy save â€” fm.TrainingModel.save() (no identity)
    â”œâ”€â”€ PredictionModel_*.pkl                          # Optional legacy save â€” fm.PredictionModel.save() (no identity)
    â”œâ”€â”€ EvaluationModel_*.pkl                          # Optional legacy save â€” fm.EvaluationModel.save() (no identity)
    â””â”€â”€ ExplanationModel_*.pkl                         # Optional legacy save â€” fm.ExplanationModel.save() (no identity)
```

**Slug mappings** (from `ClassifierModel.slug_name` / `slug_cv_name`):

| Classifier | Folder slug | | CV strategy | Folder slug |
|------|------|---|------|------|
| `rf` | `random-forest-classifier` | | `shuffle` | `shuffle-split` |
| `xgb` | `xgb-classifier` | | `stratified` | `stratified-k-fold` |
| `gb` | `gradient-boosting-classifier` | | `shuffle-stratified` | `stratified-shuffle-split` |
| `svm` | `svc` | | | |
| `lr` | `logistic-regression` | | | |

**File-name suffix** (`save_model_csv` in `utils/ml.py`): `{ClassifierName}_{CVName}_seeds-{seeds}_features-{number_of_features}`. Full registry CSV: `trained-model__{suffix}.csv`. Example: `trained-model__RandomForestClassifier_StratifiedShuffleSplit_seeds-500_features-20.csv`.

## Docstring & Comment Guidelines

- **Docstrings**: Google style. Use the `google-python-docstrings` skill when adding/updating. **Preserve double backticks** around variable names, parameter names, values (``None``, ``True``, ``False``), method calls, and attribute references. Never strip them during rewrites.
- **Comments**: Inline/block comments start â‰¥2 spaces from code, `#` then a space. Never describe *what* â€” explain *why*. Proper capitalisation, punctuation, grammar; read like narrative text.

## SHAP Plotting

Always pass `plot_size=None` to `shap.plots.beeswarm` â€” prevents SHAP from overriding pre-created `figsize`.

## Dependencies

| Library | Purpose |
|---------|---------|
| `obspy` | Seismic data processing (`Stream`, `Trace`) |
| `pandas` (â‰¥3.0.0) | Time-series manipulation |
| `tsfresh` | Automated time-series feature extraction |
| `numpy` | Numerical |
| `xgboost` (â‰¥3.x) | Gradient boosting |
| `shap` (â‰¥0.46) | Feature importance; uses Independent masker for XGBoost â‰¥3.x |

### Data Sources (`src/eruption_forecast/sources/`)

| Class | File | Purpose |
|-------|------|---------|
| `SeismicDataSource` | `base.py` | ABC with `get(date)` interface |
| `SDS` | `sds.py` | Local SeisComP Data Structure archive |
| `FDSN` | `fdsn.py` | FDSN web service with transparent local SDS caching |

## Common Workflows

### Complete Pipeline

```python
from eruption_forecast import ForecastModel

fm = ForecastModel(
    station="OJN", channel="EHZ", network="VG", location="00",
    day_to_forecast=2,            # threaded into TrainingModel/PredictionModel window_size
    root_dir="/path/to/project", n_jobs=4,
)

(
    fm.calculate(
        start_date="2020-01-01", end_date="2020-12-31",
        source="sds", sds_dir="/path/to/sds",
        methods=["rsam", "dsar", "entropy"],
    )
    .train(
        start_date="2020-01-01", end_date="2020-12-31",
        eruption_dates=["2020-06-15"],
        window_step=6, window_step_unit="hours",
        classifiers=["rf", "xgb"],          # str or list[str]
        cv_strategy="shuffle-stratified",   # shuffle | stratified | shuffle-stratified
        cv_splits=5, seeds=500, top_n_features=20,
        select_tremor_columns=["rsam_f2", "rsam_f3", "dsar_f3-f4", "entropy"],
        resample_method="auto",             # under | over | auto | None
        use_cache=True,
    )
    .predict(
        start_date="2020-07-01", end_date="2020-07-31",
        window_step=10, window_step_unit="minutes",
        plot_threshold=0.5,
    )
    .evaluate(
        model="prediction",                  # "prediction" (default) or "training"
        eruption_dates=["2020-06-15"],       # optional; falls back to train() dates
        plot_aggregate=True,
    )
    .explain(
        model="prediction",                  # "prediction" (default) or "training"
        eruption_dates=["2020-06-15"],       # optional; falls back to train() dates
        plot_per_seed=True,                  # bar + beeswarm per seed
        plot_aggregate=True,                 # per-classifier aggregate bar + beeswarm
        max_display=20,
    )
)

# Cross-classifier ranking + plots (writes under evaluation/{mode}/comparison/)
comparator = fm.EvaluationModel.compare()
print(comparator.get_ranking())
comparator.plot_all()
```

See `main.py` for a full working example.

### Evaluate Training vs Prediction Windows

```python
fm.evaluate(model="training", plot_aggregate=True)                       # in-sample
fm.evaluate(model="prediction", eruption_dates=["2020-06-15"])           # forecast window
metrics_per_classifier: dict[str, pd.DataFrame] = fm.evaluation_results
```

### Standalone Evaluation From `.pkl`

```python
from eruption_forecast import EvaluationModel

em = EvaluationModel.from_file(
    "output/VG.OJN.00.EHZ/PredictionModel_2025-03-16_2025-03-22.pkl",
    eruption_dates=["2025-03-20"],          # required for prediction reuse
)
metrics = em.evaluate(plot_aggregate=True)
comparator = em.compare()
```

### Working With Existing Data

```python
from eruption_forecast import TremorData, LabelData

tremor_data = TremorData(tremor_csv="path/to/tremor.csv")
label_data = LabelData(label_csv="path/to/label.csv")
print(label_data.parameters)  # all params from filename
```

### Persisting Stage Outputs

```python
fm.TrainingModel.save()      # â†’ {output_dir}/TrainingModel_{basename}.pkl
fm.PredictionModel.save()    # â†’ {output_dir}/PredictionModel_{basename}.pkl
fm.EvaluationModel.save()    # â†’ {output_dir}/EvaluationModel_{basename}.pkl

ensemble_path = fm.TrainingModel.classifier_ensemble_path
```

### Saving and Replaying Config

`ForecastModel.evaluate()` calls `save_config()` automatically. Explicit calls work at any stage.

```python
fm.save_config()                                              # â†’ {station_dir}/forecast.config.yaml
fm.save_config(fmt="json")                                    # â†’ forecast.config.json
fm.save_config("/tmp/custom.yaml")                            # custom path

fm2 = ForecastModel.from_config("output/VG.OJN.00.EHZ/forecast.config.yaml")
fm2.run()
```

Every stage model also exposes its own `save_config()` and auto-calls it at the end of its main run method, so a standalone run always leaves a YAML snapshot next to its artefacts:

```python
tm.save_config()      # â†’ {training_dir}/training.config.yaml       â€” auto at end of fit()
pm.save_config()      # â†’ {prediction_dir}/prediction.config.yaml   â€” auto at end of forecast()
em.save_config()      # â†’ {evaluation_dir}/evaluation.config.yaml   â€” auto at end of evaluate()
xm.save_config()      # â†’ {explanation_dir}/explanation.config.yaml â€” auto at end of explain()
```

`{evaluation_dir}` and `{explanation_dir}` are already mode-namespaced (`evaluation/{kind}/` and `explanation/{kind}/`), so training-reuse and prediction-reuse configs never collide. Each auto-call is wrapped in `try/except` so a YAML-write failure cannot regress the underlying run.

## GitHub Wiki

- Local pages: `D:\Projects\eruption-forecast\wiki\` â€” 13 pages (Home, Getting-Started, Data-Sources, Usage, Pipeline-Walkthrough, Training-Workflow, Prediction-Workflow, Evaluation-Workflow, Visualization, Configuration, Output-Structure, Architecture, API-Reference). Full rewrite landed on master via PR #109 (commit `6c00420`) â€” anchored to the post-`ft/metrics-ensemble` model stack.
- Remote: `https://github.com/martanto/eruption-forecast.wiki.git`. Sync workflow: clone the wiki repo as a sibling directory, copy `wiki/*.md` to its root, commit + push to `master`.
- Sidebar / nav structure is documented under `wiki/Home.md` â€” keep it in sync with the file list when adding or removing pages.

## Windows Development Notes

- Use `os.path.join()` for cross-platform paths.
- Use forward slashes for `uvx ty check src/` (not `.\src` â€” causes a path error on Windows).
## Local UI application architecture

The scientific distribution remains rooted at `pyproject.toml` and `src/eruption_forecast/`. `app/backend/pyproject.toml` defines the separate `eruption-forecast-app` distribution using `uv_build`, with source in `app/backend/src/eruption_forecast_app/`. Its UV development source resolves the scientific dependency to the editable repository root; published wheels use the normal PyPI dependency.

The backend contains controllers, routes, explicit Pydantic schemas, SQLite repositories/migrations, scientific adapters and subprocess workers. `app/frontend/` contains React/TypeScript/Vite/Tailwind v4/ECharts. Backend tests are in `app/backend/tests/`; frontend tests are in `app/frontend/tests/`; scientific tests remain in root `tests/`. Test outputs belong under the respective `tests/output/`; Playwright CLI captures belong in `.playwright/`.

Run the backend with `uv run --project app/backend eruption-ui dev`. Generate root OpenAPI using `uv run --project app/backend eruption-ui export-openapi`, then regenerate frontend types with `npm --prefix app/frontend run generate:api`. Parameter changes require updating the explicit backend contracts, signature/default coverage tests and generated OpenAPI/types. `wiki/UI-Workflow.md` documents the application.

General Setup offers a Windows native SDS folder chooser through `POST /api/v1/paths/pick-directory`. The dialog opens on the backend computer and returns an existing absolute directory, or null on cancellation. Manual absolute-path entry remains available for remote or headless backends; selecting a folder updates the draft until Save setup is used.

If SDS selection is already open, use **Cancel folder selection** before retrying. `DELETE /api/v1/paths/pick-directory` terminates only the dialog owned by this backend; cancellation and timeout release the picker lock.
