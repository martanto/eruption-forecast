# Local research interface

The application wraps the workflows documented in this wiki. Scientific implementations remain in `src/eruption_forecast/`; the independently packaged HTTP application lives in `app/backend/` and the browser application in `app/frontend/`.

## Start the application

From the repository root, install the backend and frontend dependencies:

```powershell
uv sync --project app/backend --group dev
npm --prefix app/frontend install
```

Start the API and its queue worker, then start Vite in another terminal:

```powershell
uv run --project app/backend eruption-ui dev
npm --prefix app/frontend run dev
```

Open `http://127.0.0.1:5173`. For a compiled frontend, run `npm --prefix app/frontend run build`, then `uv run --project app/backend eruption-ui dev` and open `http://127.0.0.1:8000`. The `serve` command runs only HTTP; use `worker` in a separate terminal when managing the processes independently. A queued job runs only while a worker is online.

The backend uses `uv_build`, has its own `pyproject.toml`, `uv.lock`, and virtual environment, and installs `eruption-forecast` as a dependency. Development resolves that dependency to the editable repository package through `[tool.uv.sources]`. To resolve the published PyPI package instead, use `uv sync --project app/backend --no-sources`; this changes the backend lock resolution. The published package must expose the same stage API as the wiki and the schema snapshot. Backend wheels declare a normal `eruption-forecast` dependency without the development path mapping.

## Directory structure

```text
app/
  backend/
    pyproject.toml                 # Independent Python package; uv_build
    uv.lock
    src/eruption_forecast_app/
      main.py                     # FastAPI factory and static hosting
      cli.py                      # Click serve/dev/worker/export-openapi
      controllers/
      routes/
      schemas/                    # Explicit Pydantic stage contracts
      repositories/migrations/    # SQLite queue and metadata
      services/pipeline/          # Scientific package adapters
      workers/                    # Queue consumer, executor, cancellation
    tests/                        # Backend tests and optional real SDS smoke
  frontend/
    package.json
    package-lock.json
    src/
      api/generated/              # Types from root openapi.json
      components/forms/
      components/charts/
      components/artifacts/
      hooks/
      pages/
      styles/
    tests/                        # Unit tests and Playwright E2E tests
      e2e/
src/eruption_forecast/             # Scientific Python package
tests/                            # Scientific package tests
openapi.json                      # Reproducible API snapshot
.playwright/                      # Local Playwright CLI captures, ignored
.app-state/                       # Local database and job logs, ignored
```

## Pages and workflows

1. **General setup:** Save SDS, tremor, output and project-root directories; station identity; job parallelism; and common defaults. Scan completeness for the calculation date range using `data-availability`. View its CSV and calendar figure in Results. Browse and discover artifacts under configured roots.
2. **Scenario setup:** Edit shared stage parameters and sparse overrides for each scenario. Add, duplicate, reorder or remove scenarios; inspect resolved values; enable evaluation and explanation. Calculation is shared once, or an existing tremor CSV can be reused. The OJN example is a retrospective smoke workflow with overlapping training/prediction periods, not an independent validation experiment.
3. **Calculate:** Select SDS or FDSN, constructor settings, frequency bands and run options. The real package computes RSAM, DSAR and entropy.
4. **Training:** Configure constructor, labels, feature extraction, externally loaded features, and fitting. The application defaults to saving complete stage models to support later reuse. The scientific package's defaults remain in the explicit schema contracts.
5. **Prediction:** Select a saved training model or supported ensemble; choose extracted, training-selected or externally loaded features; configure forecast outputs and plotting. A complete saved `TrainingModel` is required for training-selected feature mode.
6. **Evaluation:** Select a saved `TrainingModel` or `PredictionModel`, configure labels/metrics/comparison, and reuse fitted estimators. Prediction reuse requires eruption dates.
7. **Explanation:** Select a complete saved upstream stage model, configure SHAP execution and plotting, and inspect package-generated waterfalls, bars and beeswarms. Tree classifiers are supported as described in [Explanation Workflow](Explanation-Workflow.md).
8. **Results:** Search and filter actual artifacts; preview CSV/Parquet tables and time-series charts; display PNG/PDF figures; compare forecast CSVs; download outputs. Model pickles are indexed without deserialization; they are loaded only when selected as inputs to an explicit scientific job.
9. **Jobs:** Inspect lifecycle, stage events, immutable submitted/resolved settings and logs. Cancel queued/running work or explicitly retry a terminal attempt. Retry preserves the original workspace and package caches; both attempts retain artifact links.
10. **System:** Inspect actual CPU, memory, disk usage, queue counts and worker heartbeat.

Use the parameter search to find advanced fields. Nested method groups cover the constructors and public stage calls in the wiki. Reset returns to shared defaults; explicit `null`, `false`, and empty lists are preserved in sparse overrides. Named profiles and JSON/YAML import/export preserve the configuration. A local browser draft survives reloads.

## Queue, outputs and restart behavior

SQLite WAL stores configurations, jobs, events and artifact metadata. Atomic claims enforce one heavy job globally; `n_jobs` controls the scientific package's internal parallelism. A worker launches a scientific subprocess so HTTP stays responsive. Scenario failures are recorded while later scenarios continue; mixed success produces `partial`.

Each run uses `{output_dir}/ui-runs/{run_id}/`, with scenarios under `scenarios/{scenario_id}/`. Explicit stage output-directory overrides receive the same run/scenario namespace under the selected directory. Scientific subdirectories, cache identities and figure formats are retained. Application logs and submitted/resolved JSON are under `.app-state/jobs/{job_id}/`.

Cancellation terminates the executor process tree and preserves files already written. On worker restart, abandoned active attempts become `interrupted`; queued attempts remain queued. Retry is manual. Package cache hits reuse saved scientific stages; disabling `use_cache` skips stage lookup. Plot settings do not necessarily invalidate scientific caches, so use overwrite or disable cache when requesting regenerated figures.

Directory selection is explicit in General setup. File browsing/downloads and job inputs are restricted to those roots and application state. The server binds localhost and accepts local browser origins. Use `ERUPTION_APP_STATE` for isolated metadata and `ERUPTION_PROJECT_ROOT` when running an installed backend outside this checkout.

## API contract and tests

Swagger is at `/docs`, ReDoc at `/redoc`, and runtime OpenAPI at `/openapi.json`. Resources are under `/api/v1`: settings, parameters, configurations, paths, jobs, artifacts and system. Job submission returns HTTP 202.

```powershell
uv run --project app/backend eruption-ui export-openapi
npm --prefix app/frontend run generate:api
uv run --project app/backend eruption-ui export-openapi --check
npm --prefix app/frontend run check:api
uv run --project app/backend pytest app/backend/tests/ -o cache_dir=app/backend/tests/output/.pytest_cache --basetemp app/backend/tests/output/pytest-tmp
uv run --project app/backend ty check app/backend/src/
npm --prefix app/frontend test
npm --prefix app/frontend run test:e2e
uv run pytest tests/test_imports.py -v -o cache_dir=tests/output/.pytest_cache
```

Backend and frontend test outputs stay in their respective `tests/output/` directories. Playwright CLI captures stay in `.playwright/`.

Optional real-data checks read `D:\Data\OJN` and write only into backend test output:

```powershell
uv run --project app/backend python app/backend/tests/smoke_sds.py
uv run --project app/backend python app/backend/tests/smoke_pipeline.py
uv run --project app/backend python app/backend/tests/smoke_reuse.py
```

The calculation/completeness smoke uses 2025-03-16 through 2025-03-22. Model label building requires at least seven days between start/end, so the model smoke uses 2025-03-15 through 2025-03-23, one lite-rf seed and a small selected feature set. It verifies a retry leaves the saved training cache unchanged. The reuse smoke then verifies independently queued training, prediction, evaluation and explanation using the same tremor inputs and saved upstream stages. These checks establish integration behavior, not forecasting skill.

General Setup offers a Windows native SDS folder chooser through `POST /api/v1/paths/pick-directory`. The dialog opens on the backend computer and returns an existing absolute directory, or null on cancellation. Manual absolute-path entry remains available for remote or headless backends; selecting a folder updates the draft until Save setup is used.

If SDS selection is already open, use **Cancel folder selection** before retrying. `DELETE /api/v1/paths/pick-directory` terminates only the dialog owned by this backend; cancellation and timeout release the picker lock.
