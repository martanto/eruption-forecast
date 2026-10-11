# Local research application implementation

Applications live in `app/backend/` and `app/frontend/`; scientific code stays in `src/eruption_forecast/`. Branch: `experiment/ui`.

## Progress
- [x] Preserve local changes and create the requested branch from master.
- [x] Backend contracts, complete parameter inventory, OpenAPI and CLI.
- [x] Persistent SQLite queue, subprocess workers, independent stages and scenarios.
- [x] Ten-page React interface, parameter forms, availability and artifact viewers.
- [x] Results comparisons, cancellation, retry and restart recovery.
- [x] Backend, frontend and Playwright checks; comprehensive documentation.

## Decisions
Localhost, single researcher, no accounts. FastAPI/Pydantic/Loguru/Click/Uvicorn and SQLite. React/TypeScript/Vite/Tailwind v4/ECharts. Light neutral interface. SDS and FDSN. Existing artifact discovery. Every constructor and stage parameter, shared settings and per-scenario overrides. One heavy job at a time, explicit retry, partial completion after scenario failures. Package-generated SHAP and metric figures remain authoritative.

## Contracts
API resources under `/api/v1`; committed `openapi.json` at repository root; TypeScript generated from it. Runtime schema `/openapi.json`, Swagger `/docs`, ReDoc `/redoc`. Jobs return 202, persist resolved settings, events, logs and artifact manifests. Browser polls jobs every two seconds and system every five seconds.

## Acceptance
Parameter/default drift, schema/type drift, configuration round trips, field validation, transactional claims, process-tree cancellation, restart recovery, explicit retries, caches, stage reuse, scenario overrides, partial failures, completeness gaps, artifact compatibility, path containment, keyboard and responsive navigation. Backend tests/output stay in `app/backend/tests/`; frontend tests/output in `app/frontend/tests/`; scientific tests in root `tests/`. Playwright CLI captures stay in `.playwright/`. Run import checks, Ruff, ty and relevant pytest/frontend checks. Optional OJN calculation smoke: 2025-03-16 to 2025-03-22, eruption 2025-03-20. Model smoke uses a longer window to satisfy package label constraints.

## Delivered structure and checks

- Backend: `app/backend/src/eruption_forecast_app/`, separate `pyproject.toml`/`uv.lock`, `uv_build`, local editable scientific dependency with PyPI alternative.
- Frontend: `app/frontend/src/`, Vite/Tailwind v4/React/TypeScript; modular ECharts loaded with Results.
- Contract: root `openapi.json` and generated frontend types; export/check commands documented in [UI Workflow](UI-Workflow.md).
- Verification: 30 backend tests, three frontend unit tests, seven Edge browser workflows; production frontend and backend wheel/source builds; Ruff and backend ty pass. Scientific import tests pass. Scientific ty reports two existing warnings in `utils/window.py`.
- Real SDS verification: completeness, tremor calculation, training, prediction, evaluation and SHAP explanation passed using `D:\Data\OJN`; reports and outputs are local under `app/backend/tests/output/sds-smoke/`.
- Independent stage submissions also passed, reusing saved upstream models. The real scenario retry succeeded with unchanged training cache timestamps after correcting date normalization in cache lookup. A real parent/child process termination test verifies cancellation reaches descendants.
- Impeccable review approved after JSON editor focus, mobile profile access and accessibility fixes. Captures: `.playwright/ui-desktop.png`, `.playwright/ui-mobile.png`. Design tokens: root `DESIGN.md` and `.impeccable/design.json`.
- Documentation: README, AGENTS, every wiki workflow page, backend/frontend READMEs. Setup and commands: [UI Workflow](UI-Workflow.md).

### General Setup refinement
- [x] Independent critique and live browser evidence archived.
- [x] Native SDS folder picker, absolute path guidance, and grouped settings.
- [x] Backend contracts, browser interactions, and desktop/mobile polish verification (33 backend tests; 8 browser tests; 3 unit tests).
