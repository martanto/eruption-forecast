# Product

<!-- impeccable:product-schema 1 -->

## Platform
web

## Stack
React and TypeScript, Vite, Tailwind v4, ECharts; FastAPI and Pydantic, SQLite, Click and Uvicorn. Both applications live under app/.

## Users
A researcher operating a local Windows seismic archive, repeating and comparing volcanic eruption forecasting experiments.

## Product Purpose
Configure every supported pipeline parameter, run independent stages or multiple scenarios, and inspect reproducible results without editing Python scripts.

## Operating Context
Localhost only, no accounts. SDS and FDSN inputs. Long scientific jobs run independently of browser connections. Existing trusted local outputs can be reused.

## Capabilities and Constraints
Ten pages: General Setup, Scenario Setup, Calculate, Training, Prediction, Evaluation, Explanation, Results, Jobs, System. Shared scenario defaults and all-stage overrides. One heavy job at a time; explicit retry; continue other scenarios after failure. Wiki is the workflow source of truth.

## Brand Commitments
The user chose a new light, neutral scientific workspace, persistent page navigation, and expert stage forms.

## Evidence on Hand
Existing package APIs, forecast-scenario.py, wiki workflows, and optional OJN seismic data. Never fabricate scientific results.

## Product Principles
Expose effective settings. Preserve configuration and partial artifacts. Explain missing prerequisites. Keep scientific values and timestamps faithful to the package.

## Accessibility & Inclusion
Labeled controls, keyboard navigation, visible focus, readable contrast, and non-color job states.
