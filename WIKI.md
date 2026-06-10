# Wiki Rewrite Progress

**Branch:** `dev/wiki-full-rewrite` (based on `ft/metrics-ensemble`, NOT `master`)
**Started:** 2026-06-10
**Plan:** `C:\Users\Anto\.claude\plans\we-are-going-to-steady-squid.md`

Tick each row as the corresponding step finishes. Append a short note in the same line if anything deviates from the plan.

---

## Setup

- [x] Branch created: `dev/wiki-full-rewrite` (from `ft/metrics-ensemble`)
- [x] Source tree indexed (`src/eruption_forecast/**` — 64 `.py` files)
- [x] Old wiki pages audited (13 files; 5 to delete, 8 to rewrite, 5 new)
- [x] WIKI.md tracker created (this file)

## Delete superseded pages

- [x] `wiki/Installation.md` (folded into `Getting-Started.md`)
- [x] `wiki/Quick-Start.md` (folded into `Usage.md`)
- [x] `wiki/Classifiers-and-CV.md` (folded into `Training-Workflow.md`)
- [x] `wiki/Evaluation-and-Forecasting.md` (split into `Prediction-Workflow.md` + `Evaluation-Workflow.md`)
- [x] `wiki/Training-Workflows.md` (renamed to `Training-Workflow.md`)

## New / rewritten pages

| # | Page | Status | Note |
|---|------|--------|------|
| 0 | `wiki/Home.md` | [x] | Rewrite — nav table + ASCII pipeline |
| 1 | `wiki/Getting-Started.md` | [x] | NEW — merges Installation |
| 2 | `wiki/Data-Sources.md` | [x] | Rewrite — SDS vs FDSN diagram |
| 3 | `wiki/Usage.md` | [x] | NEW — Quick Start + Example |
| 4 | `wiki/Pipeline-Walkthrough.md` | [x] | Rewrite — main.py + scenarios.py |
| 5a | `wiki/Training-Workflow.md` | [x] | NEW — folds Classifiers-and-CV |
| 5b | `wiki/Prediction-Workflow.md` | [x] | NEW |
| 5c | `wiki/Evaluation-Workflow.md` | [x] | NEW |
| 6 | `wiki/Visualization.md` | [x] | Rewrite |
| 7 | `wiki/Configuration.md` | [x] | Rewrite |
| 8 | `wiki/Output-Structure.md` | [x] | Rewrite |
| 9 | `wiki/Architecture.md` | [x] | Rewrite — comprehensive class diagram |
| 10 | `wiki/API-Reference.md` | [x] | Rewrite — every public class |

## Verification

- [x] Internal-link smoke test (Grep `wiki/*.md` for dead old links) — 0 matches
- [x] API-table smoke test (signature cross-check against `src/`) — verified during writing
- [x] Wiki-render dry run (every nav link target exists) — every `Home.md` link present in `wiki/`
- [x] Final `Glob wiki/*.md` = 13 files
- [x] `changelogs/2026-06-10.md` updated

---

## Authoritative source surface

Verified during indexing:

```
src/eruption_forecast/
├── __init__.py, logger.py, data_container.py
├── config/        (base_config, constants, forecast_config, training_config)
├── dataclass/     (station_data)
├── decorators/    (decorator_class, notify)
├── ensemble/      (base_ensemble, seed_ensemble, classifier_ensemble, metrics_ensemble)
├── features/      (constants, features_builder, feature_selector, tremor_matrix_builder)
├── label/         (constants, label_builder, dynamic_label_builder, label_data, label_plots)
├── model/         (constants, base_model, cache_model, forecast_model, training_model,
│                   prediction_model, evaluation_model, classifier_model, classifier_comparator)
├── plots/         (styles, tremor_plots, feature_plots, evaluation_plots, forecast_plots)
├── sources/       (base, sds, fdsn)
├── tremor/        (calculate_tremor, rsam, dsar, shannon_entropy, tremor_data)
└── utils/         (array, dataframe, date_utils, formatting, ml, pathutils, validation, window)
```

Public exports from `eruption_forecast/__init__.py`:
`ForecastModel`, `TrainingModel`, `PredictionModel`, `EvaluationModel`, `CalculateTremor`,
`FeaturesBuilder`, `TremorMatrixBuilder`, `LabelBuilder`, `DynamicLabelBuilder`,
`LabelData`, `TremorData`, `enable_logging`, `disable_logging`, `notify`,
`send_telegram_notification`.
