# Architecture Explorer

Pipeline Atlas is a standalone, read-only guide in `app/`. It explains the current Python pipeline through a React Flow canvas, a source-backed inspector, five example scenarios and a guided tour. It does not execute training or access a backend.

## Run locally

Use Node.js 22.12+ and npm. From the repository root:

```sh
cd app
npm ci --legacy-peer-deps
npm run dev
```

Open the local URL printed by Vite. `npm run build` writes the static application to `app/dist/`. Images, fonts, content and source excerpts are bundled; a running Python service is unnecessary. No deployment is configured.

## Explore

The toolbar links to **Labelling**, **Training** and **Prediction**. Continue from `/labelling` (its link reads **Continue to Training**) to `/seed-ensemble` (the Training Guideline opens here), `/features-matrix`, `/random-under-sampler`, `/feature-selection`, `/grid-cv` and `/classifier-ensemble`. Each page uses four steps, synthetic fixtures, light/dark controls and the supplied illustration. Sampling seeds and family controls update examples locally. See [Training walkthroughs](Training-Walkthroughs.md) for calculations and implementation caveats.

Select a numbered stage to inspect its purpose, actual behavior, parameters, artifacts, implementation caveats and source excerpts. Expand stages to reveal internal steps. Search includes node text, parameters, artifacts and referenced source paths. **Caveats** filters the searchable steps and dims unaffected nodes.

**Pipeline** shows transformations. **Timeline** shows the five scenarios from the reviewed `forecast-scenario.py`, training/prediction periods and all eight dated eruptions. Historical feature lookback and label lookahead are illustrated separately from execution order. Scenario descriptions are not used to infer event counts.

View, scenario and selected node live in URL parameters. For example:

```text
/?view=timeline&scenario=scenario-3-2&node=consensus
```

Browser back/forward restores those values. Each view retains its viewport during the session. Expansion and viewports are client interaction state; canonical content and URL state are not duplicated in Zustand.

Drag or pinch the canvas, use the zoom controls, navigate the minimap, fit all nodes, or focus the selected stage. Tab and Enter select graph nodes. On narrow screens, **Stages & scenarios** and **Inspect selection** open focus-managed drawers. The tour supports previous/next controls and leaves free exploration available.

## Scientific interpretation

- Internal randomized CV and historical train/prediction boundaries are different. Overlapping windows and selection/resampling before CV do not establish leakage-free validation.
- Calculator exclusions and imputation exist. A dedicated correlation/zero/linear-value reduction stage was not found in the inspected path.
- FDR filtering and top-N p-value ranking are distinct; the fallback does not guarantee that every retained feature passes FDR.
- Lite-RF is documented using the implemented grid, not assumed parameter equivalence to Whakaari.
- Training-reuse evaluation reports fitted-model performance. Per-seed SHAP and aggregate plots explain constituent classifiers, not a separate consensus model.
- The consensus inspector labels its numerical calculation as illustrative. Mean probabilities, mean votes, population standard deviations and vote-based CI-like values are shown separately.

The script's RF/Lite-RF selection, two seeds, six-hour training stride and ten-minute prediction stride are **example settings**, not package defaults. The bundled July–August forecast image is a historical example and is not assigned to the selected scenario.

## Maintain the content

`app/src/features/architecture/` owns typed stage, node, edge, gap, source, scenario and tour records. Pipeline and Timeline consume that shared dataset through the feature's `index.ts`. Presentation features own their interactions; `ExplorerPage` and `ExplorerLayout` compose them.

`app/scripts/sources.mjs` specifies reviewed source files and excerpts. `sources.json` records the Git revision, working-tree status, review timestamp and SHA-256 fingerprints. Line endings are normalized before source hashing so Windows/Unix checkouts agree. Existing local edits are part of the reviewed snapshot.

When Python source, the scenario script, README or the referenced wiki pages change:

1. Run `npm run validate:sources` from `app/` to identify stale references.
2. Review the changed implementation and update canonical content, scenario dates, caveats, excerpts and tests as necessary.
3. Run `npm run snapshot:sources` to capture the reviewed tree, then inspect the generated diff.
4. Run the checks below. Never refresh hashes simply to silence a failure: the drift check does **not** synchronize semantics.

## Labelling walkthrough

Open **Labelling** in the Atlas toolbar (`/labelling`) to follow Continuous standard labels, dynamic per-eruption labels, and the tremor matrix. Each chapter has four selectable steps, previous/next controls and a restart action. Select a six-hour label cell or move between windows to inspect its timestamp and id.

The shared example uses a two-day tremor look-back, a six-hour slide and three positive calendar days including eruption day (`day_to_forecast=3`, `include_eruption_date=True`). Synthetic eruptions occur on 20 and 25 March 2025. Continuous covers 9–28 March, producing 80 labels, 24 positive. Dynamic history defaults to seven days before each eruption: 64 event-grid rows merge into 52 unique timestamps, retaining the same 24 positives. Choose three days to see separated event slices, or ten days for longer overlap. Missing dates are absent labels, not zero labels.

The matrix chapter accepts either label source. A timestamp t selects `[t − 2 days, t)`: 288 ten-minute tremor samples. The completeness demonstration removes every fourth sample from the selected slice (216/288, 75%), so the example’s 100% threshold rejects it. This is a per-window teaching fixture; other windows are assumed complete. Accepted rows carry the label id, datetime and metric columns; `is_erupted` stays in the separate labels table. Neighbouring six-hour windows share 252 samples. Synthetic tremor extends two days before the first label.

Source logic lives in `app/src/features/explorer/labelling.ts`, with the UI in `components/LabellingWalkthrough.tsx`. The reference PDF, `assets/How Eruption Forecasting Works.pdf`, pages 8–13, shows two forecast days plus an extra eruption day; the requested settings instead count that eruption day inside three days, producing the same positive dates. The UI is a static teaching fixture, not a Python execution surface. Tests cover date boundaries, deduplication, omitted gaps, matrix endpoints, overlap, missing samples and desktop/mobile chapter navigation.

## Validation commands

Run all commands from `app/`:

```sh
npm run typecheck
npm run lint
npm run test -- --run
npm run build
npm run validate:sources
npm run test:e2e
```

If Chromium is missing, run `npx playwright install chromium` before the browser tests. All tests, outputs, traces and reports live under `app/tests/`. Browser tests cover desktop and mobile, URL history, selection, search, caveats, expansion, tour navigation, keyboard use, zoom, pan, minimap navigation, viewport restoration and drawer focus.

Lint enforces filenames and dependency direction, rejects cross-feature deep imports and upward imports, and runs negative fixtures for both aliases and relative paths. The rule also resolves relative paths to catch normalization bypasses. Fixed lowercase entrypoints and direct imports inside one feature remain valid.

The required `eslint-plugin-check-file` currently has an unpatched development-only `micromatch → braces` advisory chain. The test runner was upgraded to remove its reported advisories. Recheck with `npm audit`; do not expose development tooling to untrusted input.

## Prediction walkthrough

Open **Prediction** in the Atlas toolbar (`/prediction`). Chapter 1, *One forecast*, has five steps over a branching workflow diagram (Tremor CSV, Window Grid, FeaturesBuilder, Trained Model, ClassifierEnsemble, `forecast()`, Forecast CSV, `plot_forecast`, Cache + Config): choose the window grid, extract features without labels (`use_features_from`), load the ensemble via `from_any`, run `forecast()`, then list the outputs. Step 4 scrubs the **recorded** OJN scenario 5.2 forecast (`assets/forecast-results_2025-08-01_2025-08-22_ws-2.csv`, via `forecastReplay.ts`) with a live threshold and a per-classifier row table.

Step 5 adds an **ROC simulation** below the forecast figure: one ROC panel per classifier plus the consensus, computed in the browser from the recorded CSV (`rocAnalysis.ts`). Windows are labelled at 10 minutes with the Labelling rule (the eruption day and the two days before it, for 2025-08-02 and 2025-08-18). A threshold slider, shared with the forecast figure, moves a dot along each curve and updates the confusion counts, TPR, FPR, precision and specificity. The shaded band shifts every probability by plus or minus its SD; the file has no per-seed columns, so it is not the per-seed range of the evaluation plots. Only two eruptions are present, so AUC is a weak estimate.

Chapter 2, *Several scenarios* (`?part=scenarios`), mirrors `forecast-scenario.py`: `calculate()` once, then train, predict, evaluate and explain per selected scenario. Reader can toggle `use_features_from`, step or play the run, inspect a scenario's `Scenario` dict, and add a custom scenario (validated: forecast starts after training ends, dates inside the calculated range). Only scenario 5.2 has recorded probabilities; other scenarios show periods and derived eruption counts. Window counts are calendar days times windows per day. `?scenario=<id>` selects a scenario.

Source: `features/explorer/predictionLessons.ts`, `components/{PredictionWalkthrough,PredictionWorkflowDiagram,PredictionReader,ScenarioRunner}.tsx`, `styles/prediction.css`. Tests: `app/tests/prediction.test.ts`; captures via `node tests/capture-prediction.mjs`.

### Reusable forecast components

The forecast figure and ROC panels are data-driven, so another page (for example an Evaluation walkthrough) can reuse them by passing a `ForecastDataset` (`features/explorer/forecastDataset.ts`): `rows` (`time` plus `{model}_probability`, `_uncertainty`, `_confidence`, `_prediction` values), `eruptionDates`, `models` (first entry is the primary series, normally the consensus), `lookaheadDays`, `stepMinutes`, optional `downloadUrl`. Exported from the explorer's `index.ts`:

| Component | Use |
|-----------|-----|
| `ForecastChart` | The curve alone: threshold line, green/ember fills, eruption-day bands, optional scrub cursor and overlay model |
| `ForecastExplorer` | Chart plus window scrubber, threshold and overlay controls, per-model table, threshold read-outs |
| `RocPanel` | One ROC curve with AUC, SD band and a threshold dot |
| `RocSimulation` | All ROC panels, shared threshold slider and confusion counts |

`RECORDED_FORECAST` is the bundled OJN scenario 5.2 dataset. `rocAnalysis.ts` holds the label, ROC and confusion maths and caches per dataset.

## Shared walkthrough components

The Labelling, Training and Prediction pages share one set of building blocks, so a change to a heading, table, select or step control lands on every page. Do not copy their markup into a new lesson; import them. The class names are unchanged, so the stylesheets in `styles/` still apply.

| Module | Provides |
|--------|----------|
| `components/lab/` | `LabSectionTitle`, `ResultOutput`, `SelectField` / `SeedSelect`, `ChoiceButtons`, `DataTable`, `ScrollRegion` / `ChartScroll`, `Legend`, `StepControls` / `StepButtons`, `ChapterNav`, `ParameterStrip`, `WhySection` / `ImplementationNotes`, `TraceToolbar`, `ChartFrame`, `YAxis` / `XAxis` / `AxisTitle` |
| `components/common/ExplanationDialog.tsx` | The "open in a full dialog" button and Radix dialog used by the wiki explanation and the prediction replay |
| `lib/scale.ts` | `linearScale`, `logScale`, `linePoints`, `linePath`: the domain-to-pixel maths and SVG point strings every chart uses |
| `hooks/useElementWidth.ts` | Container width for charts that redraw to fit instead of scrolling (`ForecastChart`) |
| `hooks/useTrace.ts` | The cursor-driven "Trace one run" state shared by `WorkflowDiagram` and `SeedWorkflowDiagram` |
| `lib/format.ts`, `lib/time.ts` | `percent`, `fixed`, `goToStepLabel`; `shortDate` (string or epoch), `utcStamp` |
| `features/explorer/components/` | Page chrome that needs routes or fixtures: `WalkthroughHeader`, `WalkthroughMenu` / `WalkthroughPager` / `WalkthroughFooter` (and the `WALKTHROUGHS` route list), `SampleCell`, `RadioCards`, `StageNumber`, `NodeBehavior` / `NodeParameters` / `NodeArtifacts` |

Each Training lesson is its own component (`UnderSamplerLesson`, `SeedEnsembleLesson`, `ClassifierEnsembleLesson`, plus the existing feature-matrix, feature-selection and grid-CV lessons); `TrainingWalkthrough` only dispatches on the topic and owns the shared `step` and `seed`. Restart remounts the lesson, so a lesson's own controls reset with it. Shared modules cannot import from `features/` (enforced by `npm run lint`).

### Charts

Draw SVG charts from `ChartFrame` (legend, scrolling `<svg role="img">`, required text label), `YAxis` / `XAxis` / `AxisTitle` (gridlines and ticks; one `chart-grid` / `chart-axis` stylesheet in `styles/charts.css`) and `lib/scale.ts`. Do not hand-roll `sx`/`sy` functions, point strings or axis text. `ForecastChart`, `RocPanel`, `ByCorrectionDemo`, `KsTestDemo`, `FisherDemo` and `LabelledDataDemo` already use them. `FisherDemo`'s "How C(n, r) works" section lives in `CombinationsLesson.tsx` to stay under the file-size limit.

### Component gallery

`/gallery` (`features/gallery/`, route `routes/GalleryPage.tsx`) renders every `components/lab/` and `components/common/` component, plus `Button`, with live demos and its source path. Check it before building a new component. When you add a shared component or change its props, add or update its `GallerySection` in `LabGallery.tsx` or `CommonGallery.tsx` in the same change. `tests/browser/gallery.spec.ts` checks the page lists the components and fits the viewport.
