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

## Checks

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
