---
version: 1
slug: "orer-components-predictionwalkthrough-tsx-59bf99fb"
primary_target: "src/features/explorer/components/PredictionWalkthrough.tsx"
related_targets: ["src/routes/PredictionPage.tsx","src/styles/prediction.css"]
---

# Prediction walkthrough

Mode: Read. Extends the Labelling and Training walkthroughs on the current branch; inherits the Atlas light/dark tokens, DM Sans and flat ruled sections. No new visual world.

## Direction contract

THESIS: Show that forecasting is the trained ensemble read over fresh, unlabelled windows, by letting the reader scrub the recorded OJN forecast and see every row the CSV holds, then repeat that under several scenarios. Refuses a static pipeline poster.

OWN-WORLD: Existing Atlas tokens: paper and surface grounds, terracotta selection, burgundy for prediction periods, ember for eruption dates, green code nodes against purple data nodes in the workflow diagram (the supplied reference image). Flat, ruled, no card shadows.

STORY: Follow tremor to a window grid, to tsfresh without labels, to the loaded ensemble, to forecast(), to the CSV, plot and cache. Then run scenarios 1.2 to 5.2 as forecast-scenario.py does: calculate once, train, predict, evaluate, explain per scenario.

FIRST VIEWPORT: Back link, title, cross-page nav (Labelling, Training, Prediction), a parameter strip, numbered step controls with the current operation, then the branching workflow diagram with the current step lit.

FORM: Code-led extension of the Training composition; signature interaction is a window scrubber over the real scenario 5.2 CSV with a live threshold, plus a scenario runner that plays the loop. No seed key applies (seed: none (unavailable)).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

Data honesty: recorded probabilities exist only for scenario 5.2; other scenarios show periods and structure, never invented probabilities. Window counts are derived from dates and labelled as such.
