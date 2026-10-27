import { ERUPTIONS, SCENARIOS, type Scenario } from "@/features/architecture";
import { DAY_MS } from "./forecastDataset";
import { THRESHOLD } from "./forecastReplay";

export const TREMOR_START = "2025-01-01";
export const TREMOR_END = "2025-12-31";
// Scenario 5.2 is the only run whose probabilities ship with the atlas.
export const RECORDED_SCENARIO_ID = "scenario-5-2";
export const DAY_TO_FORECAST = 2;
export const DEFAULT_THRESHOLD = THRESHOLD;

export const PREDICTION_STEPS = [
  [
    "Choose the forecast period",
    "PredictionModel is given a start and an end date. Its tremor is segmented into a grid of windows with the same window_size the model was trained on (day_to_forecast days) and a fine window_step, 10 minutes in the example scenarios. These windows carry no labels: no information about whether an eruption follows is used.",
  ],
  [
    "Extract features without labels",
    "FeaturesBuilder runs tsfresh over every window, exactly as in training, but in prediction mode: there is no label to test against, so no relevance filtering. use_features_from decides which columns come out of the matrix.",
  ],
  [
    "Load the trained ensemble",
    "The trained model arrives as an object, a ClassifierEnsemble json or pkl, a SeedEnsemble pkl or a registry file. ClassifierEnsemble.from_any resolves them all to one container of seed models, each with its own feature list.",
  ],
  [
    "Run forecast()",
    "Every seed reads its own feature columns for every window and returns a probability. Seeds average into a family mean, families average into the consensus, and the dispersion among them is the uncertainty. Drag across the recorded forecast to read one window.",
  ],
  [
    "Write results and cache",
    "forecast() writes the combined CSV (consensus plus every classifier), optional per-seed CSVs, the forecast plot, a prediction config and a cache entry keyed by the training hash, so retraining invalidates old forecasts.",
  ],
] as const;

export const PREDICTION_COPY = {
  title: "Prediction",
  intro: "Apply the trained ensemble to windows not seen during training.",
  why: "Training answers which windows looked like the run-up to an eruption. Prediction applies that answer to new tremor, window after window, every ten minutes. The output is a probability curve with a measure of inter-model disagreement, not a binary decision, so a reader can see when the models agree and when they diverge.",
  caveat:
    "A probability here comes from a model fitted on past eruptions and is not calibrated. Uncertainty is the dispersion between seeds and classifier families, not a confidence interval. Neighbouring windows overlap heavily and are therefore autocorrelated, so a run of high values is one episode counted repeatedly. The 0.7 threshold is a plotting and alerting choice; evaluate false positives and false negatives on later data before acting on it.",
  implementation:
    "PredictionModel.build_label and extract_features prepare the window grid and features; forecast() calls ClassifierEnsemble.predict_with_uncertainty and writes forecast-results_{basename}.csv. ForecastModelScenario.shared_prediction wraps this once per Scenario.",
  sources: [
    "model/prediction_model.py · build_label, extract_features, forecast",
    "ensemble/classifier_ensemble.py · from_any, predict_with_uncertainty",
    "model/forecast_model_scenario.py · shared_prediction, run",
    "plots/forecast_plots.py · plot_forecast",
  ],
} as const;

export const FEATURE_SOURCES = [
  {
    value: "all",
    label: "all",
    effect: "Every tsfresh feature is computed for every window.",
    cost: "Slowest. Safe default; a retrained ensemble needs nothing else.",
  },
  {
    value: "training",
    label: "training",
    effect:
      "Only the union of features that any seed selected during train() is computed.",
    cost: "Faster. Requires train() to have run in the same ForecastModel.",
  },
  {
    value: "files",
    label: "files",
    effect:
      "tsfresh is skipped. A saved features matrix and its label-features CSV are loaded instead.",
    cost: "Fastest, and it forces use_cache=False. In a scenario run, scenario 1 extracts and later scenarios reuse its matrix.",
  },
] as const;
export type FeatureSource = (typeof FEATURE_SOURCES)[number]["value"];

export interface PredictionNode {
  id: string;
  title: string;
  detail: string;
  kind: "code" | "data";
  step: number;
  col: number;
  row: number;
}
export interface PredictionEdge {
  from: string;
  to: string;
  label?: string;
  // Elbow connectors leave sideways and turn once; the rest are straight.
  elbow?: boolean;
}

// Grid positions follow the supplied reference diagram: five columns, three rows.
export const PREDICTION_NODES: readonly PredictionNode[] = [
  {
    id: "tremor",
    title: "Tremor CSV",
    detail: "forecast period",
    kind: "data",
    step: 0,
    col: 0,
    row: 1,
  },
  {
    id: "grid",
    title: "Window Grid",
    detail: "unlabelled windows",
    kind: "code",
    step: 0,
    col: 1,
    row: 1,
  },
  {
    id: "features",
    title: "FeaturesBuilder",
    detail: "tsfresh, no filtering",
    kind: "code",
    step: 1,
    col: 2,
    row: 1,
  },
  {
    id: "model",
    title: "Trained Model",
    detail: "json, pkl or object",
    kind: "data",
    step: 2,
    col: 2,
    row: 0,
  },
  {
    id: "ensemble",
    title: "ClassifierEnsemble",
    detail: "seed votes, uncertainty",
    kind: "code",
    step: 2,
    col: 3,
    row: 0,
  },
  {
    id: "forecast",
    title: "forecast()",
    detail: "PredictionModel",
    kind: "code",
    step: 3,
    col: 3,
    row: 1,
  },
  {
    id: "plot",
    title: "plot_forecast",
    detail: "PNG, optional PDF",
    kind: "code",
    step: 4,
    col: 4,
    row: 0,
  },
  {
    id: "csv",
    title: "Forecast CSV",
    detail: "consensus + per clf",
    kind: "data",
    step: 4,
    col: 4,
    row: 1,
  },
  {
    id: "cache",
    title: "Cache + Config",
    detail: "keyed by training hash",
    kind: "data",
    step: 4,
    col: 3,
    row: 2,
  },
];
export const PREDICTION_EDGES: readonly PredictionEdge[] = [
  { from: "tremor", to: "grid" },
  { from: "grid", to: "features" },
  { from: "features", to: "forecast" },
  { from: "model", to: "ensemble", label: "from_any" },
  { from: "forecast", to: "ensemble", label: "features" },
  { from: "forecast", to: "plot", elbow: true },
  { from: "forecast", to: "csv" },
  { from: "forecast", to: "cache" },
];

export const OUTPUT_FILES = [
  [
    "prediction/results/forecast-results_{basename}.csv",
    "Consensus plus every classifier, one row per window",
  ],
  [
    "prediction/results/{ClassifierName}/{seed}.csv",
    "One probability column per seed (save_seed_result=True)",
  ],
  [
    "prediction/figures/forecast_{basename}.png",
    "Probability curve, threshold and eruption markers (+ .pdf if plot_pdf)",
  ],
  [
    "prediction/prediction[.{prefix}].config.yaml",
    "Stage settings, replayable through ForecastModel.run()",
  ],
  [
    "prediction/{hash}.PredictionModel.pkl",
    "Cache entry; the hash includes the training hash",
  ],
] as const;

export const CSV_FIELDS = [
  ["{classifier}_probability", "Mean probability across that family's seeds"],
  ["{classifier}_uncertainty", "Standard deviation across seeds"],
  [
    "{classifier}_confidence",
    "Agreement-based confidence reported by the ensemble",
  ],
  [
    "{classifier}_prediction",
    "Vote fraction at the model threshold, not the thresholded mean",
  ],
  ["consensus_*", "The same four fields, averaged across classifier families"],
] as const;

// ---------------------------------------------------------------- scenarios

export interface RunScenario extends Scenario {
  custom?: boolean;
}
export const PHASES = ["train", "predict", "evaluate", "explain"] as const;
export type Phase = (typeof PHASES)[number];

const toTime = (date: string) => Date.parse(`${date}T00:00:00Z`);
export function daysBetween(start: string, end: string) {
  return Math.round((toTime(end) - toTime(start)) / DAY_MS) + 1;
}
export function eruptionsBetween(start: string, end: string) {
  return ERUPTIONS.filter((date) => date >= start && date <= end);
}
// Calendar days times windows per day; the first window lead-in is not counted.
export function windowEstimate(scenario: Scenario) {
  return (
    daysBetween(scenario.predictionStart, scenario.predictionEnd) *
    Math.round(1440 / scenario.predictionStepMinutes)
  );
}
export function validateScenario(
  draft: Pick<Scenario, "trainEnd" | "predictionStart" | "predictionEnd">,
): string | null {
  const { trainEnd, predictionStart, predictionEnd } = draft;
  if (!trainEnd || !predictionStart || !predictionEnd)
    return "Fill in every date.";
  if (trainEnd < TREMOR_START || predictionEnd > TREMOR_END)
    return `Dates must stay inside the calculated tremor range, ${TREMOR_START} to ${TREMOR_END}.`;
  if (predictionStart <= trainEnd)
    return "Prediction must start after training ends, or the forecast reuses training windows.";
  if (predictionEnd < predictionStart)
    return "Prediction must end on or after its start.";
  return null;
}
// use_features_from="files" reuses scenario 1's matrix, so its window must cover the others.
export function coveredByFirst(first: Scenario, other: Scenario) {
  return (
    other.predictionStart >= first.predictionStart &&
    other.predictionEnd <= first.predictionEnd
  );
}
export function scenarioDict(scenario: Scenario) {
  return `{
  "name": "${scenario.name}",
  "train_start_date": "${scenario.trainStart}",
  "train_end_date": "${scenario.trainEnd}",
  "prediction_start_date": "${scenario.predictionStart}",
  "prediction_end_date": "${scenario.predictionEnd}",
  "prediction_window_step": ${scenario.predictionStepMinutes},
  "prediction_window_step_unit": "minutes",
}`;
}
export function initialScenarios(): RunScenario[] {
  return SCENARIOS.map((scenario) => ({ ...scenario }));
}
