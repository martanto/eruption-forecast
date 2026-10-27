import { FORECAST_MODELS, FORECAST_ROWS } from "./forecastReplay";
import csvUrl from "../../../../assets/forecast-results_2025-08-01_2025-08-22_ws-2.csv?url";

export const DAY_MS = 86_400_000;

export interface ForecastRow {
  time: number;
  // Columns are named `{model}_{field}`, with fields probability, uncertainty,
  // confidence and prediction, as in forecast-results_*.csv.
  values: Record<string, number>;
}
export type ForecastModels = readonly (readonly [key: string, label: string])[];

/**
 * Everything the forecast chart, explorer and ROC components need. Pass a new
 * dataset (for example one built from an evaluation run) to reuse them.
 */
export interface ForecastDataset {
  id: string;
  /** Short name used in captions, for example "OJN scenario 5.2". */
  label: string;
  rows: readonly ForecastRow[];
  /** Eruption dates, `YYYY-MM-DD`, UTC. */
  eruptionDates: readonly string[];
  /** First entry is the primary series, normally the consensus. */
  models: ForecastModels;
  /** Days before an eruption that are labelled positive (the eruption day is too). */
  lookaheadDays: number;
  stepMinutes: number;
  downloadUrl?: string;
}

export const RECORDED_FORECAST: ForecastDataset = {
  id: "ojn-scenario-5-2",
  label: "OJN scenario 5.2",
  rows: FORECAST_ROWS,
  eruptionDates: ["2025-08-02", "2025-08-18"],
  models: FORECAST_MODELS,
  lookaheadDays: 2,
  stepMinutes: 10,
  downloadUrl: csvUrl,
};

export interface ReaderValue {
  probability: number;
  uncertainty: number;
  confidence: number;
  prediction: number;
}
const FIELDS = [
  "probability",
  "uncertainty",
  "confidence",
  "prediction",
] as const;

export function readRow(
  dataset: ForecastDataset,
  index: number,
  model: string,
): ReaderValue {
  const row =
    dataset.rows[Math.min(Math.max(index, 0), dataset.rows.length - 1)]!;
  const [probability, uncertainty, confidence, prediction] = FIELDS.map(
    (field) => row.values[`${model}_${field}`] ?? Number.NaN,
  );
  return {
    probability: probability!,
    uncertainty: uncertainty!,
    confidence: confidence!,
    prediction: prediction!,
  };
}

const primaryKey = (dataset: ForecastDataset) => dataset.models[0]![0];

// Bucket maxima keep short peaks visible when thousands of rows meet a few hundred pixels.
export function primarySeries(dataset: ForecastDataset, buckets: number) {
  const key = `${primaryKey(dataset)}_probability`;
  const size = Math.ceil(dataset.rows.length / buckets);
  const series: { index: number; time: number; value: number }[] = [];
  for (let start = 0; start < dataset.rows.length; start += size) {
    let best = start;
    for (let i = start; i < Math.min(start + size, dataset.rows.length); i++) {
      if (
        (dataset.rows[i]!.values[key] ?? 0) >
        (dataset.rows[best]!.values[key] ?? 0)
      )
        best = i;
    }
    series.push({
      index: best,
      time: dataset.rows[best]!.time,
      value: dataset.rows[best]!.values[key] ?? 0,
    });
  }
  return series;
}

export interface ThresholdSummary {
  above: number;
  total: number;
  eruptions: { date: string; crossings: number; leadHours: number | null }[];
}
// Descriptive only: counts windows at or above the line inside the look-ahead before each date.
export function thresholdSummary(
  dataset: ForecastDataset,
  threshold: number,
): ThresholdSummary {
  const key = `${primaryKey(dataset)}_probability`;
  const probability = (i: number) => dataset.rows[i]!.values[key] ?? 0;
  let above = 0;
  for (let i = 0; i < dataset.rows.length; i++)
    if (probability(i) >= threshold) above++;
  const first = dataset.rows[0]!.time;
  const last = dataset.rows.at(-1)!.time;
  const eruptions = dataset.eruptionDates
    .map((date) => ({ date, at: Date.parse(`${date}T00:00:00Z`) }))
    .filter(({ at }) => at >= first && at <= last + DAY_MS)
    .map(({ date, at }) => {
      const hits = dataset.rows.filter(
        (row) =>
          row.time >= at - dataset.lookaheadDays * DAY_MS &&
          row.time < at &&
          (row.values[key] ?? 0) >= threshold,
      );
      return {
        date,
        crossings: hits.length,
        leadHours: hits[0] ? (at - hits[0].time) / 3_600_000 : null,
      };
    });
  return { above, total: dataset.rows.length, eruptions };
}
