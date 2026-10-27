import { describe, expect, it } from "vitest";
import {
  RECORDED_FORECAST,
  thresholdSummary,
  type ForecastDataset,
} from "../src/features/explorer/forecastDataset";
import {
  buildLabels,
  confusionAt,
  rocFor,
} from "../src/features/explorer/rocAnalysis";
import { SCENARIOS } from "../src/features/architecture";
import {
  coveredByFirst,
  daysBetween,
  eruptionsBetween,
  validateScenario,
  windowEstimate,
} from "../src/features/explorer/predictionLessons";

describe("prediction teaching fixture", () => {
  it("counts inclusive calendar days and 10-minute windows", () => {
    const last = SCENARIOS.at(-1)!;
    expect(daysBetween(last.predictionStart, last.predictionEnd)).toBe(22);
    expect(windowEstimate(last)).toBe(22 * 144);
  });

  it("splits eruption dates between training and forecast periods", () => {
    const first = SCENARIOS[0]!;
    expect(eruptionsBetween(first.trainStart, first.trainEnd)).toEqual([
      "2025-03-20",
    ]);
    expect(
      eruptionsBetween(first.predictionStart, first.predictionEnd),
    ).toHaveLength(7);
  });

  it("rejects scenarios whose forecast overlaps training", () => {
    expect(
      validateScenario({
        trainEnd: "2025-06-01",
        predictionStart: "2025-06-01",
        predictionEnd: "2025-07-01",
      }),
    ).toMatch(/after training/);
    expect(
      validateScenario({
        trainEnd: "2025-06-01",
        predictionStart: "2025-06-02",
        predictionEnd: "2025-07-01",
      }),
    ).toBeNull();
  });

  it("lets scenario 1 cover every later forecast window for reuse", () => {
    expect(
      SCENARIOS.slice(1).every((s) => coveredByFirst(SCENARIOS[0]!, s)),
    ).toBe(true);
  });

  it("raising the threshold never increases the windows counted", () => {
    const low = thresholdSummary(RECORDED_FORECAST, 0.5);
    const high = thresholdSummary(RECORDED_FORECAST, 0.9);
    expect(high.above).toBeLessThanOrEqual(low.above);
    expect(low.total).toBe(high.total);
  });
});

describe("ROC from the recorded forecast", () => {
  it("labels the two eruption blocks at 10-minute resolution", () => {
    const { labels, positives, negatives } = buildLabels(RECORDED_FORECAST);
    expect(positives + negatives).toBe(labels.length);
    // 1 Aug starts inside the first block: 2 Aug block gives 2 days; 18 Aug gives 3 days.
    expect(positives).toBe(5 * 144);
  });

  it("gives an AUC in [0, 1] and confusion counts that always sum to all windows", () => {
    const roc = rocFor(RECORDED_FORECAST, "consensus");
    expect(roc.auc).toBeGreaterThanOrEqual(0);
    expect(roc.auc).toBeLessThanOrEqual(1);
    const total = RECORDED_FORECAST.rows.length;
    const c = confusionAt(RECORDED_FORECAST, "consensus", 0.7);
    expect(c.tp + c.fp + c.fn + c.tn).toBe(total);
    expect(confusionAt(RECORDED_FORECAST, "consensus", 0).fn).toBe(0);
    expect(confusionAt(RECORDED_FORECAST, "consensus", 1.01).tp).toBe(0);
  });
});

describe("forecast components accept any dataset", () => {
  // Hourly rows over 9 to 11 Jan; only the eruption day (10 Jan) is positive.
  const start = Date.parse("2025-01-09T00:00:00Z");
  const custom: ForecastDataset = {
    id: "synthetic",
    label: "Synthetic",
    eruptionDates: ["2025-01-10"],
    models: [["consensus", "Consensus"]],
    lookaheadDays: 0,
    stepMinutes: 60,
    rows: Array.from({ length: 72 }, (_, i) => {
      const time = start + i * 3_600_000;
      const hot =
        time >= Date.parse("2025-01-10T00:00:00Z") &&
        time < Date.parse("2025-01-11T00:00:00Z");
      return {
        time,
        values: {
          consensus_probability: hot ? 0.9 : 0.1,
          consensus_uncertainty: 0.05,
        },
      };
    }),
  };

  it("scores a perfect separator with AUC 1 and no errors at a middle threshold", () => {
    expect(buildLabels(custom).positives).toBe(24);
    expect(rocFor(custom, "consensus").auc).toBeCloseTo(1, 10);
    expect(confusionAt(custom, "consensus", 0.5)).toEqual({
      tp: 24,
      fp: 0,
      fn: 0,
      tn: 48,
    });
  });

  it("summarises crossings from the dataset's own eruption dates", () => {
    const summary = thresholdSummary(custom, 0.5);
    expect(summary.above).toBe(24);
    expect(summary.eruptions.map((item) => item.date)).toEqual(["2025-01-10"]);
  });
});
