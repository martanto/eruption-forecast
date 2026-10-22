import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
  NODES,
  EDGES,
  STAGES,
  SCENARIOS,
  TOUR,
  GAPS,
  SOURCE_SNAPSHOT,
  aggregate,
  EXAMPLE_CONSENSUS,
} from "@/features/architecture";
import { pipelineLayout, timelineLayout } from "@/features/explorer";
import { epochSeconds } from "@/lib/time";

describe("canonical content", () => {
  it("resolves every graph, stage, source, gap and tour reference", () => {
    expect(new Set(NODES.map((item) => item.id)).size).toBe(NODES.length);
    for (const item of NODES) {
      expect(STAGES.some((stage) => stage.id === item.stage)).toBe(true);
      expect(item.behavior.length).toBeGreaterThan(0);
      expect(item.sources.length).toBeGreaterThan(0);
      item.sources.forEach((id) =>
        expect(
          SOURCE_SNAPSHOT.sources.some(
            (source) =>
              source.id === id &&
              source.excerpt.length > 0 &&
              source.sha256.length === 64,
          ),
        ).toBe(true),
      );
      item.gaps.forEach((id) =>
        expect(GAPS.some((gap) => gap.id === id)).toBe(true),
      );
    }
    EDGES.forEach((edge) => {
      expect(NODES.some((item) => item.id === edge.source)).toBe(true);
      expect(NODES.some((item) => item.id === edge.target)).toBe(true);
    });
    TOUR.forEach((step) =>
      expect(NODES.some((item) => item.id === step.target)).toBe(true),
    );
  });
  it("includes five valid, progressively shorter prediction periods in UTC", () => {
    expect(SCENARIOS).toHaveLength(5);
    SCENARIOS.forEach((item) => {
      expect(epochSeconds(item.trainEnd)).toBeLessThan(
        epochSeconds(item.predictionStart),
      );
      expect(epochSeconds(item.predictionStart)).toBeLessThan(
        epochSeconds(item.predictionEnd),
      );
      expect(item.predictionStepMinutes).toBe(10);
      expect(epochSeconds(item.trainStart)).toBe(1735689600);
    });
  });
  it("matches the five scenario definitions in the reviewed script", () => {
    const script = readFileSync("../forecast-scenario.py", "utf8");
    const records = [
      ...script.matchAll(
        /"name": "(Scenario [^"]+)",[\s\S]*?"train_start_date": "([^"]+)",[\s\S]*?"train_end_date": "([^"]+)",[\s\S]*?"prediction_start_date": "([^"]+)",[\s\S]*?"prediction_end_date": "([^"]+)",[\s\S]*?"prediction_window_step": (\d+),[\s\S]*?"prediction_window_step_unit": "minutes"/g,
      ),
    ];
    expect(records).toHaveLength(SCENARIOS.length);
    records.forEach((record, index) =>
      expect(SCENARIOS[index]).toMatchObject({
        name: record[1],
        trainStart: record[2],
        trainEnd: record[3],
        predictionStart: record[4],
        predictionEnd: record[5],
        predictionStepMinutes: Number(record[6]),
      }),
    );
  });
  it("keeps overview positions fixed during expansion and shows selected hidden steps", () => {
    const overview = pipelineLayout([], "tremor", false);
    const expanded = pipelineLayout(["tremor"], "metrics", true);
    expect(expanded.nodes.length).toBeGreaterThan(overview.nodes.length);
    for (const item of overview.nodes)
      expect(
        expanded.nodes.find((node) => node.id === item.id)?.position,
      ).toEqual(item.position);
    expect(
      pipelineLayout([], "shap", false).nodes.some(
        (item) => item.id === "shap",
      ),
    ).toBe(true);
    expect(
      timelineLayout("shap", SCENARIOS[0]!.id).nodes.some(
        (item) => item.id === "shap" && item.selected,
      ),
    ).toBe(true);
  });
});
describe("illustrative consensus", () => {
  it("distinguishes probabilities, votes, population std and vote-based confidence", () => {
    expect(EXAMPLE_CONSENSUS.probability).toBeCloseTo(0.7);
    expect(EXAMPLE_CONSENSUS.vote).toBe(0.75);
    expect(EXAMPLE_CONSENSUS.uncertainty).toBeCloseTo(0.1);
    expect(EXAMPLE_CONSENSUS.confidence).toBeCloseTo(0.6001249869);
    expect(aggregate([0.2, 0.2], [0, 0]).confidence).toBe(0);
    expect(() => aggregate([], [])).toThrow();
    expect(() => aggregate([0.4], [0, 1])).toThrow();
  });
});
