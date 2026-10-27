import { describe, expect, it } from "vitest";
import {
  FAMILIES,
  retainedNegatives,
  summarize,
  TRAINING_LESSONS,
} from "../src/features/explorer/trainingLessons";

describe("training teaching fixtures", () => {
  it("retains eight unique majority rows reproducibly and varies the draw by seed", () => {
    const draws = [0, 1, 2].map(retainedNegatives);
    for (const draw of draws) {
      expect(draw).toHaveLength(8);
      expect(new Set(draw).size).toBe(8);
      expect(draw.every((id) => id >= 1 && id <= 48)).toBe(true);
    }
    expect(retainedNegatives(0)).toEqual(draws[0]);
    expect(draws[0]).not.toEqual(draws[1]);
  });
  it("uses population SD and equal weights across family means", () => {
    expect(summarize([0.62, 0.74, 0.68]).mean).toBeCloseTo(0.68);
    expect(summarize([0.62, 0.74, 0.68]).std).toBeCloseTo(0.04898979);
    expect(
      summarize(FAMILIES.map((family) => summarize(family.probabilities).mean))
        .mean,
    ).toBeCloseTo(0.64333333);
    expect(summarize([0.68]).std).toBe(0);
  });
});

describe("seed workflow fixture", () => {
  it("keeps the per-seed job inside the boundary and the merge outside it", async () => {
    const { SEED_JOB_NODES, SEED_MERGE_NODES } = await import(
      "../src/features/explorer/trainingLessons"
    );
    expect(SEED_JOB_NODES.map((n) => n.id)).toEqual([
      "sampler",
      "selector",
      "top-csv",
      "grid",
      "seed-model",
    ]);
    expect(SEED_MERGE_NODES.map((n) => n.id)).toEqual([
      "registry",
      "seed-ensemble",
      "mean-probability",
    ]);
  });
});

describe("feature selection per seed", () => {
  it("ranks rsam_f2__mean first for seed 0 and last for seed 1 under tsfresh p-values", async () => {
    const { scoreFeatures } = await import("../src/features/explorer/trainingFixtures");
    const rank = (seed: number) =>
      scoreFeatures(seed, "tsfresh").find((f) => f.name === "rsam_f2__mean")!;
    expect(rank(0).rank).toBe(1);
    expect(rank(1).rank).toBe(8);
    expect(rank(1).p).toBeGreaterThan(rank(0).p);
    // The three seeds must not share one ordering.
    const orders = [0, 1, 2].map((s) =>
      scoreFeatures(s, "tsfresh").map((f) => f.name).join(),
    );
    expect(new Set(orders).size).toBe(3);
  });
});

describe("training guideline tabs", () => {
  it("opens on seed-ensemble and lists the five component tabs in order", () => {
    expect(Object.keys(TRAINING_LESSONS)).toEqual([
      "seed-ensemble",
      "features-matrix",
      "random-under-sampler",
      "feature-selection",
      "grid-cv",
      "classifier-ensemble",
    ]);
    for (const lesson of Object.values(TRAINING_LESSONS)) {
      expect(lesson.steps).toHaveLength(4);
    }
  });
});

describe("features matrix builder workflow", () => {
  it("connects every edge to existing nodes and counts 6/3/2 kinds", async () => {
    const { FEATURES_MATRIX_NODES: nodes, FEATURES_MATRIX_EDGES: edges } =
      await import("../src/features/explorer/workflowDiagrams");
    const ids = new Set(nodes.map((n) => n.id));
    expect(edges.every((e) => ids.has(e.from) && ids.has(e.to))).toBe(true);
    const count = (kind: string) => nodes.filter((n) => n.kind === kind).length;
    expect([count("backend"), count("database"), count("external")]).toEqual([6, 3, 2]);
  });
});
