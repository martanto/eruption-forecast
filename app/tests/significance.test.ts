import { describe, expect, it } from "vitest";
import { DUMMY_FEATURES } from "../src/features/explorer/dummyDataset";
import {
  comb,
  correct,
  fisherRows,
  fisherTable,
  ksGap,
  split,
} from "../src/features/explorer/significance";

describe("significance teaching fixtures", () => {
  it("computes C(n, r) like math.comb", () => {
    expect(comb(4, 2)).toBe(6);
    expect(comb(24, 10)).toBe(1961256);
    expect(comb(10, 7)).toBe(120);
    expect(comb(5, 6)).toBe(0);
  });

  it("reproduces scipy's exact Fisher p-value for every yes/no feature", () => {
    for (const feature of DUMMY_FEATURES.filter((f) => f.kind === "binary")) {
      const { p, rows } = fisherRows(fisherTable(feature));
      expect(p).toBeCloseTo(feature.p, 9);
      // All arrangements together must equal C(24, yes).
      expect(rows.reduce((s, r) => s + r.probability, 0)).toBeCloseTo(1, 9);
    }
  });

  it("finds a KS gap between 0 and 1 and a small p for a separated feature", () => {
    const mean = DUMMY_FEATURES.find((f) => f.name === "rsam_f2__mean")!;
    const { quiet, pre } = split(mean);
    expect(quiet).toHaveLength(14);
    expect(pre).toHaveLength(10);
    expect(ksGap(quiet, pre).d).toBeCloseTo(1, 9);
    expect(mean.p).toBeLessThan(1e-5);
  });

  it("keeps fewer features as the correction gets stricter (raw > BH > BY at q = 0.05)", () => {
    const { rows, m, c } = correct(0.05);
    const raw = rows.filter((r) => r.keptRaw).length;
    const bh = rows.filter((r) => r.keptBh).length;
    const by = rows.filter((r) => r.keptBy).length;
    expect(m).toBe(10);
    expect(c).toBeCloseTo(2.9290, 3);
    expect([raw, bh, by]).toEqual([8, 6, 3]);
    // BY keeps a prefix of the sorted list.
    expect(rows.map((r) => r.keptBy)).toEqual(rows.map((_, i) => i < by));
  });
});
