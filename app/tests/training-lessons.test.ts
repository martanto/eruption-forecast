import { describe, expect, it } from "vitest";
import {
  FAMILIES,
  retainedNegatives,
  summarize,
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
