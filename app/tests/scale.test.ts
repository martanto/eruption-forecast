import { describe, expect, it } from "vitest";
import { linearScale, linePath, linePoints, logScale } from "@/lib/scale";

describe("chart scales", () => {
  it("maps a linear domain onto a range and supports flipped ranges", () => {
    const x = linearScale([0, 10], [40, 240]);
    expect(x(0)).toBe(40);
    expect(x(5)).toBe(140);
    expect(x(10)).toBe(240);
    const y = linearScale([0, 1], [200, 20]);
    expect(y(0)).toBe(200);
    expect(y(1)).toBe(20);
  });

  it("maps a log domain by decade and clamps below the minimum", () => {
    const y = logScale([1e-7, 1], [340, 16]);
    expect(y(1)).toBeCloseTo(16);
    expect(y(1e-7)).toBeCloseTo(340);
    expect(y(1e-4)).toBeCloseTo((340 + 16) / 2 + (340 - 16) / 14, 5);
    expect(y(0)).toBeCloseTo(340);
  });

  it("formats points and paths rounded to a tenth of a pixel", () => {
    const points = [
      [1, 2],
      [3.14159, 4.5],
    ] as const;
    expect(linePoints(points)).toBe("1.0,2.0 3.1,4.5");
    expect(linePath(points)).toBe("M1.0,2.0 L3.1,4.5");
  });
});
