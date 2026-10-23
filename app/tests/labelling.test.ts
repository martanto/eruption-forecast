import { describe, expect, it } from "vitest";
import {
  buildLabels,
  DAY,
  ERUPTION_DATES,
  eruptionWindows,
  matrixRows,
  positiveAt,
} from "../src/features/explorer/labelling";

describe("source-aligned labelling teaching fixture", () => {
  it("marks exactly the three calendar days including eruption day", () => {
    const eruption = ERUPTION_DATES[0]!;
    expect(positiveAt(eruption - 2 * DAY - 1)).toBe(0);
    expect(positiveAt(eruption - 2 * DAY)).toBe(1);
    expect(positiveAt(eruption + DAY - 1)).toBe(1);
    expect(positiveAt(eruption + DAY)).toBe(0);
    expect(buildLabels("continuous")).toHaveLength(80);
    expect(
      buildLabels("continuous").filter((row) => row.positive),
    ).toHaveLength(24);
  });
  it("deduplicates overlapping event grids before assigning ids and positives", () => {
    expect(eruptionWindows(7).flat()).toHaveLength(64);
    const dynamic = buildLabels("dynamic", 7);
    expect(dynamic).toHaveLength(52);
    expect(new Set(dynamic.map((row) => row.time)).size).toBe(52);
    expect(dynamic.map((row) => row.id)).toEqual(
      Array.from({ length: 52 }, (_, id) => id),
    );
    expect(dynamic.filter((row) => row.positive)).toHaveLength(24);
    expect(
      dynamic.find((row) => row.time === ERUPTION_DATES[0])?.positive,
    ).toBe(1);
  });
  it("omits timestamps between separated event histories", () => {
    const rows = buildLabels("dynamic", 3);
    expect(rows).toHaveLength(32);
    expect(rows.some((row) => row.time === ERUPTION_DATES[0]! + DAY)).toBe(
      false,
    );
  });
  it("slices 288 samples with inclusive start, exclusive end and 252 shared samples", () => {
    const labels = buildLabels("continuous");
    const first = matrixRows(labels[0]!);
    const next = matrixRows(labels[1]!);
    expect(first).toHaveLength(288);
    expect(first[0]!.time).toBe(labels[0]!.time - 2 * DAY);
    expect(first.at(-1)!.time).toBe(labels[0]!.time - 600_000);
    expect(first.every((row) => row.id === labels[0]!.id)).toBe(true);
    const times = new Set(first.map((row) => row.time));
    expect(next.filter((row) => times.has(row.time))).toHaveLength(252);
    expect(matrixRows(labels[0]!, true)).toHaveLength(216);
  });
});
