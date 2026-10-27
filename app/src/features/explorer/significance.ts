import {
  DUMMY_FEATURES,
  PRE_COUNT,
  QUIET_COUNT,
  type DummyFeature,
} from "./dummyDataset";

export function formatP(p: number) {
  return p < 0.001 ? p.toExponential(1) : p.toFixed(3);
}

export function verdict(p: number) {
  if (p < 0.001) return { label: "Strong evidence of a difference", tone: "strong" };
  if (p < 0.05) return { label: "Moderate evidence of a difference", tone: "maybe" };
  return { label: "No evidence of a difference", tone: "none" };
}

export function split(feature: DummyFeature) {
  return {
    quiet: feature.values.slice(0, QUIET_COUNT),
    pre: feature.values.slice(QUIET_COUNT),
  };
}

/** Share of values that are <= x. */
export function ecdf(values: readonly number[], x: number) {
  return values.filter((v) => v <= x).length / values.length;
}

/** Largest vertical gap between the two staircases, and where it happens. */
export function ksGap(quiet: readonly number[], pre: readonly number[]) {
  let d = 0;
  let at = Math.min(...quiet, ...pre);
  for (const x of [...quiet, ...pre]) {
    const gap = Math.abs(ecdf(pre, x) - ecdf(quiet, x));
    if (gap > d) {
      d = gap;
      at = x;
    }
  }
  return { d, at };
}

/** Number of ways to choose r items from n when order does not matter. */
export function comb(n: number, r: number) {
  if (r < 0 || r > n) return 0;
  const k = Math.min(r, n - r);
  let result = 1;
  for (let i = 1; i <= k; i++) result = (result * (n - k + i)) / i;
  return Math.round(result);
}

export interface FisherTable {
  preYes: number;
  preNo: number;
  quietYes: number;
  quietNo: number;
}

export function fisherTable(feature: DummyFeature): FisherTable {
  const { quiet, pre } = split(feature);
  const preYes = pre.filter((v) => v === 1).length;
  const quietYes = quiet.filter((v) => v === 1).length;
  return {
    preYes,
    preNo: pre.length - preYes,
    quietYes,
    quietNo: quiet.length - quietYes,
  };
}

/** One row of the hypergeometric table: x yes-windows land among pre-eruption windows. */
export function fisherRows(table: FisherTable) {
  const yes = table.preYes + table.quietYes;
  const total = PRE_COUNT + QUIET_COUNT;
  const ways = comb(total, yes);
  const rows = [];
  for (let x = Math.max(0, yes - QUIET_COUNT); x <= Math.min(PRE_COUNT, yes); x++) {
    const top = comb(PRE_COUNT, x);
    const bottom = comb(QUIET_COUNT, yes - x);
    rows.push({ x, top, bottom, count: top * bottom, probability: (top * bottom) / ways });
  }
  const observed = rows.find((row) => row.x === table.preYes)!;
  // Two-sided: every table no more likely than the observed one counts.
  const tail = rows.map((row) => ({
    ...row,
    inP: row.probability <= observed.probability * (1 + 1e-7),
  }));
  const waysInP = tail.filter((row) => row.inP).reduce((sum, row) => sum + row.count, 0);
  return { yes, total, ways, rows: tail, waysInP, p: waysInP / ways };
}

export interface ByRow {
  rank: number;
  name: string;
  test: "KS" | "Fisher";
  p: number;
  bh: number;
  by: number;
  keptBh: boolean;
  keptBy: boolean;
  keptRaw: boolean;
}

/**
 * Benjamini-Hochberg / Benjamini-Yekutieli on the dummy p-values. tsfresh uses
 * BY because its features are not independent: the bar is divided by c(m).
 */
export function correct(q: number) {
  const sorted = [...DUMMY_FEATURES].sort((a, b) => a.p - b.p);
  const m = sorted.length;
  const c = Array.from({ length: m }, (_, i) => 1 / (i + 1)).reduce((a, b) => a + b, 0);
  const bh = sorted.map((_, i) => ((i + 1) * q) / m);
  const by = bh.map((bar) => bar / c);
  const lastIndex = (bars: number[]) =>
    sorted.reduce((last, feature, i) => (feature.p <= bars[i]! ? i : last), -1);
  const lastBh = lastIndex(bh);
  const lastBy = lastIndex(by);
  const rows: ByRow[] = sorted.map((feature, i) => ({
    rank: i + 1,
    name: feature.name,
    test: feature.kind === "real" ? "KS" : "Fisher",
    p: feature.p,
    bh: bh[i]!,
    by: by[i]!,
    keptRaw: feature.p <= q,
    keptBh: i <= lastBh,
    keptBy: i <= lastBy,
  }));
  return { rows, m, c };
}
