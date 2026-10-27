import { DAY_MS, type ForecastDataset } from "./forecastDataset";

export interface RocPoint {
  threshold: number;
  fpr: number;
  tpr: number;
}
export interface Confusion {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}
export interface RocResult {
  points: RocPoint[];
  auc: number;
  // Curves from probability minus and plus one SD, sampled on a shared FPR grid.
  grid: number[];
  lower: number[];
  upper: number[];
}
export interface RocLabels {
  labels: number[];
  positives: number;
  negatives: number;
}

const labelCache = new WeakMap<ForecastDataset, RocLabels>();
// Same rule as the labelling walkthrough: the eruption day and the days before it.
export function buildLabels(dataset: ForecastDataset): RocLabels {
  const cached = labelCache.get(dataset);
  if (cached) return cached;
  const times = dataset.eruptionDates.map((date) =>
    Date.parse(`${date}T00:00:00Z`),
  );
  const labels = dataset.rows.map((row) =>
    Number(
      times.some(
        (at) =>
          row.time >= at - dataset.lookaheadDays * DAY_MS &&
          row.time < at + DAY_MS,
      ),
    ),
  );
  const positives = labels.reduce<number>((sum, value) => sum + value, 0);
  const result = { labels, positives, negatives: labels.length - positives };
  labelCache.set(dataset, result);
  return result;
}

const GRID = Array.from({ length: 101 }, (_, i) => i / 100);

export function rocCurve(
  scores: readonly number[],
  { labels, positives, negatives }: RocLabels,
): { points: RocPoint[]; auc: number } {
  const order = scores.map((_, i) => i).sort((a, b) => scores[b]! - scores[a]!);
  const points: RocPoint[] = [{ threshold: Infinity, fpr: 0, tpr: 0 }];
  let tp = 0;
  let fp = 0;
  for (let k = 0; k < order.length; k++) {
    const i = order[k]!;
    if (labels[i]) tp++;
    else fp++;
    // Emit one point per distinct score so ties move diagonally, as scikit-learn does.
    if (k === order.length - 1 || scores[order[k + 1]!] !== scores[i]) {
      points.push({
        threshold: scores[i]!,
        fpr: fp / negatives,
        tpr: tp / positives,
      });
    }
  }
  let auc = 0;
  for (let k = 1; k < points.length; k++) {
    auc +=
      ((points[k]!.fpr - points[k - 1]!.fpr) *
        (points[k]!.tpr + points[k - 1]!.tpr)) /
      2;
  }
  return { points, auc };
}

function onGrid(points: readonly RocPoint[]) {
  return GRID.map((x) => {
    let best = 0;
    for (const point of points) {
      if (point.fpr <= x) best = Math.max(best, point.tpr);
      else break;
    }
    return best;
  });
}

const rocCache = new WeakMap<ForecastDataset, Map<string, RocResult>>();
export function rocFor(dataset: ForecastDataset, model: string): RocResult {
  const perDataset = rocCache.get(dataset) ?? new Map<string, RocResult>();
  rocCache.set(dataset, perDataset);
  const cached = perDataset.get(model);
  if (cached) return cached;
  const labels = buildLabels(dataset);
  const mean = dataset.rows.map(
    (row) => row.values[`${model}_probability`] ?? 0,
  );
  const sd = dataset.rows.map((row) => row.values[`${model}_uncertainty`] ?? 0);
  const main = rocCurve(mean, labels);
  const lower = onGrid(
    rocCurve(
      mean.map((p, i) => p - sd[i]!),
      labels,
    ).points,
  );
  const upper = onGrid(
    rocCurve(
      mean.map((p, i) => p + sd[i]!),
      labels,
    ).points,
  );
  const result = { ...main, grid: GRID, lower, upper };
  perDataset.set(model, result);
  return result;
}

export function confusionAt(
  dataset: ForecastDataset,
  model: string,
  threshold: number,
): Confusion {
  const { labels } = buildLabels(dataset);
  const counts = { tp: 0, fp: 0, fn: 0, tn: 0 };
  dataset.rows.forEach((row, i) => {
    const hit = (row.values[`${model}_probability`] ?? 0) >= threshold;
    if (labels[i]) counts[hit ? "tp" : "fn"]++;
    else counts[hit ? "fp" : "tn"]++;
  });
  return counts;
}
