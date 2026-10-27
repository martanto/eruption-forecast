// Deterministic teaching fixtures for the features-matrix, feature-selection and
// grid-cv walkthroughs. Numbers are illustrative, not model output.

export const TREMOR_COLUMNS = ["rsam_f2", "dsar_f3-f4", "entropy"] as const;
export const CALCULATORS = ["mean", "maximum", "standard_deviation"] as const;
export type Calculator = (typeof CALCULATORS)[number];

// Four samples per window; real windows hold hundreds.
export const RAW_WINDOWS: Record<
  (typeof TREMOR_COLUMNS)[number],
  readonly (readonly number[])[]
> = {
  rsam_f2: [
    [1.2, 1.4, 1.1, 1.3],
    [1.5, 1.7, 1.6, 2.0],
    [2.4, 3.1, 2.8, 3.6],
  ],
  "dsar_f3-f4": [
    [0.8, 0.9, 0.8, 0.7],
    [0.9, 1.1, 1.0, 1.2],
    [1.4, 1.6, 1.9, 1.5],
  ],
  entropy: [
    [3.1, 3.0, 3.2, 3.1],
    [2.9, 3.0, 2.8, 2.9],
    [2.4, 2.2, 2.5, 2.3],
  ],
};
export const WINDOW_LABELS = [0, 0, 1] as const;

export function calculate(name: Calculator, values: readonly number[]) {
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  if (name === "mean") return mean;
  if (name === "maximum") return Math.max(...values);
  // tsfresh's standard_deviation is the population SD.
  return Math.sqrt(
    values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length,
  );
}

export interface FeatureCandidate {
  name: string;
  p: number;
  importance: number;
}

export const FEATURE_CANDIDATES: readonly FeatureCandidate[] = [
  { name: "rsam_f2__mean", p: 1e-6, importance: 0.21 },
  { name: "rsam_f3__maximum", p: 3e-5, importance: 0.05 },
  { name: "dsar_f3-f4__variance", p: 2e-4, importance: 0.17 },
  { name: "entropy__mean", p: 8e-4, importance: 0.12 },
  { name: "rsam_f0__median", p: 3e-3, importance: 0.02 },
  { name: "dsar_f0-f1__minimum", p: 2e-2, importance: 0.09 },
  { name: "entropy__standard_deviation", p: 0.2, importance: 0.04 },
  { name: "rsam_f4__mean", p: 0.6, importance: 0.01 },
];
export const FDR_ILLUSTRATIVE_CUTOFF = 0.05;

export interface ScoredFeature extends FeatureCandidate {
  score: number;
  rank: number;
}

// Each seed undersamples the majority class differently, so each seed trains
// feature selection on a different sample and can rank the features differently.
// These tables are illustrative and deliberately exaggerated: they are set by
// hand, in FEATURE_CANDIDATES order, so a reader can see a feature (here
// rsam_f2__mean) lead for one seed and trail for another. A strong feature in
// real data usually moves less.
//                         mean    max     dsar-var entropy  median  dsar-min ent-sd   f4-mean
export const SEED_P_VALUES: readonly (readonly number[])[] = [
  /* seed 0 */ [1e-6, 3e-5, 2e-4, 8e-4, 3e-3, 2e-2, 0.2, 0.6],
  /* seed 1 */ [0.45, 4e-6, 3e-5, 1e-3, 0.08, 7e-5, 0.02, 0.3],
  /* seed 2 */ [2e-3, 6e-4, 5e-6, 0.09, 1e-4, 0.4, 7e-3, 0.15],
];
export const SEED_IMPORTANCES: readonly (readonly number[])[] = [
  /* seed 0 */ [0.21, 0.05, 0.17, 0.12, 0.02, 0.09, 0.04, 0.01],
  /* seed 1 */ [0.005, 0.2, 0.15, 0.1, 0.03, 0.18, 0.06, 0.04],
  /* seed 2 */ [0.1, 0.14, 0.22, 0.02, 0.18, 0.01, 0.08, 0.06],
];

export function scoreFeatures(
  seed: number,
  method: "tsfresh" | "random_forest",
): ScoredFeature[] {
  const row = seed % SEED_P_VALUES.length;
  const scored = FEATURE_CANDIDATES.map((feature, i) => {
    const p = SEED_P_VALUES[row]![i]!;
    const importance = SEED_IMPORTANCES[row]![i]!;
    return {
      ...feature,
      p,
      importance,
      score: method === "tsfresh" ? -Math.log10(p) : importance,
      rank: 0,
    };
  });
  return [...scored]
    .sort((a, b) => b.score - a.score)
    .map((feature, i) => ({ ...feature, rank: i + 1 }));
}

export const GRID = [
  { n_estimators: 50, max_depth: 3 },
  { n_estimators: 50, max_depth: 5 },
  { n_estimators: 50, max_depth: 7 },
  { n_estimators: 100, max_depth: 3 },
  { n_estimators: 100, max_depth: 5 },
  { n_estimators: 100, max_depth: 7 },
] as const;
const BASE_SCORES = [0.62, 0.68, 0.71, 0.66, 0.74, 0.7];

export function gridScores(seed: number) {
  return BASE_SCORES.map(
    (score, i) => score + (((seed * 7 + i * 3) % 5) - 2) * 0.012,
  );
}

export const CV_ROWS = 14; // 8 negatives (ids 1-8) + 6 positives (ids 9-14)
export const CV_STRATEGIES = {
  shuffle: {
    name: "ShuffleSplit",
    note: "Random test rows. Fold 1 can hold no positives at all.",
    folds: [
      [1, 3, 6, 8],
      [2, 4, 9, 10],
      [5, 7, 11, 12],
    ],
  },
  stratified: {
    name: "StratifiedKFold",
    note: "Each row is tested exactly once, and every fold keeps positives.",
    folds: [
      [1, 2, 3, 9, 10],
      [4, 5, 6, 11, 12],
      [7, 8, 13, 14],
    ],
  },
  "shuffle-stratified": {
    name: "StratifiedShuffleSplit",
    note: "Random draws that keep the class mix; a row can reappear across folds.",
    folds: [
      [2, 6, 10, 13],
      [1, 7, 9, 14],
      [3, 6, 11, 12],
    ],
  },
} as const;
export type CvStrategy = keyof typeof CV_STRATEGIES;
