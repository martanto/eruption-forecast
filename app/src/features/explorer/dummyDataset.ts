// Generated once with scipy (ks_2samp, fisher_exact) from a seeded dummy dataset, then
// frozen. Windows 1-14 are non-eruptive (label 0); windows 15-24 are pre-eruption (label 1).
// p-values are exact scipy results, not approximations. Not real tremor data.

export const QUIET_COUNT = 14;
export const PRE_COUNT = 10;
export const DUMMY_DATA_SEED = 365;

export interface DummyFeature {
  name: string;
  kind: "real" | "binary";
  // 24 values: non-eruptive windows first, then pre-eruption windows.
  values: readonly number[];
  // Exact two-sided p-value from scipy (ks_2samp for real, fisher_exact for binary).
  p: number;
}

export const DUMMY_FEATURES: readonly DummyFeature[] = [
  {
    name: "rsam_f2__mean",
    kind: "real",
    values: [148.2, 151.3, 149.2, 164.1, 140.8, 143.2, 139.3, 159.7, 151.2, 121.8, 158.7, 143.6, 142.6, 142.2, 171.6, 199.7, 167.2, 184.0, 197.4, 188.1, 182.0, 164.6, 164.8, 197.5],
    p: 1.0197546878123e-06,
  },
  {
    name: "rsam_f3__maximum",
    kind: "real",
    values: [165.8, 139.9, 169.8, 150.2, 149.1, 153.4, 143.0, 160.6, 153.8, 174.0, 162.5, 142.8, 146.5, 126.3, 185.3, 140.6, 190.5, 163.2, 141.4, 196.5, 174.3, 153.1, 213.6, 193.8],
    p: 0.016443544340973334,
  },
  {
    name: "dsar_f3-f4__variance",
    kind: "real",
    values: [147.3, 152.7, 157.8, 155.7, 155.7, 143.5, 157.8, 148.7, 133.0, 162.6, 150.2, 133.9, 152.2, 132.0, 174.9, 176.9, 175.3, 161.9, 196.1, 178.6, 148.8, 159.7, 196.8, 161.2],
    p: 0.00018967437193308776,
  },
  {
    name: "entropy__mean",
    kind: "real",
    values: [151.1, 161.7, 176.5, 161.7, 146.7, 147.8, 165.9, 147.0, 153.0, 143.1, 146.1, 148.7, 143.2, 145.6, 153.4, 143.2, 173.7, 152.2, 165.0, 182.4, 187.0, 157.8, 158.5, 173.9],
    p: 0.04238304433485481,
  },
  {
    name: "rsam_f2__standard_deviation",
    kind: "real",
    values: [150.4, 139.5, 182.1, 153.8, 146.5, 147.8, 143.6, 143.0, 172.7, 146.7, 148.0, 144.6, 161.4, 152.9, 155.9, 158.4, 167.7, 172.3, 148.6, 148.8, 168.0, 155.9, 157.5, 158.9],
    p: 0.0225008871865784,
  },
  {
    name: "rsam_f0__median",
    kind: "real",
    values: [150.4, 138.5, 158.3, 138.5, 147.8, 147.2, 140.3, 133.7, 157.4, 136.4, 108.1, 137.0, 156.0, 149.7, 154.8, 181.7, 157.6, 160.6, 149.0, 145.7, 154.6, 148.6, 149.1, 151.0],
    p: 0.04238304433485481,
  },
  {
    name: "rsam_f4__count_above_mean",
    kind: "real",
    values: [157.5, 177.8, 148.4, 150.5, 131.6, 160.9, 146.3, 136.2, 147.9, 152.6, 133.7, 164.6, 174.7, 163.0, 148.1, 144.3, 165.5, 153.7, 150.7, 161.2, 140.8, 149.9, 128.5, 164.5],
    p: 0.9973567958491906,
  },
  {
    name: "rsam_f2__has_duplicate_max",
    kind: "binary",
    values: [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    p: 0.011088302597927042,
  },
  {
    name: "rsam_f2__symmetry_looking__r_0.05",
    kind: "binary",
    values: [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    p: 0.09180239601561446,
  },
  {
    name: "rsam_f2__large_standard_deviation__r_0.25",
    kind: "binary",
    values: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    p: 0.0005159958720330236,
  },
];
