/** Probability or share as a percentage; non-finite values read "n/a". */
export const percent = (value: number, digits = 1) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";

/** Fixed-point number; non-finite values read "n/a". */
export const fixed = (value: number, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "n/a";

/** A small probability as a percentage, with precision that grows as the value shrinks. */
export const probabilityLabel = (probability: number) => {
  const pct = probability * 100;
  if (pct < 0.01) return "<0.01%";
  if (pct < 1) return `${pct.toFixed(2)}%`;
  if (pct < 10) return `${pct.toFixed(1)}%`;
  return `${pct.toFixed(0)}%`;
};

/** Whole number with thousands separators, fixed to `en-US` so the text never depends on the visitor's locale. */
export const integer = (value: number) => value.toLocaleString("en-US");

/** Accessible name for a diagram box that jumps to a walkthrough step. */
export const goToStepLabel = (node: {
  title: string;
  detail: string;
  step: number;
}) => `${node.title}: ${node.detail}. Go to step ${node.step + 1}`;
