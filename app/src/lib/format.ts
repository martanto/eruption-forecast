/** Probability or share as a percentage; non-finite values read "n/a". */
export const percent = (value: number, digits = 1) =>
  Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "n/a";

/** Fixed-point number; non-finite values read "n/a". */
export const fixed = (value: number, digits = 3) =>
  Number.isFinite(value) ? value.toFixed(digits) : "n/a";

/** Accessible name for a diagram box that jumps to a walkthrough step. */
export const goToStepLabel = (node: {
  title: string;
  detail: string;
  step: number;
}) => `${node.title}: ${node.detail}. Go to step ${node.step + 1}`;
