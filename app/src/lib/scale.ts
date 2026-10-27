export type Scale = (value: number) => number;
type Pair = readonly [number, number];

/** Maps `domain` linearly onto `range`; a flipped range inverts the axis (screen y grows downward). */
export function linearScale([d0, d1]: Pair, [r0, r1]: Pair): Scale {
  const span = d1 - d0;
  return (value) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** Logarithmic scale; values below the domain minimum clamp to it, so a p-value of 0 still plots. */
export function logScale([d0, d1]: Pair, range: Pair): Scale {
  const scale = linearScale([Math.log10(d0), Math.log10(d1)], range);
  return (value) => scale(Math.log10(Math.max(value, d0)));
}

/** SVG `points` string for a polyline or polygon, rounded to 0.1 px. */
export const linePoints = (points: Iterable<readonly [number, number]>) =>
  Array.from(points, ([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

/** SVG path `d` string for a straight-segment line, rounded to 0.1 px. */
export const linePath = (points: Iterable<readonly [number, number]>) =>
  Array.from(
    points,
    ([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`,
  ).join(" ");
