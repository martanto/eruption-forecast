import type { ReactNode } from "react";
import type { Scale } from "@/lib/scale";
import "@/styles/charts.css";

export type GridVariant = "solid" | "dashed" | "minor";
type LabelSize = "default" | "small";

/** Horizontal gridlines with right-aligned tick labels on the left edge of a plot. */
export function YAxis({
  ticks,
  scale,
  left,
  right,
  format = String,
  grid = "solid",
  labels = true,
  labelGap = 6,
  size = "default",
}: {
  ticks: readonly number[];
  scale: Scale;
  left: number;
  right: number;
  format?: (tick: number) => ReactNode;
  /** `false` draws labels only. */
  grid?: GridVariant | false;
  /** `false` draws gridlines only, for minor ticks. */
  labels?: boolean;
  labelGap?: number;
  size?: LabelSize;
}) {
  return (
    <>
      {ticks.map((tick) => (
        <g key={tick}>
          {grid && (
            <line
              className="chart-grid"
              data-variant={grid}
              x1={left}
              x2={right}
              y1={scale(tick)}
              y2={scale(tick)}
            />
          )}
          {labels && (
            <text
              className="chart-axis"
              data-size={size}
              x={left - labelGap}
              y={scale(tick) + 4}
              textAnchor="end"
            >
              {format(tick)}
            </text>
          )}
        </g>
      ))}
    </>
  );
}

/** Centred tick labels along the bottom of a plot, with optional vertical gridlines. */
export function XAxis({
  ticks,
  scale,
  labelY,
  format = String,
  gridFrom,
  gridTo,
  grid = "solid",
  size = "default",
}: {
  ticks: readonly number[];
  scale: Scale;
  /** Baseline of the tick labels. */
  labelY: number;
  format?: (tick: number) => ReactNode;
  /** Vertical gridlines run from `gridFrom` to `gridTo`; omit both for labels only. */
  gridFrom?: number;
  gridTo?: number;
  grid?: GridVariant;
  size?: LabelSize;
}) {
  const hasGrid = gridFrom !== undefined && gridTo !== undefined;
  return (
    <>
      {ticks.map((tick) => (
        <g key={tick}>
          {hasGrid && (
            <line
              className="chart-grid"
              data-variant={grid}
              x1={scale(tick)}
              x2={scale(tick)}
              y1={gridFrom}
              y2={gridTo}
            />
          )}
          <text
            className="chart-axis"
            data-size={size}
            x={scale(tick)}
            y={labelY}
            textAnchor="middle"
          >
            {format(tick)}
          </text>
        </g>
      ))}
    </>
  );
}

/** Axis title centred on `(x, y)`; `rotate` turns it to read upward beside a y axis. */
export function AxisTitle({
  x,
  y,
  rotate = false,
  children,
}: {
  x: number;
  y: number;
  rotate?: boolean;
  children: ReactNode;
}) {
  return (
    <text
      className="chart-axis-title"
      x={x}
      y={y}
      textAnchor="middle"
      transform={rotate ? `rotate(-90 ${x} ${y})` : undefined}
    >
      {children}
    </text>
  );
}
