import type { ReactNode } from "react";
import { Legend, type LegendItem } from "./Legend";
import { ChartScroll } from "./ScrollRegion";
import "@/styles/charts.css";

/**
 * Accessible shell for a fixed-size SVG chart: optional legend above, an
 * `<svg role="img">` that scrolls sideways on narrow screens, and the draw
 * area as `children`. `label` is required so every chart has a text summary.
 */
export function ChartFrame({
  width,
  height,
  label,
  legend,
  className,
  minWidth,
  children,
}: {
  width: number;
  height: number;
  label: string;
  legend?: readonly LegendItem[];
  className?: string;
  /** Keeps a wide chart from shrinking below readable size inside the scroll region. */
  minWidth?: number;
  children: ReactNode;
}) {
  return (
    <>
      {legend && <Legend items={legend} />}
      <ChartScroll>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="img"
          aria-label={label}
          className={className}
          style={minWidth ? { minWidth } : undefined}
        >
          {children}
        </svg>
      </ChartScroll>
    </>
  );
}
