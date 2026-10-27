import type { ReactNode } from "react";

/** Keyboard-focusable scroll container, so wide tables and charts stay reachable. */
export function ScrollRegion({
  label,
  className = "training-table-scroll",
  children,
}: {
  label: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div className={className} tabIndex={0} role="region" aria-label={label}>
      {children}
    </div>
  );
}

/** Horizontally scrolling wrapper for an SVG chart on narrow screens. */
export function ChartScroll({ children }: { children: ReactNode }) {
  return (
    <ScrollRegion
      className="sig-scroll"
      label="Chart, scrolls sideways on narrow screens"
    >
      {children}
    </ScrollRegion>
  );
}
