import type { HTMLAttributes, ReactNode } from "react";

export interface LegendItem {
  /** Data attribute the stylesheet keys the swatch colour on, e.g. `["group", "quiet"]`. */
  swatch: readonly [name: string, value: string];
  label: ReactNode;
}

/** Colour key for a chart; each swatch is styled from its `data-*` attribute. */
export function Legend({
  items,
  className = "sig-legend",
}: {
  items: readonly LegendItem[];
  className?: string;
}) {
  return (
    <ul className={className} aria-label="Legend">
      {items.map(({ swatch: [name, value], label }) => {
        const attribute = {
          [`data-${name}`]: value,
        } as HTMLAttributes<HTMLLIElement>;
        return (
          <li key={`${name}-${value}`} {...attribute}>
            <i aria-hidden="true" /> {label}
          </li>
        );
      })}
    </ul>
  );
}
