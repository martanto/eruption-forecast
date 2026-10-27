import type { ReactNode } from "react";

/** Fixed settings a walkthrough uses, shown as `label / value` pairs under the heading. */
export function ParameterStrip({
  items,
}: {
  items: readonly (readonly [label: string, value: ReactNode])[];
}) {
  return (
    <div className="lab-parameters">
      {items.map(([label, value]) => (
        <span key={label}>
          {label} <strong>{value}</strong>
        </span>
      ))}
    </div>
  );
}
