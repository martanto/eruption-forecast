import type { ReactNode } from "react";

/** Heading row of a lab panel: a title on the left, a short qualifier on the right. */
export function LabSectionTitle({
  title,
  aside,
  id,
}: {
  title: ReactNode;
  aside?: ReactNode;
  id?: string;
}) {
  return (
    <div className="lab-section-title">
      <h2 id={id}>{title}</h2>
      {aside && <span>{aside}</span>}
    </div>
  );
}
