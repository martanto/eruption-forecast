import type { ReactNode } from "react";
import { cn } from "@/lib/utils";
import { ScrollRegion } from "./ScrollRegion";

/** Scrollable table: `headers` fill the head row, `children` are the body rows. */
export function DataTable({
  label,
  caption,
  headers,
  className,
  children,
}: {
  label: string;
  caption?: ReactNode;
  headers: readonly ReactNode[];
  className?: string;
  children: ReactNode;
}) {
  return (
    <ScrollRegion label={label}>
      <table className={cn("training-table", className)}>
        {caption && <caption>{caption}</caption>}
        <thead>
          <tr>
            {headers.map((header, index) => (
              <th key={index}>{header}</th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </ScrollRegion>
  );
}
