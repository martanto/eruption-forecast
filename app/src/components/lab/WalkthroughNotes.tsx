import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Short framing card; pass `id` to label the section by its heading. */
export function WhySection({
  title = "Why use it?",
  id,
  className,
  children,
}: {
  title?: string;
  id?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <section className={cn("training-why", className)} aria-labelledby={id}>
      <h2 id={id}>{title}</h2>
      {children}
    </section>
  );
}

/** Closing two-column note: the method's limits and where this repository implements it. */
export function ImplementationNotes({
  caveat,
  implementation,
  sources,
}: {
  caveat: string;
  implementation: string;
  sources: readonly string[];
}) {
  return (
    <section className="training-notes">
      <div>
        <h2>What this does—and what it cannot fix</h2>
        <p>{caveat}</p>
      </div>
      <div>
        <h2>How this repository builds it</h2>
        <p>{implementation}</p>
        <ul>
          {sources.map((source) => (
            <li key={source}>
              <code>{source}</code>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
