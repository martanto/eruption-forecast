import type { ReactNode } from "react";

/** One gallery entry: component name, source path, what it is for, and a live demo. */
export function GallerySection({
  name,
  source,
  purpose,
  children,
}: {
  name: string;
  source: string;
  purpose: string;
  children: ReactNode;
}) {
  const id = `gallery-${name}`;
  return (
    <section className="gallery-entry" aria-labelledby={id}>
      <header>
        <h2 id={id}>
          <code>{name}</code>
        </h2>
        <code className="gallery-source">{source}</code>
      </header>
      <p>{purpose}</p>
      <div className="gallery-stage">{children}</div>
    </section>
  );
}
