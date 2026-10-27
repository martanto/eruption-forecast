import { useEffect, useRef, useState } from "react";

/**
 * Tracks the rendered width of an element, for charts that redraw to fit their
 * container instead of scrolling. Attach the returned ref to the container.
 */
export function useElementWidth(initial = 720, min = 280) {
  const host = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(initial);
  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.max(min, Math.floor(entry.contentRect.width)));
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [min]);
  return [host, width] as const;
}
