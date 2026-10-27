import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import "@/styles/math.css";

/** Typeset a LaTeX expression with KaTeX; `display` centres it on its own line. */
export function MathExpr({
  tex,
  display = false,
  className,
}: {
  tex: string;
  display?: boolean;
  className?: string;
}) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        displayMode: display,
        throwOnError: false,
        output: "htmlAndMathml",
      }),
    [tex, display],
  );
  return (
    <span
      className={className}
      data-math={display ? "display" : "inline"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
