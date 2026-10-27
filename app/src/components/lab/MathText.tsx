import { Fragment } from "react";
import { MathExpr } from "./MathExpr";

/** Plain prose with inline LaTeX written between `$…$` delimiters. */
export function MathText({ children }: { children: string }) {
  return (
    <>
      {children.split(/\$([^$]+)\$/).map((part, index) =>
        index % 2 === 1 ? (
          <MathExpr key={index} tex={part} />
        ) : (
          <Fragment key={index}>{part}</Fragment>
        ),
      )}
    </>
  );
}
