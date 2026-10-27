import type { ComponentProps } from "react";
import { cn } from "@/lib/utils";

/** Live-region read-out that summarises the current state of a lab demo. */
export function ResultOutput({
  className,
  ...props
}: ComponentProps<"output">) {
  return (
    <output
      className={cn("training-result", className)}
      aria-live="polite"
      {...props}
    />
  );
}
