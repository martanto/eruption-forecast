import type { ComponentProps } from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const BUTTON_VARIANTS = cva("button", {
  variants: {
    variant: {
      default: "button-default",
      outline: "button-outline",
      ghost: "button-ghost",
    },
    size: { default: "", icon: "button-icon", small: "button-small" },
  },
  defaultVariants: { variant: "default", size: "default" },
});
export function Button({
  className,
  variant,
  size,
  asChild = false,
  ...props
}: ComponentProps<"button"> &
  VariantProps<typeof BUTTON_VARIANTS> & { asChild?: boolean }) {
  const Comp = asChild ? Slot : "button";
  return (
    <Comp
      className={cn(BUTTON_VARIANTS({ variant, size }), className)}
      {...props}
    />
  );
}
