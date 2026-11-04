import { cva, type VariantProps } from "class-variance-authority";
import { Slot } from "radix-ui";
import * as React from "react";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-control text-sm font-medium transition-[background-color,border-color,color,box-shadow] duration-150 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary: "bg-teal text-white shadow-panel hover:bg-teal-ink active:bg-teal-ink",
        secondary:
          "border border-border-strong bg-surface text-fg shadow-panel hover:bg-subtle active:bg-subtle",
        ghost: "text-fg-secondary hover:bg-subtle hover:text-fg",
      },
      size: {
        default: "h-9 px-3.5",
        sm: "h-8 px-3 text-[13px]",
        icon: "size-9",
      },
    },
    defaultVariants: { variant: "secondary", size: "default" },
  },
);

export interface ButtonProps
  extends React.ComponentProps<"button">,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

export function Button({ className, variant, size, asChild = false, ...props }: ButtonProps) {
  const Comp = asChild ? Slot.Root : "button";
  return <Comp className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}

export { buttonVariants };
