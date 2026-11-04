import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 whitespace-nowrap rounded-chip px-2 py-0.5 text-[11px] font-medium leading-4 [&_svg]:size-3",
  {
    variants: {
      tone: {
        teal: "bg-teal-soft text-teal-ink",
        "teal-outline": "border border-teal/40 bg-surface text-teal-ink",
        violet: "bg-violet-soft text-violet-ink",
        "violet-outline": "border border-violet/40 bg-surface text-violet-ink",
        amber: "bg-amber-soft text-amber-ink",
        "amber-solid": "bg-amber text-[#3f2a08]",
        slate: "bg-subtle text-fg-secondary",
        danger: "bg-danger-soft text-danger",
      },
    },
    defaultVariants: { tone: "slate" },
  },
);

export interface BadgeProps extends React.ComponentProps<"span">, VariantProps<typeof badgeVariants> {}

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />;
}
