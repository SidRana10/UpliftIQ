"use client";

import { ChevronDown } from "lucide-react";
import { Accordion as A } from "radix-ui";
import { cn } from "@/lib/utils";

/** Single collapsible section (Radix: keyboard, aria-expanded, focus management). */
export function Collapsible({
  title,
  badge,
  children,
  defaultOpen = false,
  className,
}: {
  title: React.ReactNode;
  badge?: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
  className?: string;
}) {
  return (
    <A.Root
      type="single"
      collapsible
      defaultValue={defaultOpen ? "item" : undefined}
      className={cn("min-w-0 rounded-panel border border-border bg-surface shadow-panel", className)}
    >
      <A.Item value="item">
        <A.Header className="flex">
          <A.Trigger className="group flex w-full items-center justify-between gap-3 rounded-panel px-4 py-3.5 text-left sm:px-5">
            <span className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-sm font-semibold text-fg">
              {title}
              {badge}
            </span>
            <ChevronDown
              aria-hidden
              className="size-4 shrink-0 text-fg-muted transition-transform duration-200 group-data-[state=open]:rotate-180 motion-reduce:transition-none"
            />
          </A.Trigger>
        </A.Header>
        <A.Content className="overflow-hidden border-t border-border data-[state=closed]:hidden">
          <div className="px-4 py-4 sm:px-5">{children}</div>
        </A.Content>
      </A.Item>
    </A.Root>
  );
}
