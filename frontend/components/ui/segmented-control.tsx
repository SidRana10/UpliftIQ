"use client";

import { ToggleGroup } from "radix-ui";
import { cn } from "@/lib/utils";

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
  icon?: React.ReactNode;
}

/** Single-select segmented control (arrow-key navigation, roving focus via Radix). */
export function SegmentedControl<T extends string>({
  value,
  onValueChange,
  options,
  label,
  className,
}: {
  value: T;
  onValueChange: (next: T) => void;
  options: SegmentOption<T>[];
  label: string;
  className?: string;
}) {
  return (
    <ToggleGroup.Root
      type="single"
      value={value}
      // Radix emits "" when the active item is clicked again; ignore it (always one selected)
      onValueChange={(v) => v && onValueChange(v as T)}
      aria-label={label}
      className={cn("inline-flex rounded-control bg-subtle p-0.5", className)}
    >
      {options.map((o) => (
        <ToggleGroup.Item
          key={o.value}
          value={o.value}
          className="inline-flex h-8 items-center gap-1.5 rounded-[6px] px-3 text-[13px] font-medium text-fg-secondary transition-colors duration-150 hover:text-fg data-[state=on]:bg-surface data-[state=on]:text-fg data-[state=on]:shadow-panel [&_svg]:size-3.5"
        >
          {o.icon}
          {o.label}
        </ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}
