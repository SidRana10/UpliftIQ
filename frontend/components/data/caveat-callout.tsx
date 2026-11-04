import { TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";

/** Amber limitation block. Used for limits that must stay visible (never collapsed). */
export function CaveatCallout({
  title,
  items,
  className,
  columns = false,
}: {
  title: string;
  items: React.ReactNode[];
  className?: string;
  columns?: boolean;
}) {
  return (
    <div
      role="note"
      className={cn("rounded-control border border-amber/40 bg-amber-soft p-3.5", className)}
    >
      <div className="flex items-center gap-2 text-[13px] font-semibold text-amber-ink">
        <TriangleAlert className="size-4 shrink-0" aria-hidden />
        {title}
      </div>
      <ul
        className={cn(
          "mt-2 pl-6 text-xs leading-[18px] text-fg-secondary [&>li]:list-disc [&>li]:marker:text-amber",
          columns ? "lg:columns-2 lg:gap-x-10 [&>li]:mb-1.5 [&>li]:break-inside-avoid" : "space-y-1.5",
        )}
      >
        {items.map((it, i) => (
          <li key={i}>{it}</li>
        ))}
      </ul>
    </div>
  );
}
