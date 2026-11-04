import { cn } from "@/lib/utils";

/** Sticky control strip under the top bar: the selectors that define the view below. */
export function SubBar({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div
      role="group"
      aria-label="View controls"
      className={cn(
        "z-20 -mx-4 border-b border-border bg-bg/90 px-4 py-2.5 backdrop-blur supports-[backdrop-filter]:bg-bg/80 sm:-mx-6 sm:px-6 xl:-mx-8 xl:px-8 md:sticky md:top-14",
        className,
      )}
    >
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">{children}</div>
    </div>
  );
}

export function Control({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="eyebrow shrink-0">{label}</span>
      {children}
    </div>
  );
}
