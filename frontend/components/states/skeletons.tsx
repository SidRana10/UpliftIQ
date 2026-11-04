import { Panel } from "@/components/data/panel";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export function KpiSkeleton() {
  return (
    <div
      role="status"
      aria-label="Loading"
      className="rounded-panel border border-border bg-surface p-5 shadow-panel"
    >
      <Skeleton className="h-3 w-24" />
      <Skeleton className="mt-3 h-8 w-32" />
      <Skeleton className="mt-2 h-3 w-40" />
      <Skeleton className="mt-4 h-3 w-full" />
    </div>
  );
}

export function KpiRowSkeleton() {
  return (
    <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
      {Array.from({ length: 4 }, (_, i) => (
        <KpiSkeleton key={i} />
      ))}
    </div>
  );
}

/** Matches the final panel's height to avoid layout shift when content streams in. */
export function PanelSkeleton({ height = 320, className }: { height?: number; className?: string }) {
  return (
    <Panel role="status" aria-label="Loading" className={cn("p-5", className)}>
      <Skeleton className="h-4 w-48" />
      <Skeleton className="mt-2 h-3 w-72 max-w-full" />
      <Skeleton className="mt-5 w-full" style={{ height }} />
    </Panel>
  );
}
