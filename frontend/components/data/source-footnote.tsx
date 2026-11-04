import { cn } from "@/lib/utils";

/** "Source: reports/..." line shown under charts and tables (from `envelope.source`). */
export function SourceFootnote({
  sources,
  note,
  className,
}: {
  sources: string[];
  note?: string;
  className?: string;
}) {
  return (
    <p className={cn("text-[11px] leading-4 text-fg-muted", className)}>
      {note ? <span>{note} </span> : null}
      <span>Source: </span>
      <span className="font-mono text-[10.5px] break-all">{sources.join(" · ")}</span>
    </p>
  );
}
