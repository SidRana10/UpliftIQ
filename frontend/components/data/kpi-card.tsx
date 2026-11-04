import { cn } from "@/lib/utils";
import { InfoTip } from "./info-tip";

/**
 * Headline number with its uncertainty and evidence type. `value` and `range` arrive already
 * formatted (lib/format.ts); this component never formats numbers itself.
 */
export function KpiCard({
  label,
  value,
  range,
  note,
  badge,
  info,
  accent = "teal",
  className,
}: {
  label: string;
  value: string;
  /** e.g. "[0.616, 0.715]" shown under the value as the 95% interval */
  range?: string;
  note?: React.ReactNode;
  badge?: React.ReactNode;
  info?: React.ReactNode;
  accent?: "teal" | "violet" | "amber" | "slate";
  className?: string;
}) {
  const bar = {
    teal: "bg-teal",
    violet: "bg-violet",
    amber: "bg-amber",
    slate: "bg-ref-soft",
  }[accent];
  return (
    <div
      className={cn(
        "relative flex min-w-0 flex-col overflow-hidden rounded-panel border border-border bg-surface p-4 shadow-panel sm:p-5",
        className,
      )}
    >
      <span aria-hidden className={cn("absolute inset-y-0 left-0 w-[3px]", bar)} />
      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="eyebrow">{label}</span>
          {info ? <InfoTip label={`About ${label}`}>{info}</InfoTip> : null}
        </div>
        {badge}
      </div>
      <div className="mt-2 text-[26px] font-semibold leading-8 tracking-[-0.01em] text-fg sm:text-[28px]">
        {value}
      </div>
      {range ? (
        <div className="mt-0.5 text-xs leading-4 text-fg-muted" title="95% interval">
          95% CI {range}
        </div>
      ) : null}
      {note ? <div className="mt-2.5 text-xs leading-4 text-fg-secondary">{note}</div> : null}
    </div>
  );
}
