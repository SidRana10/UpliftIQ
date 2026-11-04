/** Tooltip card shared by all charts: white panel, 1px border, popover shadow, tabular numbers. */
export function TooltipCard({
  title,
  color,
  rows,
  footnote,
}: {
  title: string;
  color?: string;
  rows: { label: string; value: string }[];
  footnote?: string;
}) {
  return (
    <div className="max-w-[260px] rounded-control border border-border bg-surface px-3 py-2.5 text-xs shadow-popover">
      <div className="flex items-center gap-1.5 font-semibold text-fg">
        {color ? <span aria-hidden className="size-2 rounded-full" style={{ background: color }} /> : null}
        {title}
      </div>
      <dl className="mt-1.5 space-y-0.5">
        {rows.map((r) => (
          <div key={r.label} className="flex justify-between gap-4">
            <dt className="text-fg-muted">{r.label}</dt>
            <dd className="font-medium text-fg">{r.value}</dd>
          </div>
        ))}
      </dl>
      {footnote ? <p className="mt-1.5 text-[11px] leading-4 text-fg-muted">{footnote}</p> : null}
    </div>
  );
}
