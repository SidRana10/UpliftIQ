import { cn } from "@/lib/utils";

export interface Column<R> {
  key: string;
  header: string;
  align?: "left" | "right";
  cell: (row: R) => React.ReactNode;
  className?: string;
}

/** Plain accessible table; wide tables scroll inside their panel, never the page. */
export function DataTable<R>({
  columns,
  rows,
  rowKey,
  caption,
  className,
}: {
  columns: Column<R>[];
  rows: R[];
  rowKey: (row: R, index: number) => string;
  caption?: string;
  className?: string;
}) {
  return (
    <div className={cn("scrollbar-thin overflow-x-auto", className)}>
      <table className="w-full border-collapse text-[13px] leading-[18px]">
        {caption ? <caption className="sr-only">{caption}</caption> : null}
        <thead>
          <tr className="border-b border-border text-fg-muted">
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={cn(
                  "whitespace-nowrap bg-subtle/60 px-3 py-2 text-[11px] font-medium uppercase tracking-[0.05em] first:rounded-tl-control last:rounded-tr-control",
                  c.align === "right" ? "text-right" : "text-left",
                )}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={rowKey(r, i)} className="border-b border-border/70 last:border-0">
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={cn(
                    "px-3 py-2.5 align-middle",
                    c.align === "right" ? "text-right" : "text-left",
                    c.className,
                  )}
                >
                  {c.cell(r)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
