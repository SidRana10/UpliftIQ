"use client";

import { ChartColumn, Table2 } from "lucide-react";
import { useId, useState } from "react";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { SourceFootnote } from "@/components/data/source-footnote";
import { Button } from "@/components/ui/button";

export interface LegendItem {
  label: string;
  color: string;
  /** "dot" = point estimate with whisker; "dash" = reference line; "band" = interval band */
  shape?: "dot" | "dot-hollow" | "dash" | "band";
}

function LegendSwatch({ item }: { item: LegendItem }) {
  const s = item.shape ?? "dot";
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-fg-secondary">
      <svg width="16" height="10" aria-hidden>
        {s === "dot" ? (
          <>
            <line x1="1" x2="15" y1="5" y2="5" stroke={item.color} strokeWidth="1.5" />
            <circle cx="8" cy="5" r="3.5" fill={item.color} />
          </>
        ) : null}
        {s === "dot-hollow" ? (
          <>
            <line x1="1" x2="15" y1="5" y2="5" stroke={item.color} strokeWidth="1.5" />
            <circle cx="8" cy="5" r="3.2" fill="#fff" stroke={item.color} strokeWidth="1.6" />
          </>
        ) : null}
        {s === "dash" ? <line x1="0" x2="16" y1="5" y2="5" stroke={item.color} strokeWidth="1.6" strokeDasharray="3 2" /> : null}
        {s === "band" ? <rect x="0" y="1" width="16" height="8" rx="2" fill={item.color} opacity="0.25" /> : null}
      </svg>
      {item.label}
    </span>
  );
}

/**
 * Panel around every chart: title, units + evidence in the subtitle, legend, a "View as table"
 * toggle (the exact plotted values, for accessibility and auditing) and a source footnote.
 */
export function ChartFrame({
  title,
  subtitle,
  badge,
  legend,
  chart,
  table,
  sources,
  note,
  className,
}: {
  title: string;
  subtitle: React.ReactNode;
  badge?: React.ReactNode;
  legend?: LegendItem[];
  chart: React.ReactNode;
  table: React.ReactNode;
  sources: string[];
  note?: string;
  className?: string;
}) {
  const [asTable, setAsTable] = useState(false);
  const id = useId();
  return (
    <Panel className={className}>
      <PanelHeader
        title={title}
        subtitle={subtitle}
        actions={
          <>
            {badge}
            <Button
              variant="ghost"
              size="sm"
              aria-pressed={asTable}
              aria-controls={id}
              onClick={() => setAsTable((v) => !v)}
              className="-mr-1 h-7 px-2 text-xs"
            >
              {asTable ? <ChartColumn /> : <Table2 />}
              <span className="hidden sm:inline">{asTable ? "View chart" : "View as table"}</span>
              <span className="sr-only sm:hidden">{asTable ? "View chart" : "View as table"}</span>
            </Button>
          </>
        }
      />
      <PanelBody className="pt-2">
        {legend && !asTable ? (
          <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1">
            {legend.map((l) => (
              <LegendSwatch key={l.label} item={l} />
            ))}
          </div>
        ) : null}
        <div id={id}>{asTable ? table : chart}</div>
        <SourceFootnote sources={sources} note={note} className="mt-3" />
      </PanelBody>
    </Panel>
  );
}
