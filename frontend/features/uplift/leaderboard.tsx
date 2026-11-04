"use client";

import { ArrowDown, ArrowUp, ArrowUpDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { SourceFootnote } from "@/components/data/source-footnote";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { InfoTip } from "@/components/data/info-tip";
import { Badge } from "@/components/ui/badge";
import { count, criteo, gigabytes, ms, seconds } from "@/lib/format";
import { cn } from "@/lib/utils";
import {
  fmtDelta,
  fmtEst,
  sortLeaderboard,
  type LeaderRow,
  type MetricKey,
  type SortKey,
} from "./select";

interface Col {
  key: SortKey;
  header: string;
  metric?: MetricKey;
  muted?: boolean;
  info?: string;
}

const COLS: Col[] = [
  { key: "v10", header: "AIPW @10%", metric: "v10" },
  { key: "v20", header: "AIPW @20%", metric: "v20" },
  { key: "v30", header: "AIPW @30%", metric: "v30" },
  { key: "vsRandom20", header: "vs random @20%", info: "Paired difference against contacting the same share at random." },
  { key: "qini", header: "Qini (AIPW)", metric: "qini", info: "Area between the gain curve and the random line, over all budgets." },
  { key: "ipw20", header: "IPW @20%", muted: true, info: "Cross-check only. IPW reads higher than AIPW on this file (see Experiment Lab)." },
  { key: "trainSeconds", header: "Train", info: "Training time at full scale, single machine." },
  { key: "p50Ms1", header: "p50 latency", info: "Median latency to score a batch of one row." },
  { key: "rowsPerS", header: "Rows/s", info: "Throughput scoring batches of 100,000 rows." },
  { key: "peakRssGb", header: "Memory", info: "Peak process memory (RSS)." },
];

function ModelBadges({ r }: { r: LeaderRow }) {
  return (
    <span className="flex flex-wrap gap-1">
      {r.selected ? <Badge tone="teal">Selected</Badge> : null}
      {r.bestByValue ? <Badge tone="violet">Best by value</Badge> : null}
      {r.bestByQini ? <Badge tone="violet">Best by Qini</Badge> : null}
    </span>
  );
}

function Cell({ r, c }: { r: LeaderRow; c: Col }) {
  switch (c.key) {
    case "v10":
    case "v20":
    case "v30":
    case "qini": {
      const f = fmtEst(r[c.key]);
      return (
        <div className="text-right">
          <div className="font-semibold tabular-nums text-fg">{f.value}</div>
          <div className="text-[11px] tabular-nums text-fg-muted">{f.range}</div>
        </div>
      );
    }
    case "vsRandom20": {
      const f = fmtDelta(r.vsRandom20);
      return (
        <div className="text-right">
          <div className="font-medium tabular-nums text-fg">{f.value}</div>
          <div className="text-[11px] tabular-nums text-fg-muted">{f.range}</div>
        </div>
      );
    }
    case "ipw20":
      return <div className="text-right tabular-nums text-fg-muted">{criteo.pp(r.ipw20)}</div>;
    case "trainSeconds":
      return <div className="text-right tabular-nums">{seconds(r.trainSeconds)}</div>;
    case "p50Ms1":
      return (
        <div className="text-right tabular-nums">
          {ms(r.p50Ms1)}
          <div className="text-[11px] text-fg-muted">p95 {ms(r.p95Ms1)}</div>
        </div>
      );
    case "rowsPerS":
      return <div className="text-right tabular-nums">{count(r.rowsPerS)}</div>;
    case "peakRssGb":
      return <div className="text-right tabular-nums">{gigabytes(r.peakRssGb)}</div>;
    default:
      return null;
  }
}

function SortIcon({ active, dir }: { active: boolean; dir: "asc" | "desc" }) {
  if (!active) return <ArrowUpDown className="size-3 opacity-40" aria-hidden />;
  return dir === "asc" ? <ArrowUp className="size-3" aria-hidden /> : <ArrowDown className="size-3" aria-hidden />;
}

/** Full-scale finalists ranked on the validation subset; sort by any column, open a row for its model card. */
export function Leaderboard({
  rows,
  compared,
  metric,
  onOpen,
  sources,
  trainRows,
}: {
  rows: LeaderRow[];
  compared: string[];
  metric: MetricKey;
  onOpen: (model: string) => void;
  sources: string[];
  trainRows: number;
}) {
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "v20", dir: "desc" });
  const sorted = sortLeaderboard(rows, sort.key, sort.dir);
  const toggle = (key: SortKey) =>
    setSort((s) => (s.key === key ? { key, dir: s.dir === "desc" ? "asc" : "desc" } : { key, dir: key === "trainSeconds" || key === "p50Ms1" || key === "peakRssGb" ? "asc" : "desc" }));

  return (
    <Panel>
      <PanelHeader
        title="Model leaderboard"
        subtitle={`Finalists trained on ${count(trainRows)} rows · scored on the 500K validation subset · 95% intervals · sort by any column`}
        actions={<EvidenceBadge kind="validation" />}
      />
      <PanelBody>
        <div className="hidden scrollbar-thin overflow-x-auto md:block">
          <table className="w-full border-collapse text-[13px]">
            <caption className="sr-only">Model leaderboard, validation evidence. Column headers sort the table.</caption>
            <thead>
              <tr className="border-b border-border text-fg-muted">
                <th scope="col" className="bg-subtle/60 px-3 py-2 text-left text-[11px] font-medium uppercase tracking-[0.05em]">
                  Model
                </th>
                {COLS.map((c) => (
                  <th
                    key={c.key}
                    scope="col"
                    aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                    className={cn("whitespace-nowrap bg-subtle/60 px-1 py-1 text-right text-[11px] font-medium uppercase tracking-[0.05em]", c.metric === metric && "bg-violet-soft/70 text-violet-ink")}
                  >
                    <span className="inline-flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => toggle(c.key)}
                        className="inline-flex items-center gap-1 rounded-control px-1.5 py-1 uppercase tracking-[0.05em] hover:text-fg"
                      >
                        {c.header}
                        <SortIcon active={sort.key === c.key} dir={sort.dir} />
                      </button>
                      {c.info ? <InfoTip label={`About ${c.header}`}>{c.info}</InfoTip> : null}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => (
                <tr key={r.model} className={cn("border-b border-border/70 last:border-0", compared.includes(r.model) && "bg-subtle/40")}>
                  <th scope="row" className="whitespace-nowrap px-3 py-3 text-left align-middle font-normal">
                    <button
                      type="button"
                      onClick={() => onOpen(r.model)}
                      className="group flex items-center gap-2 rounded-control text-left"
                      aria-label={`Open model card for ${r.label}`}
                    >
                      <span aria-hidden className="size-2.5 shrink-0 rounded-full" style={{ background: r.color }} />
                      <span>
                        <span className="flex items-center gap-1 font-semibold text-fg">
                          {r.label}
                          <ChevronRight className="size-3.5 text-fg-muted transition-transform group-hover:translate-x-0.5" aria-hidden />
                        </span>
                        <ModelBadges r={r} />
                      </span>
                    </button>
                  </th>
                  {COLS.map((c) => (
                    <td key={c.key} className={cn("px-2 py-3 align-middle", c.muted && "opacity-70", c.metric === metric && "bg-violet-soft/30")}>
                      <Cell r={r} c={c} />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <ul className="space-y-3 md:hidden">
          {sorted.map((r) => (
            <li key={r.model} className="rounded-control border border-border p-3">
              <button type="button" onClick={() => onOpen(r.model)} className="flex w-full items-start justify-between gap-2 text-left" aria-label={`Open model card for ${r.label}`}>
                <span className="min-w-0">
                  <span className="flex items-center gap-2 font-semibold text-fg">
                    <span aria-hidden className="size-2.5 rounded-full" style={{ background: r.color }} />
                    {r.label}
                  </span>
                  <span className="mt-1 block">
                    <ModelBadges r={r} />
                  </span>
                </span>
                <ChevronRight className="mt-1 size-4 shrink-0 text-fg-muted" aria-hidden />
              </button>
              <dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-2 text-[13px]">
                {(["v10", "v20", "v30", "qini"] as const).map((k) => {
                  const f = fmtEst(r[k]);
                  const label = { v10: "AIPW @10%", v20: "AIPW @20%", v30: "AIPW @30%", qini: "Qini" }[k];
                  return (
                    <div key={k}>
                      <dt className="text-[11px] text-fg-muted">{label}</dt>
                      <dd className="font-semibold tabular-nums">{f.value}</dd>
                      <dd className="text-[11px] tabular-nums text-fg-muted">{f.range}</dd>
                    </div>
                  );
                })}
                <div>
                  <dt className="text-[11px] text-fg-muted">Train time</dt>
                  <dd className="tabular-nums">{seconds(r.trainSeconds)}</dd>
                </div>
                <div>
                  <dt className="text-[11px] text-fg-muted">Latency p50 (1 row)</dt>
                  <dd className="tabular-nums">{ms(r.p50Ms1)}</dd>
                </div>
                <div>
                  <dt className="text-[11px] text-fg-muted">Peak memory</dt>
                  <dd className="tabular-nums">{gigabytes(r.peakRssGb)}</dd>
                </div>
              </dl>
            </li>
          ))}
        </ul>

        <p className="mt-3 text-xs leading-5 text-fg-muted">
          Percentage points of visit rate over treating nobody, estimated on the validation subset the models were selected on; they are not held-out results. IPW is shown muted because it is a cross-check that reads high on this file.
        </p>
        <SourceFootnote sources={sources} className="mt-2" />
      </PanelBody>
    </Panel>
  );
}
