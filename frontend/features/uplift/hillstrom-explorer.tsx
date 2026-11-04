"use client";

import { ChartFrame } from "@/components/charts/chart-frame";
import { DotWhiskerChart } from "@/components/charts/dot-whisker-chart";
import { niceAxis } from "@/components/charts/theme";
import { CaveatCallout } from "@/components/data/caveat-callout";
import { DataTable } from "@/components/data/data-table";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { Badge } from "@/components/ui/badge";
import { COLORS } from "@/lib/colors";
import type { CandidateRow, HsDiffRow } from "./select";

export function HillstromExplorer({
  candidates,
  diffs,
  summary,
  chosenLabel,
  sources,
  caveats,
}: {
  candidates: CandidateRow[];
  diffs: HsDiffRow[];
  summary: string[];
  chosenLabel: string;
  sources: string[];
  caveats: string[];
}) {
  const mens = candidates.find((c) => c.chosen);
  const valueAxis = niceAxis(Math.min(...candidates.map((c) => c.low)) - 0.3, Math.max(...candidates.map((c) => c.high)) + 0.3);
  const diffAxis = niceAxis(Math.min(0, ...diffs.map((d) => d.low)) - 0.05, Math.max(0, ...diffs.map((d) => d.high)) + 0.05);

  return (
    <div className="space-y-6 lg:space-y-8">
      <div
        role="note"
        className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-control border border-violet/30 bg-violet-soft/60 px-4 py-2.5 text-[13px] text-fg-secondary"
      >
        <EvidenceBadge kind="cross-validated" />
        <span>
          Cross-validated on the development set only. The one-time held-out evaluation is in the Overview and Decision Studio; nothing here uses the held-out customers.
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-8">
          <ChartFrame
            title="Candidate policies"
            subtitle="Expected visit rate if every customer followed the policy · AIPW, out-of-fold · 95% intervals"
            badge={<EvidenceBadge kind="cross-validated" />}
            legend={[
              { label: "Selected policy", color: COLORS.teal, shape: "dot" },
              { label: "Personalized model", color: COLORS.violet, shape: "dot-hollow" },
              { label: "Fixed baseline", color: COLORS.ref, shape: "dot-hollow" },
            ]}
            chart={
              <DotWhiskerChart
                rows={candidates.map((c) => ({
                  key: c.policy,
                  label: c.label,
                  sublabel: c.chosen ? "Selected" : c.kind === "model" ? (c.eligible ? "Eligible model" : "Ineligible model") : "Fixed baseline",
                  value: c.value,
                  low: c.low,
                  high: c.high,
                  color: c.color,
                  emphasis: c.chosen,
                  hollow: !c.chosen,
                  valueLabel: c.valueLabel,
                  rangeLabel: c.rangeLabel,
                  tooltipNote: c.config ? `Config: ${c.config}` : undefined,
                }))}
                domain={valueAxis.domain}
                ticks={valueAxis.ticks}
                tickFormat={(v) => `${v.toFixed(0)}%`}
                axisTitle="Expected visit rate (%)"
                refLines={mens ? [{ x: mens.value, label: "Men's for all", color: COLORS.teal }] : []}
                rowHeight={46}
              />
            }
            table={
              <DataTable
                caption="Candidate policies, cross-validated"
                rows={candidates}
                rowKey={(c) => c.policy}
                columns={[
                  { key: "p", header: "Policy", cell: (c) => <span className="font-medium">{c.label}</span> },
                  { key: "k", header: "Type", cell: (c) => (c.kind === "model" ? "Personalized model" : "Fixed baseline") },
                  { key: "v", header: "AIPW visit rate", align: "right", cell: (c) => c.valueLabel },
                  { key: "ci", header: "95% CI", align: "right", cell: (c) => c.rangeLabel },
                  { key: "i", header: "IPW check", align: "right", cell: (c) => <span className="text-fg-muted">{c.ipw}</span> },
                  { key: "c", header: "Config", cell: (c) => <span className="font-mono text-[11px] text-fg-muted">{c.config ?? "n/a"}</span> },
                ]}
              />
            }
            sources={sources}
            note="Hyperparameters were tuned on the same out-of-fold data, so tuned-model values are slightly optimistic; the untouched held-out set is the unbiased check."
            className="h-full"
          />
        </div>
        <div className="min-w-0 lg:col-span-4">
          <Panel className="h-full">
            <PanelHeader title="Selection rationale" subtitle="Frozen configuration" actions={<Badge tone="teal">{chosenLabel}</Badge>} />
            <PanelBody className="space-y-3">
              <ul className="space-y-2.5 pl-5 text-[13px] leading-[21px] text-fg-secondary [&>li]:list-disc [&>li]:marker:text-ref-soft">
                {summary.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
              <div className="flex flex-wrap gap-1.5 border-t border-border pt-3">
                {candidates
                  .filter((c) => c.kind === "model")
                  .map((c) => (
                    <Badge key={c.policy} tone={c.eligible ? "teal" : "slate"}>
                      {c.label}: {c.eligible ? "eligible" : "ineligible"}
                    </Badge>
                  ))}
              </div>
            </PanelBody>
          </Panel>
        </div>
      </div>

      <ChartFrame
        title="Candidates vs Men's e-mail for all"
        subtitle="Paired difference in expected visit rate, candidate minus Men's e-mail for all · cross-validated · 95% intervals"
        badge={<EvidenceBadge kind="cross-validated" />}
        legend={[
          { label: "Interval excludes zero", color: COLORS.teal, shape: "dot" },
          { label: "Interval includes zero", color: COLORS.ref, shape: "dot-hollow" },
        ]}
        chart={
          <DotWhiskerChart
            rows={diffs.map((d) => ({
              key: d.key,
              label: d.label,
              value: d.value,
              low: d.low,
              high: d.high,
              color: d.includesZero ? COLORS.ref : COLORS.teal,
              hollow: d.includesZero,
              emphasis: !d.includesZero,
              valueLabel: d.valueLabel,
              rangeLabel: d.rangeLabel,
              tooltipNote: d.includesZero ? "Interval includes zero: no demonstrated difference." : "Interval excludes zero.",
            }))}
            domain={diffAxis.domain}
            ticks={diffAxis.ticks}
            tickFormat={(v) => v.toFixed(1)}
            axisTitle="Difference in visit rate (percentage points)"
            refLines={[{ x: 0, label: "Same as Men's for all", color: COLORS.axis }]}
            rowHeight={52}
          />
        }
        table={
          <DataTable
            caption="Candidates minus Men's e-mail for all"
            rows={diffs}
            rowKey={(d) => d.key}
            columns={[
              { key: "p", header: "Policy", cell: (d) => <span className="font-medium">{d.label}</span> },
              { key: "d", header: "Difference (95% CI)", align: "right", cell: (d) => `${d.valueLabel} ${d.rangeLabel}` },
              { key: "z", header: "Includes 0", align: "right", cell: (d) => (d.includesZero ? "Yes" : "No") },
            ]}
          />
        }
        sources={sources}
      />

      {caveats.length ? <CaveatCallout title="Dataset notes" items={caveats} /> : null}
    </div>
  );
}
