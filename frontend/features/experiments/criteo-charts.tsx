"use client";

import { ChartFrame } from "@/components/charts/chart-frame";
import { DotPlot, type DotPlotRow } from "@/components/charts/dot-plot";
import { niceAxis } from "@/components/charts/theme";
import { DotWhiskerChart } from "@/components/charts/dot-whisker-chart";
import { DataTable } from "@/components/data/data-table";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { Badge } from "@/components/ui/badge";
import { COLORS } from "@/lib/colors";
import { count } from "@/lib/format";
import { StrataChart } from "./charts";
import type { EstimatorRow, StratumPoint } from "./select";

export function SmdChart({
  rows,
  thresholds,
  sources,
}: {
  rows: DotPlotRow[];
  thresholds: [number, number];
  sources: string[];
}) {
  const [pass, warn] = thresholds;
  return (
    <ChartFrame
      title="Treatment balance across features"
      subtitle="Standardized mean difference, treated minus control, for features f0–f11 · training rows"
      badge={<Badge tone="slate">Randomization check</Badge>}
      legend={[
        { label: "SMD per feature", color: COLORS.teal, shape: "dot" },
        { label: `±${pass} pass limit`, color: COLORS.refSoft, shape: "dash" },
        { label: `±${warn} warning limit`, color: COLORS.amber, shape: "dash" },
      ]}
      chart={
        <DotPlot
          rows={rows}
          series={[{ key: "smd", label: "SMD", color: COLORS.teal }]}
          domain={[-warn * 1.2, warn * 1.2]}
          band={[-pass, pass]}
          thresholds={[
            { x: -pass },
            { x: pass, label: `${pass}` },
            { x: -warn, strong: true },
            { x: warn, label: `${warn}`, strong: true },
          ]}
          tickFormat={(v) => v.toFixed(2)}
          axisTitle="Standardized mean difference (treated − control)"
          describe={`Standardized mean differences: ${rows.map((r) => `${r.label} ${r.labels.smd}`).join("; ")}`}
          rowHeight={26}
        />
      }
      table={
        <DataTable
          caption="Standardized mean differences"
          rows={rows}
          rowKey={(r) => r.key}
          columns={[
            { key: "f", header: "Feature", cell: (r) => <span className="font-medium">{r.label}</span> },
            { key: "s", header: "SMD", align: "right", cell: (r) => r.labels.smd },
            ...["Mean treated", "Mean control", "KS statistic"].map((l) => ({
              key: l,
              header: l,
              align: "right" as const,
              cell: (r: DotPlotRow) => r.extra?.find((e) => e.label === l)?.value,
            })),
          ]}
        />
      }
      sources={sources}
      className="h-full"
    />
  );
}

export function EstimatorChart({ rows, sources, n }: { rows: EstimatorRow[]; sources: string[]; n: number }) {
  const axis = niceAxis(0, Math.max(...rows.map((r) => r.high)) * 1.08);
  return (
    <ChartFrame
      title="Estimated effect of treating everyone"
      subtitle={`Percentage points of visit rate · ${count(n)}-customer validation sample · 95% intervals`}
      badge={<EvidenceBadge kind="validation" />}
      legend={[
        { label: "AIPW (primary)", color: COLORS.teal, shape: "dot" },
        { label: "IPW cross-check", color: COLORS.amber, shape: "dot-hollow" },
        { label: "Other adjustments", color: COLORS.ref, shape: "dot" },
      ]}
      chart={
        <DotWhiskerChart
          rows={rows}
          domain={axis.domain}
          ticks={axis.ticks}
          tickFormat={(v) => v.toFixed(1)}
          axisTitle="Treat-all effect (pp of visit rate)"
          rowHeight={58}
        />
      }
      table={
        <DataTable
          caption="Treat-all effect by estimator"
          rows={rows}
          rowKey={(r) => r.key}
          columns={[
            { key: "e", header: "Estimator", cell: (r) => <span className="font-medium">{r.label}</span> },
            { key: "d", header: "Adjustment", cell: (r) => <span className="text-fg-muted">{r.sublabel}</span> },
            { key: "v", header: "Estimate", align: "right", cell: (r) => r.valueLabel },
            { key: "ci", header: "95% interval", align: "right", cell: (r) => r.rangeLabel },
          ]}
        />
      }
      sources={sources}
      note="Post-stratified interval is estimate ± 1.96 SE (the report stores only the SE)."
      className="h-full"
    />
  );
}

export function StrataFrame({ points, sources }: { points: StratumPoint[]; sources: string[] }) {
  return (
    <ChartFrame
      title="Treated share and effect by predicted baseline"
      subtitle="Ten deciles of independently predicted baseline visit probability · validation sample"
      badge={<EvidenceBadge kind="validation" />}
      chart={<StrataChart points={points} />}
      table={
        <DataTable
          caption="Deciles of predicted baseline"
          rows={points}
          rowKey={(p) => String(p.stratum)}
          columns={[
            { key: "d", header: "Decile", cell: (p) => <span className="font-medium">{p.stratum}</span> },
            { key: "n", header: "Customers", align: "right", cell: (p) => count(p.n) },
            { key: "b", header: "Mean baseline", align: "right", cell: (p) => p.baselineLabel },
            { key: "t", header: "Treated share", align: "right", cell: (p) => p.treatedShareLabel },
            { key: "e", header: "Effect", align: "right", cell: (p) => p.effectLabel },
          ]}
        />
      }
      sources={sources}
      className="h-full"
    />
  );
}
