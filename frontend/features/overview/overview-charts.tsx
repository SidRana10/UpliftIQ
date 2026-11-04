"use client";

import { BudgetValueChart } from "@/components/charts/budget-value-chart";
import { ChartFrame } from "@/components/charts/chart-frame";
import { DotWhiskerChart } from "@/components/charts/dot-whisker-chart";
import { DataTable } from "@/components/data/data-table";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { COLORS } from "@/lib/colors";
import type { CriteoBudgetPoint, PolicyRowModel } from "./select";

export function HillstromPolicyChart({ rows, sources }: { rows: PolicyRowModel[]; sources: string[] }) {
  const top = Math.ceil((Math.max(...rows.map((r) => r.high)) * 1.12) / 5) * 5;
  return (
    <ChartFrame
      title="Held-out visit rate by policy"
      subtitle="Expected share of customers who visit the site if everyone followed each policy · AIPW with 95% intervals · held-out customers"
      badge={<EvidenceBadge kind="heldout" />}
      legend={[
        { label: "Frozen decision", color: COLORS.teal, shape: "dot" },
        { label: "Baseline policies", color: COLORS.ref, shape: "dot-hollow" },
      ]}
      chart={<DotWhiskerChart rows={rows} domain={[0, top]} tickFormat={(v) => `${v}%`} axisTitle="Expected visit rate (%)" />}
      table={
        <DataTable
          caption="Held-out visit rate by policy"
          rows={rows}
          rowKey={(r) => r.key}
          columns={[
            { key: "p", header: "Policy", cell: (r) => <span className="font-medium">{r.label}</span> },
            { key: "v", header: "Visit rate", align: "right", cell: (r) => r.valueLabel },
            { key: "ci", header: "95% CI", align: "right", cell: (r) => r.rangeLabel },
          ]}
        />
      }
      sources={sources}
      className="h-full"
    />
  );
}

export function CriteoBudgetChart({
  points,
  treatAll,
  sources,
}: {
  points: CriteoBudgetPoint[];
  treatAll: CriteoBudgetPoint["model"];
  sources: string[];
}) {
  return (
    <ChartFrame
      title="Held-out policy value by contact budget"
      subtitle="Extra visit rate vs treating nobody, in percentage points of the released sample · AIPW with 95% intervals · only the evaluated budgets are shown"
      badge={<EvidenceBadge kind="heldout" />}
      legend={[
        { label: "S-learner ranking", color: COLORS.teal, shape: "dot" },
        { label: "Random at same budget", color: COLORS.ref, shape: "dot-hollow" },
        { label: "Treat everyone (unconstrained)", color: COLORS.ref, shape: "dash" },
        { label: "Its 95% interval", color: COLORS.ref, shape: "band" },
      ]}
      chart={
        <BudgetValueChart
          points={points}
          treatAll={treatAll}
          treatAllLabel={`Treat everyone ${treatAll.valueLabel} pp`}
          yTick={(v) => v.toFixed(1)}
        />
      }
      table={
        <DataTable
          caption="Held-out policy value by contact budget (pp)"
          rows={points}
          rowKey={(r) => String(r.budget)}
          columns={[
            { key: "b", header: "Budget", cell: (r) => <span className="font-medium">{r.budget}%{r.primary ? " (primary)" : ""}</span> },
            { key: "m", header: "S-learner (pp)", align: "right", cell: (r) => `${r.model.valueLabel} ${r.model.rangeLabel}` },
            { key: "r", header: "Random (pp)", align: "right", cell: (r) => `${r.random.valueLabel} ${r.random.rangeLabel}` },
            { key: "d", header: "Minus random (pp)", align: "right", cell: (r) => r.vsRandomLabel },
          ]}
        />
      }
      sources={sources}
      note="Estimates apply to the released benchmark sample; values between 10%, 20% and 30% were not estimated."
      className="h-full"
    />
  );
}
