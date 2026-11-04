import { ReportFigure } from "@/components/charts/report-figure";
import { CaveatCallout } from "@/components/data/caveat-callout";
import { DataTable } from "@/components/data/data-table";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { KpiCard } from "@/components/data/kpi-card";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { SourceFootnote } from "@/components/data/source-footnote";
import { SectionError } from "@/components/states/section-error";
import { Badge } from "@/components/ui/badge";
import { Collapsible } from "@/components/ui/accordion";
import { CriteoBudgetChart } from "@/features/overview/overview-charts";
import { criteoBudgetPoints, criteoIpwCaveat, criteoKpis, criteoTreatAll } from "@/features/overview/select";
import { apiGet } from "@/lib/api/server";
import type {
  CriteoBenchmarkData,
  CriteoHeldoutData,
  CriteoHeldoutExtras,
  Envelope,
  HillstromDecisionData,
  HillstromModelsData,
} from "@/lib/api/types";
import { count } from "@/lib/format";
import { CriteoExplorer } from "./criteo-explorer";
import { HillstromExplorer } from "./hillstrom-explorer";
import {
  candidateRows,
  heldoutQini,
  heldoutRows,
  hillstromDiffRows,
  hillstromSelectionSummary,
  hsLabel,
  validationVsHeldout,
} from "./select";

type HeldoutPayload = CriteoHeldoutData & CriteoHeldoutExtras;

function HeldoutView({ d, sources, bench }: { d: HeldoutPayload; sources: string[]; bench: CriteoBenchmarkData }) {
  const kpis = criteoKpis(d);
  const points = criteoBudgetPoints(d);
  const treatAll = criteoTreatAll(d);
  const qini = heldoutQini(d);
  const rows = heldoutRows(d);
  const compare = validationVsHeldout(bench, d);
  const conv = d.exploratory_conversion;
  if (!kpis.length || !points.length || !treatAll) {
    return <SectionError error={{ kind: "report_unavailable", status: 404, message: "Held-out budget rows are missing from the report." }} what="Held-out results" />;
  }
  const convCols = conv.table.length ? Object.keys(conv.table[0]) : [];
  return (
    <div className="space-y-6 lg:space-y-8">
      <div className="grid grid-cols-1 gap-4 min-[480px]:grid-cols-2 lg:grid-cols-4">
        {kpis.slice(0, 3).map((k) => (
          <KpiCard key={k.label} label={k.label} value={k.value} range={k.range} note={k.note} info={k.info} accent={k.accent} badge={k.badge === "primary" ? <EvidenceBadge kind="primary" /> : k.badge === "supported" ? <EvidenceBadge kind="supported" /> : undefined} />
        ))}
        {qini.aipw ? (
          <KpiCard
            label="Qini (AIPW)"
            value={qini.aipw.value}
            range={qini.aipw.range}
            note={`Ranking quality over all budgets${qini.aipwShare !== null ? `; ${(qini.aipwShare * 100).toFixed(1)}% of the treat-all effect` : ""}.`}
            badge={<EvidenceBadge kind="heldout" />}
            accent="teal"
          />
        ) : null}
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-7">
          <CriteoBudgetChart points={points} treatAll={treatAll} sources={sources.slice(0, 1)} />
        </div>
        <div className="min-w-0 lg:col-span-5">
          <Panel className="h-full">
            <PanelHeader title="Held-out results" subtitle={`Frozen S-learner · ${count(d.n)} test rows`} actions={<EvidenceBadge kind="heldout" />} />
            <PanelBody>
              <DataTable
                caption="Held-out policy value by budget"
                rows={rows}
                rowKey={(r) => r.budget}
                columns={[
                  { key: "b", header: "Budget", cell: (r) => <span className="font-medium">{r.budget}{r.primary ? " ★" : ""}</span> },
                  { key: "v", header: "AIPW value", align: "right", cell: (r) => (<div><div className="font-semibold tabular-nums">{r.value}</div><div className="text-[11px] tabular-nums text-fg-muted">{r.range}</div></div>) },
                  { key: "d", header: "vs random", align: "right", cell: (r) => (<div><div className="font-medium tabular-nums">{r.vsRandom}</div><div className="text-[11px] tabular-nums text-fg-muted">{r.vsRandomRange}</div></div>) },
                  { key: "i", header: "IPW (cross-check)", align: "right", cell: (r) => <span className="tabular-nums text-fg-muted opacity-70">{r.ipw}</span> },
                ]}
              />
              <p className="mt-3 text-xs leading-5 text-fg-muted">
                ★ primary budget. IPW reads higher than AIPW at every budget, for the documented reason below; AIPW is the headline.
              </p>
              <SourceFootnote sources={sources.slice(0, 1)} className="mt-2" />
            </PanelBody>
          </Panel>
        </div>
      </div>

      <CaveatCallout title="IPW reads higher than AIPW on this file" items={[criteoIpwCaveat(d)]} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ReportFigure
          title="Held-out gain curve"
          subtitle="Stored figure: the only source of the curve shape, since no point series is stored"
          src="/api/v1/criteo/figures/heldout_gain_curve"
          alt="Held-out gain curve of the frozen S-learner against random targeting, from the stored report"
          badge={<EvidenceBadge kind="heldout" />}
          source="reports/criteo/final/heldout_gain_curve.png"
        />
        <ReportFigure
          title="Held-out value vs random"
          subtitle="Stored figure at the evaluated budgets"
          src="/api/v1/criteo/figures/heldout_value_vs_random"
          alt="Held-out policy value against random targeting at 10, 20 and 30 percent budgets, from the stored report"
          badge={<EvidenceBadge kind="heldout" />}
          source="reports/criteo/final/heldout_value_vs_random.png"
        />
      </div>

      <Panel>
        <PanelHeader
          title="Validation vs held-out, same model"
          subtitle="The selected S-learner at each evaluated budget · percentage points"
          actions={<Badge tone="slate">Different samples · not a paired comparison</Badge>}
        />
        <PanelBody>
          <DataTable
            caption="Validation versus held-out value for the selected model"
            rows={compare}
            rowKey={(r) => r.budget}
            columns={[
              { key: "b", header: "Budget", cell: (r) => <span className="font-medium">{r.budget}</span> },
              { key: "v", header: "Validation (selection sample)", align: "right", cell: (r) => <span className="tabular-nums">{r.validation}</span> },
              { key: "h", header: "Held-out (one-shot)", align: "right", cell: (r) => <span className="font-semibold tabular-nums">{r.heldout}</span> },
            ]}
          />
          <p className="mt-3 text-xs leading-5 text-fg-muted">
            The two columns come from different customers (the validation subset and the {count(d.n)}-row held-out test set). Validation values were used to choose the model, so they can only be compared loosely with the held-out result.
          </p>
        </PanelBody>
      </Panel>

      <Collapsible
        title="Exploratory: conversion"
        badge={<Badge tone="amber">Exploratory, never used for any decision</Badge>}
      >
        <p className="mb-3 text-[13px] leading-5 text-fg-secondary">
          The report&apos;s own table (display values). Conversion is reported for context only; the frozen policy was chosen on visits.
        </p>
        <DataTable
          caption="Held-out conversion value (exploratory)"
          rows={conv.table}
          rowKey={(r, i) => `${r[convCols[0]]}-${i}`}
          columns={convCols.map((c, i) => ({ key: c, header: c, align: i === 0 ? ("left" as const) : ("right" as const), cell: (r: Record<string, string>) => r[c] || "n/a" }))}
        />
        <p className="mt-2 text-[11px] text-fg-muted">{conv.note}</p>
      </Collapsible>
    </div>
  );
}

export async function CriteoExplorerSection() {
  const [bench, held] = await Promise.all([
    apiGet<Envelope<CriteoBenchmarkData>>("/criteo/benchmark"),
    apiGet<Envelope<HeldoutPayload>>("/criteo/heldout"),
  ]);
  if (!bench.ok) return <SectionError error={bench.error} what="Validation benchmark" minHeight={420} />;
  const heldoutNode = held.ok ? (
    <HeldoutView d={held.data.data} sources={held.data.source} bench={bench.data.data} />
  ) : (
    <SectionError error={held.error} what="Held-out results" minHeight={260} />
  );
  return <CriteoExplorer bench={bench.data.data} benchSources={bench.data.source} heldout={heldoutNode} />;
}

export async function HillstromExplorerSection() {
  const [models, decision] = await Promise.all([
    apiGet<Envelope<HillstromModelsData>>("/hillstrom/models"),
    apiGet<Envelope<HillstromDecisionData>>("/hillstrom/decision"),
  ]);
  if (!models.ok) return <SectionError error={models.error} what="Candidate policies" minHeight={420} />;
  if (!decision.ok) return <SectionError error={decision.error} what="Candidate policies" minHeight={420} />;
  const chosen = decision.data.data.chosen_policy.name;
  const m = models.data.data;
  return (
    <HillstromExplorer
      candidates={candidateRows(m, chosen)}
      diffs={hillstromDiffRows(m)}
      summary={hillstromSelectionSummary(m, chosen)}
      chosenLabel={hsLabel(chosen)}
      sources={models.data.source}
      caveats={models.data.caveats}
    />
  );
}
