import { ArrowRight, CircleCheck, CircleX } from "lucide-react";
import Link from "next/link";
import { CaveatCallout } from "@/components/data/caveat-callout";
import { DataTable } from "@/components/data/data-table";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { Hash } from "@/components/data/hash";
import { KpiCard } from "@/components/data/kpi-card";
import { DatasetChip } from "@/components/data/page-header";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { Badge } from "@/components/ui/badge";
import { SectionError } from "@/components/states/section-error";
import { apiGet } from "@/lib/api/server";
import type {
  CriteoHeldoutData,
  Envelope,
  Health,
  HillstromDecisionData,
  OverviewData,
} from "@/lib/api/types";
import { WORKSPACES, hrefWithDataset, type DatasetId } from "@/lib/datasets";
import { count, dateTime } from "@/lib/format";
import { cn } from "@/lib/utils";
import { CriteoBudgetChart, HillstromPolicyChart } from "./overview-charts";
import {
  criteoBudgetPoints,
  criteoInsight,
  criteoIpwCaveat,
  criteoKpis,
  criteoTreatAll,
  hillstromInsight,
  hillstromKpis,
  hillstromOutcomeRows,
  hillstromPolicyRows,
  policyLabel,
  type Insight,
  type KpiModel,
} from "./select";

const getOverview = () => apiGet<Envelope<OverviewData>>("/overview");
const getHillstrom = () => apiGet<Envelope<HillstromDecisionData>>("/hillstrom/decision");
const getCriteo = () => apiGet<Envelope<CriteoHeldoutData>>("/criteo/heldout");

function KpiGrid({ items }: { items: KpiModel[] }) {
  return (
    <div className="grid grid-cols-1 gap-4 min-[480px]:grid-cols-2 lg:grid-cols-4">
      {items.map((k) => (
        <KpiCard
          key={k.label}
          label={k.label}
          value={k.value}
          range={k.range}
          note={k.note}
          info={k.info}
          accent={k.accent}
          badge={k.badge ? <EvidenceBadge kind={k.badge} /> : undefined}
        />
      ))}
    </div>
  );
}

// ------------------------------------------------------------------------- KPI row

export async function KpiSection({ dataset }: { dataset: DatasetId }) {
  if (dataset === "hillstrom") {
    const r = await getOverview();
    if (!r.ok) return <SectionError error={r.error} what="Headline metrics" minHeight={160} />;
    return <KpiGrid items={hillstromKpis(r.data.data.hillstrom)} />;
  }
  const r = await getCriteo();
  if (!r.ok) return <SectionError error={r.error} what="Headline metrics" minHeight={160} />;
  const items = criteoKpis(r.data.data);
  if (!items.length) return <SectionError error={{ kind: "report_unavailable", status: 404, message: "No AIPW 20% budget row in the held-out report." }} what="Headline metrics" />;
  return <KpiGrid items={items} />;
}

// ------------------------------------------------------------------------- main chart

export async function ChartSection({ dataset }: { dataset: DatasetId }) {
  if (dataset === "hillstrom") {
    const r = await getHillstrom();
    if (!r.ok) return <SectionError error={r.error} what="Held-out policy values" minHeight={360} />;
    const rows = hillstromPolicyRows(r.data.data);
    return <HillstromPolicyChart rows={rows} sources={r.data.source.slice(0, 2)} />;
  }
  const r = await getCriteo();
  if (!r.ok) return <SectionError error={r.error} what="Held-out policy values" minHeight={360} />;
  const points = criteoBudgetPoints(r.data.data);
  const treatAll = criteoTreatAll(r.data.data);
  if (!points.length || !treatAll) {
    return <SectionError error={{ kind: "report_unavailable", status: 404, message: "Held-out budget rows are missing." }} what="Held-out policy values" />;
  }
  return <CriteoBudgetChart points={points} treatAll={treatAll} sources={r.data.source.slice(0, 1)} />;
}

// ------------------------------------------------------------------------- insight

function InsightPanel({ insight }: { insight: Insight }) {
  return (
    <Panel className="flex h-full flex-col">
      <PanelHeader title="The decision" subtitle="What the evidence supports" />
      <PanelBody className="flex flex-1 flex-col gap-4">
        <p className="text-lg font-semibold leading-7 tracking-[-0.01em] text-fg">{insight.headline}</p>
        <div className="space-y-3 text-[13px] leading-[21px] text-fg-secondary">
          {insight.paragraphs.map((p, i) => (
            <p key={i}>{p}</p>
          ))}
        </div>
      </PanelBody>
    </Panel>
  );
}

export async function InsightSection({ dataset }: { dataset: DatasetId }) {
  if (dataset === "hillstrom") {
    const [o, d] = await Promise.all([getOverview(), getHillstrom()]);
    if (!o.ok) return <SectionError error={o.error} what="Decision summary" minHeight={360} />;
    if (!d.ok) return <SectionError error={d.error} what="Decision summary" minHeight={360} />;
    const insight = hillstromInsight(o.data.data.hillstrom, hillstromOutcomeRows(d.data.data));
    return <InsightPanel insight={insight} />;
  }
  const r = await getCriteo();
  if (!r.ok) return <SectionError error={r.error} what="Decision summary" minHeight={360} />;
  const insight = criteoInsight(r.data.data);
  if (!insight) return <SectionError error={{ kind: "report_unavailable", status: 404, message: "Held-out budget rows are missing." }} what="Decision summary" />;
  return <InsightPanel insight={insight} />;
}

export async function CaveatSection({ dataset }: { dataset: DatasetId }) {
  if (dataset === "hillstrom") {
    const o = await getOverview();
    if (!o.ok) return null;
    return <CaveatCallout columns title="How to read this" items={o.data.data.hillstrom.caveats} />;
  }
  const r = await getCriteo();
  if (!r.ok) return null;
  return <CaveatCallout columns title="Benchmark, not business ROI" items={[...r.data.caveats, criteoIpwCaveat(r.data.data)]} />;
}

// ------------------------------------------------------------------------- Hillstrom outcomes

export async function OutcomesSection() {
  const r = await getHillstrom();
  if (!r.ok) return <SectionError error={r.error} what="Outcome evidence" minHeight={200} />;
  const rows = hillstromOutcomeRows(r.data.data);
  return (
    <Panel>
      <PanelHeader
        title="Primary vs exploratory outcomes"
        subtitle="Frozen policy compared with the pre-specified strongest alternative (Women's E-Mail for all) · AIPW · held-out customers"
        actions={<EvidenceBadge kind="heldout" />}
      />
      <PanelBody>
        <DataTable
          caption="Outcome evidence for the frozen policy"
          rows={rows}
          rowKey={(x) => x.outcome}
          columns={[
            {
              key: "o",
              header: "Outcome",
              cell: (x) => (
                <span className="flex items-center gap-2">
                  <span className="font-medium text-fg">{x.label}</span>
                  <EvidenceBadge kind={x.role} />
                </span>
              ),
            },
            { key: "f", header: "Frozen policy", align: "right", cell: (x) => <><span className="font-medium">{x.frozen}</span> <span className="text-fg-muted">{x.frozenRange}</span></> },
            { key: "d", header: `Minus ${rows[0]?.comparator ?? "comparator"}`, align: "right", cell: (x) => <><span className="font-medium">{x.diff}</span> <span className="text-fg-muted">{x.diffRange}</span></> },
            { key: "p", header: "Holm-adjusted", align: "right", cell: (x) => x.pHolm },
            {
              key: "s",
              header: "Reading",
              cell: (x) =>
                x.supported ? (
                  <EvidenceBadge kind="supported" />
                ) : x.role === "primary" ? (
                  <Badge>Not supported</Badge>
                ) : x.pHolmValue < 0.05 ? (
                  <Badge tone="amber">Exploratory signal</Badge>
                ) : (
                  <Badge>No clear difference</Badge>
                ),
            },
          ]}
        />
        <p className="mt-3 text-xs leading-5 text-fg-muted">
          Visit is the pre-specified primary outcome. Conversion and spend are rare or zero-inflated, so they are shown for
          completeness and never used to choose the policy.
        </p>
      </PanelBody>
    </Panel>
  );
}

// ------------------------------------------------------------------------- other study + status

export async function OtherStudySection({ dataset }: { dataset: DatasetId }) {
  const other: DatasetId = dataset === "hillstrom" ? "criteo" : "hillstrom";
  const r = await getOverview();
  let body: React.ReactNode;
  if (!r.ok) {
    body = <p className="text-xs leading-5 text-fg-muted">Headline unavailable: {r.error.message}</p>;
  } else if (other === "criteo") {
    const h = await getCriteo();
    const kpi = h.ok ? criteoKpis(h.data.data)[0] : undefined;
    body = kpi ? (
      <Headline label={kpi.label} value={kpi.value} range={kpi.range} note="Anonymized benchmark: percentage points of visit rate, not business ROI." />
    ) : (
      <p className="text-xs leading-5 text-fg-muted">Headline unavailable.</p>
    );
  } else {
    const k = hillstromKpis(r.data.data.hillstrom)[1];
    body = <Headline label={`${k.label} · ${policyLabel(r.data.data.hillstrom.frozen_policy)}`} value={k.value} range={k.range} note="Business experiment: expected visit rate, held-out customers." />;
  }
  return (
    <Panel className="flex h-full flex-col">
      <PanelHeader title="The other study" subtitle="Kept separate; never ranked against this one" />
      <PanelBody className="flex flex-1 flex-col gap-4">
        <DatasetChip dataset={other} />
        {body}
        <Link
          href={`/overview?dataset=${other}`}
          className="mt-auto inline-flex items-center gap-1.5 text-[13px] font-medium text-teal-ink transition-colors hover:text-teal"
        >
          Open the {other === "criteo" ? "Criteo" : "Hillstrom"} briefing <ArrowRight className="size-3.5" />
        </Link>
      </PanelBody>
    </Panel>
  );
}

function Headline({ label, value, range, note }: { label: string; value: string; range?: string; note: string }) {
  return (
    <div>
      <div className="eyebrow">{label}</div>
      <div className="mt-1 text-2xl font-semibold leading-8 tracking-[-0.01em] text-fg">{value}</div>
      {range ? <div className="text-xs text-fg-muted">95% CI {range}</div> : null}
      <p className="mt-2 text-xs leading-5 text-fg-secondary">{note}</p>
    </div>
  );
}

const REPORT_LABELS: Record<string, string> = {
  hillstrom_final_evaluation: "Hillstrom final evaluation",
  hillstrom_frozen_config: "Hillstrom frozen config",
  criteo_heldout: "Criteo held-out results",
  criteo_frozen_config: "Criteo frozen config",
  criteo_model_manifest: "Criteo model manifest",
};

export async function StatusSection() {
  const r = await apiGet<Health>("/health", { live: true });
  if (!r.ok) return <SectionError error={r.error} what="Platform status" minHeight={260} />;
  const h = r.data;
  return (
    <Panel className="h-full">
      <PanelHeader
        title="Platform status"
        subtitle="Live from the API"
        actions={<EvidenceBadge kind="live" />}
      />
      <PanelBody>
        <dl className="space-y-2.5 text-[13px]">
          <Row label="API">
            <StatusPill ok={h.status === "ok"} text={h.status === "ok" ? "Online" : "Degraded"} />
          </Row>
          <Row label="Criteo model">
            {h.criteo_model_loaded ? <StatusPill ok text={`Loaded · v${h.model_version}`} /> : <StatusPill ok={false} text="Not loaded" />}
          </Row>
          <Row label="API version">{h.api_version}</Row>
        </dl>
        <div className="mt-4 border-t border-border pt-3">
          <div className="eyebrow mb-2">Committed reports</div>
          <ul className="space-y-1.5 text-xs">
            {Object.entries(h.reports_available).map(([k, ok]) => (
              <li key={k} className="flex items-center gap-2 text-fg-secondary">
                {ok ? <CircleCheck className="size-3.5 text-teal" aria-label="available" /> : <CircleX className="size-3.5 text-danger" aria-label="missing" />}
                {REPORT_LABELS[k] ?? k}
              </li>
            ))}
          </ul>
        </div>
        {h.model_error ? <p className="mt-3 break-words font-mono text-[11px] text-danger">{h.model_error}</p> : null}
      </PanelBody>
    </Panel>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-fg-muted">{label}</dt>
      <dd className="text-right font-medium text-fg">{children}</dd>
    </div>
  );
}

function StatusPill({ ok, text }: { ok: boolean; text: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5", ok ? "text-teal-ink" : "text-danger")}>
      <span className={cn("size-1.5 rounded-full", ok ? "bg-teal" : "bg-danger")} />
      {text}
    </span>
  );
}

// ------------------------------------------------------------------------- evidence trail

export async function EvidenceSection({ dataset }: { dataset: DatasetId }) {
  if (dataset === "hillstrom") {
    const r = await getHillstrom();
    if (!r.ok) return <SectionError error={r.error} what="Evidence trail" minHeight={260} />;
    const e = r.data.data.evaluation_record;
    return (
      <Trail
        status="Completed"
        rows={[
          ["Held-out customers", count(e.n_test)],
          ["Development customers", count(e.n_dev)],
          ["Nuisance models fitted on", e.nuisance_models_fitted_on],
          ["Bootstrap", `${count(e.n_boot)} replicates · seed ${e.seed}`],
          ["Code commit", <code key="c" className="font-mono text-xs">{e.code_commit.slice(0, 8)}</code>],
          ["Working tree dirty", String(e.code_dirty) + " (recorded as-is)"],
          ["Frozen config", <Hash key="h" value={e.frozen_config_sha256} label="frozen config checksum" />],
          ["Split fingerprint", <Hash key="s" value={e.split_fingerprint} label="split fingerprint" />],
        ]}
      />
    );
  }
  const r = await getCriteo();
  if (!r.ok) return <SectionError error={r.error} what="Evidence trail" minHeight={260} />;
  const e = r.data.data.evaluation_record;
  return (
    <Trail
      status={e.status[0].toUpperCase() + e.status.slice(1)}
      oneShot={e.one_shot}
      rows={[
        ["Selected model", e.selected_model.replace("_", "-")],
        ["Test rows", count(e.test_rows)],
        ["Nuisance models fitted on", e.nuisance_fitted_on],
        ["Bootstrap", `${count(e.n_boot)} replicates · seed ${e.seed}`],
        ["Finished", dateTime(e.finished_at)],
        ["Code commit", <code key="c" className="font-mono text-xs">{e.code_commit.slice(0, 8)}</code>],
        ["Working tree dirty", String(e.code_dirty) + " (recorded as-is)"],
        ["Frozen config", <Hash key="h" value={e.frozen_config_sha256} label="frozen config checksum" />],
        ["Dataset checksum", <Hash key="d" value={e.dataset_sha256} label="dataset checksum" />],
      ]}
    />
  );
}

function Trail({ status, oneShot = true, rows }: { status: string; oneShot?: boolean; rows: [string, React.ReactNode][] }) {
  return (
    <Panel className="h-full">
      <PanelHeader
        title="Evidence trail"
        subtitle="How the held-out result was produced"
        actions={<EvidenceBadge kind="historical" />}
      />
      <PanelBody>
        <div className="mb-3 flex flex-wrap gap-1.5">
          <Badge tone="teal">{status}</Badge>
          {oneShot ? <Badge tone="teal-outline">Evaluated once</Badge> : null}
        </div>
        <dl className="space-y-2 text-[13px]">
          {rows.map(([k, v]) => (
            <Row key={k} label={k}>
              <span className="font-normal text-fg-secondary">{v}</span>
            </Row>
          ))}
        </dl>
      </PanelBody>
    </Panel>
  );
}

// ------------------------------------------------------------------------- continue

export function ContinueSection({ dataset }: { dataset: DatasetId }) {
  const next = WORKSPACES.filter((w) => w.id !== "overview");
  return (
    <div className="grid grid-cols-1 gap-4 min-[560px]:grid-cols-2 lg:grid-cols-4">
      {next.map((w) => (
        <Link
          key={w.id}
          href={hrefWithDataset(w, dataset)}
          className="group flex min-h-[104px] flex-col justify-between rounded-panel border border-border bg-surface p-4 shadow-panel transition-[border-color,box-shadow] duration-150 hover:border-border-strong hover:shadow-popover"
        >
          <div className="flex items-center justify-between">
            <span className="font-mono text-[11px] text-fg-muted">{w.number}</span>
            <ArrowRight className="size-4 text-fg-muted transition-transform duration-150 group-hover:translate-x-0.5 group-hover:text-teal" />
          </div>
          <div>
            <div className="text-sm font-semibold text-fg">{w.label}</div>
            <div className="mt-0.5 text-xs leading-4 text-fg-muted">{w.purpose}</div>
          </div>
        </Link>
      ))}
    </div>
  );
}
