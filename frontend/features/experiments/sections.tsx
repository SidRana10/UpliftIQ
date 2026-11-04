import { TriangleAlert } from "lucide-react";
import { CaveatCallout } from "@/components/data/caveat-callout";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { Hash } from "@/components/data/hash";
import { InfoTip } from "@/components/data/info-tip";
import { KpiCard } from "@/components/data/kpi-card";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { SourceFootnote } from "@/components/data/source-footnote";
import { DataTable } from "@/components/data/data-table";
import { SectionError } from "@/components/states/section-error";
import { Badge } from "@/components/ui/badge";
import { apiGet } from "@/lib/api/server";
import type {
  CriteoDiagnosticsData,
  CriteoHeldoutData,
  CriteoIpwAipwData,
  Envelope,
  HillstromBalanceRow,
  HillstromDecisionData,
  HillstromEffectsData,
  HillstromGroupRow,
} from "@/lib/api/types";
import { count, criteo, hillstrom } from "@/lib/format";
import { EstimatorChart, SmdChart, StrataFrame } from "./criteo-charts";
import { HillstromLab } from "./hillstrom-lab";
import {
  GRADE_LABEL,
  aucRows,
  checkRows,
  criteoLabKpis,
  estimatorRows,
  gapModel,
  heldoutRepeat,
  propensityQuantiles,
  smdRows,
  strataPoints,
  type Role,
} from "./select";

// ------------------------------------------------------------------------- Hillstrom

export async function HillstromLabSection() {
  const [effects, groups, balance, decision] = await Promise.all([
    apiGet<Envelope<HillstromEffectsData>>("/hillstrom/effects"),
    apiGet<Envelope<HillstromGroupRow[]>>("/hillstrom/groups"),
    apiGet<Envelope<HillstromBalanceRow[]>>("/hillstrom/balance"),
    apiGet<Envelope<HillstromDecisionData>>("/hillstrom/decision"),
  ]);
  const failed = [effects, groups, balance, decision].find((r) => !r.ok);
  if (failed && !failed.ok) return <SectionError error={failed.error} what="Experiment results" minHeight={360} />;
  if (!effects.ok || !groups.ok || !balance.ok || !decision.ok) return null;
  const roles = decision.data.data.outcome_roles as Record<"visit" | "conversion" | "spend", Role>;
  return (
    <HillstromLab
      effects={effects.data.data}
      groups={groups.data.data}
      balance={balance.data.data}
      roles={roles}
      sources={{
        effects: effects.data.source,
        groups: groups.data.source,
        balance: balance.data.source,
      }}
      caveats={effects.data.caveats}
    />
  );
}

// ------------------------------------------------------------------------- Criteo

const GRADE_TONE = { pass: "teal", warn: "amber", ordering_warning: "amber", fail: "danger" } as const;

function GradeChip({ grade }: { grade: keyof typeof GRADE_TONE }) {
  return (
    <Badge tone={GRADE_TONE[grade]} className={grade === "ordering_warning" ? "border border-amber/50 bg-surface" : undefined}>
      {GRADE_LABEL[grade]}
    </Badge>
  );
}

/** AUC above 0.5 on a 0.5–0.56 scale with the pass (0.01) and warning (0.05) limits marked. */
function AucMeter({ gap, limits }: { gap: number; limits: [number, number] }) {
  const max = 0.06;
  const pct = (v: number) => `${Math.min(100, (v / max) * 100)}%`;
  return (
    <div className="relative h-1.5 w-full rounded-full bg-subtle" aria-hidden>
      <div className="absolute inset-y-0 left-0 rounded-full bg-teal/70" style={{ width: pct(Math.max(gap, 0.0008)) }} />
      {limits.map((l) => (
        <span key={l} className="absolute -top-1 h-3.5 w-px bg-ref-soft" style={{ left: pct(l) }} />
      ))}
    </div>
  );
}

export async function CriteoLabKpis() {
  const r = await apiGet<Envelope<CriteoDiagnosticsData>>("/criteo/diagnostics");
  if (!r.ok) return <SectionError error={r.error} what="Dataset and randomization summary" minHeight={150} />;
  return (
    <div className="grid grid-cols-1 gap-4 min-[480px]:grid-cols-2 lg:grid-cols-4">
      {criteoLabKpis(r.data.data).map((k) => (
        <KpiCard key={k.label} label={k.label} value={k.value} note={k.note} info={k.info} accent={k.accent} />
      ))}
    </div>
  );
}

export async function CriteoRandomization() {
  const r = await apiGet<Envelope<CriteoDiagnosticsData>>("/criteo/diagnostics");
  if (!r.ok) return <SectionError error={r.error} what="Randomization diagnostics" minHeight={420} />;
  const d = r.data.data;
  const checks = checkRows(d.treatment.checks);
  const thresholds = d.treatment.thresholds.max_abs_smd ?? [0.05, 0.1];
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
      <div className="min-w-0 lg:col-span-5">
        <Panel className="h-full">
          <PanelHeader
            title="Randomization checks"
            subtitle="Does treatment look independent of the features? · training rows"
            actions={<EvidenceBadge kind="stored" />}
          />
          <PanelBody className="space-y-1">
            <ul className="divide-y divide-border/70">
              {checks.map((c) => (
                <li key={c.id} className="py-3 first:pt-1 last:pb-0">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-[13px] font-medium text-fg">
                        <span className="mr-1.5 font-mono text-[11px] text-fg-muted">{c.id}</span>
                        {c.name}
                      </p>
                      <p className="mt-0.5 text-xs text-fg-muted">
                        {c.measure}: <span className="font-medium tabular-nums text-fg-secondary">{c.value}</span>
                      </p>
                    </div>
                    <GradeChip grade={c.grade} />
                  </div>
                  <p className="mt-1 text-[11px] leading-4 text-fg-muted">{c.rule}</p>
                </li>
              ))}
            </ul>
            <SourceFootnote sources={r.data.source.slice(1, 2)} className="pt-3" />
          </PanelBody>
        </Panel>
      </div>
      <div className="min-w-0 lg:col-span-7">
        <SmdChart rows={smdRows(d)} thresholds={thresholds as [number, number]} sources={r.data.source.slice(1, 2)} />
      </div>
      <div className="lg:col-span-12">
        <CaveatCallout
          title="Verdict: benchmark modeling allowed, with two warnings"
          items={[
            d.treatment.verdict.statement,
          ]}
        />
      </div>
    </div>
  );
}

export async function CriteoPropensity() {
  const r = await apiGet<Envelope<CriteoDiagnosticsData>>("/criteo/diagnostics");
  if (!r.ok) return <SectionError error={r.error} what="Propensity diagnostics" minHeight={260} />;
  const d = r.data.data;
  const c = d.treatment.classifier;
  const aucs = aucRows(c);
  const q = propensityQuantiles(c);
  const limits = d.treatment.thresholds.auc_gap ?? [0.01, 0.05];
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Panel>
        <PanelHeader
          title="Can features predict treatment?"
          subtitle={`Held-out AUC of a treatment classifier · ${count(c.n_fit)} fit / ${count(c.n_eval)} evaluation rows`}
          actions={
            <InfoTip label="About the classifier check">
              If assignment is random, features should not predict it: AUC stays near 0.5. The permuted-label row is the same model on shuffled labels, a control for pure noise.
            </InfoTip>
          }
        />
        <PanelBody className="space-y-4">
          {aucs.map((a) => (
            <div key={a.key}>
              <div className="flex items-baseline justify-between gap-3 text-[13px]">
                <span className="font-medium text-fg">{a.label}</span>
                <span className="tabular-nums text-fg-secondary">
                  AUC {a.auc.toFixed(4)}
                  {a.se !== null ? <span className="text-fg-muted"> ± {a.se.toFixed(4)}</span> : null}
                  <span className="ml-2 text-xs text-fg-muted">({a.gapLabel} vs 0.5)</span>
                </span>
              </div>
              <div className="mt-1.5 flex items-center gap-3">
                <div className="flex-1">
                  <AucMeter gap={Math.max(0, a.auc - 0.5)} limits={limits as [number, number]} />
                </div>
                {a.grade ? <GradeChip grade={a.grade} /> : <Badge tone="slate">Control</Badge>}
              </div>
            </div>
          ))}
          <p className="text-[11px] leading-4 text-fg-muted">
            Ticks mark AUC − 0.5 = {limits[0]} (pass limit) and {limits[1]} (warning limit).
          </p>
          <div className="border-t border-border pt-3">
            <p className="eyebrow">Predicted treatment propensity</p>
            <dl className="mt-2 grid grid-cols-3 gap-x-3 gap-y-2 sm:grid-cols-5">
              {q.items.map((i) => (
                <div key={i.label}>
                  <dt className="text-[11px] text-fg-muted">{i.label}</dt>
                  <dd className="text-[13px] font-medium tabular-nums text-fg">{i.value}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-2 text-xs text-fg-muted">
              {q.outside} of rows fall outside [0.05, 0.95], so positivity holds. The design propensity is the constant 0.85.
            </p>
          </div>
          <SourceFootnote sources={r.data.source.slice(1, 2)} />
        </PanelBody>
      </Panel>

      <Panel>
        <PanelHeader title="Data splits" subtitle="Stratified by treatment · fixed seed" actions={<EvidenceBadge kind="stored" />} />
        <PanelBody className="space-y-4">
          <DataTable
            caption="Data splits"
            rows={(["train", "validation", "test"] as const).map((k) => ({
              k,
              rows: d.split.sizes[k],
              share: d.split.treated_share[k],
              use: { train: "Fit models and nuisance", validation: "Model selection (500K subset)", test: "One-shot held-out evaluation" }[k],
            }))}
            rowKey={(r) => r.k}
            columns={[
              { key: "s", header: "Split", cell: (r) => <span className="font-medium capitalize">{r.k}</span> },
              { key: "n", header: "Rows", align: "right", cell: (r) => count(r.rows) },
              { key: "t", header: "Treated share", align: "right", cell: (r) => hillstrom.rate(r.share, 2) },
              { key: "u", header: "Used for", cell: (r) => <span className="text-fg-muted">{r.use}</span> },
            ]}
          />
          <div>
            <p className="eyebrow">Split fingerprint</p>
            <div className="mt-1"><Hash value={d.split.split_fingerprint} /></div>
          </div>
          <p className="text-xs leading-5 text-fg-muted">
            Seed {d.split.seed}, {count(d.split.rows)} rows. The test rows are touched exactly once, by the frozen model.
          </p>
          <SourceFootnote sources={r.data.source.slice(2, 3)} />
        </PanelBody>
      </Panel>
    </div>
  );
}

export async function CriteoIpwAipw() {
  const [ipw, held] = await Promise.all([
    apiGet<Envelope<CriteoIpwAipwData>>("/criteo/ipw-aipw"),
    apiGet<Envelope<CriteoHeldoutData>>("/criteo/heldout"),
  ]);
  if (!ipw.ok) return <SectionError error={ipw.error} what="IPW vs AIPW diagnostic" minHeight={420} />;
  const d = ipw.data.data;
  const est = estimatorRows(d);
  const gap = gapModel(d);
  if (!gap || !est.length) {
    return <SectionError error={{ kind: "report_unavailable", status: 404, message: "The IPW/AIPW diagnostic report has no estimator rows." }} what="IPW vs AIPW diagnostic" />;
  }
  const h = held.ok ? heldoutRepeat(held.data.data.assumption_check) : null;
  const ipwRow = est.find((e) => e.key === "ipw")!;
  const aipwRow = est.find((e) => e.key === "aipw")!;
  const ps = est.find((e) => e.key === "ps")!;
  return (
    <section
      aria-labelledby="ipw-aipw"
      className="space-y-4 rounded-panel border border-amber/50 bg-amber-soft/30 p-4 shadow-panel sm:p-5"
    >
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-3">
          <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-control bg-amber-soft text-amber-ink">
            <TriangleAlert className="size-4" aria-hidden />
          </span>
          <div>
            <h2 id="ipw-aipw" className="text-base font-semibold text-fg">
              IPW and AIPW disagree on this file
            </h2>
            <p className="text-[13px] text-fg-secondary">A documented limitation, shown rather than hidden. AIPW is primary; IPW is a cross-check.</p>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <EvidenceBadge kind="validation" />
          <Badge tone="amber">Known limitation</Badge>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-4 text-[13px] leading-[21px] text-fg-secondary lg:grid-cols-3">
        <p>
          <strong className="font-semibold text-fg">The gap.</strong> IPW is the plain treated-minus-control difference. On the {gap.sample}-customer validation sample it reads {ipwRow.valueLabel} against {aipwRow.valueLabel} for AIPW, a difference of {gap.difference.value} {gap.difference.range}.
          {h ? ` The same pattern holds on the held-out test set: IPW ${h.ipw} vs AIPW ${h.aipw} (difference ${h.diff}, SE ${h.se}).` : ""}
        </p>
        <p>
          <strong className="font-semibold text-fg">Why.</strong> Treated customers have a higher predicted baseline visit probability than control customers ({gap.baselineTreated} vs {gap.baselineControl}; gap {gap.gap}, permutation z = {gap.z}, {gap.p}). That inflates the unadjusted contrast. AIPW removes it with an outcome model, and the propensity-free post-stratified estimate ({ps.valueLabel}) lands close to AIPW.
        </p>
        <p>
          <strong className="font-semibold text-fg">What it means.</strong> IPW is expected to read high on this file, so every benchmark table carries AIPW as the headline. AIPW is not guaranteed unbiased: it assumes treatment is as good as random given f0–f11. Gains are conditional estimates for the released sample, not population ROI.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-7">
          <EstimatorChart rows={est} sources={ipw.data.source} n={d.meta.n} />
        </div>
        <div className="min-w-0 lg:col-span-5">
          <Panel className="h-full">
            <PanelHeader
              title="Baseline imbalance"
              subtitle="Predicted visit probability without treatment"
              actions={<EvidenceBadge kind="validation" />}
            />
            <PanelBody className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-control bg-subtle p-3">
                  <p className="eyebrow">Treated arm</p>
                  <p className="mt-1 text-xl font-semibold tabular-nums text-fg">{gap.baselineTreated}</p>
                </div>
                <div className="rounded-control bg-subtle p-3">
                  <p className="eyebrow">Control arm</p>
                  <p className="mt-1 text-xl font-semibold tabular-nums text-fg">{gap.baselineControl}</p>
                </div>
              </div>
              <dl className="space-y-1.5 text-[13px]">
                {[
                  ["Gap (treated − control)", `${gap.gap} (SE ${gap.gapSe})`],
                  ["Permutation test", `z = ${gap.z}, ${gap.p} (${gap.permutations} shuffles)`],
                  ["IPW − AIPW", `${gap.difference.value} ${gap.difference.range}`],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between gap-4 border-b border-border/70 pb-1.5 last:border-0">
                    <dt className="text-fg-muted">{k}</dt>
                    <dd className="text-right font-medium tabular-nums text-fg">{v}</dd>
                  </div>
                ))}
              </dl>
              {h ? (
                <div className="rounded-control border border-teal/30 bg-teal-soft/50 p-3">
                  <p className="flex items-center gap-2 text-[13px] font-semibold text-teal-ink">
                    Repeated on the held-out test set <EvidenceBadge kind="heldout" />
                  </p>
                  <p className="mt-1 text-xs leading-5 text-fg-secondary">
                    Treat-all effect: IPW {h.ipw}, AIPW {h.aipw}. IPW − AIPW = {h.diff} (SE {h.se}).
                  </p>
                </div>
              ) : null}
            </PanelBody>
          </Panel>
        </div>
        <div className="min-w-0 lg:col-span-12">
          <StrataFrame points={strataPoints(d)} sources={ipw.data.source} />
        </div>
      </div>
    </section>
  );
}

export async function CriteoArms() {
  const r = await apiGet<Envelope<CriteoDiagnosticsData>>("/criteo/diagnostics");
  if (!r.ok) return <SectionError error={r.error} what="Observed arm rates" minHeight={160} />;
  const a = r.data.data.dataset.arms;
  const rows = (["treated", "control"] as const).map((k) => ({ k, ...a[k] }));
  return (
    <Panel>
      <PanelHeader
        title="Observed rates by arm"
        subtitle="Observed, unadjusted · released sample"
        actions={<EvidenceBadge kind="stored" />}
      />
      <PanelBody>
        <DataTable
          caption="Observed rates by arm"
          rows={rows}
          rowKey={(x) => x.k}
          columns={[
            { key: "a", header: "Arm", cell: (x) => <span className="font-medium capitalize">{x.k}</span> },
            { key: "n", header: "Customers", align: "right", cell: (x) => count(x.n) },
            { key: "s", header: "Share", align: "right", cell: (x) => hillstrom.rate(x.share, 1) },
            { key: "v", header: "Visit rate", align: "right", cell: (x) => criteo.prob(x.visit_rate, 3) },
            { key: "c", header: "Conversion rate", align: "right", cell: (x) => criteo.prob(x.conversion_rate, 3) },
            { key: "e", header: "Exposed share", align: "right", cell: (x) => criteo.prob(x.exposed_share, 2) },
          ]}
        />
        <p className="mt-3 text-xs leading-5 text-fg-muted">
          These are descriptive rates for the released benchmark file, with no adjustment and no population or revenue interpretation. Estimated effects are in the section above.
        </p>
        <SourceFootnote sources={r.data.source.slice(0, 1)} className="mt-2" />
      </PanelBody>
    </Panel>
  );
}
