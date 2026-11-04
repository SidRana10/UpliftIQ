"use client";

import { Check, CircleCheck, CircleX, Lock } from "lucide-react";
import { useState } from "react";
import { ChartFrame } from "@/components/charts/chart-frame";
import { DotWhiskerChart } from "@/components/charts/dot-whisker-chart";
import { niceAxis } from "@/components/charts/theme";
import { CaveatCallout } from "@/components/data/caveat-callout";
import { DataTable } from "@/components/data/data-table";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { InfoTip } from "@/components/data/info-tip";
import { KpiCard } from "@/components/data/kpi-card";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { Control, SubBar } from "@/components/data/sub-bar";
import { Badge } from "@/components/ui/badge";
import { SegmentedControl } from "@/components/ui/segmented-control";
import type { CriteoBenchmarkData } from "@/lib/api/types";
import { COLORS } from "@/lib/colors";
import { count, sizeLabel } from "@/lib/format";
import { useViewList, useViewState } from "@/lib/url-state";
import { cn } from "@/lib/utils";
import { CostQualityChart, LatencyChart, LearningCurveChart, ModelBudgetChart, TrainingTimeChart } from "./charts";
import { Leaderboard } from "./leaderboard";
import { ModelDetailSheet } from "./model-detail-sheet";
import {
  DEFAULT_COMPARE,
  MAX_COMPARE,
  METRICS,
  METRIC_LABEL,
  MODELS,
  SCOPES,
  budgetRefs,
  budgetSeries,
  fmtDelta,
  fmtEst,
  latencyRows,
  leaderboard,
  modelDetail,
  modelColor as modelColorOf,
  modelLabel,
  modelSeries,
  pairReferences,
  pairwiseRows,
  resourceRows,
  scalingRows,
  selectionModel,
  type MetricKey,
  type ModelKey,
  type Scope,
} from "./select";

const SCOPE_OPTIONS = [
  { value: "validation" as const, label: "Validation" },
  { value: "heldout" as const, label: "Held-out" },
];

function ScopeBanner({ scope, trainRows }: { scope: Scope; trainRows: number }) {
  const validation = scope === "validation";
  return (
    <div
      role="note"
      className={cn(
        "flex flex-wrap items-center gap-x-3 gap-y-1 rounded-control border px-4 py-2.5 text-[13px]",
        validation ? "border-violet/30 bg-violet-soft/60 text-fg-secondary" : "border-teal/30 bg-teal-soft/60 text-fg-secondary",
      )}
    >
      <EvidenceBadge kind={validation ? "validation" : "heldout"} />
      <span>
        {validation
          ? `Model-selection evidence: models trained on ${count(trainRows)} rows, scored on a 500K validation subset. The final model was chosen on these numbers, so they are not an unbiased estimate of performance.`
          : "One-shot evidence: only the frozen S-learner was scored on the untouched test set. No other model has held-out numbers, and none are shown."}
      </span>
    </div>
  );
}

function CompareChips({
  selected,
  onChange,
  disabled,
  finalists,
}: {
  selected: ModelKey[];
  onChange: (next: ModelKey[]) => void;
  disabled: boolean;
  finalists: string[];
}) {
  const full = selected.length >= MAX_COMPARE;
  return (
    <div role="group" aria-label="Compared models" className="flex flex-wrap items-center gap-1.5">
      {MODELS.map((m) => {
        const on = selected.includes(m);
        const blocked = disabled || (!on && full);
        return (
          <button
            key={m}
            type="button"
            aria-pressed={on}
            disabled={blocked}
            title={disabled ? "Held-out evidence exists only for the S-learner" : !on && full ? `Compare up to ${MAX_COMPARE} models: remove one first` : undefined}
            onClick={() => {
              if (on) {
                if (selected.length > 1) onChange(selected.filter((x) => x !== m));
              } else {
                onChange([...selected, m]);
              }
            }}
            className={cn(
              "inline-flex h-8 items-center gap-1.5 rounded-chip border px-2.5 text-[13px] font-medium transition-colors duration-150 disabled:cursor-not-allowed disabled:opacity-45",
              on ? "border-border-strong bg-surface text-fg shadow-panel" : "border-transparent bg-subtle text-fg-secondary hover:text-fg",
            )}
          >
            <span aria-hidden className="size-2.5 rounded-full" style={{ border: `2px solid ${modelColorOf(m)}`, backgroundColor: on ? modelColorOf(m) : "transparent" }} />
            {modelLabel(m)}
            {finalists.includes(m) ? null : <span className="text-[10px] font-normal text-fg-muted">≤2M</span>}
            {on ? <Check className="size-3 text-fg-muted" aria-hidden /> : null}
          </button>
        );
      })}
    </div>
  );
}

export interface ExplorerProps {
  bench: CriteoBenchmarkData;
  benchSources: string[];
  /** server-rendered held-out content (or its error panel) */
  heldout: React.ReactNode;
}

export function CriteoExplorer({ bench, benchSources, heldout }: ExplorerProps) {
  const [scope, setScope] = useViewState<Scope>("scope", "validation", SCOPES);
  const [metric, setMetric] = useViewState<MetricKey>("metric", "v20", METRICS);
  const [compared, setCompared] = useViewList<ModelKey>("models", DEFAULT_COMPARE, MODELS, MAX_COMPARE);
  const [openModel, setOpenModel] = useState<string | null>(null);

  const rows = leaderboard(bench);
  const series = modelSeries(bench);
  const finalists = series.filter((s) => s.finalist).map((s) => s.model);
  const shown = series.filter((s) => compared.includes(s.model as ModelKey));
  const manifest = openModel ? bench.numeric.models[openModel] : undefined;

  return (
    <div className="space-y-6 lg:space-y-8">
      <SubBar>
        <Control label="Evidence">
          <SegmentedControl<Scope> label="Evidence scope" value={scope} onValueChange={setScope} options={SCOPE_OPTIONS} />
        </Control>
        <Control label="Compare">
          {scope === "validation" ? (
            <CompareChips selected={compared} onChange={setCompared} disabled={false} finalists={finalists} />
          ) : (
            <span className="inline-flex items-center gap-1.5 text-[13px] text-fg-secondary">
              <Lock className="size-3.5 text-fg-muted" aria-hidden />
              Frozen S-learner only
            </span>
          )}
        </Control>
        {scope === "validation" ? (
          <Control label="Metric">
            <SegmentedControl<MetricKey>
              label="Metric"
              value={metric}
              onValueChange={setMetric}
              options={METRICS.map((m) => ({ value: m, label: METRIC_LABEL[m].replace(" (primary)", "") }))}
            />
          </Control>
        ) : null}
      </SubBar>

      <ScopeBanner scope={scope} trainRows={bench.n_train} />

      {scope === "validation" ? (
        <ValidationView
          bench={bench}
          sources={benchSources}
          metric={metric}
          shown={shown}
          compared={compared}
          rows={rows}
          onOpen={setOpenModel}
        />
      ) : (
        heldout
      )}

      <ModelDetailSheet detail={openModel && manifest ? modelDetail(openModel, manifest) : null} onClose={() => setOpenModel(null)} />
    </div>
  );
}

function ValidationView({
  bench,
  sources,
  metric,
  shown,
  compared,
  rows,
  onOpen,
}: {
  bench: CriteoBenchmarkData;
  sources: string[];
  metric: MetricKey;
  shown: ReturnType<typeof modelSeries>;
  compared: ModelKey[];
  rows: ReturnType<typeof leaderboard>;
  onOpen: (m: string) => void;
}) {
  const refsList = pairReferences(bench);
  const [reference, setReference] = useViewState<string>("ref", refsList.includes("tarnet_cpu") ? "tarnet_cpu" : refsList[0] ?? "", refsList);
  const sel = selectionModel(bench);
  const chosen = rows.find((r) => r.selected);
  const budgetModels = compared.filter((m) => rows.some((r) => r.model === m));
  const noFull = compared.filter((m) => !rows.some((r) => r.model === m));
  const bSeries = budgetSeries(bench, budgetModels);
  const refs = budgetRefs(bench);
  const pair = pairwiseRows(bench, reference, metric);
  const scaling = scalingRows(bench, metric, compared);
  const lat = latencyRows(bench, compared);
  const res = resourceRows(bench, compared);
  const metricName = METRIC_LABEL[metric].replace(" (primary)", "");
  const pairAxis = niceAxis(Math.min(0, ...pair.map((p) => p.est.low * 100)) - 0.02, Math.max(0, ...pair.map((p) => p.est.high * 100)) + 0.02);

  return (
    <>
      {chosen ? (
        <div className="grid grid-cols-1 gap-4 min-[480px]:grid-cols-2 lg:grid-cols-4">
          <KpiCard
            label="Selected model"
            value={chosen.label}
            note="Frozen by the pre-specified rule: the simplest model within one standard error of the best."
            badge={<EvidenceBadge kind="selected" />}
            accent="violet"
          />
          <KpiCard
            label="AIPW value @ 20%"
            value={fmtEst(chosen.v20).value}
            range={fmtEst(chosen.v20).range}
            note="Validation estimate, percentage points of visit rate over treating nobody."
            badge={<EvidenceBadge kind="validation" />}
            accent="violet"
          />
          <KpiCard
            label="Gain vs random @ 20%"
            value={fmtDelta(chosen.vsRandom20).value}
            range={fmtDelta(chosen.vsRandom20).range}
            note="Paired difference against random targeting at the same budget."
            accent="violet"
          />
          <KpiCard
            label="Qini (AIPW)"
            value={fmtEst(chosen.qini).value}
            range={fmtEst(chosen.qini).range}
            note="Ranking quality over all budgets; supports the choice."
            info="The Qini coefficient is the area between the model's gain curve and the random line."
            accent="violet"
          />
        </div>
      ) : null}

      <Leaderboard rows={rows} compared={compared} metric={metric} onOpen={onOpen} sources={sources.slice(0, 1)} trainRows={bench.n_train} />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-7">
          <ChartFrame
            title="Policy value by contact budget"
            subtitle="Compared finalists · percentage points of visit rate over treating nobody · AIPW with 95% intervals · only the evaluated budgets"
            badge={<EvidenceBadge kind="validation" />}
            legend={[
              ...bSeries.map((s) => ({ label: s.label, color: s.color, shape: "dot" as const })),
              { label: "Random at same budget", color: COLORS.ref, shape: "dot-hollow" as const },
              { label: "Treat everyone", color: COLORS.ref, shape: "dash" as const },
            ]}
            chart={
              bSeries.length ? (
                <ModelBudgetChart series={bSeries} refs={refs} />
              ) : (
                <p className="py-16 text-center text-sm text-fg-muted">The compared models have no full-scale run. Pick a finalist: Logistic, S-learner or TARNet (CPU).</p>
              )
            }
            table={
              <DataTable
                caption="Validation policy value by contact budget"
                rows={bSeries.flatMap((s) => s.points.map((p) => ({ s, p })))}
                rowKey={(r) => `${r.s.model}-${r.p.budget}`}
                columns={[
                  { key: "m", header: "Model", cell: (r) => <span className="font-medium">{r.s.label}</span> },
                  { key: "b", header: "Budget", align: "right", cell: (r) => `${r.p.budget}%` },
                  { key: "v", header: "AIPW value (95% CI)", align: "right", cell: (r) => `${r.p.label} ${r.p.range} pp` },
                  { key: "d", header: "Minus random", align: "right", cell: (r) => r.p.vsRandom },
                ]}
              />
            }
            sources={sources.slice(0, 1)}
            note={`Random-at-budget and treat-all are exact stored values${refs.treatAll ? ` (treat everyone: ${refs.treatAll.reportText} pp in the report)` : ""}.${noFull.length ? ` ${noFull.map(modelLabel).join(", ")} ${noFull.length > 1 ? "have" : "has"} no full-scale run, so ${noFull.length > 1 ? "they appear" : "it appears"} in the learning curve only.` : ""}`}
            className="h-full"
          />
        </div>
        <div className="min-w-0 lg:col-span-5">
          <ChartFrame
            title="Paired differences"
            subtitle={`${metricName} · each model minus the reference · same validation customers · 95% intervals`}
            badge={<EvidenceBadge kind="validation" />}
            toolbar={
              <Control label="Reference">
                <SegmentedControl<string>
                  label="Reference model"
                  value={reference}
                  onValueChange={setReference}
                  options={refsList.map((r) => ({ value: r, label: modelLabel(r) }))}
                />
              </Control>
            }
            legend={[
              { label: "Interval excludes zero", color: COLORS.teal, shape: "dot" },
              { label: "Interval includes zero", color: COLORS.ref, shape: "dot-hollow" },
            ]}
            chart={
              pair.length ? (
                <DotWhiskerChart
                  rows={pair.map((p) => ({
                    key: p.model,
                    label: `${p.label} − ${modelLabel(reference)}`,
                    value: p.est.value * 100,
                    low: p.est.low * 100,
                    high: p.est.high * 100,
                    color: p.excludesZero ? COLORS.teal : COLORS.ref,
                    hollow: !p.excludesZero,
                    emphasis: p.excludesZero,
                    valueLabel: fmtDelta(p.est).value,
                    rangeLabel: fmtDelta(p.est).range,
                    tooltipNote: "Paired difference, AIPW, validation subset.",
                  }))}
                  domain={pairAxis.domain}
                  ticks={pairAxis.ticks}
                  tickFormat={(v) => v.toFixed(2)}
                  axisTitle={`${metricName} difference (pp)`}
                  refLines={[{ x: 0, label: "No difference", color: COLORS.axis }]}
                  rowHeight={70}
                />
              ) : (
                <p className="py-12 text-center text-sm text-fg-muted">No paired differences stored for this metric.</p>
              )
            }
            table={
              <DataTable
                caption="Paired differences"
                rows={pair}
                rowKey={(p) => p.model}
                columns={[
                  { key: "m", header: `Model − ${modelLabel(reference)}`, cell: (p) => <span className="font-medium">{p.label}</span> },
                  { key: "d", header: "Difference (95% CI), pp", align: "right", cell: (p) => p.text },
                  { key: "z", header: "Excludes 0", align: "right", cell: (p) => (p.excludesZero ? "Yes" : "No") },
                ]}
              />
            }
            sources={sources.slice(0, 1)}
            note="The report stores paired differences against Logistic and TARNet (CPU); every pair among the three finalists is covered."
            className="h-full"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-8">
          <ChartFrame
            title="Learning curve"
            subtitle={`${metricName} against training rows · 95% bands · six models through 2M rows, finalists at full scale (◆)`}
            badge={<EvidenceBadge kind="validation" />}
            legend={shown.map((s) => ({ label: s.label, color: s.color, shape: "dot" as const }))}
            chart={<LearningCurveChart series={shown} metric={metric} />}
            table={
              <DataTable
                caption="Learning curve"
                rows={shown.flatMap((s) => s.points.map((p) => ({ s, p })))}
                rowKey={(r) => `${r.s.model}-${r.p.size}`}
                columns={[
                  { key: "m", header: "Model", cell: (r) => <span className="font-medium">{r.s.label}</span> },
                  { key: "n", header: "Training rows", align: "right", cell: (r) => `${count(r.p.size)}${r.p.fullScale ? " (full)" : ""}` },
                  { key: "v", header: `${metricName} (95% CI), pp`, align: "right", cell: (r) => fmtEst(r.p.est[metric]).text },
                ]}
              />
            }
            sources={sources}
            className="h-full"
          />
        </div>
        <div className="min-w-0 lg:col-span-4">
          <Panel className="h-full">
            <PanelHeader
              title="Does more data help?"
              subtitle={`Paired change in ${metricName} at each step`}
              actions={<EvidenceBadge kind="validation" />}
            />
            <PanelBody className="space-y-4">
              {scaling.length ? (
                scaling.map((s) => (
                  <div key={s.model}>
                    <p className="flex items-center gap-2 text-[13px] font-semibold text-fg">
                      <span aria-hidden className="size-2.5 rounded-full" style={{ background: s.color }} />
                      {s.label}
                    </p>
                    <ul className="mt-1.5 space-y-1">
                      {s.steps.map((st) => (
                        <li key={st.step} className="flex items-baseline justify-between gap-3 text-xs">
                          <span className="whitespace-nowrap text-fg-muted">{st.step}</span>
                          <span className={cn("text-right tabular-nums", st.excludesZero ? "font-medium text-fg" : "text-fg-secondary")}>{st.text}</span>
                        </li>
                      ))}
                    </ul>
                    <p className="mt-1.5 text-[11px] leading-4 text-fg-muted">{s.statement}</p>
                  </div>
                ))
              ) : (
                <p className="text-sm text-fg-muted">Scaling changes are stored for the finalists only.</p>
              )}
              <p className="border-t border-border pt-3 text-[11px] leading-4 text-fg-muted">
                Differences in pp with 95% intervals; bold = interval excludes zero. Full-scale models used {count(bench.n_train)} training rows.
              </p>
            </PanelBody>
          </Panel>
        </div>
      </div>

      <section aria-labelledby="eng" className="space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-2">
          <div>
            <h2 id="eng" className="text-base font-semibold text-fg">
              Engineering cost and quality
            </h2>
            <p className="text-[13px] text-fg-muted">Training time, serving latency, throughput and memory of the compared models.</p>
          </div>
          <Badge tone="slate" className="whitespace-normal">Stored offline measurement · single machine · OMP_NUM_THREADS=1</Badge>
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          <ChartFrame
            title="Training time"
            compact
            subtitle="Seconds against training rows (log-log)"
            chart={<TrainingTimeChart series={shown} />}
            table={
              <DataTable
                caption="Training time"
                rows={shown.flatMap((s) => s.points.filter((p) => p.trainSeconds !== null).map((p) => ({ s, p })))}
                rowKey={(r) => `${r.s.model}-${r.p.size}`}
                columns={[
                  { key: "m", header: "Model", cell: (r) => <span className="font-medium">{r.s.label}</span> },
                  { key: "n", header: "Rows", align: "right", cell: (r) => sizeLabel(r.p.size) },
                  { key: "t", header: "Seconds", align: "right", cell: (r) => (r.p.trainSeconds as number).toFixed(1) },
                  { key: "g", header: "Peak GB (fit)", align: "right", cell: (r) => (r.p.rssGb === null ? "n/a" : r.p.rssGb.toFixed(2)) },
                ]}
              />
            }
            sources={sources}
          />
          <ChartFrame
            title="Cost vs quality"
            compact
            subtitle={`Training seconds against ${metricName} · 2M rows and full scale (◆)`}
            badge={<EvidenceBadge kind="validation" />}
            legend={[
              ...shown.map((s) => ({ label: s.label, color: s.color, shape: "dot" as const })),
              { label: "◆ full scale · ○ 2M rows", color: COLORS.ref, shape: "dash" as const },
            ]}
            chart={<CostQualityChart series={shown} metric={metric} />}
            table={
              <DataTable
                caption="Cost versus quality"
                rows={shown.flatMap((s) => s.points.filter((p) => p.trainSeconds !== null && (p.fullScale || p.size === 2_000_000)).map((p) => ({ s, p })))}
                rowKey={(r) => `${r.s.model}-${r.p.size}`}
                columns={[
                  { key: "m", header: "Model", cell: (r) => <span className="font-medium">{r.s.label}</span> },
                  { key: "n", header: "Rows", align: "right", cell: (r) => sizeLabel(r.p.size) },
                  { key: "t", header: "Seconds", align: "right", cell: (r) => (r.p.trainSeconds as number).toFixed(1) },
                  { key: "q", header: `${metricName} (95% CI), pp`, align: "right", cell: (r) => fmtEst(r.p.est[metric]).text },
                ]}
              />
            }
            sources={sources}
          />
          <ChartFrame
            title="Prediction latency"
            compact
            subtitle="Median by batch size, whisker to p95 · full-scale finalists"
            chart={lat.length ? <LatencyChart rows={lat} /> : <p className="py-16 text-center text-sm text-fg-muted">Latency is stored for the full-scale finalists only.</p>}
            table={
              <DataTable
                caption="Prediction latency"
                rows={res}
                rowKey={(r) => r.model}
                columns={[
                  { key: "m", header: "Model", cell: (r) => <span className="font-medium">{r.label}</span> },
                  { key: "1", header: "1 row p50/p95", align: "right", cell: (r) => r.p50p95[0] },
                  { key: "2", header: "1K rows", align: "right", cell: (r) => r.p50p95[1] },
                  { key: "3", header: "100K rows", align: "right", cell: (r) => r.p50p95[2] },
                ]}
              />
            }
            sources={sources.slice(0, 1)}
          />
        </div>
        <Panel>
          <PanelHeader title="Resource profile" subtitle="Full-scale finalists among the compared models" actions={<Badge tone="slate">Stored measurement</Badge>} />
          <PanelBody>
            {res.length ? (
              <DataTable
                caption="Resource profile"
                rows={res}
                rowKey={(r) => r.model}
                columns={[
                  { key: "m", header: "Model", cell: (r) => <span className="font-medium">{r.label}</span> },
                  { key: "t", header: "Training time", align: "right", cell: (r) => r.trainSeconds },
                  { key: "g", header: "Peak memory (fit)", align: "right", cell: (r) => r.peakFit },
                  { key: "p", header: "Peak memory (process)", align: "right", cell: (r) => r.peakProcess },
                  { key: "t1", header: "Throughput, 1-row batches", align: "right", cell: (r) => r.throughput[0] },
                  { key: "t2", header: "1K batches", align: "right", cell: (r) => r.throughput[1] },
                  { key: "t3", header: "100K batches", align: "right", cell: (r) => r.throughput[2] },
                ]}
              />
            ) : (
              <p className="text-sm text-fg-muted">Select a finalist to see its resource profile.</p>
            )}
          </PanelBody>
        </Panel>
      </section>

      <Panel>
        <PanelHeader
          title={`Why the ${sel.chosenLabel} was selected`}
          subtitle="The frozen selection record · pre-specified rule applied to validation estimates"
          actions={
            <>
              <EvidenceBadge kind="validation" />
              <InfoTip label="About the one-SE rule">
                Among the models, take the simplest whose AIPW value at the 20% budget is within one paired standard error of the best. Simpler models are cheaper to run and explain.
              </InfoTip>
            </>
          }
        />
        <PanelBody className="space-y-5">
          <ol className="grid grid-cols-1 gap-3 lg:grid-cols-4">
            <li className="rounded-control border border-border p-3.5">
              <p className="eyebrow">Step 1 · order</p>
              <p className="mt-2 flex flex-wrap items-center gap-1.5 text-[13px] font-medium text-fg">
                {sel.order.map((o, i) => (
                  <span key={o.model} className="flex items-center gap-1.5">
                    {i > 0 ? <span className="text-fg-muted">→</span> : null}
                    {o.label}
                  </span>
                ))}
              </p>
              <p className="mt-2 text-xs leading-5 text-fg-muted">Simplest to most complex. Best by value: {sel.bestByValueLabel}.</p>
            </li>
            {sel.steps.map((st, i) => (
              <li key={st.simpler} className={cn("rounded-control border p-3.5", st.withinOneSe ? "border-teal/40 bg-teal-soft/40" : "border-border")}>
                <p className="eyebrow">Step {i + 2} · is {st.simplerLabel} close enough?</p>
                <p className="mt-2 flex items-center gap-1.5 text-[13px] font-semibold text-fg">
                  {st.withinOneSe ? <CircleCheck className="size-4 text-teal" aria-hidden /> : <CircleX className="size-4 text-danger" aria-hidden />}
                  {st.withinOneSe ? "Within one SE" : "More than one SE behind"}
                </p>
                <p className="mt-1.5 text-xs leading-5 text-fg-secondary">
                  {st.bestLabel} minus {st.simplerLabel}: <span className="font-medium tabular-nums">{st.diff}</span>; paired SE {st.se}.
                </p>
              </li>
            ))}
            <li className="rounded-control border border-violet/40 bg-violet-soft/50 p-3.5">
              <p className="eyebrow">Result</p>
              <p className="mt-2 text-base font-semibold text-fg">{sel.chosenLabel}</p>
              <p className="mt-1.5 text-xs leading-5 text-fg-secondary">
                Qini {sel.qiniSupports ? "supports" : "does not support"} the choice (best by Qini: {sel.bestByQiniLabel}).
              </p>
            </li>
          </ol>
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <ul className="space-y-2 pl-5 text-[13px] leading-[21px] text-fg-secondary [&>li]:list-disc [&>li]:marker:text-ref-soft">
              {sel.summary.map((t, i) => (
                <li key={i}>{t}</li>
              ))}
            </ul>
            <div>
              <p className="eyebrow">Qini, AIPW (validation)</p>
              <DataTable
                caption="Qini by finalist"
                rows={sel.qini}
                rowKey={(q) => q.model}
                columns={[
                  { key: "m", header: "Model", cell: (q) => <span className="font-medium">{q.label}</span> },
                  { key: "q", header: "Qini (95% CI), pp", align: "right", cell: (q) => q.text },
                ]}
              />
            </div>
          </div>
        </PanelBody>
      </Panel>

      <CaveatCallout
        title="Benchmark, not business ROI"
        items={[
          "Estimates are conditional on the released benchmark sample and the constant-0.85 propensity assumption; IPW and AIPW disagree on this file, so AIPW is the headline throughout.",
          "Validation numbers drove model selection and are optimistic; the held-out scope holds the unbiased one-shot result for the selected model.",
        ]}
      />
    </>
  );
}
