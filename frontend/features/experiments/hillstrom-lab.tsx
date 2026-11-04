"use client";

import { Badge } from "@/components/ui/badge";
import { Collapsible } from "@/components/ui/accordion";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ChartFrame } from "@/components/charts/chart-frame";
import { DotPlot } from "@/components/charts/dot-plot";
import { niceAxis } from "@/components/charts/theme";
import { DotWhiskerChart } from "@/components/charts/dot-whisker-chart";
import { ReportFigure } from "@/components/charts/report-figure";
import { CaveatCallout } from "@/components/data/caveat-callout";
import { DataTable } from "@/components/data/data-table";
import { EvidenceBadge } from "@/components/data/evidence-badge";
import { InfoTip } from "@/components/data/info-tip";
import { Panel, PanelBody, PanelHeader } from "@/components/data/panel";
import { Control, SubBar } from "@/components/data/sub-bar";
import type {
  HillstromBalanceRow,
  HillstromEffectsData,
  HillstromGroupRow,
  HillstromOutcome,
} from "@/lib/api/types";
import { COLORS } from "@/lib/colors";
import { count } from "@/lib/format";
import { useViewState } from "@/lib/url-state";
import { cn } from "@/lib/utils";
import { ArmBarsChart } from "./charts";
import {
  COMPARISONS,
  OUTCOMES,
  OUTCOME_LABEL,
  SAMPLES,
  SAMPLE_LABEL,
  SCALES,
  SIGNIFICANCE_LABEL,
  armBars,
  armSummary,
  balanceRows,
  effectReadout,
  forestRows,
  forestUnit,
  isRateOutcome,
  type ComparisonKey,
  type Role,
  type Sample,
  type Scale,
  type Significance,
} from "./select";

export interface HillstromLabProps {
  effects: HillstromEffectsData;
  groups: HillstromGroupRow[];
  balance: HillstromBalanceRow[];
  roles: Record<HillstromOutcome, Role>;
  sources: { effects: string[]; groups: string[]; balance: string[] };
  caveats: string[];
}

function SigBadge({ sig }: { sig: Significance }) {
  const tone = sig === "supported" ? "teal" : sig === "exploratory-significant" ? "amber" : "slate";
  return <Badge tone={tone}>{SIGNIFICANCE_LABEL[sig]}</Badge>;
}

export function HillstromLab({ effects, groups, balance, roles, sources, caveats }: HillstromLabProps) {
  const [outcome, setOutcome] = useViewState<HillstromOutcome>("outcome", "visit", OUTCOMES);
  const [sample, setSample] = useViewState<Sample>("sample", "full", SAMPLES);
  const [scale, setScale] = useViewState<Scale>("scale", "absolute", SCALES);
  const [compare, setCompare] = useViewState<ComparisonKey>("compare", "mens", COMPARISONS);

  const rows = sample === "full" ? effects.full_sample : effects.heldout;
  const role = roles[outcome];
  const n = rows.find((r) => r.outcome === outcome);
  // customers across all three arms: the shared control arm plus each e-mail arm
  const sampleSize = n ? n.n_control + rows.filter((r) => r.outcome === outcome).reduce((a, r) => a + r.n_treated, 0) : 0;
  const bars = armBars(rows, outcome);
  const summary = armSummary(rows, sample, groups);
  const forest = forestRows(rows, outcome, scale, role);
  const selected = rows.find((r) => r.outcome === outcome && (r.comparison.startsWith("Mens") ? "mens" : "womens") === compare);
  const readout = selected ? effectReadout(selected, role) : null;
  const unit = forestUnit(outcome, scale);
  const rate = isRateOutcome(outcome);
  const evidence = <EvidenceBadge kind={role === "primary" ? "primary" : "exploratory"} />;
  const sampleBadge = sample === "heldout" ? <EvidenceBadge kind="heldout" /> : <Badge tone="slate">Full sample</Badge>;

  const lo = Math.min(0, ...forest.map((r) => r.low));
  const hi = Math.max(...forest.map((r) => r.high));
  const axis = niceAxis(lo - (hi - lo) * 0.06, hi + (hi - lo) * 0.08);

  return (
    <div className="space-y-6 lg:space-y-8">
      <SubBar>
        <Control label="Outcome">
          <SegmentedControl<HillstromOutcome>
            label="Outcome"
            value={outcome}
            onValueChange={setOutcome}
            options={OUTCOMES.map((o) => ({ value: o, label: roles[o] === "primary" ? `${OUTCOME_LABEL[o]} (primary)` : OUTCOME_LABEL[o] }))}
          />
        </Control>
        <Control label="Sample">
          <SegmentedControl<Sample>
            label="Sample"
            value={sample}
            onValueChange={setSample}
            options={SAMPLES.map((s) => ({ value: s, label: SAMPLE_LABEL[s] }))}
          />
        </Control>
        <div className="ml-auto flex items-center gap-2 text-xs text-fg-muted">
          {evidence}
          {sampleBadge}
          <span className="hidden sm:inline">{count(sampleSize)} customers</span>
        </div>
      </SubBar>

      {role === "exploratory" ? (
        <CaveatCallout
          title={`${OUTCOME_LABEL[outcome]} is an exploratory outcome`}
          items={[
            "Visit is the only pre-specified primary outcome. Conversion and spend are reported for context and were not used for the campaign decision, even when they are statistically significant.",
          ]}
        />
      ) : null}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-7">
          <ChartFrame
            title={`${OUTCOME_LABEL[outcome]} by arm`}
            subtitle={`${rate ? "Share of customers" : "Mean spend per customer"} · observed and unadjusted · ${SAMPLE_LABEL[sample].toLowerCase()}`}
            badge={sampleBadge}
            chart={
              <ArmBarsChart
                bars={bars}
                tick={(v) => (rate ? `${v}%` : `$${v.toFixed(2)}`)}
                axisTitle={rate ? `${OUTCOME_LABEL[outcome]} rate (%)` : "Mean spend ($ per customer)"}
                describe={`${OUTCOME_LABEL[outcome]} by arm: ${bars.map((b) => `${b.label} ${b.valueLabel}`).join("; ")}`}
              />
            }
            table={
              <DataTable
                caption={`${OUTCOME_LABEL[outcome]} by arm`}
                rows={bars}
                rowKey={(b) => b.key}
                columns={[
                  { key: "a", header: "Arm", cell: (b) => <span className="font-medium">{b.label}</span> },
                  { key: "n", header: "Customers", align: "right", cell: (b) => count(b.n) },
                  { key: "v", header: "Observed", align: "right", cell: (b) => b.valueLabel },
                ]}
              />
            }
            sources={sources.effects}
            note="Differences and their intervals are in the forest plot below."
            className="h-full"
          />
        </div>
        <div className="min-w-0 lg:col-span-5">
          <Panel className="h-full">
            <PanelHeader title="Arm summary" subtitle={`All three outcomes · ${SAMPLE_LABEL[sample].toLowerCase()} · observed`} actions={sampleBadge} />
            <PanelBody>
              <DataTable
                caption="Arm summary"
                rows={summary}
                rowKey={(r) => r.key}
                columns={[
                  { key: "a", header: "Arm", cell: (r) => <span className="whitespace-nowrap font-medium">{r.label}</span> },
                  { key: "n", header: "n", align: "right", cell: (r) => count(r.n) },
                  { key: "v", header: "Visit", align: "right", cell: (r) => r.visit },
                  { key: "c", header: "Conv.", align: "right", cell: (r) => r.conversion },
                  { key: "s", header: "Spend", align: "right", cell: (r) => r.spend },
                  ...(sample === "full"
                    ? [{ key: "z", header: "Zero $", align: "right" as const, cell: (r: (typeof summary)[number]) => r.zeroSpend }]
                    : []),
                ]}
              />
              <p className="mt-3 text-xs leading-5 text-fg-muted">
                {sample === "full"
                  ? "Zero-spend share comes from the full-sample group summary; the held-out report stores arm sizes and rates only."
                  : "The held-out customers were not used to choose the campaign policy."}
              </p>
            </PanelBody>
          </Panel>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-8">
          <ChartFrame
            title="Treatment effects"
            subtitle={`Difference vs no e-mail · ${OUTCOME_LABEL[outcome].toLowerCase()} · 95% intervals · Holm-adjusted significance`}
            toolbar={
              <Control label="Scale">
                <SegmentedControl<Scale>
                  label="Scale"
                  value={scale}
                  onValueChange={setScale}
                  options={[
                    { value: "absolute", label: "Absolute" },
                    { value: "relative", label: "Relative" },
                  ]}
                />
              </Control>
            }
            badge={sampleBadge}
            legend={[
              { label: "Supported (primary, Holm p < 0.05)", color: COLORS.teal, shape: "dot" },
              { label: "Significant, exploratory", color: COLORS.amber, shape: "dot" },
              { label: "Not significant", color: COLORS.ref, shape: "dot-hollow" },
            ]}
            chart={
              <DotWhiskerChart
                rows={forest}
                domain={axis.domain}
                ticks={axis.ticks}
                tickFormat={unit.tick}
                axisTitle={unit.axis}
                refLines={[{ x: 0, label: "No difference", color: COLORS.axis }]}
              />
            }
            table={
              <DataTable
                caption="Treatment effects"
                rows={forest}
                rowKey={(r) => r.key}
                columns={[
                  { key: "c", header: "Comparison", cell: (r) => <span className="font-medium">{r.label}</span> },
                  { key: "a", header: "Absolute (95% CI)", align: "right", cell: (r) => `${r.absLabel} ${r.absRange}` },
                  { key: "r", header: "Relative (95% CI)", align: "right", cell: (r) => `${r.relLabel} ${r.relRange}` },
                  { key: "p", header: "p", align: "right", cell: (r) => r.pLabel },
                  { key: "h", header: "Holm p", align: "right", cell: (r) => r.pHolmLabel },
                  { key: "m", header: "Method", cell: (r) => r.method },
                  { key: "s", header: "Verdict", cell: (r) => <SigBadge sig={r.significance} /> },
                ]}
              />
            }
            sources={sources.effects}
            note={`Holm correction is applied across all six comparisons (two e-mail arms × three outcomes) within each sample. ${SAMPLE_LABEL[sample]}.`}
            className="h-full"
          />
        </div>
        <div className="min-w-0 lg:col-span-4">
          <Panel className="h-full">
            <PanelHeader
              title="What this means"
              subtitle="Plain-language reading"
              actions={
                <SegmentedControl<ComparisonKey>
                  label="Comparison"
                  value={compare}
                  onValueChange={setCompare}
                  options={[
                    { value: "mens", label: "Men's" },
                    { value: "womens", label: "Women's" },
                  ]}
                />
              }
            />
            <PanelBody className="space-y-3">
              {readout ? (
                <>
                  <div className="flex flex-wrap items-center gap-2">
                    <SigBadge sig={readout.significance} />
                    <EvidenceBadge kind={role === "primary" ? "primary" : "exploratory"} />
                  </div>
                  <p className="text-lg font-semibold leading-7 tracking-[-0.01em] text-fg">{readout.headline}</p>
                  <div className="space-y-2.5 text-[13px] leading-[21px] text-fg-secondary">
                    {readout.paragraphs.map((p, i) => (
                      <p key={i}>{p}</p>
                    ))}
                  </div>
                </>
              ) : (
                <p className="text-sm text-fg-muted">No row for this comparison in the selected sample.</p>
              )}
            </PanelBody>
          </Panel>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
        <div className="min-w-0 lg:col-span-8">
          <ChartFrame
            title="Covariate balance"
            subtitle="Largest standardized mean difference per covariate · each e-mail arm vs no e-mail · full sample"
            badge={<Badge tone="slate">Randomization check</Badge>}
            legend={[
              { label: "Men's vs control", color: COLORS.teal, shape: "dot" },
              { label: "Women's vs control", color: COLORS.violet, shape: "dot" },
              { label: "0.1 threshold", color: COLORS.refSoft, shape: "dash" },
            ]}
            chart={
              <DotPlot
                rows={balanceRows(balance)}
                rowHeight={38}
                series={[
                  { key: "mens", label: "Men's vs control", color: COLORS.teal },
                  { key: "womens", label: "Women's vs control", color: COLORS.violet },
                ]}
                domain={[0, 0.12]}
                band={[0, 0.1]}
                thresholds={[{ x: 0.1, label: "0.1" }]}
                tickFormat={(v) => v.toFixed(2)}
                axisTitle="Largest absolute standardized mean difference"
                describe={`Covariate balance: ${balance.map((b) => `${b.covariate} ${b.max_abs_smd_mens_vs_control.toFixed(3)} / ${b.max_abs_smd_womens_vs_control.toFixed(3)}`).join("; ")}`}
              />
            }
            table={
              <DataTable
                caption="Covariate balance"
                rows={balance}
                rowKey={(b) => b.covariate}
                columns={[
                  { key: "c", header: "Covariate", cell: (b) => <span className="font-medium">{b.covariate}</span> },
                  { key: "m", header: "Men's max |SMD|", align: "right", cell: (b) => b.max_abs_smd_mens_vs_control.toFixed(3) },
                  { key: "w", header: "Women's max |SMD|", align: "right", cell: (b) => b.max_abs_smd_womens_vs_control.toFixed(3) },
                ]}
              />
            }
            sources={sources.balance}
            className="h-full"
          />
        </div>
        <div className="min-w-0 lg:col-span-4">
          <Panel className="h-full">
            <PanelHeader
              title="Balance tests"
              subtitle="Do the arms differ on pre-treatment covariates?"
              actions={
                <InfoTip label="About balance tests">
                  In a randomized test the arms should look alike before treatment. A large p-value (no detectable difference) and a small standardized difference both support randomization.
                </InfoTip>
              }
            />
            <PanelBody>
              <DataTable
                caption="Balance tests"
                rows={balance}
                rowKey={(b) => b.covariate}
                columns={[
                  {
                    key: "c",
                    header: "Covariate",
                    cell: (b) => (
                      <div>
                        <div className="font-medium">{b.covariate.replace(/_/g, " ")}</div>
                        <div className="text-[11px] text-fg-muted">{b.test}</div>
                      </div>
                    ),
                  },
                  { key: "p", header: "p", align: "right", cell: (b) => b.p_value.toFixed(3) },
                  {
                    key: "b",
                    header: "Balanced",
                    align: "right",
                    cell: (b) => (
                      <span className={cn("text-xs font-medium", b["balanced_smd_lt_0.1"] ? "text-teal-ink" : "text-danger")}>
                        {b["balanced_smd_lt_0.1"] ? "Yes" : "No"}
                      </span>
                    ),
                  },
                ]}
              />
            </PanelBody>
          </Panel>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ReportFigure
          title="Spend distribution"
          subtitle="Why spend uses a bootstrap interval and a permutation test"
          src="/api/v1/hillstrom/figures/spend_distribution"
          alt="Distribution of spend per customer by arm, from the stored Hillstrom report"
          badge={<Badge tone="slate">Full sample</Badge>}
          source="reports/hillstrom/figures/spend_distribution.png"
        />
        <ReportFigure
          title="Covariate distributions"
          subtitle="Pre-treatment covariates by arm"
          src="/api/v1/hillstrom/figures/covariate_distributions"
          alt="Distributions of the pre-treatment covariates by arm, from the stored Hillstrom report"
          badge={<Badge tone="slate">Full sample</Badge>}
          source="reports/hillstrom/figures/covariate_distributions.png"
        />
      </div>

      <Collapsible title="Methodology">
        <ul className="space-y-2.5 pl-5 text-[13px] leading-[21px] text-fg-secondary [&>li]:list-disc [&>li]:marker:text-ref-soft">
          <li>
            <strong className="font-semibold text-fg">Randomization.</strong> Customers were randomly assigned to Men&apos;s e-mail, Women&apos;s e-mail or no e-mail, so differences between arms estimate the effect of the e-mail. The balance checks above are a diagnostic of that assumption.
          </li>
          <li>
            <strong className="font-semibold text-fg">Primary vs exploratory.</strong> Visit is the only pre-specified primary outcome. Conversion and spend are exploratory: they are rare (about 99% of customers spend nothing) and are never used for the decision, even when significant.
          </li>
          <li>
            <strong className="font-semibold text-fg">Holm correction.</strong> The six comparisons (two e-mail arms × three outcomes) are corrected together; badges use the Holm-adjusted p-value at 0.05, not the unadjusted one.
          </li>
          <li>
            <strong className="font-semibold text-fg">Intervals.</strong> Visit and conversion use a z-test with a Newcombe interval; spend uses a bootstrap interval and a permutation test because its distribution is heavily skewed.
          </li>
          <li>
            <strong className="font-semibold text-fg">Full sample vs held-out.</strong> Both are randomized comparisons. The held-out sample was kept aside for the single final policy evaluation, so it has fewer customers and wider intervals; significance can therefore differ between samples.
          </li>
        </ul>
        {caveats.length ? <CaveatCallout className="mt-4" title="Dataset notes" items={caveats} /> : null}
      </Collapsible>
    </div>
  );
}
