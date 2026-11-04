/**
 * Pure view-model functions for the Overview: API payload in, display-ready rows out.
 * Nothing here fetches or renders; all numbers come from the payloads, all formatting from lib/format.
 */
import type {
  CriteoHeldoutData,
  CriteoPolicyRow,
  CriteoReferenceRow,
  HillstromComparison,
  HillstromDecisionData,
  HillstromOutcome,
  HillstromPolicyValue,
  OverviewData,
} from "@/lib/api/types";
import { SERIES_COLORS } from "@/lib/colors";
import { budget, count, criteo, compact, hillstrom, interval, pValue } from "@/lib/format";

export const POLICY_LABELS: Record<string, string> = {
  mens_for_all: "Men's E-Mail for all",
  womens_for_all: "Women's E-Mail for all",
  no_email: "No e-mail",
  random: "Uniform random",
};

export const policyLabel = (key: string) => POLICY_LABELS[key] ?? key;

export interface KpiModel {
  label: string;
  value: string;
  range?: string;
  note: string;
  info?: string;
  badge?: "primary" | "supported" | "heldout" | "exploratory";
  accent: "teal" | "violet" | "amber" | "slate";
}

// ---------------------------------------------------------------------------------- Hillstrom

export function hillstromKpis(o: OverviewData["hillstrom"]): KpiModel[] {
  const v = o.value;
  const b = o.vs_strongest_alternative_baseline;
  const iv = interval(v.value, v.ci_low, v.ci_high, (n) => hillstrom.rate(n));
  const dv = interval(b.diff, b.ci_low, b.ci_high, (n) => hillstrom.rateDelta(n));
  return [
    {
      label: "Frozen decision",
      value: policyLabel(o.frozen_policy),
      note: "A fixed policy: the same e-mail for every customer. Chosen before the held-out customers were scored.",
      badge: "heldout",
      accent: "teal",
    },
    {
      label: "Held-out visit rate",
      value: iv.value,
      range: iv.range,
      note: `Expected visit rate if every customer received this e-mail (AIPW). Visit is the primary outcome.`,
      info: "AIPW (augmented inverse-propensity weighting) estimates the average outcome under a policy from randomized data, using an outcome model to reduce variance.",
      badge: "primary",
      accent: "teal",
    },
    {
      label: "Lift vs best baseline",
      value: dv.value,
      range: dv.range,
      note: `Paired difference vs ${policyLabel(b.baseline)}, the strongest alternative. ${pValue(b.p_value_holm, "Holm p")}.`,
      info: "Holm adjusts p-values for the several baseline comparisons made on the same outcome.",
      badge: b.p_value_holm < 0.05 ? "supported" : undefined,
      accent: "violet",
    },
    {
      label: "Held-out customers",
      value: count(o.heldout_customers),
      note: "Reserved test customers; never used to choose the policy.",
      accent: "slate",
    },
  ];
}

export interface PolicyRowModel {
  key: string;
  label: string;
  sublabel: string;
  value: number;
  low: number;
  high: number;
  color: string;
  emphasis: boolean;
  hollow: boolean;
  valueLabel: string;
  rangeLabel: string;
}

const POLICY_ORDER = ["mens_for_all", "womens_for_all", "random", "no_email"];

/** Held-out AIPW expected visit rate per policy, frozen policy first. Values in percent. */
export function hillstromPolicyRows(d: HillstromDecisionData): PolicyRowModel[] {
  const frozen = d.chosen_policy.name;
  const byPolicy = new Map<string, HillstromPolicyValue>();
  for (const r of d.policy_values) {
    if (r.outcome === "visit" && r.estimator === "AIPW" && r.policy !== "frozen_policy") {
      byPolicy.set(r.policy, r);
    }
  }
  const keys = [frozen, ...POLICY_ORDER.filter((k) => k !== frozen)].filter((k) => byPolicy.has(k));
  return keys.map((k) => {
    const r = byPolicy.get(k)!;
    const iv = interval(r.value, r.ci_low, r.ci_high, (n) => hillstrom.rate(n));
    return {
      key: k,
      label: policyLabel(k),
      sublabel: k === frozen ? "Frozen decision" : "Baseline",
      value: hillstrom.toPercent(r.value),
      low: hillstrom.toPercent(r.ci_low),
      high: hillstrom.toPercent(r.ci_high),
      color: SERIES_COLORS[k] ?? "#64748B",
      emphasis: k === frozen,
      hollow: k !== frozen,
      valueLabel: iv.value,
      rangeLabel: iv.range,
    };
  });
}

export interface OutcomeRowModel {
  outcome: HillstromOutcome;
  label: string;
  role: "primary" | "exploratory";
  frozen: string;
  frozenRange: string;
  comparator: string;
  diff: string;
  diffRange: string;
  pHolm: string;
  pHolmValue: number;
  supported: boolean;
}

const OUTCOME_LABEL: Record<HillstromOutcome, string> = {
  visit: "Site visit",
  conversion: "Conversion",
  spend: "Spend per customer",
};

/** One row per outcome: frozen policy value vs the pre-specified comparator, with its role. */
export function hillstromOutcomeRows(d: HillstromDecisionData): OutcomeRowModel[] {
  const rows: OutcomeRowModel[] = [];
  for (const outcome of ["visit", "conversion", "spend"] as HillstromOutcome[]) {
    const val = d.policy_values.find(
      (r) => r.outcome === outcome && r.estimator === "AIPW" && r.policy === "frozen_policy",
    );
    const cmp: HillstromComparison | undefined = d.comparisons.find(
      (c) => c.outcome === outcome && c.estimator === "AIPW" && c.primary,
    );
    if (!val || !cmp) continue;
    const role = d.outcome_roles[outcome];
    const level = outcome === "spend" ? hillstrom.spend : outcome === "conversion" ? (n: number) => hillstrom.rate(n, 2) : (n: number) => hillstrom.rate(n);
    const delta = outcome === "spend" ? hillstrom.spendDelta : (n: number) => hillstrom.rateDelta(n);
    const fi = interval(val.value, val.ci_low, val.ci_high, level);
    const di = interval(cmp.diff, cmp.ci_low, cmp.ci_high, delta);
    rows.push({
      outcome,
      label: OUTCOME_LABEL[outcome],
      role,
      frozen: fi.value,
      frozenRange: fi.range,
      comparator: policyLabel(cmp.baseline),
      diff: di.value,
      diffRange: di.range,
      pHolm: pValue(cmp.p_value_holm, "Holm p"),
      pHolmValue: cmp.p_value_holm,
      supported: role === "primary" && cmp.p_value_holm < 0.05,
    });
  }
  return rows;
}

export interface Insight {
  headline: string;
  paragraphs: string[];
}

export function hillstromInsight(o: OverviewData["hillstrom"], rows: OutcomeRowModel[]): Insight {
  const v = interval(o.value.value, o.value.ci_low, o.value.ci_high, (n) => hillstrom.rate(n));
  const b = o.vs_strongest_alternative_baseline;
  const d = interval(b.diff, b.ci_low, b.ci_high, (n) => hillstrom.rateDelta(n));
  const exploratory = rows.filter((r) => r.role === "exploratory");
  const anyExploratorySupported = exploratory.some((r) => r.pHolmValue < 0.05);
  const paragraphs = [
    `On ${count(o.heldout_customers)} held-out customers, sending the ${policyLabel(o.frozen_policy)} to everyone gave an expected visit rate of ${v.value} (95% CI ${v.range}). That is ${d.value} (95% CI ${d.range}) versus ${policyLabel(b.baseline)}, the strongest alternative, with ${pValue(b.p_value_holm, "Holm p")}.`,
    "No personalised model beat this fixed policy during model selection, so the simplest option was frozen before the held-out customers were scored.",
  ];
  if (exploratory.length) {
    paragraphs.push(
      `${exploratory.map((r) => r.outcome[0].toUpperCase() + r.outcome.slice(1)).join(" and ")} are exploratory (rare or heavily zero-inflated outcomes) and are shown beside the primary result, not used to choose the policy${anyExploratorySupported ? "" : "; neither shows a clear advantage over the comparator"}.`,
    );
  }
  return { headline: `Send the ${policyLabel(o.frozen_policy)}.`, paragraphs };
}

// ---------------------------------------------------------------------------------- Criteo

const aipwBudgetRows = (d: CriteoHeldoutData): CriteoPolicyRow[] =>
  d.numeric.policies.filter((p) => p.estimator === "AIPW" && p.policy === "budget" && p.budget_requested !== null);

function referenceRow(d: CriteoHeldoutData, name: string): CriteoReferenceRow | undefined {
  return d.numeric.references.find((r) => r.estimator === "AIPW" && r.policy === name);
}

const PRIMARY_BUDGET = 0.2;

export function criteoKpis(d: CriteoHeldoutData): KpiModel[] {
  const primary = aipwBudgetRows(d).find((p) => p.budget_requested === PRIMARY_BUDGET);
  const all = referenceRow(d, "treat_all");
  if (!primary || !all) return [];
  const v = interval(primary.value, primary.ci_low, primary.ci_high, (n) => criteo.ppValue(n));
  const g = interval(primary.diff_vs_random, primary.diff_vs_random_ci_low, primary.diff_vs_random_ci_high, (n) => criteo.ppValue(n));
  const t = interval(all.value, all.ci_low, all.ci_high, (n) => criteo.ppValue(n));
  return [
    {
      label: `AIPW value @ ${budget(PRIMARY_BUDGET)} budget`,
      value: `${v.value} pp`,
      range: v.range,
      note: "Extra visit rate vs treating nobody, if the top 20% by predicted uplift are contacted. Pre-specified primary metric.",
      info: "Percentage points (pp) of visit rate per evaluated customer, relative to treating nobody. A conditional estimate for the released sample, not ROI.",
      badge: "primary",
      accent: "teal",
    },
    {
      label: `Gain vs random @ ${budget(PRIMARY_BUDGET)}`,
      value: `${g.value} pp`,
      range: g.range,
      note: "Paired difference against contacting the same share of customers at random.",
      badge: "supported",
      accent: "violet",
    },
    {
      label: "Treat-everyone reference",
      value: `${t.value} pp`,
      range: t.range,
      note: "Unconstrained: only reachable with a 100% budget. Shown for context, not as a feasible alternative.",
      accent: "slate",
    },
    {
      label: "Held-out customers",
      value: compact(d.n),
      note: `${count(d.n)} test rows from the released benchmark sample; one-shot evaluation.`,
      badge: "heldout",
      accent: "slate",
    },
  ];
}

export interface CriteoBudgetPoint {
  budget: number; // percent
  model: { value: number; low: number; high: number; valueLabel: string; rangeLabel: string };
  random: { value: number; low: number; high: number; valueLabel: string; rangeLabel: string };
  vsRandomLabel: string;
  primary: boolean;
}

function stat(value: number, low: number, high: number) {
  const iv = interval(value, low, high, (n) => criteo.ppValue(n));
  return {
    value: criteo.toPp(value),
    low: criteo.toPp(low),
    high: criteo.toPp(high),
    valueLabel: iv.value,
    rangeLabel: `${iv.range} pp`,
  };
}

/** Evaluated budgets only (whatever the report contains); random comes from the stored reference rows. */
export function criteoBudgetPoints(d: CriteoHeldoutData): CriteoBudgetPoint[] {
  return aipwBudgetRows(d).flatMap((p) => {
    const pct = Math.round((p.budget_requested ?? 0) * 100);
    const rnd = d.numeric.references.find((r) => r.estimator === "AIPW" && r.policy === `random@${pct}%`);
    if (!rnd) return [];
    const dv = interval(p.diff_vs_random, p.diff_vs_random_ci_low, p.diff_vs_random_ci_high, (n) => criteo.ppValue(n));
    return [
      {
        budget: pct,
        model: stat(p.value, p.ci_low, p.ci_high),
        random: stat(rnd.value, rnd.ci_low, rnd.ci_high),
        vsRandomLabel: `${dv.value} pp ${dv.range}`,
        primary: p.budget_requested === PRIMARY_BUDGET,
      },
    ];
  });
}

export function criteoTreatAll(d: CriteoHeldoutData) {
  const r = referenceRow(d, "treat_all");
  return r ? stat(r.value, r.ci_low, r.ci_high) : null;
}

export function criteoInsight(d: CriteoHeldoutData): Insight | null {
  const primary = aipwBudgetRows(d).find((p) => p.budget_requested === PRIMARY_BUDGET);
  const all = referenceRow(d, "treat_all");
  if (!primary || !all) return null;
  const v = interval(primary.value, primary.ci_low, primary.ci_high, (n) => criteo.ppValue(n));
  const g = interval(primary.diff_vs_random, primary.diff_vs_random_ci_low, primary.diff_vs_random_ci_high, (n) => criteo.ppValue(n));
  const share = Math.round((primary.value / all.value) * 100);
  const overlap = primary.ci_high >= all.ci_low && all.ci_high >= primary.ci_low;
  return {
    headline: `Contact the top ${budget(PRIMARY_BUDGET)} by predicted uplift.`,
    paragraphs: [
      `On ${count(d.n)} held-out rows of the released benchmark, ranking customers with the frozen ${d.model.replace("_", "-")} and contacting the top ${budget(PRIMARY_BUDGET)} gave an AIPW policy value of ${v.value} pp (95% CI ${v.range}), ${g.value} pp above random targeting at the same budget (95% CI ${g.range}).`,
      `In point-estimate terms that recovers about ${share}% of the treat-everyone effect (${criteo.ppValue(all.value)} pp)${overlap ? "; the intervals overlap, so this is not a claim that it is smaller" : ""}. Values are shown only for the 10%, 20% and 30% budgets that were evaluated.`,
    ],
  };
}

/** The documented IPW vs AIPW discrepancy, with the held-out numbers (never hidden). */
export function criteoIpwCaveat(d: CriteoHeldoutData): string {
  const ac = d.assumption_check;
  return `IPW, the cross-check, reads higher than AIPW on this file: treat-everyone ${criteo.ppValue(ac.ate_ipw)} pp vs ${criteo.ppValue(ac.ate_aipw)} pp (difference ${criteo.ppValue(ac.ate_ipw_minus_aipw)} pp, SE ${criteo.ppValue(ac.ate_ipw_minus_aipw_se)}). AIPW is the primary estimator.`;
}
