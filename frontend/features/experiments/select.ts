/**
 * Pure view-models for the Experiment Lab: API payloads in, display-ready rows out. Significance
 * badges are derived from `significant_holm_0.05` and the outcome's role; nothing is hardcoded.
 */
import type { DotRow } from "@/components/charts/dot-whisker-chart";
import type { DotPlotRow } from "@/components/charts/dot-plot";
import type {
  CheckGrade,
  CriteoDiagnosticsData,
  CriteoIpwAipwData,
  HillstromBalanceRow,
  HillstromEffectRow,
  HillstromGroupRow,
  HillstromOutcome,
} from "@/lib/api/types";
import { COLORS } from "@/lib/colors";
import { compact, count, criteo, hillstrom, interval, pValue } from "@/lib/format";

export const OUTCOMES: readonly HillstromOutcome[] = ["visit", "conversion", "spend"];
export const SAMPLES = ["full", "heldout"] as const;
export type Sample = (typeof SAMPLES)[number];
export const SCALES = ["absolute", "relative"] as const;
export type Scale = (typeof SCALES)[number];
export const COMPARISONS = ["mens", "womens"] as const;
export type ComparisonKey = (typeof COMPARISONS)[number];
export type Role = "primary" | "exploratory";

export const OUTCOME_LABEL: Record<HillstromOutcome, string> = {
  visit: "Visit",
  conversion: "Conversion",
  spend: "Spend",
};

export const SAMPLE_LABEL: Record<Sample, string> = { full: "Full sample", heldout: "Held-out" };

export const ARM_LABEL: Record<ComparisonKey | "control", string> = {
  control: "No e-mail",
  mens: "Men's e-mail",
  womens: "Women's e-mail",
};
const ARM_COLOR: Record<ComparisonKey | "control", string> = {
  control: COLORS.ref,
  mens: COLORS.teal,
  womens: COLORS.violet,
};

export function comparisonKey(row: Pick<HillstromEffectRow, "comparison">): ComparisonKey {
  return row.comparison.startsWith("Mens") ? "mens" : "womens";
}

export const isRateOutcome = (o: HillstromOutcome) => o !== "spend";

/** Outcome value in business units: visit/conversion as a rate, spend in dollars per customer. */
export function outcomeValue(o: HillstromOutcome, x: number): string {
  if (o === "spend") return hillstrom.spend(x);
  return hillstrom.rate(x, o === "conversion" ? 2 : 1);
}

function delta(o: HillstromOutcome, x: number): string {
  return o === "spend" ? hillstrom.spendDelta(x) : hillstrom.rateDelta(x);
}

// ------------------------------------------------------------------------------ significance

export type Significance = "supported" | "exploratory-significant" | "not-significant";

export function significance(row: HillstromEffectRow, role: Role): Significance {
  if (!row["significant_holm_0.05"]) return "not-significant";
  return role === "primary" ? "supported" : "exploratory-significant";
}

export const SIGNIFICANCE_LABEL: Record<Significance, string> = {
  supported: "Supported",
  "exploratory-significant": "Significant (exploratory)",
  "not-significant": "Not significant",
};
const SIGNIFICANCE_COLOR: Record<Significance, string> = {
  supported: COLORS.teal,
  "exploratory-significant": COLORS.amber,
  "not-significant": COLORS.ref,
};

// ------------------------------------------------------------------------------ arms

export interface ArmBar {
  key: ComparisonKey | "control";
  label: string;
  n: number;
  value: number;
  valueLabel: string;
  color: string;
}

/** Observed (unadjusted) outcome per arm. The control arm is shared by both comparisons. */
export function armBars(effects: HillstromEffectRow[], outcome: HillstromOutcome): ArmBar[] {
  const rows = effects.filter((r) => r.outcome === outcome);
  if (!rows.length) return [];
  const bar = (key: ComparisonKey | "control", n: number, value: number): ArmBar => ({
    key,
    label: ARM_LABEL[key],
    n,
    value: isRateOutcome(outcome) ? hillstrom.toPercent(value) : value,
    valueLabel: outcomeValue(outcome, value),
    color: ARM_COLOR[key],
  });
  const control = rows[0];
  return [
    bar("control", control.n_control, control.control),
    ...rows.map((r) => bar(comparisonKey(r), r.n_treated, r.treated)),
  ];
}

export interface ArmSummaryRow {
  key: ComparisonKey | "control";
  label: string;
  n: number;
  visit: string;
  conversion: string;
  spend: string;
  /** only the full-sample group summary reports these */
  share?: string;
  zeroSpend?: string;
}

const SEGMENT: Record<ComparisonKey | "control", string> = {
  control: "No E-Mail",
  mens: "Mens E-Mail",
  womens: "Womens E-Mail",
};

export function armSummary(
  effects: HillstromEffectRow[],
  sample: Sample,
  groups: HillstromGroupRow[],
): ArmSummaryRow[] {
  const by = (o: HillstromOutcome) => effects.filter((r) => r.outcome === o);
  const visit = by("visit");
  const conv = by("conversion");
  const spend = by("spend");
  if (!visit.length || !conv.length || !spend.length) return [];
  const keys: (ComparisonKey | "control")[] = ["control", "mens", "womens"];
  return keys.map((key) => {
    const pick = (rows: HillstromEffectRow[], f: "control" | "treated") => {
      const r = key === "control" ? rows[0] : rows.find((x) => comparisonKey(x) === key)!;
      return r[f];
    };
    const f = key === "control" ? "control" : "treated";
    const base = visit.find((x) => key === "control" || comparisonKey(x) === key)!;
    const g = groups.find((x) => x.segment === SEGMENT[key]);
    return {
      key,
      label: ARM_LABEL[key],
      n: key === "control" ? base.n_control : base.n_treated,
      visit: outcomeValue("visit", pick(visit, f)),
      conversion: outcomeValue("conversion", pick(conv, f)),
      spend: outcomeValue("spend", pick(spend, f)),
      ...(sample === "full" && g
        ? { share: hillstrom.rate(g.share, 1), zeroSpend: hillstrom.rate(g.share_zero_spend, 1) }
        : {}),
    };
  });
}

// ------------------------------------------------------------------------------ forest

export interface ForestRow extends DotRow {
  comparison: ComparisonKey;
  significance: Significance;
  pLabel: string;
  pHolmLabel: string;
  method: string;
  absLabel: string;
  absRange: string;
  relLabel: string;
  relRange: string;
}

export function forestRows(
  effects: HillstromEffectRow[],
  outcome: HillstromOutcome,
  scale: Scale,
  role: Role,
): ForestRow[] {
  return effects
    .filter((r) => r.outcome === outcome)
    .map((r) => {
      const sig = significance(r, role);
      const rate = isRateOutcome(outcome);
      const abs = interval(r.abs_effect, r.abs_ci_low, r.abs_ci_high, (n) => delta(outcome, n));
      const rel = interval(r.rel_effect, r.rel_ci_low, r.rel_ci_high, (n) => hillstrom.relDelta(n, 0));
      const k = comparisonKey(r);
      const shown = scale === "relative" ? rel : abs;
      // plotting units: percentage points for rates, dollars for spend, percent for relative
      const u = scale === "relative" ? 100 : rate ? 100 : 1;
      const [v, lo, hi] =
        scale === "relative"
          ? [r.rel_effect, r.rel_ci_low, r.rel_ci_high]
          : [r.abs_effect, r.abs_ci_low, r.abs_ci_high];
      return {
        key: `${k}-${outcome}`,
        label: `${ARM_LABEL[k]} vs none`,
        sublabel: SIGNIFICANCE_LABEL[sig],
        value: v * u,
        low: lo * u,
        high: hi * u,
        color: SIGNIFICANCE_COLOR[sig],
        emphasis: sig === "supported",
        hollow: sig === "not-significant",
        valueLabel: shown.value,
        rangeLabel: shown.range,
        tooltipNote: `${pValue(r.p_value_holm, "Holm p")} · ${r.method}`,
        comparison: k,
        significance: sig,
        pLabel: pValue(r.p_value),
        pHolmLabel: pValue(r.p_value_holm, "Holm p"),
        method: r.method,
        absLabel: abs.value,
        absRange: abs.range,
        relLabel: rel.value,
        relRange: rel.range,
      };
    });
}

export function forestUnit(outcome: HillstromOutcome, scale: Scale) {
  if (scale === "relative") return { axis: "Relative change vs no e-mail (%)", tick: (v: number) => `${v}%` };
  if (outcome === "spend") return { axis: "Difference in spend per customer ($)", tick: (v: number) => `${v < 0 ? "−" : ""}$${Math.abs(v).toFixed(2)}` };
  return { axis: "Difference in rate (percentage points)", tick: (v: number) => `${v.toFixed(1)}` };
}

export interface Readout {
  headline: string;
  paragraphs: string[];
  significance: Significance;
}

const VERB: Record<HillstromOutcome, string> = {
  visit: "visited the site",
  conversion: "converted",
  spend: "spent on average",
};

/** Plain-language reading of one comparison, built only from the stored numbers. */
export function effectReadout(row: HillstromEffectRow, role: Role): Readout {
  const o = row.outcome;
  const k = comparisonKey(row);
  const sig = significance(row, role);
  const abs = interval(row.abs_effect, row.abs_ci_low, row.abs_ci_high, (n) => delta(o, n));
  const rel = interval(row.rel_effect, row.rel_ci_low, row.rel_ci_high, (n) => hillstrom.relDelta(n, 0));
  const verdict: Record<Significance, string> = {
    supported: `Supported: significant after Holm correction, on the pre-specified primary outcome (${pValue(row.p_value_holm, "Holm p")}).`,
    "exploratory-significant": `Significant after Holm correction (${pValue(row.p_value_holm, "Holm p")}), but ${OUTCOME_LABEL[o].toLowerCase()} is an exploratory outcome and was not used for the campaign decision.`,
    "not-significant": `Not significant after Holm correction (${pValue(row.p_value_holm, "Holm p")}); this sample does not show a reliable difference on this exploratory outcome.`,
  };
  return {
    headline: `${ARM_LABEL[k]}: ${abs.value}${o === "spend" ? " per customer" : ""}`,
    significance: sig,
    paragraphs: [
      `Customers sent the ${ARM_LABEL[k].toLowerCase()} ${VERB[o]} ${o === "spend" ? "" : "at "}${outcomeValue(o, row.treated)}, against ${outcomeValue(o, row.control)} with no e-mail (${count(row.n_treated)} vs ${count(row.n_control)} customers). Difference ${abs.value} ${abs.range} (95% CI), ${rel.value} relative ${rel.range}.`,
      verdict[sig],
      `Method: ${row.method}. Unadjusted ${pValue(row.p_value)}${row.welch_p_value !== null ? `; Welch t-test ${pValue(row.welch_p_value)}` : ""}.`,
    ],
  };
}

// ------------------------------------------------------------------------------ balance

export function balanceRows(balance: HillstromBalanceRow[]): DotPlotRow[] {
  return balance.map((b) => ({
    key: b.covariate,
    label: b.covariate.replace(/_/g, " "),
    values: { mens: b.max_abs_smd_mens_vs_control, womens: b.max_abs_smd_womens_vs_control },
    labels: {
      mens: criteo.smd(b.max_abs_smd_mens_vs_control),
      womens: criteo.smd(b.max_abs_smd_womens_vs_control),
    },
    extra: [
      { label: "Test", value: `${b.test}, ${pValue(b.p_value)}` },
      { label: "Balanced (< 0.1)", value: b["balanced_smd_lt_0.1"] ? "Yes" : "No" },
    ],
  }));
}

// ------------------------------------------------------------------------------ Criteo

export interface Kpi {
  label: string;
  value: string;
  note: string;
  accent: "teal" | "violet" | "amber" | "slate";
  info?: string;
}

export function criteoLabKpis(d: CriteoDiagnosticsData): Kpi[] {
  const { dataset, split, treatment } = d;
  return [
    {
      label: "Rows in the file",
      value: compact(dataset.n_rows),
      note: `${count(dataset.n_rows)} customers across a treated and a control arm (anonymized benchmark).`,
      accent: "slate",
    },
    {
      label: "Treated share",
      value: hillstrom.rate(dataset.arms.treated.share, 1),
      note: `${count(dataset.arms.treated.n)} treated · ${count(dataset.arms.control.n)} control.`,
      accent: "slate",
      info: "Criteo's published design assigns treatment with a constant 0.85 probability. It is supported by the publisher but cannot be verified from the released file.",
    },
    {
      label: "Train · validation · test",
      value: `${compact(split.sizes.train)} · ${compact(split.sizes.validation)} · ${compact(split.sizes.test)}`,
      note: `Stratified split (seed ${split.seed}). Treated share ${hillstrom.rate(split.treated_share.test, 1)} in every split.`,
      accent: "slate",
    },
    {
      label: "Randomization verdict",
      value: statusLabel(treatment.verdict.status.replace(/^benchmark_/, "")),
      note: `Benchmark verdict (${treatment.verdict.status}). No feature-based check failed; the warnings stay visible below.`,
      accent: "amber",
    },
  ];
}

export function statusLabel(status: string): string {
  const s = status.replace(/_/g, " ");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

export interface CheckRow {
  id: string;
  name: string;
  measure: string;
  value: string;
  grade: CheckGrade;
  rule: string;
}

const CHECK_META: Record<string, { name: string; measure: string }> = {
  A1a: { name: "Standardized mean differences", measure: "max |SMD|" },
  A1b: { name: "Treatment predicted from features", measure: "best AUC − 0.5" },
  A1c: { name: "Treated share by feature bin", measure: "max |bin share − overall|" },
  A1d: { name: "Stability across row blocks", measure: "max |block share − overall|" },
  A2: { name: "Positivity", measure: "min / max bin share" },
};

export function checkRows(checks: CriteoDiagnosticsData["treatment"]["checks"]): CheckRow[] {
  return Object.entries(checks).map(([key, c]) => {
    const id = key.split("_")[0];
    const meta = CHECK_META[id] ?? { name: key, measure: "value" };
    const value =
      typeof c.value === "number"
        ? c.value.toFixed(c.value < 0.1 ? 4 : 3)
        : `${(c.value.min_bin_share ?? 0).toFixed(3)} / ${(c.value.max_bin_share ?? 0).toFixed(3)}`;
    return { id, name: meta.name, measure: meta.measure, value, grade: c.grade, rule: c.rule };
  });
}

export const GRADE_LABEL: Record<CheckGrade, string> = {
  pass: "Pass",
  warn: "Warning",
  ordering_warning: "Ordering warning",
  fail: "Fail",
};

export function smdRows(d: CriteoDiagnosticsData): DotPlotRow[] {
  const ks = new Map(d.treatment.ks.map((k) => [k.feature, k.ks]));
  return d.treatment.smd.map((s) => ({
    key: s.feature,
    label: s.feature,
    values: { smd: s.smd },
    labels: { smd: criteo.smd(s.smd) },
    extra: [
      { label: "Mean treated", value: s.mean_treated.toFixed(3) },
      { label: "Mean control", value: s.mean_control.toFixed(3) },
      { label: "KS statistic", value: criteo.smd(ks.get(s.feature) ?? NaN) },
    ],
  }));
}

export interface AucRow {
  key: string;
  label: string;
  auc: number;
  se: number | null;
  grade: CheckGrade | null;
  gapLabel: string;
}

export function aucRows(c: CriteoDiagnosticsData["treatment"]["classifier"]): AucRow[] {
  const row = (key: string, label: string, auc: number, se: number | null, grade: CheckGrade | null): AucRow => ({
    key,
    label,
    auc,
    se,
    grade,
    gapLabel: `${auc - 0.5 >= 0 ? "+" : "−"}${Math.abs(auc - 0.5).toFixed(4)}`,
  });
  return [
    row("lgbm", "LightGBM classifier", c.lightgbm.auc, c.lightgbm.auc_se, c.lightgbm.grade),
    row("logit", "Logistic classifier", c.logistic.auc, c.logistic.auc_se, c.logistic.grade),
    row("perm", "Permuted-label control", c.permuted_label_auc, null, null),
  ];
}

export function propensityQuantiles(c: CriteoDiagnosticsData["treatment"]["classifier"]) {
  const p = c.predicted_propensity;
  return {
    items: [
      { label: "Minimum", value: p.min.toFixed(3) },
      { label: "1st percentile", value: p.q01.toFixed(3) },
      { label: "Median", value: p.median.toFixed(3) },
      { label: "99th percentile", value: p.q99.toFixed(3) },
      { label: "Maximum", value: p.max.toFixed(3) },
    ],
    outside: `${(p["share_outside_0.05_0.95"] * 100).toFixed(2)}%`,
  };
}

// ---- IPW vs AIPW

export interface EstimatorRow extends DotRow {
  note?: string;
}

const pp = (n: number) => criteo.pp(n);

export function estimatorRows(d: CriteoIpwAipwData): EstimatorRow[] {
  const get = (q: string) => d.paired_intervals.find((p) => p.quantity === q);
  const ipw = get("ipw");
  const aipw = get("aipw");
  const g = get("gcomp");
  if (!ipw || !aipw || !g) return [];
  const ps = d.poststratified_on_predicted_baseline;
  const mk = (
    key: string,
    label: string,
    sublabel: string,
    est: number,
    lo: number,
    hi: number,
    color: string,
    extra: Partial<DotRow> = {},
  ): EstimatorRow => {
    const iv = interval(est, lo, hi, (n) => pp(n));
    return {
      key,
      label,
      sublabel,
      value: criteo.toPp(est),
      low: criteo.toPp(lo),
      high: criteo.toPp(hi),
      color,
      valueLabel: iv.value,
      rangeLabel: iv.range,
      ...extra,
    };
  };
  return [
    mk("ipw", "IPW", "unadjusted contrast", ipw.estimate, ipw.ci_low, ipw.ci_high, COLORS.amber, { hollow: true }),
    mk("aipw", "AIPW (primary)", "outcome-model adjusted", aipw.estimate, aipw.ci_low, aipw.ci_high, COLORS.teal, { emphasis: true }),
    mk("gcomp", "G-computation", "outcome model only", g.estimate, g.ci_low, g.ci_high, COLORS.violet),
    mk("ps", "Post-stratified", "deciles of predicted baseline", ps.estimate, ps.estimate - 1.96 * ps.se, ps.estimate + 1.96 * ps.se, COLORS.ref, {
      tooltipNote: "Interval is estimate ± 1.96 SE (normal approximation); the report stores only the SE.",
    }),
  ];
}

export interface GapModel {
  difference: { value: string; range: string };
  baselineTreated: string;
  baselineControl: string;
  gap: string;
  gapSe: string;
  z: string;
  p: string;
  permutations: string;
  sample: string;
}

export function gapModel(d: CriteoIpwAipwData): GapModel | null {
  const diff = d.paired_intervals.find((p) => p.quantity === "ipw_minus_aipw");
  if (!diff) return null;
  return {
    difference: interval(diff.estimate, diff.ci_low, diff.ci_high, (n) => criteo.ppDelta(n)),
    baselineTreated: criteo.prob(d.arm_rates.predicted_mu0_among_treated, 3),
    baselineControl: criteo.prob(d.arm_rates.predicted_mu0_among_control, 3),
    gap: criteo.ppDelta(d.arm_rates.baseline_gap_treated_minus_control),
    gapSe: criteo.pp(d.arm_rates.baseline_gap_se),
    z: d.permutation_test.z.toFixed(1),
    p: pValue(d.permutation_test.p_two_sided),
    permutations: count(d.permutation_test.n_perm),
    sample: count(d.meta.n),
  };
}

export interface StratumPoint {
  stratum: number;
  n: number;
  /** percentage points */
  effect: number;
  effectLabel: string;
  /** percent */
  treatedShare: number;
  treatedShareLabel: string;
  baselineLabel: string;
  rates: string;
}

export function strataPoints(d: CriteoIpwAipwData): StratumPoint[] {
  return d.strata.map((s) => ({
    stratum: s.stratum,
    n: s.n,
    effect: criteo.toPp(s.effect),
    effectLabel: criteo.ppDelta(s.effect),
    treatedShare: s.treated_share * 100,
    treatedShareLabel: hillstrom.rate(s.treated_share, 2),
    baselineLabel: criteo.prob(s.mean_predicted_baseline, 2),
    rates: `${criteo.prob(s.rate_treated, 2)} treated · ${criteo.prob(s.rate_control, 2)} control`,
  }));
}

export function heldoutRepeat(a: {
  ate_ipw: number;
  ate_aipw: number;
  ate_ipw_minus_aipw: number;
  ate_ipw_minus_aipw_se: number;
}) {
  return {
    ipw: criteo.pp(a.ate_ipw),
    aipw: criteo.pp(a.ate_aipw),
    diff: criteo.pp(a.ate_ipw_minus_aipw),
    se: criteo.pp(a.ate_ipw_minus_aipw_se),
  };
}
