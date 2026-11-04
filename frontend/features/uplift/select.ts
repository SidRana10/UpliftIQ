/**
 * Pure view-models for the Uplift Explorer. Everything is derived from the `/criteo/benchmark`,
 * `/criteo/heldout` and `/hillstrom/models` payloads; no number is typed in here. Values stay as
 * stored fractions of visit rate and are formatted to percentage points on the way out.
 */
import type {
  CriteoBenchmarkData,
  CriteoHeldoutData,
  CriteoModelManifest,
  HillstromCandidate,
  HillstromModelsData,
} from "@/lib/api/types";
import { SERIES_COLORS } from "@/lib/colors";
import { budget, criteo, count, gigabytes, hillstrom, interval, ms, seconds, sizeLabel } from "@/lib/format";

export const MODELS = ["logistic", "s_learner", "t_learner", "dr_learner", "tarnet_cpu", "tarnet_mps"] as const;
export type ModelKey = (typeof MODELS)[number];
export const FINALISTS: readonly ModelKey[] = ["logistic", "s_learner", "tarnet_cpu"];
export const DEFAULT_COMPARE: readonly ModelKey[] = ["s_learner", "tarnet_cpu", "logistic"];
export const MAX_COMPARE = 3;

export const MODEL_LABEL: Record<string, string> = {
  logistic: "Logistic",
  s_learner: "S-learner",
  t_learner: "T-learner",
  dr_learner: "DR-learner",
  tarnet_cpu: "TARNet (CPU)",
  tarnet_mps: "TARNet (MPS)",
};
export const modelLabel = (k: string) => MODEL_LABEL[k] ?? k;
export const modelColor = (k: string) => SERIES_COLORS[k] ?? "#64748B";

export const SCOPES = ["validation", "heldout"] as const;
export type Scope = (typeof SCOPES)[number];
export const METRICS = ["v10", "v20", "v30", "qini"] as const;
export type MetricKey = (typeof METRICS)[number];
export const METRIC_LABEL: Record<MetricKey, string> = {
  v10: "AIPW @10%",
  v20: "AIPW @20% (primary)",
  v30: "AIPW @30%",
  qini: "Qini",
};
export const METRIC_REPORT_KEY: Record<MetricKey, string> = {
  v10: "value@10%",
  v20: "value@20%",
  v30: "value@30%",
  qini: "qini",
};
export const BUDGET_METRICS = ["v10", "v20", "v30"] as const;

export interface Est {
  value: number;
  low: number;
  high: number;
}

const est = (t: [number, number, number]): Est => ({ value: t[0], low: t[1], high: t[2] });

/** Estimate as "0.748 pp" and "[0.635, 0.850]". */
export function fmtEst(e: Est) {
  const iv = interval(e.value, e.low, e.high, (n) => criteo.ppValue(n));
  return { value: `${iv.value} pp`, range: iv.range, text: `${iv.value} ${iv.range}` };
}
/** Paired difference: value carries the unit ("+0.587 pp"), the interval does not ("[+0.497, +0.668]"). */
export const fmtDelta = (e: Est) => {
  const iv = interval(e.value, e.low, e.high, (n) => criteo.ppDeltaValue(n));
  return { value: `${iv.value} pp`, range: iv.range, text: `${iv.value} ${iv.range} pp` };
};
/** the interval excludes zero */
export const excludesZero = (e: Est) => e.low > 0 || e.high < 0;

// ------------------------------------------------------------------------- per-model series

export interface CurvePoint {
  size: number;
  est: Record<MetricKey, Est>;
  fullScale: boolean;
  trainSeconds: number | null;
  rssGb: number | null;
}

export interface ModelSeries {
  model: string;
  label: string;
  color: string;
  finalist: boolean;
  points: CurvePoint[];
}

/** Learning-curve series: every model at 100K/500K/2M, finalists also at full scale. */
export function modelSeries(d: CriteoBenchmarkData): ModelSeries[] {
  const lc = d.learning_curve;
  const out: ModelSeries[] = [];
  for (const m of MODELS) {
    const levels = lc.levels.filter((l) => l.model === m).sort((a, b) => a.size - b.size);
    if (!levels.length) continue;
    const eng = (size: number) => lc.engineering.find((e) => e.model === m && e.size === size);
    const points: CurvePoint[] = levels.map((l) => ({
      size: l.size,
      est: { v10: est(l["value@10%"]), v20: est(l["value@20%"]), v30: est(l["value@30%"]), qini: est(l.qini) },
      fullScale: false,
      trainSeconds: eng(l.size)?.train_s ?? null,
      rssGb: eng(l.size)?.rss_peak_fit_gb ?? null,
    }));
    const manifest = d.numeric.models[m];
    if (manifest && FINALISTS.includes(m)) {
      const full = fullScaleEstimates(d, m);
      if (full) {
        points.push({
          size: manifest.n_fit,
          est: full,
          fullScale: true,
          trainSeconds: manifest.fit_seconds,
          rssGb: manifest.rss_gb_peak_during_fit,
        });
      }
    }
    out.push({ model: m, label: modelLabel(m), color: modelColor(m), finalist: FINALISTS.includes(m), points });
  }
  return out;
}

const policyRows = (d: CriteoBenchmarkData, model: string, estimator: "AIPW" | "IPW") =>
  d.numeric.policies.filter((p) => p.model === model && p.estimator === estimator && p.policy === "budget");

function fullScaleEstimates(d: CriteoBenchmarkData, model: string): Record<MetricKey, Est> | null {
  const rows = policyRows(d, model, "AIPW");
  const q = d.numeric.qini.find((x) => x.model === model && x.estimator === "AIPW");
  const at = (b: number) => rows.find((r) => r.budget_requested === b);
  const a = at(0.1);
  const b = at(0.2);
  const c = at(0.3);
  if (!a || !b || !c || !q || q.ci_low === null || q.ci_high === null) return null;
  const e = (r: typeof a): Est => ({ value: r.value, low: r.ci_low, high: r.ci_high });
  return { v10: e(a), v20: e(b), v30: e(c), qini: { value: q.qini, low: q.ci_low, high: q.ci_high } };
}

// ------------------------------------------------------------------------- leaderboard

export interface LeaderRow {
  model: string;
  label: string;
  color: string;
  selected: boolean;
  bestByValue: boolean;
  bestByQini: boolean;
  v10: Est;
  v20: Est;
  v30: Est;
  qini: Est;
  vsRandom20: Est;
  ipw20: number;
  ipwQini: number;
  trainSeconds: number;
  p50Ms1: number;
  p95Ms1: number;
  rowsPerS: number;
  peakRssGb: number;
}

export type SortKey =
  | "label"
  | "v10"
  | "v20"
  | "v30"
  | "vsRandom20"
  | "qini"
  | "ipw20"
  | "trainSeconds"
  | "p50Ms1"
  | "rowsPerS"
  | "peakRssGb";

export function leaderboard(d: CriteoBenchmarkData): LeaderRow[] {
  const sel = d.selection;
  return FINALISTS.flatMap((m) => {
    const manifest = d.numeric.models[m];
    const full = fullScaleEstimates(d, m);
    const b20 = policyRows(d, m, "AIPW").find((r) => r.budget_requested === 0.2);
    const i20 = policyRows(d, m, "IPW").find((r) => r.budget_requested === 0.2);
    const iq = d.numeric.qini.find((x) => x.model === m && x.estimator === "IPW");
    if (!manifest || !full || !b20 || !i20 || !iq) return [];
    const l1 = manifest.latency.find((l) => l.batch === 1);
    const l100k = manifest.latency.find((l) => l.batch === 100000);
    return [
      {
        model: m,
        label: modelLabel(m),
        color: modelColor(m),
        selected: sel.chosen === m,
        bestByValue: sel.best_by_value === m,
        bestByQini: sel.best_by_qini === m,
        ...full,
        vsRandom20: { value: b20.diff_vs_random, low: b20.diff_vs_random_ci_low, high: b20.diff_vs_random_ci_high },
        ipw20: i20.value,
        ipwQini: iq.qini,
        trainSeconds: manifest.fit_seconds,
        p50Ms1: l1?.p50_ms ?? NaN,
        p95Ms1: l1?.p95_ms ?? NaN,
        rowsPerS: l100k?.rows_per_s ?? NaN,
        peakRssGb: manifest.peak_rss_gb,
      },
    ];
  });
}

const sortValue = (r: LeaderRow, k: SortKey): number | string => {
  switch (k) {
    case "label":
      return r.label;
    case "v10":
    case "v20":
    case "v30":
    case "vsRandom20":
    case "qini":
      return r[k].value;
    default:
      return r[k];
  }
};

export function sortLeaderboard(rows: LeaderRow[], key: SortKey, dir: "asc" | "desc"): LeaderRow[] {
  const sign = dir === "asc" ? 1 : -1;
  return [...rows].sort((a, b) => {
    const x = sortValue(a, key);
    const y = sortValue(b, key);
    if (typeof x === "string" || typeof y === "string") return sign * String(x).localeCompare(String(y));
    // missing measurements sort last whichever way the column is ordered
    if (Number.isNaN(x)) return 1;
    if (Number.isNaN(y)) return -1;
    return sign * (x - y);
  });
}

// ------------------------------------------------------------------------- budget chart

export interface BudgetPointModel {
  budget: number; // percent
  est: Est;
  label: string;
  range: string;
  vsRandom: string;
}

export interface BudgetSeries {
  model: string;
  label: string;
  color: string;
  points: BudgetPointModel[];
}

export interface BudgetRefs {
  /** exact stored point values (the validation report stores their intervals as text only) */
  random: { budget: number; value: number; label: string; reportText: string }[];
  treatAll: { value: number; label: string; reportText: string } | null;
}

export function budgetSeries(d: CriteoBenchmarkData, models: string[]): BudgetSeries[] {
  return models
    .filter((m) => FINALISTS.includes(m as ModelKey))
    .flatMap((m) => {
      const rows = policyRows(d, m, "AIPW");
      const points = ([0.1, 0.2, 0.3] as const).flatMap((b): BudgetPointModel[] => {
        const r = rows.find((x) => x.budget_requested === b);
        if (!r) return [];
        const e = { value: r.value, low: r.ci_low, high: r.ci_high };
        const f = fmtEst(e);
        const dv = fmtDelta({ value: r.diff_vs_random, low: r.diff_vs_random_ci_low, high: r.diff_vs_random_ci_high });
        return [{ budget: Math.round(b * 100), est: e, label: f.value, range: f.range, vsRandom: `${dv.value} ${dv.range}` }];
      });
      return points.length ? [{ model: m, label: modelLabel(m), color: modelColor(m), points }] : [];
    });
}

/** Random-at-budget and treat-all references (exact stored values; interval text verbatim from the report). */
export function budgetRefs(d: CriteoBenchmarkData): BudgetRefs {
  const table = (policy: string) =>
    d.tables.references.find((r) => r.policy === policy);
  const any = d.numeric.policies.filter((p) => p.estimator === "AIPW" && p.policy === "budget");
  const random = ([0.1, 0.2, 0.3] as const).flatMap((b) => {
    const r = any.find((x) => x.budget_requested === b);
    if (!r) return [];
    return [
      {
        budget: Math.round(b * 100),
        value: r.random_value,
        label: criteo.pp(r.random_value),
        reportText: table(`random@${Math.round(b * 100)}%`)?.["AIPW value (pp)"] ?? "",
      },
    ];
  });
  const first = any[0];
  return {
    random,
    treatAll: first
      ? {
          value: first.treat_all_value,
          label: criteo.pp(first.treat_all_value),
          reportText: table("treat_all")?.["AIPW value (pp)"] ?? "",
        }
      : null,
  };
}

// ------------------------------------------------------------------------- pairwise

export interface PairRow {
  model: string;
  label: string;
  est: Est;
  text: string;
  excludesZero: boolean;
}

export const pairReferences = (d: CriteoBenchmarkData): string[] => Object.keys(d.numeric.pairwise);

/** Paired differences (model minus reference, AIPW) for one metric: what the report stores, nothing derived. */
export function pairwiseRows(d: CriteoBenchmarkData, reference: string, metric: MetricKey): PairRow[] {
  const rows = d.numeric.pairwise[reference] ?? [];
  return rows
    .filter((r) => r.estimator === "AIPW" && r.metric === METRIC_REPORT_KEY[metric])
    .map((r) => {
      const e = { value: r.diff, low: r.ci_low, high: r.ci_high };
      return { model: r.model, label: modelLabel(r.model), est: e, text: fmtDelta(e).text, excludesZero: excludesZero(e) };
    });
}

// ------------------------------------------------------------------------- scaling

export interface ScalingStep {
  step: string;
  est: Est;
  text: string;
  excludesZero: boolean;
}

export interface ScalingRow {
  model: string;
  label: string;
  color: string;
  steps: ScalingStep[];
  /** plain statement of the last step, computed from its interval */
  statement: string;
}

export function scalingRows(d: CriteoBenchmarkData, metric: MetricKey, models: readonly string[]): ScalingRow[] {
  const key = METRIC_REPORT_KEY[metric];
  return models.flatMap((m) => {
    if (!FINALISTS.includes(m as ModelKey)) return [];
    const steps: ScalingStep[] = d.learning_curve.changes
      .filter((c) => c.model === m && c.metric === key)
      .map((c) => {
        const e = { value: c.diff, low: c.ci_low, high: c.ci_high };
        return { step: c.step.replace(" to ", " → "), est: e, text: fmtDelta(e).text, excludesZero: excludesZero(e) };
      });
    const full = d.change_from_2m.find((c) => c.model === m && c.metric === key);
    if (full) {
      const e = est(full.change);
      steps.push({ step: "2M → full", est: e, text: fmtDelta(e).text, excludesZero: excludesZero(e) });
    }
    const last = steps[steps.length - 1];
    const fullN = d.numeric.models[m]?.n_fit;
    const statement = last
      ? `Going from 2M to ${fullN ? sizeLabel(fullN) : "full"} rows changed ${METRIC_LABEL[metric].replace(" (primary)", "")} by ${fmtDelta(last.est).value}; the interval ${last.excludesZero ? "excludes" : "includes"} zero.`
      : "";
    return [{ model: m, label: modelLabel(m), color: modelColor(m), steps, statement }];
  });
}

// ------------------------------------------------------------------------- engineering

export interface LatencyRow {
  model: string;
  label: string;
  color: string;
  batches: { batch: number; p50: number; p95: number; rowsPerS: number }[];
}

export function latencyRows(d: CriteoBenchmarkData, models: readonly string[]): LatencyRow[] {
  return models.flatMap((m) => {
    const mf = d.numeric.models[m];
    if (!mf || !FINALISTS.includes(m as ModelKey)) return [];
    return [
      {
        model: m,
        label: modelLabel(m),
        color: modelColor(m),
        batches: mf.latency.map((l) => ({ batch: l.batch, p50: l.p50_ms, p95: l.p95_ms, rowsPerS: l.rows_per_s })),
      },
    ];
  });
}

export const batchLabel = (b: number) => (b === 1 ? "1 row" : b === 1000 ? "1,000 rows" : `${count(b)} rows`);

export interface ResourceRow {
  model: string;
  label: string;
  trainSeconds: string;
  peakFit: string;
  peakProcess: string;
  p50p95: string[];
  throughput: string[];
}

export function resourceRows(d: CriteoBenchmarkData, models: readonly string[]): ResourceRow[] {
  return models.flatMap((m) => {
    const mf = d.numeric.models[m];
    if (!mf || !FINALISTS.includes(m as ModelKey)) return [];
    return [
      {
        model: m,
        label: modelLabel(m),
        trainSeconds: seconds(mf.fit_seconds),
        peakFit: gigabytes(mf.rss_gb_peak_during_fit),
        peakProcess: gigabytes(mf.peak_rss_gb),
        p50p95: mf.latency.map((l) => `${ms(l.p50_ms)} / ${ms(l.p95_ms)}`),
        throughput: mf.latency.map((l) => `${count(l.rows_per_s)} rows/s`),
      },
    ];
  });
}

// ------------------------------------------------------------------------- selection

export interface SelectionStep {
  simpler: string;
  simplerLabel: string;
  best: string;
  bestLabel: string;
  diff: string;
  se: string;
  withinOneSe: boolean;
  /** simpler model kept (true) or rejected (false) */
  verdict: string;
}

export interface SelectionModel {
  order: { model: string; label: string }[];
  steps: SelectionStep[];
  chosen: string;
  chosenLabel: string;
  bestByValue: string;
  bestByValueLabel: string;
  bestByQini: string;
  bestByQiniLabel: string;
  qiniSupports: boolean;
  qini: { model: string; label: string; text: string }[];
  summary: string[];
}

export function selectionModel(d: CriteoBenchmarkData): SelectionModel {
  const s = d.selection;
  const steps: SelectionStep[] = s.steps.map((st) => {
    const e = { value: st.diff, low: st.ci_low, high: st.ci_high };
    const f = fmtDelta(e);
    return {
      simpler: st.simpler,
      simplerLabel: modelLabel(st.simpler),
      best: st.best,
      bestLabel: modelLabel(st.best),
      diff: `${f.value} ${f.range}`,
      se: criteo.pp(st.se),
      withinOneSe: st.within_one_se,
      verdict: st.within_one_se ? `${modelLabel(st.simpler)} is within one SE: eligible` : `${modelLabel(st.simpler)} is more than one SE behind: rejected`,
    };
  });
  const qini = d.numeric.qini
    .filter((q) => q.estimator === "AIPW" && q.ci_low !== null && q.ci_high !== null)
    .map((q) => ({ model: q.model, label: modelLabel(q.model), text: fmtEst({ value: q.qini, low: q.ci_low as number, high: q.ci_high as number }).text }));
  const chosenLabel = modelLabel(s.chosen);
  const summary = [
    `Candidates are ordered from simplest to most complex: ${s.order.map(modelLabel).join(" → ")}. The rule keeps the simplest model whose AIPW value at the 20% budget is within one paired standard error of the best model (${modelLabel(s.best_by_value)}).`,
    ...steps.map((x) => `${x.simplerLabel} vs ${x.bestLabel}: difference ${x.diff}, SE ${x.se}. ${x.verdict}.`),
    s.chosen === s.best_by_value
      ? `${chosenLabel} is both the best and the simplest eligible model.`
      : `${chosenLabel} is the simplest eligible model, so it is chosen over the higher-valued ${modelLabel(s.best_by_value)}.`,
    s.qini_supports_choice
      ? `Cross-check: the Qini coefficient also favours ${modelLabel(s.best_by_qini)}, so ranking quality across all budgets supports the choice.`
      : `Cross-check: Qini favours ${modelLabel(s.best_by_qini)}, which does not support the choice; the pre-specified rule was still followed.`,
  ];
  return {
    order: s.order.map((m) => ({ model: m, label: modelLabel(m) })),
    steps,
    chosen: s.chosen,
    chosenLabel,
    bestByValue: s.best_by_value,
    bestByValueLabel: modelLabel(s.best_by_value),
    bestByQini: s.best_by_qini,
    bestByQiniLabel: modelLabel(s.best_by_qini),
    qiniSupports: s.qini_supports_choice,
    qini,
    summary,
  };
}

// ------------------------------------------------------------------------- model detail

export interface DetailRow {
  label: string;
  value: string;
}

function flatten(prefix: string, v: unknown, out: DetailRow[]) {
  if (v !== null && typeof v === "object" && !Array.isArray(v)) {
    for (const [k, x] of Object.entries(v)) flatten(prefix ? `${prefix}.${k}` : k, x, out);
  } else {
    out.push({ label: prefix, value: Array.isArray(v) ? v.join(", ") : String(v) });
  }
}

export interface ModelDetail {
  model: string;
  label: string;
  family: string;
  configKey: string;
  overview: DetailRow[];
  hyperparameters: DetailRow[];
  training: DetailRow[];
  libraries: DetailRow[];
  commit: string;
  dirty: boolean;
}

export function modelDetail(model: string, m: CriteoModelManifest): ModelDetail {
  const hyper: DetailRow[] = [];
  flatten("", m.hyperparameters, hyper);
  const training: DetailRow[] = [];
  if (m.training) {
    training.push(
      { label: "Device", value: m.training.device },
      { label: "Epochs run", value: String(m.training.epochs_run) },
      { label: "Best epoch", value: String(m.training.best_epoch) },
      { label: "Best validation loss", value: m.training.best_val_loss.toFixed(5) },
    );
  }
  return {
    model,
    label: modelLabel(model),
    family: m.family,
    configKey: m.config_key,
    overview: [
      { label: "Training rows", value: count(m.n_fit) },
      { label: "Training time", value: seconds(m.fit_seconds) },
      { label: "Peak memory (fit)", value: gigabytes(m.rss_gb_peak_during_fit) },
      { label: "Peak memory (process)", value: gigabytes(m.peak_rss_gb) },
      { label: "Predicted uplift, mean", value: criteo.ppDelta(m.uplift_mean) },
      { label: "Predicted uplift, std", value: criteo.pp(m.uplift_std) },
      { label: "OMP threads", value: m.omp_num_threads },
    ],
    hyperparameters: hyper,
    training,
    libraries: Object.entries(m.libraries).map(([label, value]) => ({ label, value })),
    commit: m.code.commit,
    dirty: m.code.dirty,
  };
}

// ------------------------------------------------------------------------- held-out scope

export interface HeldoutQini {
  estimator: string;
  text: string;
}

export function heldoutQini(d: CriteoHeldoutData) {
  const a = d.numeric.qini.find((q) => q.estimator === "AIPW");
  const i = d.numeric.qini.find((q) => q.estimator === "IPW");
  return {
    aipw: a && a.ci_low !== null && a.ci_high !== null ? fmtEst({ value: a.qini, low: a.ci_low, high: a.ci_high }) : null,
    aipwShare: a ? a.qini / a.ate_treat_all : null,
    ipw: i ? criteo.pp(i.qini) : null,
  };
}

export interface HeldoutTableRow {
  budget: string;
  value: string;
  range: string;
  random: string;
  vsRandom: string;
  vsRandomRange: string;
  ipw: string;
  primary: boolean;
}

export function heldoutRows(d: CriteoHeldoutData): HeldoutTableRow[] {
  return d.numeric.policies
    .filter((p) => p.estimator === "AIPW" && p.policy === "budget")
    .map((p) => {
      const ipw = d.numeric.policies.find(
        (x) => x.estimator === "IPW" && x.policy === "budget" && x.budget_requested === p.budget_requested,
      );
      const v = fmtEst({ value: p.value, low: p.ci_low, high: p.ci_high });
      const dv = fmtDelta({ value: p.diff_vs_random, low: p.diff_vs_random_ci_low, high: p.diff_vs_random_ci_high });
      return {
        budget: budget(p.budget_requested ?? p.budget),
        value: v.value,
        range: v.range,
        random: criteo.pp(p.random_value),
        vsRandom: dv.value,
        vsRandomRange: dv.range,
        ipw: ipw ? criteo.pp(ipw.value) : "n/a",
        primary: p.budget_requested === 0.2,
      };
    });
}

export interface CompareRow {
  budget: string;
  validation: string;
  heldout: string;
}

/** The selected model's validation and held-out value at each budget (different samples, not paired). */
export function validationVsHeldout(bench: CriteoBenchmarkData, held: CriteoHeldoutData): CompareRow[] {
  const model = held.model;
  return ([0.1, 0.2, 0.3] as const).flatMap((b) => {
    const v = policyRows(bench, model, "AIPW").find((r) => r.budget_requested === b);
    const h = held.numeric.policies.find((r) => r.estimator === "AIPW" && r.policy === "budget" && r.budget_requested === b);
    if (!v || !h) return [];
    return [
      {
        budget: budget(b),
        validation: fmtEst({ value: v.value, low: v.ci_low, high: v.ci_high }).text + " pp",
        heldout: fmtEst({ value: h.value, low: h.ci_low, high: h.ci_high }).text + " pp",
      },
    ];
  });
}

// ------------------------------------------------------------------------- Hillstrom

export const HS_LABEL: Record<string, string> = {
  logistic: "Logistic (interactions)",
  s_learner: "LightGBM S-learner",
  t_learner: "LightGBM T-learner",
  dr_learner: "DR-learner",
  no_email: "No e-mail for all",
  mens_for_all: "Men's e-mail for all",
  womens_for_all: "Women's e-mail for all",
  random: "Uniform random",
};
export const hsLabel = (k: string) => HS_LABEL[k] ?? k;
const HS_MODELS = ["logistic", "s_learner", "t_learner", "dr_learner"];

export interface CandidateRow {
  policy: string;
  label: string;
  kind: "model" | "baseline";
  chosen: boolean;
  eligible: boolean | null;
  bestByValue: boolean;
  value: number; // percent
  low: number;
  high: number;
  valueLabel: string;
  rangeLabel: string;
  ipw: string;
  vsMens: string | null;
  vsMensRange: string | null;
  vsMensIncludesZero: boolean | null;
  config: string | null;
  color: string;
}

export function candidateRows(d: HillstromModelsData, chosenPolicy: string): CandidateRow[] {
  const sel = d.selection;
  return d.candidates.map((c: HillstromCandidate) => {
    const kind = HS_MODELS.includes(c.policy) ? "model" : "baseline";
    const v = interval(c.aipw_value, c.ci_low, c.ci_high, (n) => hillstrom.rate(n, 2));
    const hasDiff = c.diff_vs_mens_for_all !== null && c.diff_vs_mens_for_all_ci_low !== null && c.diff_vs_mens_for_all_ci_high !== null;
    const dv = hasDiff
      ? interval(c.diff_vs_mens_for_all as number, c.diff_vs_mens_for_all_ci_low as number, c.diff_vs_mens_for_all_ci_high as number, (n) => hillstrom.rateDelta(n))
      : null;
    return {
      policy: c.policy,
      label: hsLabel(c.policy),
      kind,
      chosen: c.policy === chosenPolicy,
      eligible: kind === "model" ? sel.eligible_models.includes(c.policy) : null,
      bestByValue: c.policy === sel.best_by_value,
      value: hillstrom.toPercent(c.aipw_value),
      low: hillstrom.toPercent(c.ci_low),
      high: hillstrom.toPercent(c.ci_high),
      valueLabel: v.value,
      rangeLabel: v.range,
      ipw: hillstrom.rate(c.ipw_value, 2),
      vsMens: dv ? dv.value : null,
      vsMensRange: dv ? dv.range : null,
      vsMensIncludesZero: hasDiff
        ? (c.diff_vs_mens_for_all_ci_low as number) <= 0 && (c.diff_vs_mens_for_all_ci_high as number) >= 0
        : null,
      config: c.config && c.config !== "fixed baseline" ? c.config : null,
      color: HS_COLOR(c.policy, c.policy === chosenPolicy),
    };
  });
}

function HS_COLOR(policy: string, chosen: boolean): string {
  if (chosen) return "#117E74";
  return HS_MODELS.includes(policy) ? "#7467C9" : "#64748B";
}

export interface HsDiffRow {
  key: string;
  label: string;
  value: number; // pp
  low: number;
  high: number;
  valueLabel: string;
  rangeLabel: string;
  includesZero: boolean;
}

/** Candidate minus Men's-for-all, cross-validated, as stored. */
export function hillstromDiffRows(d: HillstromModelsData): HsDiffRow[] {
  return d.candidates.flatMap((c) => {
    if (c.diff_vs_mens_for_all === null || c.diff_vs_mens_for_all_ci_low === null || c.diff_vs_mens_for_all_ci_high === null) return [];
    const iv = interval(c.diff_vs_mens_for_all, c.diff_vs_mens_for_all_ci_low, c.diff_vs_mens_for_all_ci_high, (n) => hillstrom.rateDelta(n));
    return [
      {
        key: c.policy,
        label: hsLabel(c.policy),
        value: c.diff_vs_mens_for_all * 100,
        low: c.diff_vs_mens_for_all_ci_low * 100,
        high: c.diff_vs_mens_for_all_ci_high * 100,
        valueLabel: iv.value,
        rangeLabel: iv.range,
        includesZero: c.diff_vs_mens_for_all_ci_low <= 0 && c.diff_vs_mens_for_all_ci_high >= 0,
      },
    ];
  });
}

export function hillstromSelectionSummary(d: HillstromModelsData, chosenPolicy: string): string[] {
  const sel = d.selection;
  const best = d.candidates.find((c) => c.policy === sel.best_by_value);
  const bestDiff =
    best && best.diff_vs_mens_for_all !== null && best.diff_vs_mens_for_all_ci_low !== null && best.diff_vs_mens_for_all_ci_high !== null
      ? interval(best.diff_vs_mens_for_all, best.diff_vs_mens_for_all_ci_low, best.diff_vs_mens_for_all_ci_high, (n) => hillstrom.rateDelta(n))
      : null;
  const out = [
    `Selected policy: ${hsLabel(chosenPolicy)}. The rule keeps the simplest option within one paired standard error of the best, with fixed baselines allowed to win.`,
    `Best by cross-validated value: ${hsLabel(sel.best_by_value)}${bestDiff ? `, ${bestDiff.value} ${bestDiff.range} against ${hsLabel(sel.best_baseline)}` : ""}.`,
  ];
  if (bestDiff && best && (best.diff_vs_mens_for_all_ci_low as number) <= 0 && (best.diff_vs_mens_for_all_ci_high as number) >= 0) {
    out.push("That interval includes zero: the development data do not show that personalization beats sending the best fixed e-mail to everyone.");
  }
  out.push(
    `Eligible personalized models (not below the best fixed baseline): ${sel.eligible_models.map(hsLabel).join(", ") || "none"}. Ineligible: ${sel.ineligible_models.map(hsLabel).join(", ") || "none"}.`,
  );
  return out;
}
