/**
 * TEST-ONLY fixtures, trimmed from the structure (and rounded values) of the committed reports.
 * Production code never imports this file; the app reads everything from the API.
 */
import type { CriteoHeldoutData, HillstromDecisionData, OverviewData } from "@/lib/api/types";

export const overviewHillstrom: OverviewData["hillstrom"] = {
  frozen_policy: "mens_for_all",
  primary_metric: "AIPW expected visit rate (proportion)",
  value: { value: 0.1819, ci_low: 0.171, ci_high: 0.1933 },
  vs_strongest_alternative_baseline: {
    baseline: "womens_for_all",
    diff: 0.0294,
    ci_low: 0.0144,
    ci_high: 0.0444,
    p_value_holm: 0.00016,
  },
  heldout_customers: 12800,
  caveats: ["Visit is the primary outcome."],
};

const pv = (outcome: string, policy: string, value: number, lo: number, hi: number) => ({
  outcome,
  estimator: "AIPW",
  policy,
  value,
  se: 0.01,
  ci_low: lo,
  ci_high: hi,
});

const cmp = (outcome: string, diff: number, lo: number, hi: number, holm: number, primary = true) => ({
  outcome,
  estimator: "AIPW",
  baseline: "womens_for_all",
  diff,
  ci_low: lo,
  ci_high: hi,
  paired_se: 0.01,
  p_value: holm,
  primary,
  p_value_holm: holm,
});

export const hillstromDecision = {
  chosen_policy: { kind: "baseline", name: "mens_for_all" },
  policy_values: [
    pv("visit", "frozen_policy", 0.1819, 0.171, 0.1933),
    pv("visit", "mens_for_all", 0.1819, 0.171, 0.1933),
    pv("visit", "womens_for_all", 0.1526, 0.1426, 0.1631),
    pv("visit", "no_email", 0.1055, 0.0965, 0.1149),
    pv("visit", "random", 0.1467, 0.1409, 0.153),
    { ...pv("visit", "frozen_policy", 0.2, 0.19, 0.21), estimator: "IPW" },
    pv("conversion", "frozen_policy", 0.0091, 0.0064, 0.012),
    pv("spend", "frozen_policy", 1.1556, 0.6761, 1.6805),
  ],
  comparisons: [
    cmp("visit", 0.0294, 0.0144, 0.0444, 0.00016),
    cmp("visit", 0.0764, 0.0615, 0.0907, 0.00001, false),
    cmp("conversion", 0.0004, -0.0037, 0.0043, 0.8334),
    cmp("spend", -0.1091, -0.8344, 0.6067, 0.7633),
  ],
  evaluation_record: {} as HillstromDecisionData["evaluation_record"],
  outcome_roles: { visit: "primary", conversion: "exploratory", spend: "exploratory" },
} as HillstromDecisionData;

const budgetRow = (b: number, value: number, lo: number, hi: number, vsRandom: number, vlo: number, vhi: number) => ({
  model: "s_learner",
  estimator: "AIPW",
  policy: "budget",
  budget_requested: b,
  budget: b,
  n_treated: 1,
  value,
  se: 0.0002,
  ci_low: lo,
  ci_high: hi,
  random_value: value - vsRandom,
  diff_vs_random: vsRandom,
  diff_vs_random_ci_low: vlo,
  diff_vs_random_ci_high: vhi,
  treat_all_value: 0.0074,
  feasibility: "feasible (budget-constrained)",
});

const ref = (policy: string, value: number, lo: number, hi: number, estimator = "AIPW") => ({
  policy,
  estimator,
  budget: 0,
  value,
  ci_low: lo,
  ci_high: hi,
  feasibility: "",
});

export const criteoHeldout = {
  model: "s_learner",
  n: 2795918,
  primary_metric: "AIPW policy value at 20% contact budget",
  assumption_check: {
    ate_ipw: 0.010197,
    ate_aipw: 0.007442,
    ate_ipw_minus_aipw: 0.002755,
    ate_ipw_minus_aipw_se: 0.000192,
  },
  evaluation_record: {} as CriteoHeldoutData["evaluation_record"],
  numeric: {
    policies: [
      budgetRow(0.1, 0.005819, 0.005404, 0.006174, 0.005074, 0.0047, 0.005411),
      budgetRow(0.2, 0.006741, 0.00616, 0.00715, 0.005253, 0.0048, 0.00558),
      budgetRow(0.3, 0.00711, 0.00656, 0.00756, 0.004879, 0.00448, 0.00523),
      { ...budgetRow(0.2, 0.0097, 0.009, 0.0105, 0.0076, 0.007, 0.008), estimator: "IPW" },
      { ...budgetRow(0, 0.0075, 0.007, 0.008, 0.0001, 0, 0.0002), policy: "uplift_gt_0", budget_requested: null },
    ],
    references: [
      ref("treat_nobody", 0, 0, 0),
      ref("treat_all", 0.007442, 0.006867, 0.007931),
      ref("random@10%", 0.000744, 0.000687, 0.000793),
      ref("random@20%", 0.001488, 0.001373, 0.001586),
      ref("random@30%", 0.002233, 0.00206, 0.002379),
      ref("treat_all", 0.0102, 0.0095, 0.0107, "IPW"),
    ],
    qini: [],
  },
} as unknown as CriteoHeldoutData;
