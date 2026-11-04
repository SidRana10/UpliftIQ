/**
 * Hand-written types for the FastAPI payloads (the OpenAPI schema types `Envelope.data` as `any`,
 * so generation would add nothing). Fractions are as stored: Criteo values are fractions of visit
 * rate (formatted to percentage points only in lib/format.ts).
 */

export interface Envelope<T> {
  dataset: "hillstrom" | "criteo" | "platform";
  source: string[];
  provenance: "historical" | "new_execution";
  frozen_config_sha256: string | null;
  caveats: string[];
  data: T;
}

export interface Health {
  status: "ok" | "degraded";
  api_version: string;
  criteo_model_loaded: boolean;
  model_version: string | null;
  model_error: string | null;
  reports_available: Record<string, boolean>;
}

export interface Interval {
  value: number;
  ci_low: number;
  ci_high: number;
}

// ------------------------------------------------------------------ /overview

export interface OverviewData {
  hillstrom: {
    frozen_policy: string;
    primary_metric: string;
    value: { value: number; ci_low: number; ci_high: number };
    vs_strongest_alternative_baseline: {
      baseline: string;
      diff: number;
      ci_low: number;
      ci_high: number;
      p_value_holm: number;
    };
    heldout_customers: number;
    caveats: string[];
  };
  criteo: {
    model: string;
    primary_metric: string;
    heldout_customers: number;
    evaluation_record_status: string;
    code_dirty: boolean;
    caveats: string[];
  };
}

// ------------------------------------------------------------------ Hillstrom decision

export type HillstromOutcome = "visit" | "conversion" | "spend";

export interface HillstromPolicyValue {
  outcome: HillstromOutcome;
  estimator: "AIPW" | "IPW";
  policy: string;
  value: number;
  se: number;
  ci_low: number;
  ci_high: number;
}

export interface HillstromComparison {
  outcome: HillstromOutcome;
  estimator: "AIPW" | "IPW";
  baseline: string;
  diff: number;
  ci_low: number;
  ci_high: number;
  paired_se: number;
  p_value: number;
  primary: boolean;
  p_value_holm: number;
}

export interface HillstromDecisionData {
  chosen_policy: { kind: string; name: string };
  policy_values: HillstromPolicyValue[];
  comparisons: HillstromComparison[];
  evaluation_record: {
    frozen_config_sha256: string;
    primary_outcome: string;
    primary_comparator: string;
    data_sha256: string;
    split_fingerprint: string;
    n_dev: number;
    n_test: number;
    seed: number;
    n_boot: number;
    code_commit: string;
    code_dirty: boolean;
    nuisance_models_fitted_on: string;
  };
  outcome_roles: Record<HillstromOutcome, "primary" | "exploratory">;
}

// ------------------------------------------------------------------ Criteo held-out

export interface CriteoPolicyRow {
  model: string;
  estimator: "AIPW" | "IPW";
  policy: string;
  budget_requested: number | null;
  budget: number;
  n_treated: number;
  value: number;
  se: number;
  ci_low: number;
  ci_high: number;
  random_value: number;
  diff_vs_random: number;
  diff_vs_random_ci_low: number;
  diff_vs_random_ci_high: number;
  treat_all_value: number;
  feasibility: string;
}

export interface CriteoReferenceRow {
  policy: string;
  estimator: "AIPW" | "IPW";
  budget: number;
  value: number;
  ci_low: number;
  ci_high: number;
  feasibility: string;
}

export interface CriteoQiniRow {
  model: string;
  estimator: "AIPW" | "IPW";
  qini: number;
  ci_low: number | null;
  ci_high: number | null;
  ate_treat_all: number;
}

export interface CriteoHeldoutData {
  model: string;
  n: number;
  primary_metric: string;
  assumption_check: {
    ate_ipw: number;
    ate_aipw: number;
    ate_ipw_minus_aipw: number;
    ate_ipw_minus_aipw_se: number;
  };
  evaluation_record: {
    status: string;
    one_shot: boolean;
    frozen_config_sha256: string;
    selected_model: string;
    code_commit: string;
    code_dirty: boolean;
    test_rows: number;
    dataset_sha256: string;
    split_fingerprint: string;
    nuisance_fitted_on: string;
    finished_at: string;
    seed: number;
    n_boot: number;
  };
  numeric: {
    policies: CriteoPolicyRow[];
    references: CriteoReferenceRow[];
    qini: CriteoQiniRow[];
  };
}

// ------------------------------------------------------------------ Hillstrom experiment & models

export interface HillstromGroupRow {
  segment: string;
  n: number;
  share: number;
  visit_rate: number;
  conversion_rate: number;
  mean_spend_per_customer: number;
  median_spend_per_customer: number;
  share_zero_spend: number;
  n_converters: number;
  mean_spend_per_converter: number;
  max_spend: number;
}

export interface HillstromEffectRow {
  comparison: string;
  outcome: HillstromOutcome;
  n_treated: number;
  n_control: number;
  control: number;
  treated: number;
  abs_effect: number;
  abs_ci_low: number;
  abs_ci_high: number;
  rel_effect: number;
  rel_ci_low: number;
  rel_ci_high: number;
  p_value: number;
  welch_p_value: number | null;
  method: string;
  p_value_holm: number;
  "significant_holm_0.05": boolean;
}

export interface HillstromEffectsData {
  full_sample: HillstromEffectRow[];
  heldout: HillstromEffectRow[];
}

export interface HillstromBalanceRow {
  covariate: string;
  test: string;
  p_value: number;
  max_abs_smd_mens_vs_control: number;
  max_abs_smd_womens_vs_control: number;
  "balanced_smd_lt_0.1": boolean;
}

export interface HillstromCandidate {
  policy: string;
  config: string | null;
  aipw_value: number;
  ci_low: number;
  ci_high: number;
  ipw_value: number;
  diff_vs_no_email: number | null;
  diff_vs_no_email_ci_low: number | null;
  diff_vs_no_email_ci_high: number | null;
  diff_vs_mens_for_all: number | null;
  diff_vs_mens_for_all_ci_low: number | null;
  diff_vs_mens_for_all_ci_high: number | null;
  diff_vs_womens_for_all: number | null;
  diff_vs_womens_for_all_ci_low: number | null;
  diff_vs_womens_for_all_ci_high: number | null;
  diff_vs_random: number | null;
  diff_vs_random_ci_low: number | null;
  diff_vs_random_ci_high: number | null;
}

export interface HillstromModelsData {
  candidates: HillstromCandidate[];
  selection: {
    best_by_value: string;
    best_baseline: string;
    eligible_models: string[];
    ineligible_models: string[];
  };
}

// ------------------------------------------------------------------ Criteo diagnostics

export type CheckGrade = "pass" | "warn" | "ordering_warning" | "fail";

export interface CriteoCheck {
  value: number | Record<string, number>;
  grade: CheckGrade;
  rule: string;
}

export interface CriteoDiagnosticsData {
  dataset: {
    n_rows: number;
    arms: Record<
      "treated" | "control",
      { n: number; share: number; visit_rate: number; conversion_rate: number; exposed_share: number }
    >;
    overall: { visit_rate: number; conversion_rate: number };
  };
  treatment: {
    n_train_rows: number;
    overall_treated_share: number;
    smd: { feature: string; mean_treated: number; mean_control: number; smd: number }[];
    ks: { feature: string; ks: number }[];
    classifier: {
      n_fit: number;
      n_eval: number;
      base_rate_eval: number;
      lightgbm: { auc: number; auc_se: number; grade: CheckGrade };
      logistic: { auc: number; auc_se: number; grade: CheckGrade };
      permuted_label_auc: number;
      predicted_propensity: {
        min: number;
        q01: number;
        median: number;
        q99: number;
        max: number;
        "share_outside_0.05_0.95": number;
      };
    };
    checks: Record<string, CriteoCheck>;
    verdict: { status: string; statement: string };
    thresholds: Record<string, [number, number]>;
    split_fingerprint: string;
  };
  split: {
    rows: number;
    seed: number;
    sizes: Record<"train" | "validation" | "test", number>;
    treated_share: Record<"train" | "validation" | "test", number>;
    split_fingerprint: string;
  };
}

export interface PairedInterval {
  quantity: string;
  estimate: number;
  se: number;
  ci_low: number;
  ci_high: number;
}

export interface CriteoIpwAipwData {
  meta: { n: number; e: number; nuisance_n_train: number };
  estimates: { ipw: number; aipw: number; gcomp: number; ipw_minus_aipw: number };
  paired_intervals: PairedInterval[];
  arm_rates: {
    observed_rate_treated: number;
    observed_rate_control: number;
    predicted_mu0_among_treated: number;
    predicted_mu0_among_control: number;
    baseline_gap_treated_minus_control: number;
    baseline_gap_se: number;
  };
  permutation_test: { observed_gap: number; null_sd: number; z: number; p_two_sided: number; n_perm: number };
  poststratified_on_predicted_baseline: { estimate: number; se: number };
  strata: {
    stratum: number;
    n: number;
    mean_predicted_baseline: number;
    treated_share: number;
    rate_treated: number;
    rate_control: number;
    effect: number;
  }[];
}

// ------------------------------------------------------------------ Criteo validation benchmark

export type CriteoModelKey = "logistic" | "s_learner" | "t_learner" | "dr_learner" | "tarnet_cpu" | "tarnet_mps";

/** [value, ci_low, ci_high] as stored in the scaling report. */
export type Triple = [number, number, number];

export interface CriteoLatency {
  batch: number;
  reps: number;
  p50_ms: number;
  p95_ms: number;
  rows_per_s: number;
}

export interface CriteoModelManifest {
  family: string;
  config_key: string;
  n_fit: number;
  fit_seconds: number;
  predict_val_seconds: number;
  latency: CriteoLatency[];
  rss_gb_before_fit: number;
  rss_gb_peak_during_fit: number;
  peak_rss_gb: number;
  omp_num_threads: string;
  hyperparameters: Record<string, unknown>;
  training?: {
    device: string;
    epochs_run: number;
    best_epoch: number;
    fit_seconds: number;
    best_val_loss: number;
  };
  libraries: Record<string, string>;
  code: { commit: string; dirty: boolean };
  uplift_mean: number;
  uplift_std: number;
}

export interface CriteoBenchmarkData {
  split: string;
  n_train: number;
  /** the report's own display tables (strings), kept verbatim */
  tables: { references: Record<string, string>[] } & Record<string, Record<string, string | number | boolean>[]>;
  selection: {
    order: string[];
    eligible: string[];
    steps: {
      simpler: string;
      best: string;
      diff: number;
      ci_low: number;
      ci_high: number;
      se: number;
      within_one_se: boolean;
    }[];
    chosen: string;
    best_by_value: string;
    best_by_qini: string;
    qini_supports_choice: boolean;
  };
  change_from_2m: { model: string; metric: string; base: Triple; full: Triple; change: Triple }[];
  learning_curve: {
    levels: {
      model: string;
      size: number;
      "value@10%": Triple;
      "value@20%": Triple;
      "value@30%": Triple;
      qini: Triple;
    }[];
    changes: { model: string; step: string; metric: string; diff: number; ci_low: number; ci_high: number }[];
    engineering: { model: string; size: number; train_s: number; rss_peak_fit_gb: number; epochs: number | null }[];
  };
  numeric: {
    policies: (CriteoPolicyRow & {
      diff_vs_treat_all: number;
      diff_vs_treat_all_ci_low: number;
      diff_vs_treat_all_ci_high: number;
    })[];
    qini: (CriteoQiniRow & { note?: string })[];
    pairwise: Record<
      string,
      {
        model: string;
        reference_model: string;
        estimator: "AIPW" | "IPW";
        metric: string;
        diff: number;
        ci_low: number;
        ci_high: number;
      }[]
    >;
    models: Record<string, CriteoModelManifest>;
  };
}

/** Exploratory conversion table, kept as the report's own display strings. */
export interface CriteoHeldoutExtras {
  exploratory_conversion: { table: Record<string, string>[]; note: string };
  evaluation_record: CriteoHeldoutData["evaluation_record"];
  assumption_check: CriteoHeldoutData["assumption_check"];
}
