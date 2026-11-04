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
