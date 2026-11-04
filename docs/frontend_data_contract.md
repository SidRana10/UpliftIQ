# Backend data contract for the five frontend workspaces

Status: design. Nothing here requires a new statistical analysis: every field is served from an
already-generated, committed report or from the packaged Criteo model. FastAPI will expose these as
read-only JSON; this document fixes the shapes and sources first.

## Two datasets, two kinds of claim (must be visible in the UI)

| | Hillstrom (business-facing) | Criteo (anonymized benchmark) |
|:--|:--|:--|
| What it is | 64,000 customers, randomized 3-arm email test (Men's / Women's / none) | 13.98M rows, randomized treatment (85/15), features `f0`-`f11` anonymized |
| Outcomes | visit (primary), conversion and spend (exploratory), in business units | visit (primary), conversion (exploratory); no spend |
| Features | interpretable (recency, history, channel, ...) | not interpretable; never label them with business meaning |
| Frozen decision | **Men's E-Mail for all** (no personalised model beat the fixed policy; the simplest option won) | **LightGBM S-learner, top-20% budget** (primary metric: AIPW value at 20%) |
| Evidence | one-shot held-out evaluation, 12,800 customers | one-shot held-out evaluation, 2,795,918 customers |
| Allowed language | expected visit/conversion/spend rates, uplift for this experiment | conditional estimates for the released sample only; **no ROI, no population claims** |

Every Criteo payload carries `caveats` (below). Never render Criteo numbers in currency, and never
place Hillstrom and Criteo numbers on one axis or in one ranking.

Common `caveats` for Criteo (verbatim strings from the manifest / reports):
- conditional estimates for the released sample; not population ROI or guaranteed causal gains
- the constant 0.85 propensity is publisher-supported but unverifiable; IPW and AIPW disagree on this file (AIPW primary, IPW a cross-check)
- outcome-dependent privacy subsampling limits causal interpretation

## Envelope

Every endpoint returns
```json
{ "dataset": "hillstrom | criteo", "source": ["reports/..."], "provenance": "historical | new_execution",
  "frozen_config_sha256": "...", "caveats": ["..."], "data": { } }
```
`provenance` is `historical` for anything read from a committed report and `new_execution` only for
live model inference. Intervals are `{ "value", "ci_low", "ci_high" }`. Criteo gains are in
percentage points (pp) of visit rate vs treating nobody; Hillstrom rates are proportions, spend in dollars.

## Workspace contracts

### 1. Overview
Purpose: the headline of each study, side by side but never merged.
- `GET overview` -> `{ hillstrom: { frozen_policy, primary_metric, value: Interval, vs_best_baseline: Interval }, criteo: { model, primary_metric: "aipw_value_20pct_pp", value: Interval, vs_random: Interval, treat_all: Interval, n_test } }`
- Sources: Hillstrom `reports/final_evaluation/{report.md, policy_values.csv, comparisons.csv, evaluation_record.json}`; Criteo `reports/criteo/final/{final_results.json, heldout_value.csv, evaluation_record.json}`.
- Required: a "benchmark, not business ROI" label on the Criteo card.

### 2. Experiment Lab (what the randomized experiments show)
- `GET experiments/hillstrom/groups` <- `reports/hillstrom/tables/group_summary.csv` (segment, n, visit/conversion rate, spend stats)
- `GET experiments/hillstrom/effects` <- `reports/hillstrom/tables/effects.csv` and `reports/final_evaluation/observed_effects.csv` (comparison, outcome, control, treated, abs/rel effect with CI, Holm p; mark `significant_holm_0.05`)
- `GET experiments/hillstrom/balance` <- `balance.csv` (covariate balance of the randomization)
- `GET experiments/criteo/diagnostics` <- `reports/criteo/treatment_diagnostics.json`, `dataset_summary.json`, `split_summary.json` (treatment share, balance, split sizes + fingerprint)
- `GET experiments/criteo/ipw-aipw` <- `reports/criteo/benchmark/ipw_aipw_diagnostic.json` (the documented IPW vs AIPW discrepancy; show it, do not hide it)
- Figures are served as static files from the same report folders.

### 3. Uplift Explorer (how uplift differs across customers/models)
- Hillstrom: `GET uplift/hillstrom/models` <- `reports/model_selection/selection_summary.csv` (policy, config, AIPW value + CI, differences vs baselines). Personalised models were statistically indistinguishable from the fixed Men's-email policy.
- Criteo: `GET uplift/criteo/curve` <- `reports/criteo/final/final_results.json` (`curves` are not stored there; use `heldout_value.csv`, `heldout_qini.csv`, `heldout_references.csv` and figure `heldout_gain_curve.png`). Budgets 10/20/30%, Qini, references (treat-nobody = zero baseline, treat-all = unconstrained, random = budget-feasible).
- Criteo model comparison (validation): `reports/criteo/benchmark_full/final_results.json` (logistic, S-learner, TARNet CPU) and `reports/criteo/benchmark_2m/scaling_results.json` (learning curve). Label as validation, not held-out.
- Per-customer exploration (Criteo only): `POST predict` below, with `f0`-`f11` inputs.

### 4. Decision Studio (what to do, and with how much confidence)
- `GET decision/hillstrom` -> frozen policy (`reports/model_selection/frozen_config.json` `chosen_policy`), its held-out value vs each baseline (`comparisons.csv`, with `primary` flag), outcome roles (visit primary; conversion/spend exploratory).
- `GET decision/criteo` -> budget table (`heldout_value.csv`: budget, share treated, AIPW value, random, value minus random, IPW cross-check), references with `type` (zero baseline / budget-feasible / unconstrained reference).
- `POST predict` (Criteo): body is one record or a list of records with exactly `f0`...`f11` (floats). Response rows: `p_visit_control`, `p_visit_treatment`, `uplift`, `recommend_treatment`, plus `model_version`. Validation errors return HTTP 422 with the message from `InputValidationError` (missing/unexpected/non-finite/non-numeric fields, empty or oversized batch, max 200,000 records). Backed by `CriteoUpliftModel` (`upliftiq.modeling.criteo_package`); needs no raw dataset.
- The UI states that `recommend_treatment` is `uplift > 0`, a per-customer rule; the evaluated policy is "treat the top-k% by uplift", which needs a batch ranking.

### 5. Model Operations
- `GET models/criteo` <- `reports/criteo/model_package.json` (= `models/criteo_s_learner/1.0.0/manifest.json`): version, input schema, hyperparameters, frozen config sha, dataset sha, split fingerprint, artifact sha256, dependencies, reproducibility (`matches_frozen_validation_fingerprint`), limitations.
- `GET models/criteo/runs` <- MLflow (local SQLite, `mlruns/mlflow.db`, experiment `UpliftIQ-Criteo`). Run tags separate **`provenance=historical`** (imported from committed reports; `executed_by_this_run=false`) from **`provenance=new_execution`** (packaging and verification runs). Metric names: `validation_*` and `heldout_*` (pp), e.g. `heldout_aipw_value_20pct_pp`.
- `GET models/criteo/evaluation-record` <- `reports/criteo/final/evaluation_record.json` (one-shot status, `code_dirty`, hashes). Display `code_dirty: true` as recorded.
- `GET models/hillstrom` <- `reports/model_selection/frozen_config.json` + `reports/final_evaluation/evaluation_record.json` (no served model: the frozen policy is a fixed rule).
- Health: package loads and checksum matches (`CriteoUpliftModel.load`).

## Out of scope for this contract
New analyses, retraining, re-running either held-out evaluation, auth, multi-user state, cloud
deployment, and ROI/currency modelling for Criteo.
