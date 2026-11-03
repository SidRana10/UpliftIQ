# Criteo Large-Scale Causal ML and Deep Learning Plan

Status: design only. Builds on `docs/causal_modeling_plan.md` (Hillstrom protocol, completed and frozen) and
`docs/datasets.md`. Hillstrom code paths, results and reports are not modified; shared code is only
generalized in backward-compatible ways with Hillstrom regression tests.

Data: Criteo Uplift v2.1 (corrected), 13,979,592 rows, `f0`-`f11` (anonymized dense floats), binary
`treatment` (85.00% treated), `visit` (4.70%), `conversion` (0.29%), and `exposure` (post-treatment).
Machine: Apple M4, 10 cores, 16 GB RAM, so everything must fit comfortably in memory without a cluster.

## 1. Decisions at a glance

| Topic | Decision |
|---|---|
| Estimand | Effect of *assignment* (`treatment`) on visit, within the released, non-uniformly subsampled file. Intention-to-treat. No population-level ROI claims. |
| Splits | One stratified random split: **train 64% / validation 16% / test 20%** (stratified on treatment x visit x conversion, seed 42). Single holdout validation (CV is too expensive at this scale and unnecessary with 2.2M validation rows). Test touched once. |
| Features | `f0`-`f11` only. `treatment` is the arm indicator; `exposure`, `visit`, `conversion` are never inputs. |
| Propensity | **Not assumed.** Observed 0.85/0.15 is *tested* (section 3). If checks pass, a constant `e = (0.15, 0.85)` estimated on train is used; if not, a cross-fitted propensity model and clipped weights. |
| Models | Logistic (interactions), LightGBM S-learner, LightGBM T-learner, DR-learner (challenger), TARNet (primary neural). DragonNet is **not** built (section 5). |
| Primary metric | Qini coefficient (area under the uplift curve) from AIPW scores on validation; policy value at budgets and for the `uplift > 0` rule as co-primary views. |
| Baselines | Treat none, treat all, random targeting at the same budget. |
| Winner | One-SE rule on validation Qini, simplest model wins, must beat random targeting; config frozen, then one-shot test evaluation. |
| Development | Representative subset ladder (100k, 500k, 2M, full train) before any full-scale run; a hard compute budget. |

## 2. Splits, preparation, leakage prevention

- **Preparation (once):** stream the gzip CSV in chunks into `data/processed/criteo/` as float32 features
  (`X.npy`, about 670 MB), `uint8` columns for treatment, exposure, visit and conversion. Memory-map for use. Git-ignored;
  no new dependency (NumPy only, no Parquet/Arrow).
- **Split:** one `int8` array aligned with rows (0 train, 1 validation, 2 test), saved with its SHA-256
  fingerprint, same pattern as `modeling/split.py`. Stratified so the 0.29% conversions are balanced. There is no
  user or advertiser id; each row is treated as an independent user (documented; duplicated feature
  vectors are expected because many features are heavily concentrated on single values and are not a leak by themselves).
- **Subsets for development:** stratified, seeded, nested subsets of **train** only (100k, 500k, 2M), saved as index
  arrays. The validation and test sets are never subsampled for the final numbers.
- **Preprocessing fit on train rows only:** robust clipping at train 0.1 and 99.9 percentiles, then standardization,
  for logistic and neural models; none for trees. Same fold-safe pattern as `features.py`.
- **Leakage guards:** a whitelist function (extended from `data/criteo.py:assert_no_leakage`) used at the
  model entry; tests assert that scrambling `exposure`, `visit`, `conversion` leaves the feature matrix
  unchanged and that test rows are never loaded during development (poison test as in Stage 4).
  Early stopping uses a 10% inner holdout of **train**, never validation or test.
- **Why `exposure` is excluded:** it is a post-treatment variable (only treated users can be exposed; exposed treated
  users visit at 41% vs 3.5%). Using it would leak the outcome. It is analyzed only descriptively (non-compliance).

## 3. Propensity, subsampling and the assumptions behind policy evaluation

Hillstrom's `e = 1/3` was fixed by design. Criteo's is not: the file merges several incrementality tests and was
sub-sampled non-uniformly, with no advertiser identifier. The evaluators therefore need assumptions that we
**state and test rather than inherit**:

| Assumption | Why needed | Validation (on train+validation, before any modeling) |
|---|---|---|
| A1: Within the file, treatment is independent of `f0`-`f11` (unconfoundedness given X, constant propensity) | IPW/AIPW with a constant `e` | (a) standardized mean differences treated vs control on all features, target < 0.05; (b) LightGBM treatment-from-features classifier on train, **held-out AUC** ideally within 0.50-0.51; (c) treatment rate by feature decile within a band around 0.85; (d) stability of the rate across row-order blocks |
| A2: Positivity (0 < e < 1 everywhere) | weights bounded | min/max treated rate across feature bins; max weight is about 6.7 (1/0.15) under constant e |
| A3: Subsampling is ignorable for *relative* comparisons of targeting rules | ranking and policy comparisons mean something | cannot be proven; document that subsampling probability may depend on treatment/outcome, so absolute base rates and lift are not true campaign incrementality; only **comparative** claims (model vs baseline on this file) are made |

Decision gate (end of Stage C1): if A1 holds, use the constant propensity (estimated on train with a CI; the
estimate is essentially exact at this n). If (b) or (c) fail materially, switch **all** policy estimates to a
cross-fitted propensity model with clipping (e in [0.05, 0.95], clipping rate reported), label them
"propensity-adjusted", and add DragonNet to the plan. In either case we claim **sample-level policy value
within this released dataset** only: no population-level advertising ROI, no revenue or cost conclusions (outcomes are
visits and conversions, and contact budgets are not dollars).

## 4. Reuse of the Hillstrom infrastructure

| Hillstrom module | Criteo use |
|---|---|
| `evaluation/policy.py` (IPW/AIPW, out-of-sample `OutcomeModelPredictions`, paired bootstrap, baselines) | Generalize `N_ARMS` from a constant to a parameter (default 3, **no behavior change for Hillstrom**); support binary treatment with per-arm propensity `(0.15, 0.85)`. Existing tests must still pass unchanged. |
| `modeling/learners.py` (Logistic, S, T, DR) | Make the number of arms a parameter (2 for Criteo); same interface `fit / predict_uplift / recommend`; LightGBM settings scaled up. |
| `modeling/select.py` + `select_report.py` | Reuse `one_se_choice`, frozen-config and report patterns; replace 5-fold CV with train/validation holdout. |
| `evaluation/final_evaluation.py` | Reuse the verify-frozen-config, data-checksum, split-fingerprint and one-shot guard. |
| `evaluation/stats.py`, bootstrap utilities | Reuse; a Poisson bootstrap is added for 2.8M-row curves. |
| New | `data/criteo_prepare.py` (npy conversion, split, subsets), Qini/budget curves, TARNet, run manifests, benchmarks. |

Binary treatment ("treat" vs "don't treat") makes the policy `pi(x) = 1{tau_hat(x) > c}`, with contact-budget policies
"treat the top fraction q by tau_hat". As agreed earlier, **contact-budget curves are not monetary cost analysis**:
no cost or margin is available, and none is assumed.

## 5. Models

All models predict `tau(x) = P(visit | x, treated) - P(visit | x, control)`.

1. **Logistic regression (interactions):** L2, `[x, T, T*x]`, solver saga/lbfgs on the train split; C tuned on a small grid. Fast, transparent baseline.
2. **LightGBM S-learner:** one classifier with `T` as a feature. Known weakness: treatment effects are small relative to
   main effects, so the model may underuse `T`; tracked via the variance of `tau_hat`.
3. **LightGBM T-learner:** separate treated (about 7.6M train rows) and control (about 1.3M) classifiers. The control model has
   much less data, which creates variance asymmetry; we report it.
4. **Advanced challenger: DR-learner** (cross-fitted, doubly robust). Why: it targets `tau` directly rather than subtracting
   two noisy predictions; it is robust to moderate outcome-model error; it extends our already-tested implementation;
   with validated near-constant propensity its pseudo-outcome is unbiased. The unbalanced allocation inflates the variance of the control term (weight 6.7), which the
   large n and a regularized regressor absorb. The X-learner is the documented alternative for imbalanced arms and is not built
   unless the DR-learner fails validation.
5. **TARNet (primary neural benchmark):** PyTorch MLP, shared trunk (e.g. 3 x 128, ReLU/BatchNorm-free, dropout 0.1) and two outcome
   heads (control, treated), BCE on the observed arm's head, AdamW, early stopping on an inner train holdout, mini-batches
   of 8-16k, fixed seeds, MPS if available else CPU. `tau_hat = sigmoid(head_1) - sigmoid(head_0)`. It is the
   cleanest neural CATE architecture and a fair deep counterpart to the T-learner with shared representation learning.
   **New dependency: `torch` only.** We do not assume it will beat LightGBM; tabular data with 12 features often
   favors boosted trees, and the benchmark is reported either way.
6. **DragonNet: excluded for now.** Its extras (a propensity head and targeted regularization) address
   confounding that is absent in a randomized trial with a (validated) constant propensity. It adds complexity and a tuning
   burden without a clear benefit. Revisit only if the A1 gate fails.

Tuning grids stay small (LightGBM: leaves {16, 64, 255} x min_child_samples {1000, 5000}; TARNet: width {64, 128} x lr {1e-3, 3e-4};
logistic C {0.01, 0.1, 1}), applied on the subset ladder first.

## 6. Outcomes and evaluation

- **Primary: visit uplift.** Selection uses visit only. **Secondary: conversion uplift** (about 40k events, very rare):
  refit of the **selected family** with fixed hyperparameters, descriptive, wide intervals, no selection.
- **AIPW scores** with the validated propensity and an outcome model that is **out of sample** for the scored rows (nuisance trained on train, scored
  on validation; on test, trained on train+validation only). IPW reported as a check.
- **Metrics on validation:**
  1. **Qini coefficient / AUUC** (primary): `G(q) = (1/N) * sum_i 1{rank_i <= qN} * (psi_i(1) - psi_i(0))`; coefficient = area between
     `G(q)` and the random line `q * G(1)`. Also gain at q = 10/20/30/50%.
  2. **Policy value** of `1{tau_hat > 0}` and budget policies vs baselines, as AIPW expected visit rate.
  3. Decile calibration (predicted vs observed uplift) as a diagnostic.
- **Baselines:** treat none, treat all, random targeting at the same budget (random line). The model must beat random targeting
  and is compared with treat-all.
- **Uncertainty:** Poisson bootstrap over customers (B = 200, fixed seed, shared weights across models, so differences are
  paired) on binned cumulative statistics (100 bins) to keep 2-3M row curves cheap; analytic SEs for policy values. With millions of rows,
  intervals are tight, so we also report **effect sizes and practical significance** (for example Qini relative to treat-all).
- **Selection without leakage:** all tuning and model choice use validation only; early stopping uses an inner train holdout;
  winner by the one-SE rule (order: logistic < S < T < DR < TARNet; ties to the simpler), must beat random targeting;
  config, seed, split fingerprint, data checksum and code commit are frozen in JSON **before** test scoring; test is scored
  **once** with the existing one-shot guard; the secondary outcome and engineering benchmarks never feed selection.

## 7. Large-scale engineering

- **Data loading:** npy memmap, float32/uint8; no pandas on the hot path. Memory: features 0.67 GB, labels 0.06 GB,
  LightGBM bin matrix about 0.2 GB, torch tensors about 0.7 GB; peak well under 6 GB of the 16 GB.
- **Training:** LightGBM with `num_threads=10`, histogram binning, `deterministic=True`, `force_row_wise`; bin the dataset once per split and cache it.
  Time estimates (to be **measured**, not assumed): logistic minutes, LightGBM several minutes per model at full train, DR
  about 3x the T-learner, TARNet on the order of tens of minutes on CPU/MPS. Full-scale runs are capped at about 3 hours total.
- **Inference:** report p50/p95 latency at batch sizes 1, 1,000 and 100,000, and throughput (rows/s) per model, on CPU, with model
  size on disk. No serving infrastructure yet (a thin backend is a later milestone).
- **Reproducibility:** each run writes a manifest (`reports/criteo/runs/<run_id>.json`: code commit, split fingerprint,
  data checksum, subset size, seed, hyperparameters, thread count, timings, metrics). Seeds fixed for NumPy, LightGBM, torch; torch
  results may differ slightly across devices (CPU vs MPS), so the device is recorded and the final benchmark uses one device.

## 8. Development strategy

| Rung | Train rows | Purpose |
|---|---|---|
| S0 | 100k | unit tests, debugging, shapes, determinism |
| S1 | 500k | implement and smoke-tune every model (minutes per run) |
| S2 | 2M | learning curve and pre-selection of 2-3 configs per family |
| S3 | full train (8.9M) | controlled full-scale benchmark of the finalists, validation = full 2.2M rows |

Output: a **learning curve** (validation Qini vs train size for each model) that shows where deep learning and boosting scale differently.
No full-scale runs until the S2 results and the A1 gate are in. Validation uses a 500k stratified subset during S0-S1 and the full set afterwards.

## 9. Implementation stages and acceptance criteria

| Stage | Work | Acceptance criteria |
|---|---|---|
| C0 Data prep | Install `torch` (verify import and MPS), generalize `policy.py` to `n_arms`, npy conversion, split array and subset indices | Row count 13,979,592 and column checks match the validated summary; split reproducible (same fingerprint), strata balanced within 0.01 pp; **all existing Hillstrom tests pass unchanged**; conversion takes < 10 min and < 6 GB RAM |
| C1 Assumption checks | Balance SMDs, treatment-from-features AUC, propensity by bins, block stability, exposure-leakage demo; written report and gate decision | Report committed; explicit pass/fail on A1-A3 with numbers; chosen propensity mode recorded in config |
| C2 Evaluator | Binary-treatment Qini/budget curves, AIPW/IPW, Poisson bootstrap | Unbiased and covered by CIs on synthetic data with 85/15 allocation, a known `tau(x)`, and an A/A case (Qini near 0); unchanged results for Hillstrom |
| C3 Traditional learners | Binary-arm logistic, S, T, DR at S0/S1 | Recover planted heterogeneity on synthetic data; deterministic; leakage tests (exposure/outcome scrambling changes nothing); fit time recorded |
| C4 TARNet | PyTorch implementation, inner-holdout early stopping, CPU/MPS | Recovers planted CATE on synthetic data; same seed same result on one device; throughput recorded |
| C5 Subset benchmark | S1/S2 grids and learning curves | Table of validation Qini +/- CI per model and size; shortlist; no test data accessed (poison test) |
| C6 Full benchmark and selection | S3 runs for finalists, engineering benchmarks (time, memory, latency), one-SE selection, frozen config | Selection report with paired CIs vs baselines; winner decided by the written rule (baselines or a non-neural model can win); runtime within the budget; frozen JSON committed |
| C7 Test and report | One-shot test evaluation, conversion secondary, model card and limitations | Config verified before scoring; one-shot guard; test tables, figures, report committed; claims limited to sample-level comparative uplift; DL vs GBM comparison reported honestly |

## 10. Tradeoffs and limitations

- **Holdout instead of CV:** far cheaper and sufficient at 2.2M validation rows; gives up fold-to-fold variance estimates (we report bootstrap variability instead).
- **Test size (2.8M):** intervals become tiny, so "significant" does not mean "useful"; we report practical effect sizes.
- **Constant-propensity assumption** is testable but not provable given the hidden test/advertiser structure and subsampling; the gate and the comparative-only claims limit the damage.
- **Deep learning:** adds a dependency, tuning burden and device nondeterminism; may match but not beat LightGBM. A null result is acceptable.
- **Visit as outcome:** a proxy, not revenue. Conversion is too rare for reliable heterogeneity estimates.
- **Out of scope:** DragonNet (unless gated in), X-learner, ensembling/stacking, hyperparameter frameworks, GPUs/clusters, serving infrastructure, frontend, any Hillstrom changes.
