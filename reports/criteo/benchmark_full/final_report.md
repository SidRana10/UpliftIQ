# Criteo C6 final: full-scale validation benchmark of three finalists

**Validation only.** The held-out test set was not accessed. Hyperparameters are the fixed
C3/C4 defaults (no tuning); the three finalists were trained on the entire training split
(8,946,939 rows). Values are **conditional estimates for the released Criteo sample** in percentage
points of visit rate per evaluated customer relative to treating nobody, estimated with AIPW
(IPW as a diagnostic). They are not population-level incrementality, guaranteed causal gains, or
advertising ROI, and because validation results drove this choice they are not evidence of final
generalization performance.

## Setup
- Finalists: logistic, s_learner, tarnet_cpu (isolated worker processes, `OMP_NUM_THREADS=1`).
- Training rows: 8,946,939 (whole training split). Validation subset: 500,000 rows (treatment-stratified, fixed, seed 42). Bootstrap: 200 Poisson replicates, shared weights (paired).
- AIPW nuisance (shared by every model): LightGBM T-learner (defaults) trained on 500,000 train rows, predicting the validation subset out of sample. Those rows were chosen to be disjoint from the 2M candidate subset, so they **overlap the full training split** (about 5.6% of it). Both models are fitted on train rows only and neither sees a validation outcome, so the out-of-sample property for scored rows holds; the earlier rungs' strict disjointness does not.
- Split fingerprint `c6bcbcbf543b4f73…`, data sha256 `2716e1bf0fd157a9…`.
- Code commit `54261d0774` (dirty: True); python 3.12.14, lightgbm 4.7.0, torch 2.15.0, scikit-learn 1.9.1.

## Policy value at matched contact budgets (AIPW)
| model      | AIPW value@10% (pp)   | vs random@10% (pp)   |   IPW value@10% (pp) | AIPW value@20% (pp)   | vs random@20% (pp)   |   IPW value@20% (pp) | AIPW value@30% (pp)   | vs random@30% (pp)   |   IPW value@30% (pp) |
|:-----------|:----------------------|:---------------------|---------------------:|:----------------------|:---------------------|---------------------:|:----------------------|:---------------------|---------------------:|
| logistic   | 0.562 [0.480, 0.651]  | 0.481 [0.403, 0.555] |                0.914 | 0.667 [0.572, 0.764]  | 0.505 [0.431, 0.581] |                1.003 | 0.700 [0.596, 0.812]  | 0.457 [0.386, 0.538] |                1.037 |
| s_learner  | 0.614 [0.539, 0.693]  | 0.533 [0.460, 0.605] |                0.963 | 0.748 [0.635, 0.850]  | 0.587 [0.497, 0.668] |                1.066 | 0.774 [0.658, 0.878]  | 0.532 [0.451, 0.602] |                1.088 |
| tarnet_cpu | 0.623 [0.540, 0.704]  | 0.542 [0.467, 0.617] |                0.965 | 0.748 [0.627, 0.861]  | 0.587 [0.493, 0.678] |                1.078 | 0.756 [0.641, 0.865]  | 0.514 [0.431, 0.590] |                1.074 |

References:

| policy       |   budget | AIPW value (pp)      | feasibility                                                  |
|:-------------|---------:|:---------------------|:-------------------------------------------------------------|
| treat_nobody |      0   | 0.000 [0.000, 0.000] | reference (budget 0)                                         |
| treat_all    |      1   | 0.808 [0.677, 0.923] | unconstrained reference (feasible only at budget 100%)       |
| random@10%   |      0.1 | 0.081 [0.068, 0.092] | feasible (budget-constrained, expectation over random draws) |
| random@20%   |      0.2 | 0.162 [0.135, 0.185] | feasible (budget-constrained, expectation over random draws) |
| random@30%   |      0.3 | 0.242 [0.203, 0.277] | feasible (budget-constrained, expectation over random draws) |

## Qini (supporting diagnostic)
| model      | Qini AIPW (pp)       |   Qini IPW (pp) |   Qini / treat-all effect |
|:-----------|:---------------------|----------------:|--------------------------:|
| logistic   | 0.279 [0.231, 0.327] |           0.456 |                     0.346 |
| s_learner  | 0.335 [0.287, 0.385] |           0.498 |                     0.415 |
| tarnet_cpu | 0.331 [0.283, 0.379] |           0.497 |                     0.409 |

## Paired model differences (AIPW)
Differences come from joint bootstrap runs with shared weights (each pair appears once).

| comparison                 | metric    | difference (pp)        | excludes 0   |
|:---------------------------|:----------|:-----------------------|:-------------|
| s_learner minus logistic   | value@10% | 0.051 [-0.003, 0.106]  | False        |
| s_learner minus logistic   | value@20% | 0.081 [0.033, 0.148]   | True         |
| s_learner minus logistic   | value@30% | 0.074 [0.025, 0.127]   | True         |
| s_learner minus logistic   | qini      | 0.056 [0.012, 0.096]   | True         |
| tarnet_cpu minus logistic  | value@10% | 0.061 [0.010, 0.121]   | True         |
| tarnet_cpu minus logistic  | value@20% | 0.081 [0.014, 0.154]   | True         |
| tarnet_cpu minus logistic  | value@30% | 0.057 [-0.000, 0.111]  | False        |
| tarnet_cpu minus logistic  | qini      | 0.051 [0.007, 0.100]   | True         |
| s_learner minus tarnet_cpu | value@10% | -0.009 [-0.074, 0.051] | False        |
| s_learner minus tarnet_cpu | value@20% | -0.000 [-0.047, 0.044] | False        |
| s_learner minus tarnet_cpu | value@30% | 0.018 [-0.020, 0.068]  | False        |
| s_learner minus tarnet_cpu | qini      | 0.004 [-0.029, 0.038]  | False        |

## Selection (predefined rule)
Primary metric: AIPW policy value at a 20% contact budget on the fixed validation subset (Qini is a supporting diagnostic only). 1) A model is eligible only if its paired difference from random targeting at 20% has a 95% interval above 0. 2) The best eligible model by point value is found. 3) One-SE simplicity rule: walking from the simplest model (logistic < S-learner < TARNet), the first eligible model whose value is within one paired standard error of the best wins. The paired SE is the bootstrap SE of the difference, taken as the 95% percentile-interval width / 3.92. Hyperparameters are never tuned.

Eligibility at the 20% budget:

| model      | AIPW value@20% (pp)   | vs random@20% (pp)   | beats random   |
|:-----------|:----------------------|:---------------------|:---------------|
| logistic   | 0.667 [0.572, 0.764]  | 0.505 [0.431, 0.581] | yes            |
| s_learner  | 0.748 [0.635, 0.850]  | 0.587 [0.497, 0.668] | yes            |
| tarnet_cpu | 0.748 [0.627, 0.861]  | 0.587 [0.493, 0.678] | yes            |

One-SE walk:

- tarnet_cpu minus logistic: 0.081 pp vs one paired SE 0.036 pp, outside one SE (continue)
- tarnet_cpu minus s_learner: 0.000 pp vs one paired SE 0.023 pp, within one SE (simpler model chosen)

**Selected finalist: `s_learner`.** Best eligible by point value: `tarnet_cpu`; best by Qini: `s_learner` (agrees with the selection).

Selection is a statement about this validation subset under fixed untuned hyperparameters. A model
that fails to separate from simpler ones is not evidence it is equivalent; it only means the data
cannot distinguish them at this size, which is why the simpler one wins ties.

## Change from the previous 2M benchmark (paired, same validation customers)
`*` marks a paired interval that excludes 0. Intervals reflect scoring noise on the validation
subset, not retraining variability.

| model      | metric    | 2,000,000 rows       | 8,946,939 rows       | change (pp, paired)    |
|:-----------|:----------|:---------------------|:---------------------|:-----------------------|
| logistic   | value@10% | 0.537 [0.447, 0.622] | 0.562 [0.480, 0.651] | 0.025 [-0.000, 0.054]  |
| logistic   | value@20% | 0.654 [0.556, 0.742] | 0.667 [0.572, 0.764] | 0.013 [-0.014, 0.039]  |
| logistic   | value@30% | 0.685 [0.577, 0.781] | 0.700 [0.596, 0.812] | 0.014 [-0.010, 0.044]  |
| logistic   | qini      | 0.266 [0.215, 0.318] | 0.279 [0.231, 0.327] | 0.013 [-0.004, 0.031]  |
| s_learner  | value@10% | 0.630 [0.550, 0.716] | 0.614 [0.539, 0.693] | -0.016 [-0.063, 0.033] |
| s_learner  | value@20% | 0.757 [0.640, 0.858] | 0.748 [0.635, 0.850] | -0.009 [-0.051, 0.030] |
| s_learner  | value@30% | 0.803 [0.687, 0.910] | 0.774 [0.658, 0.878] | -0.029 [-0.066, 0.015] |
| s_learner  | qini      | 0.359 [0.309, 0.402] | 0.335 [0.287, 0.385] | -0.024 [-0.048, 0.004] |
| tarnet_cpu | value@10% | 0.622 [0.546, 0.685] | 0.623 [0.540, 0.704] | 0.001 [-0.049, 0.051]  |
| tarnet_cpu | value@20% | 0.705 [0.602, 0.793] | 0.748 [0.627, 0.861] | 0.043 [-0.018, 0.109]  |
| tarnet_cpu | value@30% | 0.723 [0.623, 0.821] | 0.756 [0.641, 0.865] | 0.033 [-0.033, 0.087]  |
| tarnet_cpu | qini      | 0.294 [0.249, 0.342] | 0.331 [0.283, 0.379] | 0.037 [-0.010, 0.085]  |

## Engineering: training cost, latency, memory
Latency uses `predict_uplift` on validation rows (includes preprocessing). RSS peak in fit is
sampled every 50 ms; process peak includes data loading.

| model      |   n_fit |   train_s |   predict 500k rows (s) |   rows/s @100k | p50/p95 ms @1   | p50/p95 ms @1k   | p50/p95 ms @100k   |   RSS before fit (GB) |   RSS peak in fit (GB) |   process peak RSS (GB) |   epochs |
|:-----------|--------:|----------:|------------------------:|---------------:|:----------------|:-----------------|:-------------------|----------------------:|-----------------------:|------------------------:|---------:|
| logistic   | 8946939 |       9.9 |                    0.05 |     12,459,766 | 0.08 / 0.10     | 0.2 / 0.2        | 8 / 8              |                  0.71 |                   2.55 |                    2.55 |          |
| s_learner  | 8946939 |      48.3 |                    2.88 |        210,575 | 0.33 / 0.36     | 5.3 / 6.1        | 475 / 587          |                  1    |                   2.15 |                    2.15 |          |
| tarnet_cpu | 8946939 |     149   |                    0.14 |      4,454,244 | 0.04 / 0.04     | 0.2 / 0.2        | 22 / 23            |                  1    |                   1.77 |                    1.78 |       15 |

## Assumption check: IPW vs AIPW (identification caveat preserved)
| quantity | value |
|:--|--:|
| treated share (validation subset) | 0.8500 |
| treat-all effect, IPW, pp | 1.100 |
| treat-all effect, AIPW, pp | 0.808 |
| IPW minus AIPW, pp (SE) | 0.292 (0.046) |
| predicted baseline, treated vs control, pp | 4.120 vs 3.848 (z = 6.4) |

The discrepancy documented in `reports/criteo/benchmark/ipw_aipw_diagnostic.md` persists: the
constant 0.85 propensity does not fully hold in the released file, IPW is likely biased upward, and
AIPW is primary. AIPW is not guaranteed unbiased if the propensity assumption and the outcome
nuisance are both wrong, and one shared estimator makes models comparable without making their
relative rankings unbiased.

## Frozen configuration
`frozen_config.json` in this folder records the selected model, effective hyperparameters, seed,
dataset checksum, split fingerprint, validation/nuisance/bootstrap settings, the SHA-256 of the
selected model's validation scores and the library and code state. It was written before any test
evaluation, and `test_set_accessed` is `false`. The code state is marked dirty because these results
are intentionally uncommitted; commit them before the one-shot test evaluation so the freeze is
tied to a clean commit.
