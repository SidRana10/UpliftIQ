# Criteo C7 development-only rehearsal (NOT test results)

This dry run uses the **validation split** (2,236,735 customers) as a stand-in holdout to exercise the exact final pipeline. The held-out test set was not accessed. Numbers below are validation numbers and carry no weight as final results.

**Read first: identification and scope limits.**
- Values are **conditional estimates for the released Criteo sample**, not population-level
  incrementality, advertising ROI, or guaranteed causal gains. Criteo's outcome-dependent privacy
  subsampling limits causal interpretation.
- The constant 0.85 treatment propensity is publisher-supported but cannot be verified from the
  file. IPW (which uses it directly) and AIPW disagree on this file (documented in
  `reports/criteo/benchmark/ipw_aipw_diagnostic.md`): IPW is likely biased upward, so **AIPW is
  primary** and IPW is shown only as a cross-check. AIPW is not guaranteed unbiased if the
  propensity assumption and the outcome nuisance are both wrong.
- The model was selected on validation data; the test set was used once, for this report only.

Units: percentage points (pp) of visit rate per evaluated customer, relative to treating nobody.
Budgets are shares of customers contacted, not monetary cost.

## Setup and verification
- Frozen model: `s_learner` (LightGBM, effective hyperparameters in `frozen_config.json`), seed 42, trained on the full training split (8,946,939 rows).
- Dataset sha256 `2716e1bf0fd157a9…`; split fingerprint `c6bcbcbf543b4f73…`.
- AIPW nuisance: LightGBM T-learner, same configuration as validation, fitted on 500,000 **training rows only** (fingerprint `27035f46f49e7f3a`); no validation or test rows. The plan's wording (train+validation nuisance for test) was replaced by this training-only rule at the owner's direction.
- Bootstrap: 200 Poisson replicates, shared weights (paired); seed 42.
- frozen config, dataset checksum, split fingerprint, hyperparameters, protocol: verified
- refit validation-score SHA-256 equals frozen hash: True
- refit AIPW nuisance equals the validated nuisance (SHA-256): True
- pipeline reproduces frozen validation AIPW value@20% (500K subset): 0.748210 pp (frozen 0.748210 pp)
- test rows opened: no (workers scored the validation split only)

## Primary result: AIPW policy value by contact budget (20% is primary)
`value minus random` is the paired difference from random targeting at the same budget.

| budget   | customers treated (share)   | AIPW value (pp)      |   random at same budget (pp) | value minus random (pp)   | excludes 0   |   IPW value (pp, cross-check) | primary   |
|:---------|:----------------------------|:---------------------|-----------------------------:|:--------------------------|:-------------|------------------------------:|:----------|
| 10%      | 10.0%                       | 0.578 [0.540, 0.620] |                        0.075 | 0.503 [0.469, 0.540]      | True         |                         0.904 |           |
| 20%      | 20.0%                       | 0.680 [0.625, 0.733] |                        0.149 | 0.531 [0.487, 0.575]      | True         |                         0.985 | yes       |
| 30%      | 30.0%                       | 0.707 [0.650, 0.767] |                        0.224 | 0.484 [0.443, 0.524]      | True         |                         0.999 |           |

## Reference policies
Budget-feasible policies respect the contact budget; treat-all is an unconstrained reference
(feasible only at a 100% budget) and treat-nobody is the zero baseline.

| policy       | budget   | AIPW value (pp)      | type                    |
|:-------------|:---------|:---------------------|:------------------------|
| treat_nobody | 0%       | 0.000 [0.000, 0.000] | zero baseline           |
| treat_all    | 100%     | 0.745 [0.688, 0.809] | unconstrained reference |
| random@10%   | 10%      | 0.075 [0.069, 0.081] | budget-feasible         |
| random@20%   | 20%      | 0.149 [0.138, 0.162] | budget-feasible         |
| random@30%   | 30%      | 0.224 [0.207, 0.243] | budget-feasible         |

Treat-all effect (AIPW): 0.745 pp [0.688, 0.809].

## Qini / AUUC
Area between the cumulative-gain curve and the random line (100-point trapezoid).

| estimator         | Qini (pp)            |   Qini / treat-all effect |
|:------------------|:---------------------|--------------------------:|
| AIPW (primary)    | 0.309 [0.288, 0.333] |                     0.415 |
| IPW (cross-check) | 0.461                |                     0.451 |

## IPW vs AIPW (known limitation, preserved)
| quantity | value |
|:--|--:|
| treated share | 0.8500 |
| treat-all effect, IPW (treated minus control mean), pp | 1.021 |
| treat-all effect, AIPW, pp | 0.745 |
| IPW minus AIPW, pp (SE) | 0.276 (0.022) |
| predicted baseline, treated vs control, pp | 4.091 vs 3.834 (z = 12.9) |

The IPW column in the table above is higher than AIPW at every budget for the same reason as on
validation: the arms differ in predicted baseline, which the constant-propensity estimator ignores.

## Exploratory: conversion uplift (not used for any decision)
Conversion is very rare, so intervals are wide and the conversion nuisance (trained on the same
500,000 rows) is weak. Rows show the **frozen visit policy**
(unchanged) scored on conversion, and, for description only, a conversion-trained S-learner of the
same family and fixed hyperparameters. Neither affects the selected model or targeting policy.
Conversion IPW vs AIPW treat-all: 0.1250 vs 0.1043 pp.

| budget                          | frozen visit policy: conversion value (pp)   | conversion-model ranking (pp)   |   random (pp) |
|:--------------------------------|:---------------------------------------------|:--------------------------------|--------------:|
| 10%                             | 0.0855 [0.0745, 0.0982]                      | 0.0912 [0.0799, 0.1066]         |        0.0104 |
| 20%                             | 0.0932 [0.0812, 0.1074]                      | 0.1002 [0.0882, 0.1151]         |        0.0209 |
| 30%                             | 0.0953 [0.0826, 0.1097]                      | 0.1018 [0.0898, 0.1167]         |        0.0313 |
| 100% (treat all, unconstrained) | 0.1043 [0.0909, 0.1197]                      |                                 |               |

Figures: `heldout_gain_curve.png`, `heldout_value_vs_random.png`.

## Caveats
- One seed, one split, one fixed configuration; intervals reflect scoring noise on this sample, not refit variability.
- Released-sample estimates only: no claim about advertising ROI, cost, or any customer population outside this file.
