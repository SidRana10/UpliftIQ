# Criteo final held-out evaluation (one-shot)

The frozen `s_learner` was evaluated **once** on the untouched held-out test set (2,795,918 customers). No other model was scored on it, nothing was retuned, and these results were not used to change the model, hyperparameters, selection rule or policy.

**Read first: identification and scope limits.**
- Values are **conditional estimates for the released Criteo sample**, not population-level
  incrementality, advertising ROI, or guaranteed causal gains. Criteo's outcome-dependent privacy
  subsampling limits causal interpretation.
- The constant 0.85 treatment propensity is publisher-supported but cannot be verified from the
  file. IPW (which uses it directly) and AIPW disagree on this file (documented in
  `reports/criteo/benchmark/ipw_aipw_diagnostic.md`): IPW is likely biased upward, so **AIPW is
  primary** and IPW is shown only as a cross-check. AIPW is not guaranteed unbiased if the
  propensity assumption and the outcome nuisance are both wrong.
- The model was selected on validation data.
- The held-out test set was used once, for this report only; nothing was changed afterwards.

Units: percentage points (pp) of visit rate per evaluated customer, relative to treating nobody.
Budgets are shares of customers contacted, not monetary cost.

## Setup and verification
- Frozen model: `s_learner` (LightGBM, effective hyperparameters in `frozen_config.json`), seed 42, trained on the full training split (8,946,939 rows).
- Dataset sha256 `2716e1bf0fd157a9…`; split fingerprint `c6bcbcbf543b4f73…`.
- AIPW nuisance: LightGBM T-learner, same configuration as validation, fitted on 500,000 **training rows only** (fingerprint `27035f46f49e7f3a`); no validation or test rows. The plan's wording (train+validation nuisance for test) was replaced by this training-only rule at the owner's direction.
- Bootstrap: 200 Poisson replicates, shared weights (paired); seed 42.
- frozen config, dataset checksum, split fingerprint, hyperparameters, protocol: verified before test access
- refit validation-score SHA-256 equals frozen hash: True
- development rehearsal: passed before this evaluation
- models scored on the test set: s_learner only

## Primary result: AIPW policy value by contact budget (20% is primary)
`value minus random` is the paired difference from random targeting at the same budget.

| budget   | customers treated (share)   | AIPW value (pp)      |   random at same budget (pp) | value minus random (pp)   | excludes 0   |   IPW value (pp, cross-check) | primary   |
|:---------|:----------------------------|:---------------------|-----------------------------:|:--------------------------|:-------------|------------------------------:|:----------|
| 10%      | 10.0%                       | 0.582 [0.540, 0.617] |                        0.074 | 0.507 [0.470, 0.541]      | True         |                         0.909 |           |
| 20%      | 20.0%                       | 0.674 [0.616, 0.715] |                        0.149 | 0.525 [0.480, 0.558]      | True         |                         0.969 | yes       |
| 30%      | 30.0%                       | 0.711 [0.656, 0.756] |                        0.223 | 0.488 [0.448, 0.523]      | True         |                         0.994 |           |

## Reference policies
Budget-feasible policies respect the contact budget; treat-all is an unconstrained reference
(feasible only at a 100% budget) and treat-nobody is the zero baseline.

| policy       | budget   | AIPW value (pp)      | type                    |
|:-------------|:---------|:---------------------|:------------------------|
| treat_nobody | 0%       | 0.000 [0.000, 0.000] | zero baseline           |
| treat_all    | 100%     | 0.744 [0.687, 0.793] | unconstrained reference |
| random@10%   | 10%      | 0.074 [0.069, 0.079] | budget-feasible         |
| random@20%   | 20%      | 0.149 [0.137, 0.159] | budget-feasible         |
| random@30%   | 30%      | 0.223 [0.206, 0.238] | budget-feasible         |

Treat-all effect (AIPW): 0.744 pp [0.687, 0.793].

## Qini / AUUC
Area between the cumulative-gain curve and the random line (100-point trapezoid).

| estimator         | Qini (pp)            |   Qini / treat-all effect |
|:------------------|:---------------------|--------------------------:|
| AIPW (primary)    | 0.318 [0.293, 0.340] |                     0.428 |
| IPW (cross-check) | 0.461                |                     0.452 |

## IPW vs AIPW (known limitation, preserved)
| quantity | value |
|:--|--:|
| treated share | 0.8500 |
| treat-all effect, IPW (treated minus control mean), pp | 1.020 |
| treat-all effect, AIPW, pp | 0.744 |
| IPW minus AIPW, pp (SE) | 0.276 (0.019) |
| predicted baseline, treated vs control, pp | 4.079 vs 3.823 (z = 14.4) |

The IPW column in the table above is higher than AIPW at every budget for the same reason as on
validation: the arms differ in predicted baseline, which the constant-propensity estimator ignores.

## Exploratory: conversion uplift (not used for any decision)
Conversion is very rare, so intervals are wide and the conversion nuisance (trained on the same
500,000 rows) is weak. Rows show the **frozen visit policy**
(unchanged) scored on conversion, and, for description only, a conversion-trained S-learner of the
same family and fixed hyperparameters. Neither affects the selected model or targeting policy.
Conversion IPW vs AIPW treat-all: 0.1119 vs 0.0984 pp.

| budget                          | frozen visit policy: conversion value (pp)   | conversion-model ranking (pp)   |   random (pp) |
|:--------------------------------|:---------------------------------------------|:--------------------------------|--------------:|
| 10%                             | 0.0763 [0.0641, 0.0857]                      | 0.0859 [0.0718, 0.0968]         |        0.0098 |
| 20%                             | 0.0877 [0.0749, 0.0980]                      | 0.0924 [0.0787, 0.1046]         |        0.0197 |
| 30%                             | 0.0889 [0.0759, 0.1001]                      | 0.0929 [0.0782, 0.1056]         |        0.0295 |
| 100% (treat all, unconstrained) | 0.0984 [0.0839, 0.1112]                      |                                 |               |

Figures: `heldout_gain_curve.png`, `heldout_value_vs_random.png`.

## Caveats
- One seed, one split, one fixed configuration; intervals reflect scoring noise on this sample, not refit variability.
- Released-sample estimates only: no claim about advertising ROI, cost, or any customer population outside this file.
