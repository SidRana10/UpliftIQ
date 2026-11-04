# Criteo C6 preliminary validation benchmark: 2,000,000 training rows

**Preliminary.** These are validation-subset benchmarks on training subsets of 2,000,000 rows. They are
not final held-out results: the Criteo test set was not touched, and **no model has been selected or
frozen**. Hyperparameters are the fixed C3/C4 defaults (no tuning). Differences between models are
reported with paired intervals and should be read as effect sizes, not as a ranking decision.

Units: gains are **percentage points of visit rate per evaluated customer relative to treating
nobody**, estimated with AIPW (IPW as a check). Budgets are contact shares, not monetary cost.
They describe this released sample, not original-population incrementality or advertising ROI:
the constant 0.85 propensity is publisher-supported but cannot be verified from the file, and
Criteo's outcome-dependent privacy subsampling limits causal interpretation.

## Setup
- Training rows: 2,000,000 (nested, treatment-stratified subset of the train split; identical for every model).
- Validation subset: 500,000 rows (treatment-stratified, fixed). Bootstrap: 200 Poisson replicates, seed 42, shared weights (paired).
- AIPW nuisance: one LightGBM T-learner trained on 500,000 train rows disjoint from every candidate subset up to 2,000,000 rows, predicting the validation subset (out of sample, same for all models).
- Split fingerprint `c6bcbcbf543b4f73…`, data sha256 `2716e1bf0fd157a9…`, seed 42.
- Code commit `1abfb4fcf4` (dirty: True); python 3.12.14, lightgbm 4.7.0, torch 2.15.0, scikit-learn 1.9.1; `OMP_NUM_THREADS=1`; each model ran in its own process.

## Policy value at matched contact budgets (AIPW)
Value = expected visit-rate gain vs treating nobody (pp) if the top share by predicted uplift is treated.
"vs random" is the paired difference from random targeting at the same budget.

| model      | AIPW value@10% (pp)   | vs random@10% (pp)   |   IPW value@10% (pp) | AIPW value@20% (pp)   | vs random@20% (pp)   |   IPW value@20% (pp) | AIPW value@30% (pp)   | vs random@30% (pp)   |   IPW value@30% (pp) |
|:-----------|:----------------------|:---------------------|---------------------:|:----------------------|:---------------------|---------------------:|:----------------------|:---------------------|---------------------:|
| logistic   | 0.537 [0.447, 0.622]  | 0.456 [0.376, 0.533] |                0.881 | 0.654 [0.556, 0.742]  | 0.492 [0.413, 0.564] |                0.993 | 0.685 [0.577, 0.781]  | 0.443 [0.374, 0.512] |                1.022 |
| s_learner  | 0.630 [0.550, 0.716]  | 0.549 [0.483, 0.629] |                0.981 | 0.757 [0.640, 0.858]  | 0.596 [0.503, 0.677] |                1.058 | 0.803 [0.687, 0.910]  | 0.560 [0.482, 0.639] |                1.11  |
| t_learner  | 0.572 [0.496, 0.652]  | 0.491 [0.428, 0.564] |                0.867 | 0.677 [0.586, 0.776]  | 0.516 [0.450, 0.593] |                0.975 | 0.718 [0.612, 0.822]  | 0.475 [0.402, 0.553] |                1.014 |
| dr_learner | 0.596 [0.508, 0.680]  | 0.516 [0.440, 0.591] |                0.937 | 0.671 [0.558, 0.763]  | 0.510 [0.427, 0.581] |                0.984 | 0.717 [0.614, 0.828]  | 0.474 [0.407, 0.557] |                1.039 |
| tarnet_cpu | 0.622 [0.546, 0.685]  | 0.541 [0.476, 0.596] |                0.986 | 0.705 [0.602, 0.793]  | 0.544 [0.465, 0.616] |                1.064 | 0.723 [0.623, 0.821]  | 0.481 [0.416, 0.554] |                1.072 |
| tarnet_mps | 0.566 [0.488, 0.637]  | 0.485 [0.413, 0.551] |                0.912 | 0.717 [0.612, 0.814]  | 0.556 [0.475, 0.635] |                1.052 | 0.762 [0.652, 0.865]  | 0.520 [0.449, 0.588] |                1.101 |

References (feasible = respects the budget; treat-all is an unconstrained reference below a 100% budget):

| policy       |   budget | AIPW value (pp)      | feasibility                                                  |
|:-------------|---------:|:---------------------|:-------------------------------------------------------------|
| treat_nobody |      0   | 0.000 [0.000, 0.000] | reference (budget 0)                                         |
| treat_all    |      1   | 0.808 [0.677, 0.923] | unconstrained reference (feasible only at budget 100%)       |
| random@10%   |      0.1 | 0.081 [0.068, 0.092] | feasible (budget-constrained, expectation over random draws) |
| random@20%   |      0.2 | 0.162 [0.135, 0.185] | feasible (budget-constrained, expectation over random draws) |
| random@30%   |      0.3 | 0.242 [0.203, 0.277] | feasible (budget-constrained, expectation over random draws) |

## Qini (AUUC-style) ranking metric
Area between the cumulative-gain curve and the random line; curve in `fit2000000_gain_curves.png`.

| model      | Qini AIPW (pp)       |   Qini IPW (pp) |   Qini / treat-all effect |
|:-----------|:---------------------|----------------:|--------------------------:|
| logistic   | 0.266 [0.215, 0.318] |           0.447 |                     0.329 |
| s_learner  | 0.359 [0.309, 0.402] |           0.521 |                     0.444 |
| t_learner  | 0.283 [0.224, 0.337] |           0.428 |                     0.351 |
| dr_learner | 0.299 [0.257, 0.346] |           0.467 |                     0.37  |
| tarnet_cpu | 0.294 [0.249, 0.342] |           0.488 |                     0.364 |
| tarnet_mps | 0.320 [0.272, 0.361] |           0.492 |                     0.396 |

## Paired differences between models (AIPW, selected metrics)
Reference rows come from two joint bootstrap runs with shared weights; full table in the JSON.

| reference   | model      | metric    | difference (pp)         | excludes 0   |
|:------------|:-----------|:----------|:------------------------|:-------------|
| logistic    | s_learner  | value@20% | 0.103 [0.049, 0.163]    | True         |
| logistic    | s_learner  | qini      | 0.093 [0.056, 0.134]    | True         |
| logistic    | t_learner  | value@20% | 0.023 [-0.052, 0.100]   | False        |
| logistic    | t_learner  | qini      | 0.017 [-0.039, 0.079]   | False        |
| logistic    | dr_learner | value@20% | 0.018 [-0.031, 0.079]   | False        |
| logistic    | dr_learner | qini      | 0.032 [-0.012, 0.078]   | False        |
| logistic    | tarnet_cpu | value@20% | 0.052 [0.010, 0.113]    | True         |
| logistic    | tarnet_cpu | qini      | 0.028 [-0.017, 0.077]   | False        |
| logistic    | tarnet_mps | value@20% | 0.063 [0.007, 0.122]    | True         |
| logistic    | tarnet_mps | qini      | 0.054 [0.008, 0.098]    | True         |
| tarnet_cpu  | logistic   | value@20% | -0.052 [-0.113, -0.010] | True         |
| tarnet_cpu  | logistic   | qini      | -0.028 [-0.077, 0.017]  | False        |
| tarnet_cpu  | s_learner  | value@20% | 0.052 [-0.007, 0.112]   | False        |
| tarnet_cpu  | s_learner  | qini      | 0.065 [0.014, 0.118]    | True         |
| tarnet_cpu  | t_learner  | value@20% | -0.028 [-0.091, 0.042]  | False        |
| tarnet_cpu  | t_learner  | qini      | -0.011 [-0.065, 0.049]  | False        |
| tarnet_cpu  | dr_learner | value@20% | -0.034 [-0.084, 0.029]  | False        |
| tarnet_cpu  | dr_learner | qini      | 0.005 [-0.044, 0.058]   | False        |
| tarnet_cpu  | tarnet_mps | value@20% | 0.012 [-0.040, 0.059]   | False        |
| tarnet_cpu  | tarnet_mps | qini      | 0.026 [-0.018, 0.066]   | False        |

## Engineering: training cost, latency, memory
Latency measured on the validation rows with `predict_uplift` (includes preprocessing), CPU inference
for tree/linear models; TARNet on its own device. Peak RSS is the process high-water mark, including
data loading (the memory-mapped feature file is paged in while gathering rows). "RSS peak in fit"
is sampled every 50 ms during `fit`; MPS unified memory is not fully visible in RSS. TARNet
`epochs` is the number of epochs run (cap 30, early-stopping patience 3 on a 10% internal split).

| model      |   n_fit |   train_s |   predict 500k rows (s) |   rows/s @100k | p50/p95 ms @1   | p50/p95 ms @1k   | p50/p95 ms @100k   |   RSS before fit (GB) |   RSS peak in fit (GB) |   process peak RSS (GB) |   epochs |
|:-----------|--------:|----------:|------------------------:|---------------:|:----------------|:-----------------|:-------------------|----------------------:|-----------------------:|------------------------:|---------:|
| logistic   | 2000000 |       2.3 |                    0.05 |     12,801,093 | 0.08 / 0.09     | 0.2 / 0.2        | 8 / 8              |                  0.62 |                   1.05 |                    1.27 |          |
| s_learner  | 2000000 |       8.8 |                    1.94 |        295,595 | 0.32 / 1.38     | 6.2 / 9.6        | 338 / 396          |                  0.71 |                   1.11 |                    1.36 |          |
| t_learner  | 2000000 |       9.5 |                    2.36 |        261,209 | 0.32 / 0.35     | 4.3 / 4.5        | 383 / 446          |                  0.5  |                   0.69 |                    1.15 |          |
| dr_learner | 2000000 |      15.9 |                    0.14 |      4,220,374 | 0.15 / 0.17     | 0.4 / 0.4        | 24 / 27            |                  0.71 |                   0.73 |                    1.36 |          |
| tarnet_cpu | 2000000 |      24.9 |                    0.14 |      4,459,905 | 0.04 / 0.04     | 0.2 / 0.2        | 22 / 23            |                  0.71 |                   1.01 |                    1.36 |       11 |
| tarnet_mps | 2000000 |      18.4 |                    0.12 |      2,699,292 | 0.44 / 0.56     | 0.7 / 0.8        | 37 / 44            |                  0.34 |                   1.06 |                    1.14 |       12 |

## Training cost vs quality
`fit2000000_cost_quality.png`.

| model      |   train_s |   Qini AIPW (pp) |   Qini CI low |   Qini CI high |
|:-----------|----------:|-----------------:|--------------:|---------------:|
| logistic   |    2.2944 |           0.2662 |        0.2147 |         0.3185 |
| s_learner  |    8.7697 |           0.3588 |        0.309  |         0.4022 |
| t_learner  |    9.4818 |           0.2833 |        0.2242 |         0.3372 |
| dr_learner |   15.8703 |           0.2986 |        0.2573 |         0.3462 |
| tarnet_cpu |   24.9428 |           0.294  |        0.249  |         0.3417 |
| tarnet_mps |   18.4375 |           0.3198 |        0.2715 |         0.3612 |

## Assumption check: IPW vs AIPW
No propensity model was fitted; this only compares the two estimators and arm balance.

| quantity | value |
|:--|--:|
| treated share (validation subset) | 0.8500 |
| treat-all effect, IPW (= treated minus control mean), pp | 1.100 |
| treat-all effect, AIPW, pp | 0.808 |
| IPW minus AIPW, pp (SE) | 0.292 (0.046) |
| predicted baseline visit rate, treated vs control, pp | 4.120 vs 3.848 |
| difference (SE), pp | 0.272 (0.042) |

**Flag:** IPW and AIPW disagree by more than twice their paired standard error, or the arms differ in predicted baseline (z = 6.4). This is consistent with covariate imbalance between treated and control in the released file, i.e. the constant 0.85 propensity does not fully hold even though the C1 balance thresholds passed. AIPW (primary) adjusts through the outcome model and stays consistent when that model is adequate; IPW values above are shown only as a check and are likely biased upward. All models are scored with the same estimator, so model comparisons are less affected, but absolute gains should be read with this caveat.

## Caveats
- A single preliminary rung and a single seed; model orderings may change with more training data, tuning, or other seeds.
- TARNet appears on CPU and MPS: CPU is bit-reproducible, MPS is not guaranteed to be. Selection (later) will use one device.
- The shared nuisance is itself a LightGBM T-learner. With an out-of-sample nuisance, AIPW is unbiased if the constant 0.85 propensity holds and is consistent if the nuisance is adequate; it is not guaranteed to be unbiased if both are wrong. IPW is shown as a check only.
- Using one shared estimator makes models comparable but does not by itself make their relative rankings unbiased.
