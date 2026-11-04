# Criteo C5 preliminary validation benchmark: 100,000 training rows

**Preliminary.** These are validation-subset benchmarks on training subsets of 100,000 rows. They are
not final held-out results: the Criteo test set was not touched, and **no model has been selected or
frozen**. Hyperparameters are the fixed C3/C4 defaults (no tuning). Differences between models are
reported with paired intervals and should be read as effect sizes, not as a ranking decision.

Units: gains are **percentage points of visit rate per evaluated customer relative to treating
nobody**, estimated with AIPW (IPW as a check). Budgets are contact shares, not monetary cost.
They describe this released sample, not original-population incrementality or advertising ROI:
the constant 0.85 propensity is publisher-supported but cannot be verified from the file, and
Criteo's outcome-dependent privacy subsampling limits causal interpretation.

## Setup
- Training rows: 100,000 (nested, treatment-stratified subset of the train split; identical for every model).
- Validation subset: 500,000 rows (treatment-stratified, fixed). Bootstrap: 200 Poisson replicates, seed 42, shared weights (paired).
- AIPW nuisance: one LightGBM T-learner trained on 500,000 train rows disjoint from every candidate subset up to 2,000,000 rows, predicting the validation subset (out of sample, same for all models).
- Split fingerprint `c6bcbcbf543b4f73…`, data sha256 `2716e1bf0fd157a9…`, seed 42.
- Code commit `9c2d5568e3` (dirty: True); python 3.12.14, lightgbm 4.7.0, torch 2.15.0, scikit-learn 1.9.1; `OMP_NUM_THREADS=1`; each model ran in its own process.

## Policy value at matched contact budgets (AIPW)
Value = expected visit-rate gain vs treating nobody (pp) if the top share by predicted uplift is treated.
"vs random" is the paired difference from random targeting at the same budget.

| model      | AIPW value@10% (pp)   | vs random@10% (pp)   |   IPW value@10% (pp) | AIPW value@20% (pp)   | vs random@20% (pp)   |   IPW value@20% (pp) | AIPW value@30% (pp)   | vs random@30% (pp)   |   IPW value@30% (pp) |
|:-----------|:----------------------|:---------------------|---------------------:|:----------------------|:---------------------|---------------------:|:----------------------|:---------------------|---------------------:|
| logistic   | 0.576 [0.507, 0.637]  | 0.495 [0.432, 0.549] |                0.893 | 0.675 [0.598, 0.753]  | 0.513 [0.443, 0.583] |                1.001 | 0.706 [0.618, 0.786]  | 0.463 [0.392, 0.523] |                1.037 |
| s_learner  | 0.573 [0.500, 0.652]  | 0.493 [0.425, 0.567] |                0.885 | 0.699 [0.611, 0.781]  | 0.538 [0.468, 0.603] |                1.043 | 0.723 [0.636, 0.817]  | 0.481 [0.423, 0.551] |                1.068 |
| t_learner  | 0.529 [0.445, 0.593]  | 0.448 [0.373, 0.511] |                0.796 | 0.632 [0.534, 0.710]  | 0.471 [0.397, 0.536] |                0.906 | 0.661 [0.571, 0.749]  | 0.418 [0.351, 0.482] |                0.928 |
| dr_learner | 0.534 [0.467, 0.594]  | 0.453 [0.390, 0.514] |                0.844 | 0.627 [0.548, 0.699]  | 0.466 [0.399, 0.530] |                0.96  | 0.656 [0.574, 0.745]  | 0.414 [0.352, 0.478] |                0.994 |
| tarnet_cpu | 0.614 [0.535, 0.691]  | 0.534 [0.464, 0.605] |                0.909 | 0.700 [0.610, 0.783]  | 0.538 [0.465, 0.607] |                0.996 | 0.723 [0.625, 0.810]  | 0.481 [0.414, 0.550] |                1.013 |
| tarnet_mps | 0.578 [0.505, 0.654]  | 0.498 [0.433, 0.566] |                0.896 | 0.672 [0.594, 0.751]  | 0.510 [0.441, 0.578] |                0.988 | 0.684 [0.600, 0.773]  | 0.442 [0.376, 0.506] |                1.001 |

References (feasible = respects the budget; treat-all is an unconstrained reference below a 100% budget):

| policy       |   budget | AIPW value (pp)      | feasibility                                                  |
|:-------------|---------:|:---------------------|:-------------------------------------------------------------|
| treat_nobody |      0   | 0.000 [0.000, 0.000] | reference (budget 0)                                         |
| treat_all    |      1   | 0.808 [0.677, 0.923] | unconstrained reference (feasible only at budget 100%)       |
| random@10%   |      0.1 | 0.081 [0.068, 0.092] | feasible (budget-constrained, expectation over random draws) |
| random@20%   |      0.2 | 0.162 [0.135, 0.185] | feasible (budget-constrained, expectation over random draws) |
| random@30%   |      0.3 | 0.242 [0.203, 0.277] | feasible (budget-constrained, expectation over random draws) |

## Qini (AUUC-style) ranking metric
Area between the cumulative-gain curve and the random line; curve in `fit100000_gain_curves.png`.

| model      | Qini AIPW (pp)       |   Qini IPW (pp) |   Qini / treat-all effect |
|:-----------|:---------------------|----------------:|--------------------------:|
| logistic   | 0.280 [0.225, 0.336] |           0.456 |                     0.347 |
| s_learner  | 0.292 [0.239, 0.344] |           0.472 |                     0.361 |
| t_learner  | 0.238 [0.187, 0.290] |           0.355 |                     0.295 |
| dr_learner | 0.244 [0.190, 0.292] |           0.422 |                     0.302 |
| tarnet_cpu | 0.287 [0.231, 0.336] |           0.426 |                     0.355 |
| tarnet_mps | 0.268 [0.220, 0.321] |           0.43  |                     0.331 |

## Paired differences between models (AIPW, selected metrics)
Reference rows come from two joint bootstrap runs with shared weights; full table in the JSON.

| reference   | model      | metric    | difference (pp)         | excludes 0   |
|:------------|:-----------|:----------|:------------------------|:-------------|
| logistic    | s_learner  | value@20% | 0.024 [-0.033, 0.078]   | False        |
| logistic    | s_learner  | qini      | 0.012 [-0.029, 0.056]   | False        |
| logistic    | t_learner  | value@20% | -0.043 [-0.115, 0.026]  | False        |
| logistic    | t_learner  | qini      | -0.042 [-0.104, 0.017]  | False        |
| logistic    | dr_learner | value@20% | -0.048 [-0.102, 0.009]  | False        |
| logistic    | dr_learner | qini      | -0.036 [-0.081, 0.018]  | False        |
| logistic    | tarnet_cpu | value@20% | 0.025 [-0.019, 0.076]   | False        |
| logistic    | tarnet_cpu | qini      | 0.007 [-0.026, 0.044]   | False        |
| logistic    | tarnet_mps | value@20% | -0.003 [-0.048, 0.042]  | False        |
| logistic    | tarnet_mps | qini      | -0.012 [-0.049, 0.027]  | False        |
| tarnet_cpu  | logistic   | value@20% | -0.025 [-0.076, 0.019]  | False        |
| tarnet_cpu  | logistic   | qini      | -0.007 [-0.044, 0.026]  | False        |
| tarnet_cpu  | s_learner  | value@20% | -0.000 [-0.063, 0.050]  | False        |
| tarnet_cpu  | s_learner  | qini      | 0.005 [-0.049, 0.052]   | False        |
| tarnet_cpu  | t_learner  | value@20% | -0.067 [-0.139, -0.007] | True         |
| tarnet_cpu  | t_learner  | qini      | -0.048 [-0.110, 0.001]  | False        |
| tarnet_cpu  | dr_learner | value@20% | -0.072 [-0.130, -0.011] | True         |
| tarnet_cpu  | dr_learner | qini      | -0.043 [-0.095, 0.005]  | False        |
| tarnet_cpu  | tarnet_mps | value@20% | -0.028 [-0.060, 0.006]  | False        |
| tarnet_cpu  | tarnet_mps | qini      | -0.019 [-0.048, 0.013]  | False        |

## Engineering: training cost, latency, memory
Latency measured on the validation rows with `predict_uplift` (includes preprocessing), CPU inference
for tree/linear models; TARNet on its own device. Peak RSS is the process high-water mark, including
data loading (the memory-mapped feature file is paged in while gathering rows). "RSS peak in fit"
is sampled every 50 ms during `fit`; MPS unified memory is not fully visible in RSS. TARNet
epochs: 30 is the epoch cap (early stopping did not trigger).

| model      |   n_fit |   train_s |   predict 500k rows (s) |   rows/s @100k | p50/p95 ms @1   | p50/p95 ms @1k   | p50/p95 ms @100k   |   RSS before fit (GB) |   RSS peak in fit (GB) |   process peak RSS (GB) |   epochs |
|:-----------|--------:|----------:|------------------------:|---------------:|:----------------|:-----------------|:-------------------|----------------------:|-----------------------:|------------------------:|---------:|
| logistic   |  100000 |       0.1 |                    0.06 |     12,704,462 | 0.08 / 0.09     | 0.2 / 0.2        | 8 / 8              |                   0.6 |                   0.62 |                    1.25 |          |
| s_learner  |  100000 |       0.5 |                    1.4  |        492,058 | 0.32 / 0.35     | 2.4 / 2.5        | 203 / 275          |                   0.6 |                   0.64 |                    1.25 |          |
| t_learner  |  100000 |       0.8 |                    1.71 |        337,503 | 0.32 / 0.33     | 3.5 / 3.5        | 296 / 335          |                   0.6 |                   0.64 |                    1.25 |          |
| dr_learner |  100000 |       1   |                    0.18 |      2,824,081 | 0.16 / 0.18     | 0.5 / 0.5        | 35 / 35            |                   0.6 |                   0.64 |                    1.25 |          |
| tarnet_cpu |  100000 |       3.7 |                    0.14 |      3,854,097 | 0.04 / 0.04     | 0.2 / 0.2        | 26 / 27            |                   0.6 |                   0.82 |                    1.25 |       27 |
| tarnet_mps |  100000 |       3.5 |                    0.31 |      4,209,522 | 0.46 / 0.62     | 0.6 / 0.8        | 24 / 24            |                   0.6 |                   1    |                    1.25 |       30 |

## Training cost vs quality
`fit100000_cost_quality.png`.

| model      |   train_s |   Qini AIPW (pp) |   Qini CI low |   Qini CI high |
|:-----------|----------:|-----------------:|--------------:|---------------:|
| logistic   |    0.1226 |           0.28   |        0.2252 |         0.3363 |
| s_learner  |    0.5351 |           0.2918 |        0.2389 |         0.3439 |
| t_learner  |    0.7605 |           0.2383 |        0.1867 |         0.2895 |
| dr_learner |    0.9679 |           0.2439 |        0.19   |         0.292  |
| tarnet_cpu |    3.6869 |           0.2865 |        0.2311 |         0.3363 |
| tarnet_mps |    3.539  |           0.2678 |        0.2204 |         0.3208 |

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
- The shared nuisance is itself a LightGBM T-learner; AIPW remains unbiased for any out-of-sample nuisance, and IPW is shown as a check.
