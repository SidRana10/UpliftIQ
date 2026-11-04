# Criteo C5 preliminary validation benchmark: 500,000 training rows

**Preliminary.** These are validation-subset benchmarks on training subsets of 500,000 rows. They are
not final held-out results: the Criteo test set was not touched, and **no model has been selected or
frozen**. Hyperparameters are the fixed C3/C4 defaults (no tuning). Differences between models are
reported with paired intervals and should be read as effect sizes, not as a ranking decision.

Units: gains are **percentage points of visit rate per evaluated customer relative to treating
nobody**, estimated with AIPW (IPW as a check). Budgets are contact shares, not monetary cost.
They describe this released sample, not original-population incrementality or advertising ROI:
the constant 0.85 propensity is publisher-supported but cannot be verified from the file, and
Criteo's outcome-dependent privacy subsampling limits causal interpretation.

## Setup
- Training rows: 500,000 (nested, treatment-stratified subset of the train split; identical for every model).
- Validation subset: 500,000 rows (treatment-stratified, fixed). Bootstrap: 200 Poisson replicates, seed 42, shared weights (paired).
- AIPW nuisance: one LightGBM T-learner trained on 500,000 train rows disjoint from every candidate subset up to 2,000,000 rows, predicting the validation subset (out of sample, same for all models).
- Split fingerprint `c6bcbcbf543b4f73…`, data sha256 `2716e1bf0fd157a9…`, seed 42.
- Code commit `9c2d5568e3` (dirty: True); python 3.12.14, lightgbm 4.7.0, torch 2.15.0, scikit-learn 1.9.1; `OMP_NUM_THREADS=1`; each model ran in its own process.

## Policy value at matched contact budgets (AIPW)
Value = expected visit-rate gain vs treating nobody (pp) if the top share by predicted uplift is treated.
"vs random" is the paired difference from random targeting at the same budget.

| model      | AIPW value@10% (pp)   | vs random@10% (pp)   |   IPW value@10% (pp) | AIPW value@20% (pp)   | vs random@20% (pp)   |   IPW value@20% (pp) | AIPW value@30% (pp)   | vs random@30% (pp)   |   IPW value@30% (pp) |
|:-----------|:----------------------|:---------------------|---------------------:|:----------------------|:---------------------|---------------------:|:----------------------|:---------------------|---------------------:|
| logistic   | 0.502 [0.410, 0.580]  | 0.422 [0.345, 0.491] |                0.843 | 0.652 [0.552, 0.745]  | 0.490 [0.405, 0.565] |                0.968 | 0.691 [0.577, 0.786]  | 0.449 [0.378, 0.521] |                1.003 |
| s_learner  | 0.599 [0.516, 0.681]  | 0.519 [0.443, 0.591] |                0.913 | 0.717 [0.602, 0.822]  | 0.555 [0.474, 0.640] |                1.021 | 0.742 [0.617, 0.845]  | 0.500 [0.416, 0.572] |                1.044 |
| t_learner  | 0.550 [0.457, 0.627]  | 0.469 [0.387, 0.539] |                0.844 | 0.628 [0.521, 0.714]  | 0.466 [0.373, 0.533] |                0.911 | 0.653 [0.545, 0.745]  | 0.410 [0.330, 0.480] |                0.933 |
| dr_learner | 0.553 [0.471, 0.641]  | 0.472 [0.396, 0.549] |                0.862 | 0.649 [0.548, 0.758]  | 0.487 [0.406, 0.567] |                0.975 | 0.692 [0.584, 0.804]  | 0.449 [0.373, 0.534] |                1.018 |
| tarnet_cpu | 0.579 [0.501, 0.662]  | 0.498 [0.423, 0.572] |                0.906 | 0.726 [0.628, 0.820]  | 0.564 [0.483, 0.638] |                1.03  | 0.738 [0.628, 0.839]  | 0.495 [0.418, 0.570] |                1.055 |
| tarnet_mps | 0.558 [0.487, 0.641]  | 0.478 [0.411, 0.555] |                0.902 | 0.706 [0.611, 0.804]  | 0.544 [0.472, 0.626] |                1.037 | 0.720 [0.604, 0.823]  | 0.477 [0.399, 0.551] |                1.06  |

References (feasible = respects the budget; treat-all is an unconstrained reference below a 100% budget):

| policy       |   budget | AIPW value (pp)      | feasibility                                                  |
|:-------------|---------:|:---------------------|:-------------------------------------------------------------|
| treat_nobody |      0   | 0.000 [0.000, 0.000] | reference (budget 0)                                         |
| treat_all    |      1   | 0.808 [0.677, 0.923] | unconstrained reference (feasible only at budget 100%)       |
| random@10%   |      0.1 | 0.081 [0.068, 0.092] | feasible (budget-constrained, expectation over random draws) |
| random@20%   |      0.2 | 0.162 [0.135, 0.185] | feasible (budget-constrained, expectation over random draws) |
| random@30%   |      0.3 | 0.242 [0.203, 0.277] | feasible (budget-constrained, expectation over random draws) |

## Qini (AUUC-style) ranking metric
Area between the cumulative-gain curve and the random line; curve in `fit500000_gain_curves.png`.

| model      | Qini AIPW (pp)       |   Qini IPW (pp) |   Qini / treat-all effect |
|:-----------|:---------------------|----------------:|--------------------------:|
| logistic   | 0.265 [0.222, 0.312] |           0.428 |                     0.328 |
| s_learner  | 0.311 [0.261, 0.356] |           0.466 |                     0.385 |
| t_learner  | 0.239 [0.182, 0.287] |           0.364 |                     0.296 |
| dr_learner | 0.267 [0.220, 0.319] |           0.438 |                     0.33  |
| tarnet_cpu | 0.302 [0.249, 0.352] |           0.472 |                     0.374 |
| tarnet_mps | 0.287 [0.238, 0.335] |           0.473 |                     0.356 |

## Paired differences between models (AIPW, selected metrics)
Reference rows come from two joint bootstrap runs with shared weights; full table in the JSON.

| reference   | model      | metric    | difference (pp)         | excludes 0   |
|:------------|:-----------|:----------|:------------------------|:-------------|
| logistic    | s_learner  | value@20% | 0.065 [0.009, 0.131]    | True         |
| logistic    | s_learner  | qini      | 0.046 [0.002, 0.092]    | True         |
| logistic    | t_learner  | value@20% | -0.024 [-0.096, 0.032]  | False        |
| logistic    | t_learner  | qini      | -0.027 [-0.080, 0.027]  | False        |
| logistic    | dr_learner | value@20% | -0.003 [-0.067, 0.053]  | False        |
| logistic    | dr_learner | qini      | 0.002 [-0.047, 0.048]   | False        |
| logistic    | tarnet_cpu | value@20% | 0.074 [0.020, 0.136]    | True         |
| logistic    | tarnet_cpu | qini      | 0.037 [-0.009, 0.088]   | False        |
| logistic    | tarnet_mps | value@20% | 0.054 [0.003, 0.125]    | True         |
| logistic    | tarnet_mps | qini      | 0.022 [-0.022, 0.067]   | False        |
| tarnet_cpu  | logistic   | value@20% | -0.074 [-0.136, -0.020] | True         |
| tarnet_cpu  | logistic   | qini      | -0.037 [-0.088, 0.009]  | False        |
| tarnet_cpu  | s_learner  | value@20% | -0.009 [-0.076, 0.055]  | False        |
| tarnet_cpu  | s_learner  | qini      | 0.009 [-0.038, 0.056]   | False        |
| tarnet_cpu  | t_learner  | value@20% | -0.098 [-0.173, -0.027] | True         |
| tarnet_cpu  | t_learner  | qini      | -0.063 [-0.122, -0.019] | True         |
| tarnet_cpu  | dr_learner | value@20% | -0.077 [-0.136, -0.014] | True         |
| tarnet_cpu  | dr_learner | qini      | -0.035 [-0.076, 0.016]  | False        |
| tarnet_cpu  | tarnet_mps | value@20% | -0.020 [-0.050, 0.009]  | False        |
| tarnet_cpu  | tarnet_mps | qini      | -0.015 [-0.033, 0.005]  | False        |

## Engineering: training cost, latency, memory
Latency measured on the validation rows with `predict_uplift` (includes preprocessing), CPU inference
for tree/linear models; TARNet on its own device. Peak RSS is the process high-water mark, including
data loading (the memory-mapped feature file is paged in while gathering rows). "RSS peak in fit"
is sampled every 50 ms during `fit`; MPS unified memory is not fully visible in RSS. TARNet
epochs: 30 is the epoch cap (early stopping did not trigger).

| model      |   n_fit |   train_s |   predict 500k rows (s) |   rows/s @100k | p50/p95 ms @1   | p50/p95 ms @1k   | p50/p95 ms @100k   |   RSS before fit (GB) |   RSS peak in fit (GB) |   process peak RSS (GB) |   epochs |
|:-----------|--------:|----------:|------------------------:|---------------:|:----------------|:-----------------|:-------------------|----------------------:|-----------------------:|------------------------:|---------:|
| logistic   |  500000 |       0.5 |                    0.05 |     12,193,512 | 0.08 / 0.09     | 0.2 / 0.2        | 8 / 9              |                  0.62 |                   0.73 |                    1.27 |          |
| s_learner  |  500000 |       1.6 |                    1.23 |        542,020 | 0.32 / 0.35     | 2.3 / 2.4        | 184 / 239          |                  0.62 |                   0.75 |                    1.27 |          |
| t_learner  |  500000 |       1.7 |                    0.88 |        920,940 | 0.32 / 0.34     | 1.5 / 1.6        | 109 / 171          |                  0.62 |                   0.72 |                    1.27 |          |
| dr_learner |  500000 |       3.6 |                    0.16 |      3,851,171 | 0.16 / 0.17     | 0.4 / 0.4        | 26 / 31            |                  0.62 |                   0.72 |                    1.27 |          |
| tarnet_cpu |  500000 |       7.6 |                    0.14 |      3,890,540 | 0.04 / 0.04     | 0.2 / 0.2        | 26 / 26            |                  0.62 |                   0.78 |                    1.27 |       12 |
| tarnet_mps |  500000 |       6.5 |                    0.14 |      4,047,743 | 0.47 / 0.66     | 0.7 / 0.9        | 25 / 25            |                  0.62 |                   0.9  |                    1.27 |       12 |

## Training cost vs quality
`fit500000_cost_quality.png`.

| model      |   train_s |   Qini AIPW (pp) |   Qini CI low |   Qini CI high |
|:-----------|----------:|-----------------:|--------------:|---------------:|
| logistic   |    0.5012 |           0.2653 |        0.2218 |         0.3117 |
| s_learner  |    1.6201 |           0.3113 |        0.2608 |         0.3563 |
| t_learner  |    1.654  |           0.2387 |        0.1817 |         0.2869 |
| dr_learner |    3.6457 |           0.2669 |        0.2202 |         0.3188 |
| tarnet_cpu |    7.5695 |           0.302  |        0.2494 |         0.3521 |
| tarnet_mps |    6.4783 |           0.2872 |        0.2378 |         0.3354 |

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
