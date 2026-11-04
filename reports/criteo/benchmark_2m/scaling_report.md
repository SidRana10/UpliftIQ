# Criteo C6 learning curve: 100K, 500K, 2M training rows

**Preliminary and validation-only.** Same validation subset (500,000 rows), shared out-of-sample
AIPW nuisance, fixed C3/C4 hyperparameters, seed 42 and 200 paired Poisson bootstrap
replicates as the per-rung reports. The held-out test set was not touched and **no model has been
selected or frozen**. Values are conditional estimates for this released sample (percentage points
of visit rate per evaluated customer relative to treating nobody), not population-level causal
effects or advertising ROI; they are not evidence of final generalization performance.

## AIPW policy value and Qini by training size
| model      | train rows   | AIPW value@10% (pp)   | AIPW value@20% (pp)   | AIPW value@30% (pp)   | AIPW qini (pp)       |
|:-----------|:-------------|:----------------------|:----------------------|:----------------------|:---------------------|
| logistic   | 100K         | 0.576 [0.507, 0.637]  | 0.675 [0.598, 0.753]  | 0.706 [0.618, 0.786]  | 0.280 [0.225, 0.336] |
| logistic   | 500K         | 0.502 [0.410, 0.580]  | 0.652 [0.552, 0.745]  | 0.691 [0.577, 0.786]  | 0.265 [0.222, 0.312] |
| logistic   | 2M           | 0.537 [0.447, 0.622]  | 0.654 [0.556, 0.742]  | 0.685 [0.577, 0.781]  | 0.266 [0.215, 0.318] |
| s_learner  | 100K         | 0.573 [0.500, 0.652]  | 0.699 [0.611, 0.781]  | 0.723 [0.636, 0.817]  | 0.292 [0.239, 0.344] |
| s_learner  | 500K         | 0.599 [0.516, 0.681]  | 0.717 [0.602, 0.822]  | 0.742 [0.617, 0.845]  | 0.311 [0.261, 0.356] |
| s_learner  | 2M           | 0.630 [0.550, 0.716]  | 0.757 [0.640, 0.858]  | 0.803 [0.687, 0.910]  | 0.359 [0.309, 0.402] |
| t_learner  | 100K         | 0.529 [0.445, 0.593]  | 0.632 [0.534, 0.710]  | 0.661 [0.571, 0.749]  | 0.238 [0.187, 0.290] |
| t_learner  | 500K         | 0.550 [0.457, 0.627]  | 0.628 [0.521, 0.714]  | 0.653 [0.545, 0.745]  | 0.239 [0.182, 0.287] |
| t_learner  | 2M           | 0.572 [0.496, 0.652]  | 0.677 [0.586, 0.776]  | 0.718 [0.612, 0.822]  | 0.283 [0.224, 0.337] |
| dr_learner | 100K         | 0.534 [0.467, 0.594]  | 0.627 [0.548, 0.699]  | 0.656 [0.574, 0.745]  | 0.244 [0.190, 0.292] |
| dr_learner | 500K         | 0.553 [0.471, 0.641]  | 0.649 [0.548, 0.758]  | 0.692 [0.584, 0.804]  | 0.267 [0.220, 0.319] |
| dr_learner | 2M           | 0.596 [0.508, 0.680]  | 0.671 [0.558, 0.763]  | 0.717 [0.614, 0.828]  | 0.299 [0.257, 0.346] |
| tarnet_cpu | 100K         | 0.614 [0.535, 0.691]  | 0.700 [0.610, 0.783]  | 0.723 [0.625, 0.810]  | 0.287 [0.231, 0.336] |
| tarnet_cpu | 500K         | 0.579 [0.501, 0.662]  | 0.726 [0.628, 0.820]  | 0.738 [0.628, 0.839]  | 0.302 [0.249, 0.352] |
| tarnet_cpu | 2M           | 0.622 [0.546, 0.685]  | 0.705 [0.602, 0.793]  | 0.723 [0.623, 0.821]  | 0.294 [0.249, 0.342] |
| tarnet_mps | 100K         | 0.578 [0.505, 0.654]  | 0.672 [0.594, 0.751]  | 0.684 [0.600, 0.773]  | 0.268 [0.220, 0.321] |
| tarnet_mps | 500K         | 0.558 [0.487, 0.641]  | 0.706 [0.611, 0.804]  | 0.720 [0.604, 0.823]  | 0.287 [0.238, 0.335] |
| tarnet_mps | 2M           | 0.566 [0.488, 0.637]  | 0.717 [0.612, 0.814]  | 0.762 [0.652, 0.865]  | 0.320 [0.272, 0.361] |

## Change from adding training data (AIPW, paired 95% interval)
Each change is the bigger training set minus the smaller one for the same model, scored on the same
validation customers with shared bootstrap weights. `*` marks a paired interval that excludes 0.
Only the 20% budget and Qini are shown here; all budgets are in the JSON. Each interval covers
scoring noise on the validation subset; it does not include the variability of re-training with
another seed, so a "significant" change is not guaranteed to reproduce.

| model      | 100K to 500K: value@20% (pp)   | 100K to 500K: qini (pp)   | 500K to 2M: value@20% (pp)   | 500K to 2M: qini (pp)   |
|:-----------|:-------------------------------|:--------------------------|:-----------------------------|:------------------------|
| logistic   | -0.023 [-0.080, 0.038]         | -0.015 [-0.065, 0.036]    | 0.002 [-0.024, 0.034]        | 0.001 [-0.020, 0.023]   |
| s_learner  | 0.017 [-0.051, 0.093]          | 0.020 [-0.039, 0.077]     | 0.040 [-0.003, 0.089]        | 0.047 [0.015, 0.077] *  |
| t_learner  | -0.005 [-0.077, 0.061]         | 0.000 [-0.064, 0.055]     | 0.049 [-0.005, 0.116]        | 0.045 [-0.013, 0.112]   |
| dr_learner | 0.021 [-0.052, 0.086]          | 0.023 [-0.039, 0.079]     | 0.023 [-0.029, 0.073]        | 0.032 [-0.006, 0.071]   |
| tarnet_cpu | 0.026 [-0.027, 0.082]          | 0.015 [-0.028, 0.067]     | -0.020 [-0.076, 0.036]       | -0.008 [-0.054, 0.035]  |
| tarnet_mps | 0.034 [-0.028, 0.101]          | 0.019 [-0.033, 0.072]     | 0.011 [-0.039, 0.070]        | 0.033 [-0.007, 0.072]   |

## Training cost by size
| model      |   train s @100K |   train s @500K |   train s @2M |   2M / 500K |   fit RSS peak @2M (GB) |
|:-----------|----------------:|----------------:|--------------:|------------:|------------------------:|
| dr_learner |            0.97 |            3.65 |         15.87 |        4.35 |                    0.73 |
| logistic   |            0.12 |            0.5  |          2.29 |        4.58 |                    1.05 |
| s_learner  |            0.54 |            1.62 |          8.77 |        5.41 |                    1.11 |
| t_learner  |            0.76 |            1.65 |          9.48 |        5.73 |                    0.69 |
| tarnet_cpu |            3.69 |            7.57 |         24.94 |        3.3  |                    1.01 |
| tarnet_mps |            3.54 |            6.48 |         18.44 |        2.85 |                    1.06 |

Figure: `scaling_curves.png`.

## How to read this
- Diminishing returns are judged from the paired changes above, not from the point estimates: a step
  whose interval contains 0 is not distinguishable from no improvement on this validation subset.
- AIPW remains consistent only if the constant 0.85 propensity or the nuisance model is adequate;
  the IPW/AIPW discrepancy documented in `ipw_aipw_diagnostic.md` is still present and is shown in
  each rung's report. Using one shared estimator makes comparisons across models and sizes
  consistent but does not by itself make rankings unbiased.
- A single seed and fixed, untuned hyperparameters: changes may reflect the defaults' capacity
  (for example LightGBM's `min_child_samples`, TARNet's early stopping) as much as data
  limits.
