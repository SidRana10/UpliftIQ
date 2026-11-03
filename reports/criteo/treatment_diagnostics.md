# Criteo C1: treatment diagnostics (train split only)

**Verdict: benchmark_modeling_allowed.** No feature-based check failed: treatment looks independent of f0-f11 with adequate positivity. Benchmark modeling and sample-level, comparative evaluation may proceed under the publisher-supported constant-0.85 propensity assumption. Warnings (not failures): A1c_treatment_share_by_feature_bin, A1d_stability_across_row_blocks; the row-order finding is an ordering warning, not proof of heterogeneous propensities. The constant 0.85 treatment propensity is supported by the publisher (Diemert et al., arXiv:2111.10106: all source tests were subsampled to a global 85% treatment ratio, with treatment-independence validation reported), but it cannot be independently verified from the released file, which has no test identifier. Criteo's outcome-dependent privacy subsampling limits causal interpretation and prevents claims about original-population incrementality or ROI. This is a diagnostic judgment, not proof of identifiability.

Train rows: 8,946,939; overall treated share 0.8500.
Thresholds were fixed before looking at the results.

| Check | Value | Grade | Rule |
|---|---|---|---|
| A1a_standardized_mean_differences | 0.04903542082196859 | pass | max |SMD| <= 0.05 pass, <= 0.10 warn |
| A1b_treatment_from_features_auc | 0.008766809803921527 | pass | best held-out AUC - 0.5 <= 0.01 pass, <= 0.05 warn |
| A1c_treatment_share_by_feature_bin | 0.02227367240001954 | warn | max |bin share - overall| (bins n >= 1000) <= 0.01 pass, <= 0.03 warn |
| A1d_stability_across_row_blocks | 0.8500000950045596 | ordering_warning | max |block share - overall| <= 0.01 pass, else ordering_warning (does not fail the verdict: row order is not the randomization unit) |
| A2_positivity | {'min_bin_share': 0.845671546436762, 'max_bin_share': 0.8722737674045792, 'max_ipw_weight': 7.829245251189312} | pass | all bin shares, and the 1st-99th percentile of predicted propensity, within [0.05, 0.95] |

## Treatment-from-features classifier
Fitted on 1,000,000 train rows, evaluated on a disjoint 1,000,000-row train-derived set.
AUC: LightGBM 0.5078 (SE 0.0008), logistic 0.5088
(SE 0.0008); permuted-label control 0.5011.
Predicted propensity range 0.595 to 0.992
(median 0.849).

## Balance by feature
| feature   |   mean_treated |   mean_control |     smd |     ks |
|:----------|---------------:|---------------:|--------:|-------:|
| f0        |        19.6134 |        19.6498 | -0.0068 | 0.0089 |
| f1        |        10.0704 |        10.0678 |  0.0253 | 0.003  |
| f2        |         8.4464 |         8.4483 | -0.0064 | 0.0055 |
| f3        |         4.1691 |         4.2328 | -0.049  | 0.0154 |
| f4        |        10.3394 |        10.3364 |  0.0088 | 0.002  |
| f5        |         4.0265 |         4.0397 | -0.0317 | 0.0066 |
| f6        |        -4.1824 |        -4.0012 | -0.0401 | 0.0151 |
| f7        |         5.1056 |         5.0795 |  0.022  | 0.0066 |
| f8        |         3.9334 |         3.9347 | -0.0233 | 0.0082 |
| f9        |        16.0542 |        15.8824 |  0.0248 | 0.0074 |
| f10       |         5.3337 |         5.3319 |  0.0104 | 0.0021 |
| f11       |        -0.171  |        -0.1709 | -0.0057 | 0.0008 |

Treated share by feature bin: `treatment_balance.png`; per-bin and per-block tables: `treatment_diagnostics.json`.

## Row-order structure
The train rows consist of only **13 contiguous single-treatment runs** (12 transitions); a shuffled file would have millions. The file is ordered, which is why the row-block check raises an **ordering warning**. Feature means shift between runs (below), so the file is probably a concatenation of sources.

|       start |                n |   treatment |   mean_f0 |   mean_f6 |   mean_f9 |
|------------:|-----------------:|------------:|----------:|----------:|----------:|
| 0           |      3.56533e+06 |           1 |     21.62 |     -4.48 |     14.43 |
| 3.56533e+06 | 788429           |           0 |     20.7  |     -4.14 |     15.08 |
| 4.35376e+06 | 898478           |           1 |     17.03 |     -2.97 |     17.88 |
| 5.25224e+06 | 193012           |           0 |     17.57 |     -3.64 |     17.56 |
| 5.44525e+06 |      1.09546e+06 |           1 |     17.45 |     -4.2  |     18.03 |
| 6.5407e+06  | 181475           |           0 |     17.92 |     -3.65 |     17.55 |
| 6.72218e+06 |      1.30677e+06 |           1 |     18.58 |     -3.89 |     17    |
| 8.02895e+06 |  49048           |           0 |     21.25 |     -3.47 |     13.65 |
| 8.078e+06   |  74836           |           1 |     19.08 |     -5.9  |     13.59 |
| 8.15284e+06 |  13052           |           0 |     18.94 |     -5.85 |     13.59 |
| 8.16589e+06 | 144593           |           1 |     22.36 |     -4.47 |     13.51 |
| 8.31048e+06 | 117024           |           0 |     18.07 |     -4.26 |     17.15 |
| 8.42751e+06 | 519433           |           1 |     16.81 |     -4.63 |     18.56 |

This is a warning about row order, not proof of heterogeneous treatment propensities: the publishers state that all source tests were subsampled to a common 85% treatment ratio. Run boundaries are not used as experiment IDs, and treated shares of adjacent runs are not interpreted as test-level propensities. Do not rely on row order in any split or evaluation (the C0 split is a random, treatment-stratified permutation).

## Limitations (read before using any result)
- **Non-uniform subsampling.** Criteo sub-sampled the data non-uniformly for privacy; the sampling
  mechanism is unknown. Balance on f0-f11 does not rule out selection that depends on unobserved
  variables (advertiser, campaign, time) or on outcomes. These diagnostics see only the observed features.
- **Publisher-supported, unverifiable propensity.** The paper (Section 3, "Considerations to concatenate
  data from different tests") says CRITEO-UPLIFTv2 was subsampled to a global 85% treatment ratio, and
  Section 5.1 reports treatment-independence validation. The released file has no test identifier, so the
  constant 0.85 propensity is an assumption we use, not one we can verify here.
- **Outcome-dependent privacy subsampling.** It limits causal interpretation: estimates describe this
  released sample, and original-population incrementality or ROI claims are not supported.
- **Large n.** With millions of rows even trivial imbalances are statistically detectable, so the
  thresholds are practical (effect-size) rules, not significance tests.
- A pass supports benchmark modeling and sample-level, comparative evaluation within this file. It does
  not establish population-level incrementality or advertising ROI. A fail is reported as-is: no
  propensity model or alternative architecture is introduced automatically.
